//! 势 / Shi 命令行工具

use shi_core::{AgentState, Engine, PheromoneField, ChannelSpec};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("bench");

    match cmd {
        "bench" => bench(),
        "verify" => verify(),
        "syssim" => syssim(),
        _ => {
            eprintln!("用法: shi <bench|verify|syssim>");
            std::process::exit(1);
        }
    }
}

/// 性能基准：N 个体 x M tick
fn bench() {
    for &n_agents in &[1_000usize, 10_000, 100_000] {
        let n_ticks = 1000u64;

        let world = PheromoneField::new(200, 200, vec![ChannelSpec::new(1.0, 0.0)]);
        let mut agents = AgentState::new(n_agents);
        let mut rng = Pcg64Mcg::seed_from_u64(0);
        agents.place_random(200.0, 200.0, &mut rng);

        let stepper = |ag: &mut AgentState, w: &mut PheromoneField, _tick: u64| {
            for i in 0..ag.n {
                let (gx, gy) = w.gradient(ag.x[i], ag.y[i], 0);
                let norm = (gx * gx + gy * gy).sqrt();
                if norm > 1e-12 {
                    ag.vx[i] = 0.8 * gx / norm;
                    ag.vy[i] = 0.8 * gy / norm;
                } else {
                    ag.vx[i] = 0.0;
                    ag.vy[i] = 0.0;
                }
                ag.x[i] = (ag.x[i] + ag.vx[i]).clamp(0.0, 199.0);
                ag.y[i] = (ag.y[i] + ag.vy[i]).clamp(0.0, 199.0);
            }
        };

        let mut engine = Engine::new(world, agents, 0, stepper, 0);

        let t0 = Instant::now();
        engine.run(n_ticks);
        let dt = t0.elapsed().as_secs_f64();

        let agent_ticks = n_agents as f64 * n_ticks as f64;
        println!(
            "  {:>7} agents x {:>5} ticks: {:>8.3}s  ({:.2}M agent-ticks/s)",
            n_agents,
            n_ticks,
            dt,
            agent_ticks / dt / 1e6
        );
    }
}

/// 确定性验证：同种子跑 N 次，哈希序列必须一致
fn verify() {
    let runs = 20;
    let n_agents = 1000;
    let n_ticks = 500u64;

    let build = || {
        let world = PheromoneField::new(100, 100, vec![ChannelSpec::new(0.98, 0.0)]);
        let mut agents = AgentState::new(n_agents);
        let mut rng = Pcg64Mcg::seed_from_u64(42);
        agents.place_random(100.0, 100.0, &mut rng);

        let stepper = |ag: &mut AgentState, w: &mut PheromoneField, _tick: u64| {
            for i in 0..ag.n {
                let (gx, gy) = w.gradient(ag.x[i], ag.y[i], 0);
                let norm = (gx * gx + gy * gy).sqrt();
                if norm > 1e-12 {
                    ag.vx[i] = 0.5 * gx / norm;
                    ag.vy[i] = 0.5 * gy / norm;
                }
                ag.x[i] = (ag.x[i] + ag.vx[i]).clamp(0.0, 99.0);
                ag.y[i] = (ag.y[i] + ag.vy[i]).clamp(0.0, 99.0);
                w.deposit(ag.x[i], ag.y[i], 0, 0.01);
            }
        };

        Engine::new(world, agents, 42, stepper, 50)
    };

    let mut baseline: Option<Vec<u64>> = None;
    let mut mismatches = 0;

    for i in 0..runs {
        let mut e = build();
        e.run(n_ticks);
        let seq = e.hash_sequence();

        match &baseline {
            None => {
                println!("  基准哈希序列: {} 个", seq.len());
                println!("  首个: {:016x}", seq[0]);
                println!("  末个: {:016x}", seq[seq.len() - 1]);
                baseline = Some(seq);
            }
            Some(b) => {
                if &seq != b {
                    mismatches += 1;
                    println!("  [FAIL] 第 {} 次运行哈希不一致", i + 1);
                }
            }
        }
    }

    println!();
    if mismatches == 0 {
        println!("[PASS] {} 次运行，哈希序列 100% 一致", runs);
    } else {
        println!("[FAIL] {}/{} 次运行哈希不一致", mismatches, runs);
        std::process::exit(1);
    }
}

/// 分布式系统自愈实验
fn syssim() {
    use shi_syssim::{run, ExpConfig};

    println!("=== 分布式系统自愈：去中心化 vs 中心化 ===");
    println!("模型：1000 节点、持续任务到达（每 tick 15 个任务）、需求 10/任务");
    println!();

    // 先测容量上限
    let base = ExpConfig {
        n_nodes: 1000,
        n_tasks: 100,
        ticks: 300,
        central_fail_at: None,
        node_fail_prob: 0.0,
        arrival_rate: 15.0,
        ha_replicas: 3,
        seed: 0,
    };

    // 场景 1：正常运行
    println!("[场景 1] 正常运行（无故障）");
    let d = run("decentralized", &base);
    let c = run("centralized", &base);
    println!(
        "  去中心化: 完成率 {:>5.1}%  任务 {} 个  负载均衡 {:.3}",
        d.completed_ratio * 100.0,
        d.total_tasks,
        d.load_balance
    );
    println!(
        "  中心化:   完成率 {:>5.1}%  任务 {} 个  负载均衡 {:.3}",
        c.completed_ratio * 100.0,
        c.total_tasks,
        c.load_balance
    );
    println!();

    // 场景 2：中心失效
    println!("[场景 2] 中心在 tick 150 失效");
    let cfg2 = ExpConfig {
        central_fail_at: Some(150),
        ..base.clone()
    };
    let d2 = run("decentralized", &cfg2);
    let c2 = run("centralized", &cfg2);
    println!(
        "  去中心化: 完成率 {:>5.1}%",
        d2.completed_ratio * 100.0
    );
    println!(
        "  中心化:   完成率 {:>5.1}%  （中心失效后停止调度）",
        c2.completed_ratio * 100.0
    );
    println!(
        "  -> 差异 {:+.1} 个百分点",
        (d2.completed_ratio - c2.completed_ratio) * 100.0
    );
    println!();

    // 场景 3：节点故障
    println!("[场景 3] 持续节点故障（每 50 tick 失效 2%）");
    let cfg3 = ExpConfig {
        node_fail_prob: 0.02,
        ..base.clone()
    };
    let d3 = run("decentralized", &cfg3);
    let c3 = run("centralized", &cfg3);
    println!(
        "  去中心化: 完成率 {:>5.1}%  失效节点 {}",
        d3.completed_ratio * 100.0,
        d3.failed_nodes
    );
    println!(
        "  中心化:   完成率 {:>5.1}%  失效节点 {}",
        c3.completed_ratio * 100.0,
        c3.failed_nodes
    );
    println!();

    // 场景 4：中心失效 + 节点故障
    println!("[场景 4] 中心失效 + 持续节点故障（最坏情况）");
    let cfg4 = ExpConfig {
        central_fail_at: Some(150),
        node_fail_prob: 0.02,
        ..base.clone()
    };
    let d4 = run("decentralized", &cfg4);
    let c4 = run("centralized", &cfg4);
    println!(
        "  去中心化: 完成率 {:>5.1}%",
        d4.completed_ratio * 100.0
    );
    println!(
        "  中心化:   完成率 {:>5.1}%",
        c4.completed_ratio * 100.0
    );
    println!(
        "  差异: {:+.1} 个百分点",
        (d4.completed_ratio - c4.completed_ratio) * 100.0
    );
}
