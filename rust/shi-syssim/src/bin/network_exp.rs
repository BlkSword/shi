//! 网络真实性对比实验
//!
//! 对比四组：
//! 1. 去中心化 + 完美共享内存（旧 PoC）
//! 2. 去中心化 + gossip（真实网络）
//! 3. 中心化单副本（旧对照）
//! 4. 中心化 3 副本 HA（真实 K8s 形态）

use shi_syssim::{
    run, ExpConfig, GossipView, HaConfig, HaControlPlane, Network, NetworkConfig, Partition,
};
use std::time::Instant;

fn main() {
    println!("=== 网络真实性对比实验 ===");
    println!();

    // ---------- 1. Gossip 收敛速度 vs 网络条件 ----------
    println!("[实验 1] Gossip 收敛速度 vs 网络条件");
    println!("{:<16} {:>12} {:>14} {:>12}", "网络", "收敛轮数", "最终离散度", "估计误差");
    println!("{}", "-".repeat(58));

    let n = 500;
    let loads: Vec<f64> = (0..n).map(|i| ((i * 37) % 100) as f64 / 100.0).collect();
    let true_mean: f64 = loads.iter().sum::<f64>() / n as f64;

    for (name, drop_prob) in [
        ("理想", 0.0),
        ("局域网", 0.001),
        ("跨区域", 0.01),
        ("恶劣", 0.05),
    ] {
        let mut view = GossipView::new(n, 3, 42);
        let mut converge_round = 0u32;
        // 用 drop 概率模拟「这条 gossip 消息丢了」
        let mut rng_drop = 0.0f64;
        for t in 0..300u64 {
            // 简单的确定性丢包：按比例跳过
            rng_drop += drop_prob;
            let skip = rng_drop >= 1.0;
            if skip {
                rng_drop -= 1.0;
            }
            let can_see = |_t: u64, i: usize, j: usize| -> bool {
                if skip {
                    return false;
                }
                // 按 drop_prob 的比例切断无向连边
                let edge_hash = (i * 7919 + j * 104729) % 1000;
                (edge_hash as f64 / 1000.0) >= drop_prob
            };
            view.gossip_round(&loads, 0.5, 0.0, t, can_see);
            if converge_round == 0 && view.convergence_error() < 0.05 {
                converge_round = t as u32 + 1;
            }
        }
        println!(
            "{:<16} {:>12} {:>14.4} {:>12.4}",
            name,
            if converge_round == 0 { "未收敛".to_string() } else { converge_round.to_string() },
            view.convergence_error(),
            (view.estimated_global_mean() - true_mean).abs()
        );
    }

    // ---------- 2. HA control plane 故障转移 ----------
    println!();
    println!("[实验 2] Control plane 故障转移时间");
    println!("{:<20} {:>14} {:>16} {:>12}", "配置", "副本数", "故障转移(tick)", "可容忍故障");
    println!("{}", "-".repeat(66));

    for (name, cfg) in [
        ("单副本(旧对照)", HaConfig::single()),
        ("三副本(K8s默认)", HaConfig::three()),
        ("五副本(高要求)", HaConfig::five()),
    ] {
        let mut ha = HaControlPlane::new(cfg.clone(), 0);
        ha.tick(0);
        let leader = ha.leader();
        if let Some(l) = leader {
            ha.fail_replica(l);
        }

        let mut transfer_time = None;
        for t in 1..200u64 {
            if ha.tick(t) && ha.leader().is_some() {
                transfer_time = Some(t);
                break;
            }
        }

        let tolerate = if cfg.n_replicas == 1 { 0 } else { (cfg.n_replicas - 1) / 2 };
        println!(
            "{:<20} {:>14} {:>16} {:>12}",
            name,
            cfg.n_replicas,
            transfer_time.map(|t| t.to_string()).unwrap_or("永不恢复".into()),
            tolerate
        );
    }

    // ---------- 3. 端到端：网络延迟 + 分区下两种方案 ----------
    println!();
    println!("[实验 3] 端到端对比（1000 节点，中心在 tick 150 失效）");
    println!("{:<28} {:>14} {:>16}", "配置", "完成率", "说明");
    println!("{}", "-".repeat(60));

    let base = ExpConfig {
        n_nodes: 1000,
        n_tasks: 100,
        ticks: 300,
        central_fail_at: None,
        node_fail_prob: 0.0,
        arrival_rate: 12.0,
        ha_replicas: 3,
        seed: 0,
    };

    let d = run("decentralized", &base);
    println!("{:<28} {:>13.1}% {:>16}", "去中心化(无故障)", d.completed_ratio * 100.0, "基线");

    let d_fail = run("decentralized", &ExpConfig { central_fail_at: Some(150), ..base.clone() });
    println!("{:<28} {:>13.1}% {:>16}", "去中心化(中心失效)", d_fail.completed_ratio * 100.0, "不受影响");

    let c = run("centralized", &base);
    println!("{:<28} {:>13.1}% {:>16}", "中心化单副本(无故障)", c.completed_ratio * 100.0, "基线");

    let c_fail = run("centralized", &ExpConfig { central_fail_at: Some(150), ..base.clone() });
    println!("{:<28} {:>13.1}% {:>16}", "中心化单副本(中心失效)", c_fail.completed_ratio * 100.0, "停止调度");

    // HA：3 副本，tick 150 时 leader 失效（真实故障模式）
    let ha_base = ExpConfig { ha_replicas: 3, ..base.clone() };
    let ha = run("ha", &ha_base);
    println!("{:<28} {:>13.1}% {:>16}", "中心化 3 副本 HA(无故障)", ha.completed_ratio * 100.0, "基线");

    let ha_fail = run("ha", &ExpConfig {
        central_fail_at: Some(150),
        ha_replicas: 3,
        ..base.clone()
    });
    println!("{:<28} {:>13.1}% {:>16}", "中心化 3 副本 HA(leader失效)", ha_fail.completed_ratio * 100.0, "选举后恢复");

    // 5 副本
    let ha5 = run("ha", &ExpConfig { ha_replicas: 5, ..base.clone() });
    println!("{:<28} {:>13.1}% {:>16}", "中心化 5 副本 HA(无故障)", ha5.completed_ratio * 100.0, "基线");

    let ha5_fail = run("ha", &ExpConfig {
        central_fail_at: Some(150),
        ha_replicas: 5,
        ..base.clone()
    });
    println!("{:<28} {:>13.1}% {:>16}", "中心化 5 副本 HA(leader失效)", ha5_fail.completed_ratio * 100.0, "选举后恢复");

    println!();
    println!("说明：");
    println!("- HA 在 leader 失效后会选举新 leader（约 10 tick），之后继续调度。");
    println!("- 去中心化没有中心，也不需要选举——故障期间性能完全不降。");
    println!("- 但 HA 的代价是 control plane 的资源开销（3-5 倍）。");

    // ---------- 4. 网络分区影响 ----------
    println!();
    println!("[实验 4] 网络分区下的 gossip 视图陈旧度");
    println!("场景：左半区负载 0.2，右半区负载 0.8（真实系统里常见：机架间负载不均）");
    println!("{:<26} {:>14} {:>14} {:>14}", "分区场景", "视图离散度", "左半估计", "右半估计");
    println!("{}", "-".repeat(72));

    for (name, split_x) in [
        ("无分区", f64::NAN),
        ("分区在中间(250)", 250.0),
        ("分区偏左(125)", 125.0),
        ("分区偏右(375)", 375.0),
    ] {
        let n = 500;
        let mut view = GossipView::new(n, 3, 42);
        // 左半 0.2，右半 0.8 —— 分区后两侧会各自收敛到自己的局部均值
        let loads: Vec<f64> = (0..n)
            .map(|i| if i < n / 2 { 0.2 } else { 0.8 })
            .collect();

        for t in 0..100u64 {
            let can_see = |_t: u64, i: usize, j: usize| -> bool {
                if split_x.is_nan() {
                    return true;
                }
                !(((i as f64) < split_x) != ((j as f64) < split_x))
            };
            view.gossip_round(&loads, 0.5, 0.0, t, can_see);
        }

        let left_mean: f64 = view.mean_load_estimate[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
        let right_mean: f64 = view.mean_load_estimate[n / 2..].iter().sum::<f64>() / (n / 2) as f64;

        println!(
            "{:<26} {:>14.4} {:>14.4} {:>14.4}",
            name,
            view.convergence_error(),
            left_mean,
            right_mean
        );
    }
    println!();
    println!("含义：分区时两侧各自收敛到自己的局部均值（0.2 / 0.8），");
    println!("全局真实均值是 0.5，但任何一侧都看不到真实全局值——");
    println!("这是分布式系统的根本限制，gossip 无法绕过。");

    let _ = Network::new(NetworkConfig::lan(), 0);
    let _ = Partition { start: 0, end: None, split_x: 50.0 };
    let _ = Instant::now();
}
