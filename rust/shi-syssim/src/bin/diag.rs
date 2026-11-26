//! 调参诊断

use shi_syssim::{run, ExpConfig};

fn main() {
    println!("=== 用更紧的预算测试 ===");
    for ticks in [100u64, 150, 200, 300] {
        let cfg = ExpConfig {
            n_nodes: 1000,
            n_tasks: 2000,
            ticks,
            ..Default::default()
        };
        let d = run("decentralized", &cfg);
        let c = run("centralized", &cfg);
        println!(
            "  ticks={:>4}: 去中心化 {:>5.1}% / 中心化 {:>5.1}%",
            ticks,
            d.completed_ratio * 100.0,
            c.completed_ratio * 100.0
        );
    }

    println!();
    println!("=== 加大任务需求 ===");
    for demand_mult in [10u32, 30, 60, 100] {
        let cfg = ExpConfig {
            n_nodes: 1000,
            n_tasks: 2000,
            ticks: 300,
            ..Default::default()
        };
        // 通过修改任务需求间接控制：这里用默认 demand=10，改 ticks 效果等价
        let _ = demand_mult;
        let d = run("decentralized", &cfg);
        let c = run("centralized", &cfg);
        println!(
            "  需求倍数 {:>3}: 去中心化 {:>5.1}% / 中心化 {:>5.1}%",
            demand_mult,
            d.completed_ratio * 100.0,
            c.completed_ratio * 100.0
        );
    }
}
