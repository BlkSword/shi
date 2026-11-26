//! 超大规模测试：单机极限在哪里

use shi_syssim::{run, ExpConfig};
use std::time::Instant;

fn main() {
    println!("=== 超大规模测试（中心失效场景） ===");
    println!("{:<10} {:>12} {:>14} {:>14} {:>10}", "节点数", "耗时(s)", "去中心化", "中心化(失效)", "差异");
    println!("{}", "-".repeat(64));

    for &n_nodes in &[10000usize, 25000, 50000, 100000] {
        let cfg = ExpConfig {
            n_nodes,
            n_tasks: 100,
            ticks: 50,
            central_fail_at: Some(50),
            node_fail_prob: 0.0,
            arrival_rate: 12.0 * (n_nodes as f64 / 1000.0),
            seed: 0,
        };

        let t0 = Instant::now();
        let d = run("decentralized", &cfg);
        let dt = t0.elapsed().as_secs_f64();

        let c = run("centralized", &cfg);

        println!(
            "{:<10} {:>12.2} {:>13.1}% {:>13.1}% {:>+9.1}pp",
            n_nodes,
            dt,
            d.completed_ratio * 100.0,
            c.completed_ratio * 100.0,
            (d.completed_ratio - c.completed_ratio) * 100.0
        );
    }

    println!();
    println!("说明:");
    println!("- 每 tick 有 arrival_rate 个任务到达，需求 10/任务");
    println!("- 中心在 tick 50 失效，之后停止调度");
    println!("- 去中心化方案不依赖中心，持续工作");
}
