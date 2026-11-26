//! 多种子统计实验：验证结论的稳健性

use shi_syssim::{run, ExpConfig};
use std::time::Instant;

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

fn main() {
    let seeds: Vec<u64> = (0..10).collect();
    let base = ExpConfig {
        n_nodes: 1000,
        n_tasks: 100,
        ticks: 300,
        central_fail_at: None,
        node_fail_prob: 0.0,
        arrival_rate: 15.0,
        seed: 0,
    };

    println!("=== 多种子统计（10 seeds, 1000 节点, 300 ticks） ===");
    println!();

    let scenarios: Vec<(&str, ExpConfig)> = vec![
        ("正常运行", base.clone()),
        ("中心失效", ExpConfig { central_fail_at: Some(150), ..base.clone() }),
        ("节点故障", ExpConfig { node_fail_prob: 0.02, ..base.clone() }),
        ("失效+故障", ExpConfig { central_fail_at: Some(150), node_fail_prob: 0.02, ..base.clone() }),
    ];

    println!("{:<14} {:>20} {:>20} {:>10}", "场景", "去中心化", "中心化", "差异");
    println!("{}", "-".repeat(68));

    for (name, cfg) in &scenarios {
        let mut dec = Vec::new();
        let mut cen = Vec::new();

        for &seed in &seeds {
            let c = ExpConfig { seed, ..cfg.clone() };
            dec.push(run("decentralized", &c).completed_ratio);
            cen.push(run("centralized", &c).completed_ratio);
        }

        let (dm, ds) = mean_std(&dec);
        let (cm, cs) = mean_std(&cen);
        println!(
            "{:<14} {:>8.1}% ± {:<5.1} {:>8.1}% ± {:<5.1} {:>+8.1}pp",
            name,
            dm * 100.0,
            ds * 100.0,
            cm * 100.0,
            cs * 100.0,
            (dm - cm) * 100.0
        );
    }

    println!();
    println!("=== 规模扩展测试（无故障 vs 中心失效） ===");
    println!("{:<10} {:>12} {:>14} {:>14} {:>10}", "节点数", "耗时(ms)", "去中心化", "中心化(失效)", "差异");
    println!("{}", "-".repeat(64));

    for &n_nodes in &[1000usize, 5000, 10000, 20000] {
        let cfg = ExpConfig {
            n_nodes,
            n_tasks: 100,
            ticks: 200,
            central_fail_at: Some(100),
            node_fail_prob: 0.0,
            arrival_rate: 12.0 * (n_nodes as f64 / 1000.0),
            seed: 0,
        };

        let t0 = Instant::now();
        let d = run("decentralized", &cfg);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;

        let c = run("centralized", &cfg);

        println!(
            "{:<10} {:>12.1} {:>13.1}% {:>13.1}% {:>+9.1}pp",
            n_nodes,
            dt,
            d.completed_ratio * 100.0,
            c.completed_ratio * 100.0,
            (d.completed_ratio - c.completed_ratio) * 100.0
        );
    }
}
