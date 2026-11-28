//! 公平性检验：四类攻击在相同"标记预算"下的对比

use shi_syssim::{Swarm, SwarmConfig};

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    if v.len() < 2 { return (mean, 0.0); }
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn run(cfg: SwarmConfig, ticks: u64) -> f64 {
    let mut s = Swarm::new(cfg);
    for _ in 0..ticks { s.step(); }
    s.coverage()
}

fn main() {
    let seeds: Vec<u64> = (0..8).collect();
    let base = SwarmConfig {
        n_drones: 80, n_targets: 20, n_channels: 20,
        command_post_until: Some(0),
        field_size: 200.0, sense_radius: 8.0,
        ..Default::default()
    };
    let ticks = 1500u64;

    let baseline = {
        let mut v = Vec::new();
        for &s in &seeds { v.push(run(SwarmConfig { seed: s, ..base.clone() }, ticks)); }
        mean_std(&v).0
    };

    println!("基准: {:.1}%", baseline * 100.0);
    println!();
    println!("=== 等预算对比（每 tick 2 个标记）===");
    println!("{:<18} {:>18} {:>12}", "攻击", "覆盖率", "损失(pp)");
    println!("{}", "-".repeat(50));

    // 注入：2 个随机假标记
    let mut v = Vec::new();
    for &s in &seeds {
        v.push(run(SwarmConfig { decoy_rate: 1.0, decoys_per_tick: 2, seed: s, ..base.clone() }, ticks));
    }
    let (m, sd) = mean_std(&v);
    println!("{:<18} {:>9.1}% ± {:>5.1} {:>+11.1}", "注入（2 标记）", m*100.0, sd*100.0, (m-baseline)*100.0);

    // 淹没：2 个随机噪声
    let mut v = Vec::new();
    for &s in &seeds {
        v.push(run(SwarmConfig { flood_marks: 2, seed: s, ..base.clone() }, ticks));
    }
    let (m, sd) = mean_std(&v);
    println!("{:<18} {:>9.1}% ± {:>5.1} {:>+11.1}", "淹没（2 标记）", m*100.0, sd*100.0, (m-baseline)*100.0);

    // 淹没阶梯：看多少噪声才起作用
    println!();
    println!("=== 淹没攻击的剂量反应 ===");
    println!("{:<12} {:>18} {:>12}", "噪声/ tick", "覆盖率", "损失(pp)");
    println!("{}", "-".repeat(44));
    for n in [1usize, 2, 4, 8, 16, 32] {
        let mut v = Vec::new();
        for &s in &seeds {
            v.push(run(SwarmConfig { flood_marks: n, seed: s, ..base.clone() }, ticks));
        }
        let (m, sd) = mean_std(&v);
        println!("{:<12} {:>9.1}% ± {:>5.1} {:>+11.1}", n, m*100.0, sd*100.0, (m-baseline)*100.0);
    }

    // 注入阶梯对照
    println!();
    println!("=== 注入攻击的剂量反应（对照）===");
    println!("{:<12} {:>18} {:>12}", "假标记/tick", "覆盖率", "损失(pp)");
    println!("{}", "-".repeat(44));
    for n in [1usize, 2, 4, 8, 16, 32] {
        let mut v = Vec::new();
        for &s in &seeds {
            v.push(run(SwarmConfig { decoy_rate: 1.0, decoys_per_tick: n, seed: s, ..base.clone() }, ticks));
        }
        let (m, sd) = mean_std(&v);
        println!("{:<12} {:>9.1}% ± {:>5.1} {:>+11.1}", n, m*100.0, sd*100.0, (m-baseline)*100.0);
    }
}
