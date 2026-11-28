//! 时间对齐后的四类攻击对比
//!
//! 修正上一版的时间混淆：重放改用预设幽灵位置，从 tick 0 开始生效。

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
    let seeds: Vec<u64> = (0..10).collect();
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

    println!("基准（无攻击）: {:.1}%", baseline * 100.0);
    println!();
    println!("=== 等预算对比（每 tick 2 单位，重放已时间对齐）===");
    println!("{:<20} {:>18} {:>12} {:>10}", "攻击", "覆盖率", "损失(pp)", "相对损失");
    println!("{}", "-".repeat(64));

    // 注入：2 个随机假标记
    let mut v = Vec::new();
    for &s in &seeds {
        v.push(run(SwarmConfig { decoy_rate: 1.0, decoys_per_tick: 2, seed: s, ..base.clone() }, ticks));
    }
    let (m, sd) = mean_std(&v);
    println!("{:<20} {:>9.1}% ± {:>5.1} {:>+11.1} {:>+9.1}%", "注入（伪造）", m*100.0, sd*100.0, (m-baseline)*100.0, (m-baseline)/baseline*100.0);
    let inject_loss = m - baseline;

    // 淹没：2 个随机噪声
    let mut v = Vec::new();
    for &s in &seeds {
        v.push(run(SwarmConfig { flood_marks: 2, seed: s, ..base.clone() }, ticks));
    }
    let (m, sd) = mean_std(&v);
    println!("{:<20} {:>9.1}% ± {:>5.1} {:>+11.1} {:>+9.1}%", "淹没（稀释）", m*100.0, sd*100.0, (m-baseline)*100.0, (m-baseline)/baseline*100.0);

    // 重放·时间对齐：预设幽灵位置，从 tick 0 生效
    let mut v = Vec::new();
    for &s in &seeds {
        v.push(run(SwarmConfig { replay_rate: 0.1, replay_fixed: true, seed: s, ..base.clone() }, ticks));
    }
    let (m, sd) = mean_std(&v);
    println!("{:<20} {:>9.1}% ± {:>5.1} {:>+11.1} {:>+9.1}%", "重放（时间对齐）", m*100.0, sd*100.0, (m-baseline)*100.0, (m-baseline)/baseline*100.0);
    let replay_aligned = m;

    // 重放·经典（依赖摧毁事件）
    let mut v = Vec::new();
    for &s in &seeds {
        v.push(run(SwarmConfig { replay_rate: 0.5, seed: s, ..base.clone() }, ticks));
    }
    let (m, sd) = mean_std(&v);
    println!("{:<20} {:>9.1}% ± {:>5.1} {:>+11.1} {:>+9.1}%", "重放（经典·依赖摧毁）", m*100.0, sd*100.0, (m-baseline)*100.0, (m-baseline)/baseline*100.0);

    // 拒绝：压制 50% 信道
    let mut v = Vec::new();
    for &s in &seeds {
        v.push(run(SwarmConfig { jam_from: Some(200), jam_ratio: 0.5, seed: s, ..base.clone() }, ticks));
    }
    let (m, sd) = mean_std(&v);
    println!("{:<20} {:>9.1}% ± {:>5.1} {:>+11.1} {:>+9.1}%", "拒绝（删除）", m*100.0, sd*100.0, (m-baseline)*100.0, (m-baseline)/baseline*100.0);

    // 判定
    println!();
    println!("=== 判定 ===");
    println!(
        "时间对齐后，重放损失从 -10.6pp 变成 {:+.1}pp（提升 {:.1}pp）",
        (replay_aligned - baseline) * 100.0,
        ((replay_aligned - baseline) - (-0.106)) * 100.0
    );
    println!(
        "注入 vs 重放（同预算）: {:.1}pp vs {:.1}pp",
        inject_loss * 100.0,
        (replay_aligned - baseline) * 100.0
    );
}
