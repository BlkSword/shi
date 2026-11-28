//! 对抗实验：通信干扰 + 假标记诱骗

use shi_syssim::{Swarm, SwarmConfig};

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    if v.len() < 2 {
        return (mean, 0.0);
    }
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn main() {
    let seeds: Vec<u64> = (0..8).collect();
    let base = SwarmConfig {
        n_drones: 80,
        n_targets: 20,
        n_channels: 20,
        command_post_until: Some(0),
        field_size: 200.0,
        sense_radius: 8.0,
        ..Default::default()
    };
    let ticks = 2000u64;

    // ============ 实验 1：通信干扰 ============
    println!("=== 实验 1: 通信干扰（信道数 × 干扰比例）===");
    println!("80 架无人机 / 20 目标 / 无指挥所 / tick 300 起干扰");
    println!();
    println!("{:<12} {:>18} {:>18} {:>18}", "干扰比例", "20 信道覆盖率", "10 信道覆盖率", "4 信道覆盖率");
    println!("{}", "-".repeat(70));

    for &jam in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let mut row = Vec::new();
        for &n_ch in &[20usize, 10, 4] {
            let mut covs = Vec::new();
            for &seed in &seeds {
                let cfg = SwarmConfig {
                    n_channels: n_ch,
                    jam_from: if jam > 0.0 { Some(300) } else { None },
                    jam_ratio: jam,
                    seed,
                    ..base.clone()
                };
                let mut s = Swarm::new(cfg);
                for _ in 0..ticks {
                    s.step();
                }
                covs.push(s.coverage());
            }
            let (m, s) = mean_std(&covs);
            row.push((m, s));
        }
        println!(
            "{:<12} {:>10.1}% ± {:>5.1} {:>10.1}% ± {:>5.1} {:>10.1}% ± {:>5.1}",
            format!("{:.0}%", jam * 100.0),
            row[0].0 * 100.0, row[0].1 * 100.0,
            row[1].0 * 100.0, row[1].1 * 100.0,
            row[2].0 * 100.0, row[2].1 * 100.0
        );
    }

    // ============ 实验 2：假标记诱骗 ============
    println!();
    println!("=== 实验 2: 假标记诱骗（信道数 × 假标记注入率）===");
    println!("{:<12} {:>18} {:>18} {:>16}", "假标记强度", "20 信道覆盖率", "4 信道覆盖率", "被骗次数");
    println!("{}", "-".repeat(68));

    for &rate in &[0.0, 0.05, 0.2, 0.5, 1.0] {
        let mut row20 = Vec::new();
        let mut row4 = Vec::new();
        let mut visits = Vec::new();
        for &seed in &seeds {
            for (n_ch, bucket) in [(20usize, 0), (4usize, 1)] {
                let cfg = SwarmConfig {
                    n_channels: n_ch,
                    decoy_rate: rate,
                    decoys_per_tick: if rate > 0.0 { 2 } else { 0 },
                    seed,
                    ..base.clone()
                };
                let mut s = Swarm::new(cfg);
                for _ in 0..ticks {
                    s.step();
                }
                if bucket == 0 {
                    row20.push(s.coverage());
                    visits.push(s.adversary.decoy_visits as f64);
                } else {
                    row4.push(s.coverage());
                }
            }
        }
        let (m20, s20) = mean_std(&row20);
        let (m4, s4) = mean_std(&row4);
        let (v, _) = mean_std(&visits);
        println!(
            "{:<12} {:>10.1}% ± {:>5.1} {:>10.1}% ± {:>5.1} {:>16.0}",
            format!("{:.0}%", rate * 100.0),
            m20 * 100.0, s20 * 100.0,
            m4 * 100.0, s4 * 100.0,
            v
        );
    }

    // ============ 实验 3：双重对抗 ============
    println!();
    println!("=== 实验 3: 干扰 + 假标记（双重对抗）===");
    println!("{:<24} {:>16} {:>16}", "场景", "覆盖率", "相对无对抗");
    println!("{}", "-".repeat(58));

    let scenarios: Vec<(&str, Option<u64>, f64, f64)> = vec![
        ("无对抗", None, 0.0, 0.0),
        ("仅干扰 50%", Some(300), 0.5, 0.0),
        ("仅假标记 20%", None, 0.0, 0.2),
        ("干扰 50% + 假标记 20%", Some(300), 0.5, 0.2),
    ];

    let mut baseline = 0.0;
    for (i, (name, jam_from, jam_ratio, decoy)) in scenarios.iter().enumerate() {
        let mut covs = Vec::new();
        for &seed in &seeds {
            let cfg = SwarmConfig {
                n_channels: 20,
                jam_from: *jam_from,
                jam_ratio: *jam_ratio,
                decoy_rate: *decoy,
                decoys_per_tick: if *decoy > 0.0 { 2 } else { 0 },
                seed,
                ..base.clone()
            };
            let mut s = Swarm::new(cfg);
            for _ in 0..ticks {
                s.step();
            }
            covs.push(s.coverage());
        }
        let (m, _) = mean_std(&covs);
        if i == 0 {
            baseline = m;
        }
        println!(
            "{:<24} {:>15.1}% {:>+15.1}pp",
            name,
            m * 100.0,
            (m - baseline) * 100.0
        );
    }
}
