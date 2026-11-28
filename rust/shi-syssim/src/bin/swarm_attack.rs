//! 协调通道污染：四种攻击的分类学 + 三种防御
//!
//! 核心问题
//! --------
//! 当攻击者能写入你的协调介质时，哪种攻击最有效？哪种防御最划算？
//!
//! 四种攻击（按等强度归一：都是"每 tick 注入/压制 N 单位"）
//! --------------------------------------------------------
//! - 拒绝（Denial）   ：压制信道，删信息
//! - 注入（Injection）：随机位置写假标记，伪造信息
//! - 重放（Replay）   ：在已摧毁目标处重写标记，伪造"过期真相"
//! - 淹没（Flooding） ：高频随机噪声，稀释信息

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

fn run(cfg: SwarmConfig, ticks: u64) -> f64 {
    let mut s = Swarm::new(cfg);
    for _ in 0..ticks {
        s.step();
    }
    s.coverage()
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
    let ticks = 1500u64;

    // 基准
    let baseline = {
        let mut v = Vec::new();
        for &seed in &seeds {
            v.push(run(SwarmConfig { seed, ..base.clone() }, ticks));
        }
        mean_std(&v).0
    };

    println!("=== 协调通道污染：四种攻击对比 ===");
    println!(
        "基准（无攻击）: {:.1}%   80 架无人机 / 20 目标 / 20 信道",
        baseline * 100.0
    );
    println!();
    println!(
        "{:<16} {:>8} {:>18} {:>12} {:>14}",
        "攻击类型", "强度", "覆盖率", "损失(pp)", "相对损失"
    );
    println!("{}", "-".repeat(72));

    // 强度档位：轻度 / 中度 / 重度
    let levels = [
        ("轻度", 0.25, 1usize),
        ("中度", 0.5, 2),
        ("重度", 0.75, 4),
    ];

    let mut results: Vec<(String, f64, f64)> = Vec::new();

    for (name, ratio, n_marks) in levels {
        // ---- 拒绝：压制 ratio 比例的信道 ----
        let mut v = Vec::new();
        for &seed in &seeds {
            v.push(run(
                SwarmConfig {
                    jam_from: Some(200),
                    jam_ratio: ratio,
                    seed,
                    ..base.clone()
                },
                ticks,
            ));
        }
        let (m, _) = mean_std(&v);
        println!(
            "{:<16} {:>8} {:>9.1}% ± {:>5.1} {:>+11.1} {:>+13.1}%",
            "拒绝（干扰）",
            name,
            m * 100.0,
            mean_std(&v).1 * 100.0,
            (m - baseline) * 100.0,
            (m - baseline) / baseline * 100.0
        );
        results.push(("拒绝".into(), ratio, m));

        // ---- 注入：随机假标记 ----
        let mut v = Vec::new();
        for &seed in &seeds {
            v.push(run(
                SwarmConfig {
                    decoy_rate: ratio,
                    decoys_per_tick: n_marks,
                    seed,
                    ..base.clone()
                },
                ticks,
            ));
        }
        let (m, s) = mean_std(&v);
        println!(
            "{:<16} {:>8} {:>9.1}% ± {:>5.1} {:>+11.1} {:>+13.1}%",
            "注入（假标记）",
            name,
            m * 100.0,
            s * 100.0,
            (m - baseline) * 100.0,
            (m - baseline) / baseline * 100.0
        );
        results.push(("注入".into(), ratio, m));

        // ---- 重放：已摧毁目标处重写 ----
        let mut v = Vec::new();
        for &seed in &seeds {
            v.push(run(
                SwarmConfig {
                    replay_rate: ratio,
                    seed,
                    ..base.clone()
                },
                ticks,
            ));
        }
        let (m, s) = mean_std(&v);
        println!(
            "{:<16} {:>8} {:>9.1}% ± {:>5.1} {:>+11.1} {:>+13.1}%",
            "重放（过期真相）",
            name,
            m * 100.0,
            s * 100.0,
            (m - baseline) * 100.0,
            (m - baseline) / baseline * 100.0
        );
        results.push(("重放".into(), ratio, m));

        // ---- 淹没：高频随机噪声 ----
        let mut v = Vec::new();
        for &seed in &seeds {
            v.push(run(
                SwarmConfig {
                    flood_marks: n_marks * 4, // 噪声需要更大量才有效
                    seed,
                    ..base.clone()
                },
                ticks,
            ));
        }
        let (m, s) = mean_std(&v);
        println!(
            "{:<16} {:>8} {:>9.1}% ± {:>5.1} {:>+11.1} {:>+13.1}%",
            "淹没（噪声）",
            name,
            m * 100.0,
            s * 100.0,
            (m - baseline) * 100.0,
            (m - baseline) / baseline * 100.0
        );
        results.push(("淹没".into(), ratio, m));

        println!();
    }

    // ============ 防御效果 ============
    println!("=== 三种防御 vs 中度注入（最有效的攻击）===");
    println!();

    // 先测中度注入的基线
    let mut v = Vec::new();
    for &seed in &seeds {
        v.push(run(
            SwarmConfig {
                decoy_rate: 0.5,
                decoys_per_tick: 2,
                seed,
                ..base.clone()
            },
            ticks,
        ));
    }
    let undefended = mean_std(&v).0;

    println!(
        "{:<22} {:>18} {:>14} {:>12}",
        "防御策略", "覆盖率", "相对无防御", "恢复比例"
    );
    println!("{}", "-".repeat(70));

    println!(
        "{:<22} {:>9.1}% ± {:>5.1} {:>14} {:>12}",
        "无防御",
        undefended * 100.0,
        mean_std(&v).1 * 100.0,
        "—",
        "—"
    );

    // 防御 1：信源认证（敌方标记全部拒绝）
    let mut v = Vec::new();
    for &seed in &seeds {
        v.push(run(
            SwarmConfig {
                decoy_rate: 0.5,
                decoys_per_tick: 2,
                auth_ratio: 1.0,
                seed,
                ..base.clone()
            },
            ticks,
        ));
    }
    let (auth, auth_s) = mean_std(&v);
    let recover = if (baseline - undefended).abs() > 1e-6 {
        (auth - undefended) / (baseline - undefended) * 100.0
    } else {
        0.0
    };
    println!(
        "{:<22} {:>9.1}% ± {:>5.1} {:>+13.1}pp {:>11.0}%",
        "信源认证（完美）",
        auth * 100.0,
        auth_s * 100.0,
        (auth - undefended) * 100.0,
        recover
    );

    // 防御 2：快速衰减（标记寿命缩短）
    for decay in [0.99, 0.95, 0.90] {
        let mut v = Vec::new();
        for &seed in &seeds {
            v.push(run(
                SwarmConfig {
                    decoy_rate: 0.5,
                    decoys_per_tick: 2,
                    mark_decay: Some(decay),
                    seed,
                    ..base.clone()
                },
                ticks,
            ));
        }
        let (m, s) = mean_std(&v);
        let rec = if (baseline - undefended).abs() > 1e-6 {
            (m - undefended) / (baseline - undefended) * 100.0
        } else {
            0.0
        };
        println!(
            "{:<22} {:>9.1}% ± {:>5.1} {:>+13.1}pp {:>11.0}%",
            format!("快速衰减（{:.2}）", decay),
            m * 100.0,
            s * 100.0,
            (m - undefended) * 100.0,
            rec
        );
    }

    // 防御 3：短衰减对无攻击时的代价
    println!();
    println!("=== 快速衰减的副作用（无攻击时）===");
    for decay in [0.998, 0.99, 0.95, 0.90] {
        let mut v = Vec::new();
        for &seed in &seeds {
            v.push(run(
                SwarmConfig {
                    mark_decay: Some(decay),
                    seed,
                    ..base.clone()
                },
                ticks,
            ));
        }
        let (m, s) = mean_std(&v);
        println!(
            "  衰减 {:.3}: {:.1}% ± {:.1}  ({:+.1}pp)",
            decay,
            m * 100.0,
            s * 100.0,
            (m - baseline) * 100.0
        );
    }

    // ============ 结论 ============
    println!();
    println!("=== 判定 ===");

    // 找最有效的攻击（重度档）
    let heavy: Vec<_> = results.iter().filter(|(_, r, _)| (*r - 0.75).abs() < 1e-9).collect();
    if let Some((name, _, cov)) = heavy.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap()) {
        println!(
            "最有效的攻击（重度档）: {} （覆盖率 {:.1}%）",
            name,
            cov * 100.0
        );
    }

    if (baseline - undefended).abs() > 1e-6 {
        println!(
            "信源认证能恢复 {:.0}% 的被注入损失",
            (auth - undefended) / (baseline - undefended) * 100.0
        );
    }
}
