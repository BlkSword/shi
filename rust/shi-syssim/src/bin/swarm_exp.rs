//! 蜂群协调实验：通信容量 × 规模 × 指挥所有无
//!
//! 三个假设
//! --------
//! H-A: 信道数 >= 目标数时覆盖率显著更高（信息容量决定上限）
//! H-B: 固定信道数下，蜂群规模存在最优点（密度上限）
//! H-C: 指挥所失效后的性能损失，随信道数增加而减小

use shi_syssim::{DroneState, Swarm, SwarmConfig};
use std::time::Instant;

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    if v.len() < 2 {
        return (mean, 0.0);
    }
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

struct RunOutcome {
    coverage: f64,
    destroyed: usize,
    attacks: u32,
    /// 达到 50% 覆盖率所需 tick（未达到 = ticks）
    t_half: u64,
    /// 活跃（未损毁）无人机数
    active: usize,
}

fn run_one(cfg: SwarmConfig, ticks: u64) -> RunOutcome {
    let mut s = Swarm::new(cfg);
    let mut t_half = ticks;
    let half = s.targets.len() / 2;

    for t in 0..ticks {
        s.step();
        if t_half == ticks && s.destroyed_count() >= half {
            t_half = t + 1;
        }
    }

    let active = s
        .drones
        .iter()
        .filter(|d| d.state != DroneState::Destroyed)
        .count();

    RunOutcome {
        coverage: s.coverage(),
        destroyed: s.destroyed_count(),
        attacks: s.total_attacks,
        t_half,
        active,
    }
}

fn main() {
    let seeds: Vec<u64> = (0..8).collect();
    let n_targets = 20usize;
    let field = 200.0;
    let sense = 8.0;
    let ticks = 2000u64;

    // ============ H-A: 信道数 vs 覆盖率 ============
    println!("=== H-A: 通信信道数 vs 目标覆盖率（蜂群 40，目标 20，无指挥所）===");
    println!("{:<10} {:>18} {:>12} {:>14}", "信道数", "覆盖率", "摧毁数", "50%用时(tick)");
    println!("{}", "-".repeat(58));

    let channels_list = [1usize, 2, 4, 10, 20, 40];
    let mut h_a = Vec::new();
    for &n_ch in &channels_list {
        let mut covs = Vec::new();
        let mut des = Vec::new();
        let mut ths = Vec::new();
        for &seed in &seeds {
            let r = run_one(
                SwarmConfig {
                    n_drones: 40,
                    n_targets,
                    n_channels: n_ch,
                    command_post_until: Some(0),
                    field_size: field,
                    sense_radius: sense,
                    seed,
                    ..Default::default()
                },
                ticks,
            );
            covs.push(r.coverage);
            des.push(r.destroyed as f64);
            ths.push(r.t_half as f64);
        }
        let (cm, cs) = mean_std(&covs);
        let (dm, _) = mean_std(&des);
        let (tm, _) = mean_std(&ths);
        h_a.push((n_ch, cm));
        println!(
            "{:<10} {:>9.1}% ± {:>5.1} {:>12.1} {:>14.0}",
            n_ch,
            cm * 100.0,
            cs * 100.0,
            dm,
            tm
        );
    }

    // ============ H-B: 蜂群规模 vs 覆盖率（固定信道） ============
    println!();
    println!("=== H-B: 蜂群规模 vs 覆盖率（信道=1 严重冲突 / 信道=20 无冲突）===");
    println!("{:<10} {:>20} {:>20}", "蜂群规模", "1 信道", "20 信道");
    println!("{}", "-".repeat(54));

    let swarm_sizes = [10usize, 20, 40, 80, 160, 320];
    let mut h_b = Vec::new();
    for &n_d in &swarm_sizes {
        let mut row = Vec::new();
        for &n_ch in &[1usize, 20] {
            let mut covs = Vec::new();
            for &seed in &seeds {
                let r = run_one(
                    SwarmConfig {
                        n_drones: n_d,
                        n_targets,
                        n_channels: n_ch,
                        command_post_until: Some(0),
                        field_size: field,
                        sense_radius: sense,
                        seed,
                        ..Default::default()
                    },
                    ticks,
                );
                covs.push(r.coverage);
            }
            let (m, s) = mean_std(&covs);
            row.push((m, s));
        }
        h_b.push((n_d, row[0].0, row[1].0));
        println!(
            "{:<10} {:>9.1}% ± {:>5.1} {:>9.1}% ± {:>5.1}",
            n_d,
            row[0].0 * 100.0,
            row[0].1 * 100.0,
            row[1].0 * 100.0,
            row[1].1 * 100.0
        );
    }

    // ============ H-C: 指挥所失效的代价 vs 信道数 ============
    println!();
    println!("=== H-C: 指挥所失效的代价 vs 信道数（蜂群 40，目标 20）===");
    println!("{:<10} {:>16} {:>16} {:>12}", "信道数", "有指挥所", "无指挥所", "损失(pp)");
    println!("{}", "-".repeat(58));

    for &n_ch in &channels_list {
        let mut with_cp = Vec::new();
        let mut without_cp = Vec::new();
        for &seed in &seeds {
            let base = SwarmConfig {
                n_drones: 40,
                n_targets,
                n_channels: n_ch,
                command_post_until: None,
                field_size: field,
                sense_radius: sense,
                seed,
                ..Default::default()
            };
            with_cp.push(run_one(base.clone(), ticks).coverage);
            without_cp.push(
                run_one(
                    SwarmConfig {
                        command_post_until: Some(0),
                        ..base
                    },
                    ticks,
                )
                .coverage,
            );
        }
        let (wm, _) = mean_std(&with_cp);
        let (wom, _) = mean_std(&without_cp);
        println!(
            "{:<10} {:>15.1}% {:>15.1}% {:>+11.1}",
            n_ch,
            wm * 100.0,
            wom * 100.0,
            (wom - wm) * 100.0
        );
    }

    // ============ 判定 ============
    println!();
    println!("=== 判定 ===");

    // H-A: 20 信道（>= 目标数）应显著高于 1 信道
    let cov_1 = h_a.iter().find(|(c, _)| *c == 1).map(|(_, v)| *v).unwrap_or(0.0);
    let cov_20 = h_a.iter().find(|(c, _)| *c == 20).map(|(_, v)| *v).unwrap_or(0.0);
    let diff_a = (cov_20 - cov_1) * 100.0;
    println!(
        "H-A 信道数 1 -> 20: 覆盖 {:.1}% -> {:.1}% ({:+.1}pp) => {}",
        cov_1 * 100.0,
        cov_20 * 100.0,
        diff_a,
        if diff_a > 15.0 { "成立" } else { "不成立" }
    );

    // H-B: 找最优点
    let mut best_size = 0;
    let mut best_cov = 0.0;
    for &(n_d, c1, _) in &h_b {
        if c1 > best_cov {
            best_cov = c1;
            best_size = n_d;
        }
    }
    let last_cov = h_b.last().map(|(_, c, _)| *c).unwrap_or(0.0);
    println!(
        "H-B 1 信道下最优点: {} 架（覆盖 {:.1}%），最大规模 {} 架（覆盖 {:.1}%）=> {}",
        best_size,
        best_cov * 100.0,
        swarm_sizes.last().unwrap(),
        last_cov * 100.0,
        if best_size != *swarm_sizes.last().unwrap() && last_cov < best_cov - 0.05 {
            "存在最优规模"
        } else {
            "未发现最优点"
        }
    );

    let _ = Instant::now();
}
