//! 多智能体共享记忆的污染实验（跨域验证）

use shi_syssim::{MultiMem, MultiMemConfig};

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    if v.len() < 2 { return (mean, 0.0); }
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn run(cfg: MultiMemConfig, ticks: u64) -> MultiMem {
    let mut m = MultiMem::new(cfg);
    for _ in 0..ticks { m.step(); }
    m
}

fn main() {
    let seeds: Vec<u64> = (0..10).collect();
    let base = MultiMemConfig {
        n_agents: 20,
        n_tasks: 40,
        work_per_task: 1.0,
        work_rate: 0.2,
        ..Default::default()
    };
    let ticks = 1500u64;

    let baseline = {
        let mut v = Vec::new();
        for &s in &seeds {
            v.push(run(MultiMemConfig { seed: s, ..base.clone() }, ticks).completion());
        }
        mean_std(&v).0
    };

    println!("=== 多智能体共享记忆：污染实验 ===");
    println!("基准（无攻击，20 agent / 40 任务）: {:.1}%", baseline * 100.0);
    println!();

    // ---------- 攻击强度 ----------
    println!("=== 注入攻击：被劫持 agent 数量 vs 完成率 ===");
    println!("{:<14} {:>18} {:>12} {:>14}", "被劫持数", "完成率", "损失(pp)", "假记录数");
    println!("{}", "-".repeat(62));

    for &n_comp in &[0usize, 1, 2, 3, 5, 8] {
        let mut comps = Vec::new();
        let mut fakes = Vec::new();
        for &s in &seeds {
            let m = run(
                MultiMemConfig {
                    n_compromised: n_comp,
                    attack_rate: 1.0,
                    seed: s,
                    ..base.clone()
                },
                ticks,
            );
            comps.push(m.completion());
            fakes.push(m.stats.fake_entries as f64);
        }
        let (c, cs) = mean_std(&comps);
        let (f, _) = mean_std(&fakes);
        println!(
            "{:<14} {:>9.1}% ± {:>5.1} {:>+11.1} {:>14.0}",
            format!("{} / 20", n_comp),
            c * 100.0,
            cs * 100.0,
            (c - baseline) * 100.0,
            f
        );
    }

    // ---------- 防御 ----------
    println!();
    println!("=== 防御对比（3 个被劫持 agent，攻击率 100%）===");
    println!("{:<26} {:>18} {:>14}", "防御", "完成率", "相对无防御");
    println!("{}", "-".repeat(60));

    let attack_cfg = MultiMemConfig {
        n_compromised: 3,
        attack_rate: 1.0,
        ..base.clone()
    };

    let mut v = Vec::new();
    for &s in &seeds {
        v.push(run(MultiMemConfig { seed: s, ..attack_cfg.clone() }, ticks).completion());
    }
    let (undef, undef_s) = mean_std(&v);
    println!(
        "{:<26} {:>9.1}% ± {:>5.1} {:>14}",
        "无防御",
        undef * 100.0,
        undef_s * 100.0,
        "—"
    );

    // 防御 1：溯源认证
    let mut v = Vec::new();
    let mut rej = Vec::new();
    for &s in &seeds {
        let m = run(
            MultiMemConfig {
                defense_provenance: true,
                seed: s,
                ..attack_cfg.clone()
            },
            ticks,
        );
        v.push(m.completion());
        rej.push(m.stats.rejected as f64);
    }
    let (prov, prov_s) = mean_std(&v);
    let (rj, _) = mean_std(&rej);
    println!(
        "{:<26} {:>9.1}% ± {:>5.1} {:>+13.1}pp  (拒绝 {:.0} 条)",
        "溯源认证",
        prov * 100.0,
        prov_s * 100.0,
        (prov - undef) * 100.0,
        rj
    );

    // 防御 2：交叉验证
    for k in [2usize, 3] {
        let mut v = Vec::new();
        for &s in &seeds {
            v.push(
                run(
                    MultiMemConfig {
                        defense_cross_validate: k,
                        seed: s,
                        ..attack_cfg.clone()
                    },
                    ticks,
                )
                .completion(),
            );
        }
        let (m, sd) = mean_std(&v);
        println!(
            "{:<26} {:>9.1}% ± {:>5.1} {:>+13.1}pp",
            format!("交叉验证（需 {} 源）", k),
            m * 100.0,
            sd * 100.0,
            (m - undef) * 100.0
        );
    }

    // ---------- 防御的代价（无攻击时）----------
    println!();
    println!("=== 防御的代价（无攻击时）===");
    for (name, cfg) in [
        ("无防御", base.clone()),
        ("溯源认证", MultiMemConfig { defense_provenance: true, ..base.clone() }),
        ("交叉验证 2", MultiMemConfig { defense_cross_validate: 2, ..base.clone() }),
        ("交叉验证 3", MultiMemConfig { defense_cross_validate: 3, ..base.clone() }),
    ] {
        let mut v = Vec::new();
        for &s in &seeds {
            v.push(run(MultiMemConfig { seed: s, ..cfg.clone() }, ticks).completion());
        }
        let (m, sd) = mean_std(&v);
        println!(
            "  {:<14} {:.1}% ± {:.1}  ({:+.1}pp)",
            name,
            m * 100.0,
            sd * 100.0,
            (m - baseline) * 100.0
        );
    }

    println!();
    println!("=== 跨域判定 ===");
    println!("蜂群结论：「伪造信息比删除信息更致命」「能区分真假的防御才有用」");
    println!("多智能体：见上表 —— 溯源认证恢复 {:.0}% 的损失", (prov - undef) / (baseline - undef) * 100.0);
}
