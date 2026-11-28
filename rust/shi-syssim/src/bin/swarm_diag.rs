use shi_syssim::{Swarm, SwarmConfig};

fn run(cfg: SwarmConfig, ticks: u64) -> (f64, usize, u32) {
    let mut s = Swarm::new(cfg);
    for _ in 0..ticks {
        s.step();
    }
    (s.coverage(), s.destroyed_count(), s.total_attacks)
}

fn main() {
    // 更难的任务：更大的战场、更小的感知半径、更短的时限、更少的无人机
    println!("=== 调参：找信道起作用的区间 ===");
    println!("{:<28} {:>10} {:>10} {:>12}", "配置", "覆盖率", "摧毁", "攻击数");
    println!("{}", "-".repeat(64));

    for (label, field_size, sense, drones, targets, ticks) in [
        ("200x200 sense5 20机 20目标", 200.0, 5.0, 20, 20, 1000u64),
        ("200x200 sense5 40机 20目标", 200.0, 5.0, 40, 20, 1000),
        ("200x200 sense8 40机 20目标", 200.0, 8.0, 40, 20, 1000),
        ("300x300 sense6 40机 20目标", 300.0, 6.0, 40, 20, 1500),
        ("300x300 sense6 40机 40目标", 300.0, 6.0, 40, 40, 1500),
    ] {
        let c1 = run(
            SwarmConfig {
                n_drones: drones,
                n_targets: targets,
                n_channels: 1,
                command_post_until: Some(0),
                field_size,
                sense_radius: sense,
                ..Default::default()
            },
            ticks,
        );
        let c20 = run(
            SwarmConfig {
                n_drones: drones,
                n_targets: targets,
                n_channels: targets, // 无冲突
                command_post_until: Some(0),
                field_size,
                sense_radius: sense,
                ..Default::default()
            },
            ticks,
        );
        println!(
            "{:<28} 1ch:{:>5.1}% {:>3} {:>4} | {}ch:{:>5.1}% {:>3} {:>4}",
            label,
            c1.0 * 100.0, c1.1, c1.2,
            targets,
            c20.0 * 100.0, c20.1, c20.2
        );
    }
}
