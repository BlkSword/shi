//! 导出蜂群仿真轨迹，供前端演示使用
//!
//! 输出 JSON：每个场景的每帧状态

use shi_syssim::{DroneState, Swarm, SwarmConfig};
use std::fs;

fn serialize_frame(s: &Swarm) -> String {
    let drones: Vec<String> = s
        .drones
        .iter()
        .map(|d| {
            let st = match d.state {
                DroneState::Searching => 0,
                DroneState::Attacking => 1,
                DroneState::Destroyed => 2,
            };
            format!("[{:.1},{:.1},{}]", d.x, d.y, st)
        })
        .collect();

    let targets: Vec<String> = s
        .targets
        .iter()
        .map(|t| {
            format!(
                "[{:.1},{:.1},{:.1},{},{}]",
                t.x,
                t.y,
                (t.health / t.max_health).max(0.0),
                if t.destroyed { 1 } else { 0 },
                if t.discovered { 1 } else { 0 }
            )
        })
        .collect();

    let decoys: Vec<String> = s
        .decoy_positions
        .iter()
        .rev()
        .take(40)
        .map(|(x, y)| format!("[{:.1},{:.1}]", x, y))
        .collect();

    format!(
        "{{\"t\":{},\"d\":[{}],\"g\":[{}],\"y\":[{}],\"cov\":{:.3},\"atk\":{},\"decoy\":{},\"jam\":{}}}",
        s.tick,
        drones.join(","),
        targets.join(","),
        decoys.join(","),
        s.coverage(),
        s.total_attacks,
        s.adversary.decoy_visits,
        s.adversary.jammed_channels
    )
}

fn main() {
    let scenarios: Vec<(&str, &str, Option<u64>, f64, f64)> = vec![
        ("baseline", "无对抗（20 信道）", None, 0.0, 0.0),
        ("jam", "通信干扰 50%（tick 300 起）", Some(300), 0.5, 0.0),
        ("decoy", "假标记诱骗 20%", None, 0.0, 0.2),
        ("both", "干扰 50% + 假标记 20%", Some(300), 0.5, 0.2),
        ("fewch", "通信瓶颈（仅 2 信道）", None, 0.0, 0.0),
    ];

    let ticks = 1200u64;
    let record_every = 20u64;

    let mut out = String::from("window.DEMO_DATA = {\n");
    out.push_str("  \"config\": {\"fieldSize\": 200, \"ticks\": ");
    out.push_str(&ticks.to_string());
    out.push_str(", \"recordEvery\": ");
    out.push_str(&record_every.to_string());
    out.push_str("},\n  \"scenarios\": [\n");

    for (idx, (key, label, jam_from, jam_ratio, decoy)) in scenarios.iter().enumerate() {
        let n_channels = if *key == "fewch" { 2 } else { 20 };
        let cfg = SwarmConfig {
            n_drones: 80,
            n_targets: 20,
            n_channels,
            command_post_until: Some(0),
            field_size: 200.0,
            sense_radius: 8.0,
            jam_from: *jam_from,
            jam_ratio: *jam_ratio,
            decoy_rate: *decoy,
            decoys_per_tick: if *decoy > 0.0 { 2 } else { 0 },
            seed: 7,
            ..Default::default()
        };

        let mut s = Swarm::new(cfg);
        let mut frames = Vec::new();

        for t in 0..ticks {
            s.step();
            if t % record_every == 0 || t == ticks - 1 {
                frames.push(serialize_frame(&s));
            }
        }

        out.push_str("    {\n");
        out.push_str(&format!("      \"key\": \"{}\",\n", key));
        out.push_str(&format!("      \"label\": \"{}\",\n", label));
        out.push_str(&format!("      \"channels\": {},\n", n_channels));
        out.push_str(&format!("      \"finalCoverage\": {:.3},\n", s.coverage()));
        out.push_str(&format!("      \"attacks\": {},\n", s.total_attacks));
        out.push_str(&format!("      \"decoyVisits\": {},\n", s.adversary.decoy_visits));
        out.push_str(&format!("      \"jammed\": {},\n", s.adversary.jammed_channels));
        out.push_str("      \"frames\": [\n");
        out.push_str(&frames.join(",\n"));
        out.push_str("\n      ]\n    }");
        if idx + 1 < scenarios.len() {
            out.push(',');
        }
        out.push('\n');
    }

    out.push_str("  ]\n};\n");

    fs::create_dir_all("../demo").ok();
    fs::write("../demo/demo_data.js", out).unwrap();
    println!("written ../demo/demo_data.js");
}
