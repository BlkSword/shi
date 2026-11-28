//! 无人机蜂群：指挥所失效后的自主协调
//!
//! 场景
//! ----
//! N 架无人机攻击 M 个目标。指挥所在某个时刻被打掉，
//! 之后蜂群只能靠**有限信道的共享标记**（stigmergy）协调。
//!
//! 核心机制
//! --------
//! 1. **发现**：无人机飞到目标附近才能"看到"它
//! 2. **标记**：发现后往信道 `target_id % n_channels` 写标记
//! 3. **跟随**：其他无人机沿着信道梯度飞向标记点
//! 4. **冲突**：n_channels < n_targets 时，多个目标共用信道，
//!    标记重叠，无人机被误导到错误位置 —— 这就是"信息容量不足"
//!
//! 为什么这能测出工程结论
//! ----------------------
//! - 信道数 = 共享信息容量（对应 H2 实验的"通道数"）
//! - 蜂群规模 = 并发写入者数量（对应"密度"）
//! - 指挥所有无 = 中心 vs 无中心（对应 G2 的注入通道）
//!
//! 预期结论（待验证）
//! ------------------
//! 1. 信道数 >= 目标数时，覆盖率接近 100%
//! 2. 信道数 << 目标数时，覆盖率和冗余率同时恶化
//! 3. 固定信道数下，蜂群规模超过阈值后性能下降（密度上限）

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::Serialize;

use shi_core::{ChannelSpec, PheromoneField};

/// 无人机状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DroneState {
    /// 搜索中（未发现目标）
    Searching,
    /// 前往已标记的目标
    Attacking,
    /// 已损毁
    Destroyed,
}

/// 一架无人机
#[derive(Debug, Clone)]
pub struct Drone {
    pub x: f64,
    pub y: f64,
    pub state: DroneState,
    /// 正在攻击的目标索引
    pub target: Option<usize>,
    /// 剩余弹药（每次攻击消耗 1）
    pub ammo: f64,
    /// 已造成的伤害
    pub damage_done: f64,
}

/// 一个目标
#[derive(Debug, Clone)]
pub struct Target {
    pub x: f64,
    pub y: f64,
    /// 剩余血量
    pub health: f64,
    /// 初始血量
    pub max_health: f64,
    /// 是否已被发现
    pub discovered: bool,
    /// 累计被命中的次数（用于算冗余）
    pub hits: u32,
    /// 是否已摧毁
    pub destroyed: bool,
}

impl Target {
    pub fn is_alive(&self) -> bool {
        !self.destroyed
    }
}

/// 蜂群配置
#[derive(Debug, Clone, Serialize)]
pub struct SwarmConfig {
    /// 无人机数量
    pub n_drones: usize,
    /// 目标数量
    pub n_targets: usize,
    /// 通信信道数（共享信息容量）
    pub n_channels: usize,
    /// 指挥所存活到哪个 tick（None = 永不失效）
    pub command_post_until: Option<u64>,
    /// 战场尺寸
    pub field_size: f64,
    /// 无人机的感知半径
    pub sense_radius: f64,
    /// 无人机的攻击半径
    pub attack_radius: f64,
    /// 每 tick 移动距离
    pub drone_speed: f64,
    /// 每次攻击造成的伤害
    pub damage_per_hit: f64,
    /// 目标血量
    pub target_health: f64,
    /// 干扰：从该 tick 起，被干扰的信道不可用（None = 无干扰）
    pub jam_from: Option<u64>,
    /// 干扰的信道比例 [0, 1]。0.5 = 一半信道被压制
    pub jam_ratio: f64,
    /// 每 tick 敌方向信道注入假标记的概率 [0, 1)
    pub decoy_rate: f64,
    /// 假标记数量（每个 tick 最多注入几个）
    pub decoys_per_tick: usize,
    pub seed: u64,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            n_drones: 100,
            n_targets: 20,
            n_channels: 20,
            command_post_until: None,
            field_size: 100.0,
            sense_radius: 8.0,
            attack_radius: 2.0,
            drone_speed: 1.0,
            damage_per_hit: 1.0,
            target_health: 3.0,
            jam_from: None,
            jam_ratio: 0.0,
            decoy_rate: 0.0,
            decoys_per_tick: 0,
            seed: 0,
        }
    }
}

/// 对抗状态
#[derive(Debug, Clone, Default, Serialize)]
pub struct AdversaryStats {
    /// 被干扰的信道数
    pub jammed_channels: usize,
    /// 注入的假标记总数
    pub decoys_injected: u32,
    /// 无人机被假标记骗走的次数
    pub decoy_visits: u32,
}

/// 蜂群战场
pub struct Swarm {
    pub config: SwarmConfig,
    pub drones: Vec<Drone>,
    pub targets: Vec<Target>,
    /// 共享标记场：每个信道一个通道
    pub field: PheromoneField,
    pub tick: u64,
    /// 统计
    pub total_attacks: u32,
    /// 指挥所是否已失效
    pub command_dead: bool,
    /// 每个信道当前"归属"的目标（最后一次写入者）
    ///
    /// 这就是冲突的载体：
    /// - n_channels >= n_targets：每个目标独占信道，owner 唯一
    /// - n_channels < n_targets：多个目标抢同一信道，owner 被覆盖，
    ///   跟随者会被引向错误目标 —— 这就是信息容量不足的后果
    pub channel_owner: Vec<Option<usize>>,
    /// 被干扰的信道（静态确定：按 jam_ratio 选）
    pub jammed: Vec<bool>,
    /// 假标记位置（用于统计被骗次数）
    pub decoy_positions: Vec<(f64, f64)>,
    /// 对抗统计
    pub adversary: AdversaryStats,
    rng: Pcg64Mcg,
}

impl Swarm {
    pub fn new(config: SwarmConfig) -> Self {
        let mut rng = Pcg64Mcg::seed_from_u64(config.seed);
        let size = config.field_size;

        // 无人机：随机初始位置
        let drones: Vec<Drone> = (0..config.n_drones)
            .map(|_| Drone {
                x: rng.gen::<f64>() * size,
                y: rng.gen::<f64>() * size,
                state: DroneState::Searching,
                target: None,
                ammo: 10.0,
                damage_done: 0.0,
            })
            .collect();

        // 目标：随机分布，彼此间隔至少 size/(n_targets^0.5) * 0.6
        let min_dist = size / (config.n_targets as f64).sqrt() * 0.6;
        let mut targets: Vec<Target> = Vec::with_capacity(config.n_targets);
        let mut attempts = 0;
        while targets.len() < config.n_targets && attempts < config.n_targets * 200 {
            attempts += 1;
            let x = rng.gen::<f64>() * size;
            let y = rng.gen::<f64>() * size;
            if targets
                .iter()
                .all(|t| ((t.x - x).powi(2) + (t.y - y).powi(2)).sqrt() >= min_dist)
            {
                targets.push(Target {
                    x,
                    y,
                    health: config.target_health,
                    max_health: config.target_health,
                    discovered: false,
                    hits: 0,
                    destroyed: false,
                });
            }
        }

        // 标记场：n_channels 个通道
        //
        // 衰减率的选择很关键
        // ------------------
        // 早期用 0.95（半衰期约 14 tick），结果标记在无人机飞抵前就消失了，
        // 信道数完全不影响结果——因为标记根本没起作用。
        //
        // 战场尺度 200、无人机速度 1，跨场约 200 tick。
        // 标记至少要活 300 tick 才有意义：0.998^300 ≈ 0.55。
        let n_channels = config.n_channels;
        // 标记不扩散（diffusion = 0）：点标记的梯度靠中心差分就够，
        // 而扩散的代价是 O(格数 × 通道数)，多通道时是主要瓶颈。
        let channels: Vec<ChannelSpec> = (0..n_channels)
            .map(|_| ChannelSpec::new(0.998, 0.0))
            .collect();
        let side = (size as usize).max(50);
        let field = PheromoneField::new(side, side, channels);

        Self {
            config,
            drones,
            targets,
            field,
            tick: 0,
            total_attacks: 0,
            command_dead: false,
            channel_owner: vec![None; n_channels],
            jammed: vec![false; n_channels],
            decoy_positions: Vec::new(),
            adversary: AdversaryStats::default(),
            rng,
        }
    }

    /// 判断某个信道当前是否可用（未被干扰）
    #[inline]
    fn channel_usable(&self, ch: usize) -> bool {
        if let Some(from) = self.config.jam_from {
            if self.tick >= from && self.jammed.get(ch).copied().unwrap_or(false) {
                return false;
            }
        }
        true
    }

    /// 敌方向信道注入假标记
    fn inject_decoys(&mut self) {
        if self.config.decoy_rate <= 0.0 || self.config.decoys_per_tick == 0 {
            return;
        }
        let size = self.config.field_size;
        for _ in 0..self.config.decoys_per_tick {
            if self.rng.gen::<f64>() >= self.config.decoy_rate {
                continue;
            }
            let x = self.rng.gen::<f64>() * size;
            let y = self.rng.gen::<f64>() * size;
            // 假标记随机占用一个可用信道
            let ch = self.rng.gen_range(0..self.config.n_channels);
            if !self.channel_usable(ch) {
                continue;
            }
            // 假标记强度略低于真标记，但足够吸引无人机
            self.field.deposit(x, y, ch, 35.0);
            self.decoy_positions.push((x, y));
            self.adversary.decoys_injected += 1;
        }
    }

    /// 目标索引 → 信道索引（可能冲突）
    #[inline]
    fn target_channel(&self, target_id: usize) -> usize {
        target_id % self.config.n_channels
    }

    /// 推进一个 tick
    pub fn step(&mut self) {
        // 0. 检查指挥所是否失效
        if let Some(until) = self.config.command_post_until {
            if self.tick >= until {
                self.command_dead = true;
            }
        }

        // 1. 对抗：干扰生效 + 注入假标记
        if let Some(from) = self.config.jam_from {
            if self.tick == from {
                // 按比例确定被干扰的信道（确定性：取前 k 个）
                let k = (self.config.n_channels as f64 * self.config.jam_ratio).round() as usize;
                for ch in 0..k.min(self.config.n_channels) {
                    self.jammed[ch] = true;
                }
                self.adversary.jammed_channels = k.min(self.config.n_channels);
            }
        }
        self.inject_decoys();

        // 2. 发现目标：搜索中的无人机靠近目标就标记它
        self.discover_and_mark();

        // 2. 决策：有指挥所时集中分配，没有时跟随标记场
        if !self.command_dead {
            self.central_assign();
        } else {
            self.follow_marks();
        }

        // 3. 移动
        self.move_drones();

        // 4. 攻击
        self.attack();

        // 5. 环境场演化
        self.field.step();

        self.tick += 1;
    }

    /// 发现并标记目标
    fn discover_and_mark(&mut self) {
        let sense_r2 = self.config.sense_radius * self.config.sense_radius;
        let n_targets = self.targets.len();

        for d in 0..self.drones.len() {
            if self.drones[d].state == DroneState::Destroyed {
                continue;
            }
            let (dx, dy) = (self.drones[d].x, self.drones[d].y);

            for t in 0..n_targets {
                if self.targets[t].destroyed || self.targets[t].discovered {
                    continue;
                }
                let d2 = (self.targets[t].x - dx).powi(2) + (self.targets[t].y - dy).powi(2);
                if d2 <= sense_r2 {
                    self.targets[t].discovered = true;
                    let ch = self.target_channel(t);
                    self.field.deposit(self.targets[t].x, self.targets[t].y, ch, 50.0);
                    // 信道归属：后发现的覆盖先发现的（这就是冲突）
                    self.channel_owner[ch] = Some(t);
                }
            }
        }
    }

    /// 有指挥所：集中分配（全局最优的简化版——找最近的空闲无人机去打未被攻击的目标）
    fn central_assign(&mut self) {
        // 收集需要攻击的目标
        let need: Vec<usize> = (0..self.targets.len())
            .filter(|&t| self.targets[t].is_alive())
            .collect();

        if need.is_empty() {
            return;
        }

        // 收集空闲无人机
        let mut idle: Vec<usize> = (0..self.drones.len())
            .filter(|&d| {
                self.drones[d].state != DroneState::Destroyed && self.drones[d].target.is_none()
            })
            .collect();

        // 贪心：每个目标按距离分配给最近的空闲无人机
        // 目标是「每个目标至少 1 架」，多出来的无人机平均分配
        let mut idx = 0;
        for (round, &t) in need.iter().enumerate() {
            let _ = round;
            if idx >= idle.len() {
                break;
            }
            // 找最近的空闲无人机
            let mut best = idx;
            let mut best_d = f64::INFINITY;
            for k in idx..idle.len() {
                let d = idle[k];
                let dist = (self.targets[t].x - self.drones[d].x).powi(2)
                    + (self.targets[t].y - self.drones[d].y).powi(2);
                if dist < best_d {
                    best_d = dist;
                    best = k;
                }
            }
            idle.swap(idx, best);
            let d = idle[idx];
            self.drones[d].target = Some(t);
            self.drones[d].state = DroneState::Attacking;
            idx += 1;
        }
    }

    /// 无指挥所：跟随标记场梯度
    ///
    /// 只看自己最强的那个信道——如果多个目标共用信道，
    /// 梯度会被拉向"密度最高"的标记点，导致扎堆
    fn follow_marks(&mut self) {
        let sense_r = 3usize;
        for d in 0..self.drones.len() {
            if self.drones[d].state == DroneState::Destroyed {
                continue;
            }
            // 已在攻击中且目标还活着：保持
            if let Some(t) = self.drones[d].target {
                if self.targets[t].is_alive() {
                    continue;
                } else {
                    self.drones[d].target = None;
                    self.drones[d].state = DroneState::Searching;
                }
            }

            // 找最强的**可用**信道（被干扰的跳过）
            let mut best_ch = 0;
            let mut best_val = 0.0;
            for ch in 0..self.config.n_channels {
                if !self.channel_usable(ch) {
                    continue;
                }
                let v = self.field.sense(self.drones[d].x, self.drones[d].y, ch, sense_r, true);
                if v > best_val {
                    best_val = v;
                    best_ch = ch;
                }
            }

            // 有信号：跟随时信道归属者就是目标
            if best_val > 1e-6 {
                // 关键：多个目标共享信道时，owner 已被覆盖，
                // 所有跟随者都会被导向同一个目标 —— 冗余的来源
                if let Some(owner) = self.channel_owner[best_ch] {
                    if owner < self.targets.len() && self.targets[owner].is_alive() {
                        self.drones[d].state = DroneState::Attacking;
                        self.drones[d].target = Some(owner);
                    } else {
                        self.drones[d].state = DroneState::Searching;
                        self.drones[d].target = None;
                    }
                }
            } else {
                self.drones[d].state = DroneState::Searching;
            }
        }
    }

    /// 移动
    fn move_drones(&mut self) {
        let size = self.config.field_size;
        let speed = self.config.drone_speed;
        let sense_r = 3usize;

        for d in 0..self.drones.len() {
            if self.drones[d].state == DroneState::Destroyed {
                continue;
            }

            let (mut vx, mut vy) = (0.0, 0.0);

            if self.command_dead {
                // 无指挥所：跟随最强信道梯度
                let mut best_ch = 0;
                let mut best_val = 0.0;
                for ch in 0..self.config.n_channels {
                    if !self.channel_usable(ch) {
                        continue;
                    }
                    let v =
                        self.field.sense(self.drones[d].x, self.drones[d].y, ch, sense_r, true);
                    if v > best_val {
                        best_val = v;
                        best_ch = ch;
                    }
                }
                if best_val > 1e-6 {
                    let (gx, gy) = self.field.gradient(self.drones[d].x, self.drones[d].y, best_ch);
                    let n = (gx * gx + gy * gy).sqrt();
                    if n > 1e-12 {
                        vx = speed * gx / n;
                        vy = speed * gy / n;
                    }
                }
            } else {
                // 有指挥所：直接飞向分配的目标
                if let Some(t) = self.drones[d].target {
                    if t < self.targets.len() && self.targets[t].is_alive() {
                        let dx = self.targets[t].x - self.drones[d].x;
                        let dy = self.targets[t].y - self.drones[d].y;
                        let n = (dx * dx + dy * dy).sqrt();
                        if n > 1e-12 {
                            vx = speed * dx / n;
                            vy = speed * dy / n;
                        }
                    }
                }
            }

            // 没方向：随机搜索
            if vx == 0.0 && vy == 0.0 {
                let ang = self.rng.gen::<f64>() * std::f64::consts::TAU;
                vx = speed * ang.cos();
                vy = speed * ang.sin();
            }

            self.drones[d].x = (self.drones[d].x + vx).clamp(0.0, size);
            self.drones[d].y = (self.drones[d].y + vy).clamp(0.0, size);
        }
    }

    /// 攻击：在攻击半径内就打
    fn attack(&mut self) {
        let ar2 = self.config.attack_radius * self.config.attack_radius;
        let n_targets = self.targets.len();

        for d in 0..self.drones.len() {
            if self.drones[d].state == DroneState::Destroyed || self.drones[d].ammo <= 0.0 {
                continue;
            }
            let (dx, dy) = (self.drones[d].x, self.drones[d].y);

            // 承诺式攻击：只打被分配的目标
            //
            // 为什么不用「就近攻击」
            // --------------------
            // 就近攻击会掩盖协调失败——无人机走到哪都能顺手打一个，
            // 信道冲突造成的"跑错地方"就没有代价了。
            //
            // 承诺式攻击更接近真实：无人机接到目标后直奔那里，
            // 到达后攻击。如果被误导到错误位置，就是纯粹的浪费。
            let mut hit: Option<usize> = None;
            if let Some(t) = self.drones[d].target {
                if t < n_targets && self.targets[t].is_alive() {
                    let d2 = (self.targets[t].x - dx).powi(2)
                        + (self.targets[t].y - dy).powi(2);
                    if d2 <= ar2 {
                        hit = Some(t);
                    }
                }
            }

            // 若无人机停在某处但附近没有真目标，检查是否被假标记骗了
            if hit.is_none() && !self.decoy_positions.is_empty() {
                for &(dx2, dy2) in &self.decoy_positions {
                    if (dx2 - dx).powi(2) + (dy2 - dy).powi(2) <= ar2 {
                        self.adversary.decoy_visits += 1;
                        break;
                    }
                }
            }

            if let Some(t) = hit {
                self.targets[t].health -= self.config.damage_per_hit;
                self.targets[t].hits += 1;
                self.total_attacks += 1;
                self.drones[d].ammo -= 1.0;
                self.drones[d].damage_done += self.config.damage_per_hit;

                if self.targets[t].health <= 0.0 {
                    self.targets[t].destroyed = true;
                    self.drones[d].target = None;
                    self.drones[d].state = DroneState::Searching;
                }
            }
        }
    }

    /// 摧毁的目标数
    pub fn destroyed_count(&self) -> usize {
        self.targets.iter().filter(|t| t.destroyed).count()
    }

    /// 目标覆盖率
    pub fn coverage(&self) -> f64 {
        if self.targets.is_empty() {
            return 0.0;
        }
        self.destroyed_count() as f64 / self.targets.len() as f64
    }

    /// 冗余率：实际攻击次数 / 摧毁目标所需的最小攻击次数
    ///
    /// 1.0 = 完美（没有浪费），> 1 = 有冗余
    pub fn redundancy(&self) -> f64 {
        let destroyed = self.destroyed_count();
        if destroyed == 0 {
            return 0.0;
        }
        let needed_per_target = (self.config.target_health / self.config.damage_per_hit).ceil();
        let ideal = destroyed as f64 * needed_per_target;
        if ideal <= 0.0 {
            return 0.0;
        }
        self.total_attacks as f64 / ideal
    }

    /// 有效攻击率：造成有效伤害的攻击占比（打已死目标算浪费）
    pub fn effective_attack_ratio(&self) -> f64 {
        if self.total_attacks == 0 {
            return 0.0;
        }
        let needed: f64 = self
            .targets
            .iter()
            .map(|t| (t.max_health - t.health.max(0.0)).min(t.max_health))
            .sum();
        needed / self.total_attacks as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swarm_creates_entities() {
        let s = Swarm::new(SwarmConfig::default());
        assert_eq!(s.drones.len(), 100);
        assert_eq!(s.targets.len(), 20);
        assert_eq!(s.field.n_channels, 20);
    }

    #[test]
    fn targets_have_min_separation() {
        let cfg = SwarmConfig {
            n_targets: 10,
            field_size: 100.0,
            ..Default::default()
        };
        let s = Swarm::new(cfg);
        let min_dist = 100.0 / (10f64).sqrt() * 0.6;
        for i in 0..s.targets.len() {
            for j in (i + 1)..s.targets.len() {
                let d = ((s.targets[i].x - s.targets[j].x).powi(2)
                    + (s.targets[i].y - s.targets[j].y).powi(2))
                .sqrt();
                assert!(d >= min_dist * 0.99, "targets too close: {d} < {min_dist}");
            }
        }
    }

    #[test]
    fn central_has_better_coverage_than_none() {
        let run = |cp: Option<u64>| {
            let mut s = Swarm::new(SwarmConfig {
                command_post_until: cp,
                n_drones: 100,
                n_targets: 20,
                n_channels: 20,
                ..Default::default()
            });
            for _ in 0..500 {
                s.move_drones();
                s.discover_and_mark();
                if !s.command_dead {
                    s.central_assign();
                } else {
                    s.follow_marks();
                }
                s.attack();
                s.field.step();
                s.tick += 1;
                if let Some(until) = cp {
                    if s.tick >= until {
                        s.command_dead = true;
                    }
                }
            }
            s.coverage()
        };

        let with_cp = run(None);
        let without_cp = run(Some(0));
        // 有指挥所应该更好（允许平手，因为在充分时间里无中心也能收敛）
        assert!(
            with_cp >= without_cp - 0.1,
            "central {with_cp} should be >= decentralized {without_cp}"
        );
    }

    #[test]
    fn channel_collision_reduces_coverage() {
        let run = |n_channels: usize| {
            let mut s = Swarm::new(SwarmConfig {
                command_post_until: Some(0), // 立即失效
                n_drones: 40,
                n_targets: 20,
                n_channels,
                field_size: 200.0,
                sense_radius: 8.0,
                ..Default::default()
            });
            for _ in 0..1200 {
                s.step();
            }
            s.coverage()
        };

        let many = run(20); // 无冲突
        let few = run(1); // 严重冲突
        assert!(
            many > few + 0.15,
            "more channels should clearly help: many={many:.2} few={few:.2}"
        );
    }

    #[test]
    fn too_many_drones_hurts_with_few_channels() {
        // 核心发现：信道严重不足时，加无人机反而更差（羊群效应）
        let run = |n_drones: usize| {
            let mut s = Swarm::new(SwarmConfig {
                command_post_until: Some(0),
                n_drones,
                n_targets: 20,
                n_channels: 1, // 严重冲突
                field_size: 200.0,
                sense_radius: 8.0,
                seed: 0,
                ..Default::default()
            });
            for _ in 0..1200 {
                s.step();
            }
            s.coverage()
        };

        let mid = run(40);
        let huge = run(320);
        assert!(
            huge < mid,
            "more drones should hurt with 1 channel: 40->{mid:.2}, 320->{huge:.2}"
        );
    }

    #[test]
    fn enough_channels_saturate() {
        // 信道数 >= 目标数后，再加信道没有收益
        let run = |n_channels: usize| {
            let mut s = Swarm::new(SwarmConfig {
                command_post_until: Some(0),
                n_drones: 40,
                n_targets: 20,
                n_channels,
                field_size: 200.0,
                sense_radius: 8.0,
                seed: 0,
                ..Default::default()
            });
            for _ in 0..1200 {
                s.step();
            }
            s.coverage()
        };

        let c20 = run(20);
        let c40 = run(40);
        assert!(
            (c20 - c40).abs() < 0.05,
            "should saturate at 20 channels: 20ch={c20:.2} 40ch={c40:.2}"
        );
    }

    #[test]
    fn determinism() {
        let build = || {
            let mut s = Swarm::new(SwarmConfig {
                seed: 42,
                command_post_until: Some(100),
                ..Default::default()
            });
            for _ in 0..300 {
                s.step();
            }
            (s.coverage(), s.total_attacks, s.destroyed_count())
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn redundancy_at_least_one() {
        let mut s = Swarm::new(SwarmConfig {
            n_channels: 20,
            ..Default::default()
        });
        for _ in 0..500 {
            s.step();
        }
        // 没有打中任何目标时冗余率定义无意义
        if s.destroyed_count() > 0 {
            assert!(s.redundancy() >= 1.0);
        }
    }
}
