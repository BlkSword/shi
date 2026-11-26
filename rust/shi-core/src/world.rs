//! 多通道环境场
//!
//! 形式化定义
//! ----------
//! 环境 E = (C, W, D)
//! - C: 通道集合
//! - W: 写入算子
//! - D: 衰减/扩散算子
//!
//! 数据布局：(n_channels, height, width) 的 f64 数组。
//! 通道维度在最前面，便于按通道做整体衰减。

use serde::{Deserialize, Serialize};

/// 单个通道的配置。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChannelSpec {
    pub decay: f64,
    pub diffusion: f64,
}

impl ChannelSpec {
    pub fn new(decay: f64, diffusion: f64) -> Self {
        assert!(decay > 0.0 && decay <= 1.0, "decay must be in (0, 1]");
        assert!(
            (0.0..=0.25).contains(&diffusion),
            "diffusion must be in [0, 0.25]"
        );
        Self { decay, diffusion }
    }

    /// 静态通道（不衰减不扩散）。
    pub fn static_channel() -> Self {
        Self {
            decay: 1.0,
            diffusion: 0.0,
        }
    }
}

/// 多通道信息素场。
#[derive(Debug, Clone)]
pub struct PheromoneField {
    pub width: usize,
    pub height: usize,
    pub n_channels: usize,
    /// 数据布局：(n_channels, height, width)
    pub data: Vec<f64>,
    pub specs: Vec<ChannelSpec>,
}

impl PheromoneField {
    pub fn new(width: usize, height: usize, specs: Vec<ChannelSpec>) -> Self {
        assert!(width > 0 && height > 0, "grid size must be positive");
        assert!(!specs.is_empty(), "at least one channel required");
        let n_channels = specs.len();
        Self {
            width,
            height,
            n_channels,
            data: vec![0.0; n_channels * height * width],
            specs,
        }
    }

    /// 单通道静态场（最常用）。
    pub fn single_channel(width: usize, height: usize) -> Self {
        Self::new(width, height, vec![ChannelSpec::static_channel()])
    }

    #[inline]
    fn idx(&self, c: usize, y: usize, x: usize) -> usize {
        (c * self.height + y) * self.width + x
    }

    /// 把连续坐标映射到格子索引，越界时钳制到边界。
    #[inline]
    pub fn cell(&self, x: f64, y: f64) -> (usize, usize) {
        let cx = (x.floor() as isize).clamp(0, self.width as isize - 1) as usize;
        let cy = (y.floor() as isize).clamp(0, self.height as isize - 1) as usize;
        (cy, cx)
    }

    /// 在 (x, y) 所在格叠加浓度。
    #[inline]
    pub fn deposit(&mut self, x: f64, y: f64, channel: usize, amount: f64) {
        let (cy, cx) = self.cell(x, y);
        let i = self.idx(channel, cy, cx);
        self.data[i] += amount;
    }

    /// 读取 (x, y) 所在格的浓度。
    #[inline]
    pub fn read(&self, x: f64, y: f64, channel: usize) -> f64 {
        let (cy, cx) = self.cell(x, y);
        self.data[self.idx(channel, cy, cx)]
    }

    /// 读取邻域浓度。
    ///
    /// - `normalize = false`：返回总和（体现「有多少人在做这件事」）
    /// - `normalize = true`：返回均值（与单格浓度同量级，做阈值实验时必须用）
    pub fn sense(
        &self,
        x: f64,
        y: f64,
        channel: usize,
        radius: usize,
        normalize: bool,
    ) -> f64 {
        let (cy, cx) = self.cell(x, y);
        let y0 = cy.saturating_sub(radius);
        let y1 = (cy + radius + 1).min(self.height);
        let x0 = cx.saturating_sub(radius);
        let x1 = (cx + radius + 1).min(self.width);

        let mut sum = 0.0;
        for yy in y0..y1 {
            for xx in x0..x1 {
                sum += self.data[self.idx(channel, yy, xx)];
            }
        }
        if normalize {
            let count = (y1 - y0) * (x1 - x0);
            sum / count as f64
        } else {
            sum
        }
    }

    /// 用中心差分估计梯度。
    #[inline]
    pub fn gradient(&self, x: f64, y: f64, channel: usize) -> (f64, f64) {
        let (cy, cx) = self.cell(x, y);
        let xm = cx.saturating_sub(1);
        let xp = (cx + 1).min(self.width - 1);
        let ym = cy.saturating_sub(1);
        let yp = (cy + 1).min(self.height - 1);

        let gx = self.data[self.idx(channel, cy, xp)] - self.data[self.idx(channel, cy, xm)];
        let gy = self.data[self.idx(channel, yp, cx)] - self.data[self.idx(channel, ym, cx)];
        (gx, gy)
    }

    /// 通道总浓度。
    pub fn total(&self, channel: usize) -> f64 {
        let start = channel * self.height * self.width;
        let end = start + self.height * self.width;
        self.data[start..end].iter().sum()
    }

    /// 推进一个 tick：先扩散，再衰减。
    pub fn step(&mut self) {
        // 扩散
        let needs_diffusion = self.specs.iter().any(|s| s.diffusion > 0.0);
        if needs_diffusion {
            self.diffuse();
        }
        // 衰减
        for c in 0..self.n_channels {
            let decay = self.specs[c].decay;
            if decay != 1.0 {
                let start = c * self.height * self.width;
                let end = start + self.height * self.width;
                for v in &mut self.data[start..end] {
                    *v *= decay;
                }
            }
        }
    }

    /// 五点差分扩散（零通量边界）。
    fn diffuse(&mut self) {
        let w = self.width;
        let h = self.height;
        let plane = w * h;
        let mut new_data = self.data.clone();

        for c in 0..self.n_channels {
            let d = self.specs[c].diffusion;
            if d == 0.0 {
                continue;
            }
            let base = c * plane;
            for y in 0..h {
                for x in 0..w {
                    let i = base + y * w + x;
                    let up = if y > 0 { self.data[i - w] } else { self.data[i] };
                    let down = if y + 1 < h {
                        self.data[i + w]
                    } else {
                        self.data[i]
                    };
                    let left = if x > 0 { self.data[i - 1] } else { self.data[i] };
                    let right = if x + 1 < w {
                        self.data[i + 1]
                    } else {
                        self.data[i]
                    };
                    let laplacian = up + down + left + right - 4.0 * self.data[i];
                    new_data[i] = self.data[i] + d * laplacian;
                }
            }
        }
        self.data = new_data;
    }

    /// 设置静态径向梯度场：concentration = amplitude * exp(-dist / length_scale)
    pub fn set_radial_gradient(
        &mut self,
        channel: usize,
        cx: f64,
        cy: f64,
        amplitude: f64,
        length_scale: f64,
    ) {
        assert!(length_scale > 0.0, "length_scale must be positive");
        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let v = amplitude * (-dist / length_scale).exp();
                let i = self.idx(channel, y, x);
                self.data[i] = v;
            }
        }
    }

    /// 设置高斯吸引子：amplitude * exp(-dist^2 / (2 * sigma^2))
    pub fn set_gaussian_attractor(
        &mut self,
        channel: usize,
        cx: f64,
        cy: f64,
        amplitude: f64,
        sigma: f64,
    ) {
        assert!(sigma > 0.0, "sigma must be positive");
        let two_sigma_sq = 2.0 * sigma * sigma;
        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let d2 = dx * dx + dy * dy;
                let v = amplitude * (-d2 / two_sigma_sq).exp();
                let i = self.idx(channel, y, x);
                self.data[i] = v;
            }
        }
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// 状态快照（用于哈希）。
    pub fn snapshot(&self) -> &[f64] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deposit_then_read() {
        let mut f = PheromoneField::single_channel(10, 10);
        f.deposit(3.5, 4.5, 0, 2.0);
        assert_eq!(f.read(3.5, 4.5, 0), 2.0);
    }

    #[test]
    fn deposit_accumulates() {
        let mut f = PheromoneField::single_channel(10, 10);
        f.deposit(3.0, 3.0, 0, 1.0);
        f.deposit(3.0, 3.0, 0, 2.0);
        assert_eq!(f.read(3.0, 3.0, 0), 3.0);
    }

    #[test]
    fn out_of_bounds_clamps() {
        let mut f = PheromoneField::single_channel(10, 10);
        f.deposit(-5.0, -5.0, 0, 1.0);
        assert_eq!(f.read(0.0, 0.0, 0), 1.0);
        f.deposit(100.0, 100.0, 0, 1.0);
        assert_eq!(f.read(9.0, 9.0, 0), 1.0);
    }

    #[test]
    fn sense_normalize() {
        let mut f = PheromoneField::single_channel(10, 10);
        f.deposit(5.0, 5.0, 0, 25.0);
        let sum = f.sense(5.0, 5.0, 0, 2, false);
        let mean = f.sense(5.0, 5.0, 0, 2, true);
        assert_eq!(sum, 25.0);
        assert!((mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gradient_points_uphill() {
        let mut f = PheromoneField::single_channel(10, 10);
        for x in 0..10 {
            f.deposit(x as f64, 5.0, 0, x as f64);
        }
        let (gx, gy) = f.gradient(5.0, 5.0, 0);
        assert!(gx > 0.0);
        assert_eq!(gy, 0.0);
    }

    #[test]
    fn decay_reduces() {
        let mut f = PheromoneField::new(10, 10, vec![ChannelSpec::new(0.5, 0.0)]);
        f.deposit(5.0, 5.0, 0, 8.0);
        f.step();
        assert!((f.read(5.0, 5.0, 0) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn diffusion_conserves_total() {
        let mut f = PheromoneField::new(10, 10, vec![ChannelSpec::new(1.0, 0.2)]);
        f.deposit(5.0, 5.0, 0, 100.0);
        let before = f.total(0);
        f.step();
        let after = f.total(0);
        assert!((before - after).abs() < 1e-9, "before={before} after={after}");
    }

    #[test]
    fn radial_gradient_peaks_at_center() {
        let mut f = PheromoneField::single_channel(20, 20);
        f.set_radial_gradient(0, 10.0, 10.0, 100.0, 5.0);
        // 中心处应为 amplitude
        assert!((f.read(10.0, 10.0, 0) - 100.0).abs() < 1e-9);
        // 角落处距离约 14.14，浓度 = 100 * exp(-14.14/5) ≈ 5.9
        let corner = f.read(0.0, 0.0, 0);
        assert!(corner < 100.0, "corner should be less than center");
        assert!(corner > 0.0, "corner should be positive");
        // 距离越远浓度越低
        assert!(f.read(5.0, 5.0, 0) < f.read(10.0, 10.0, 0));
    }

    #[test]
    fn deterministic_step() {
        let build = || {
            let mut f = PheromoneField::single_channel(20, 20);
            for i in 0..50 {
                f.deposit((i % 20) as f64, (i / 20) as f64, 0, i as f64);
                f.step();
            }
            f
        };
        let a = build();
        let b = build();
        assert_eq!(a.data, b.data);
    }
}
