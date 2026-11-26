//! 势 / Shi — 无中枢决策系统仿真内核
//!
//! 设计原则
//! --------
//! 1. **确定性优先**：同种子必然产生逐位相同的结果
//! 2. **无全局状态**：每个个体只感知局部
//! 3. **可并行**：环境更新和个体更新可以 rayon 并行，但不破坏确定性
//!
//! 与 Python 版的关系
//! ------------------
//! Python 版用于快速验证假设，Rust 版用于大规模实验。
//! 两者的算法必须逐位一致（通过状态哈希验证）。

pub mod agent;
pub mod engine;
pub mod rng;
pub mod statehash;
pub mod world;

pub use agent::{AgentState, response_probability, sample_thresholds};
pub use engine::{Engine, Stepper};
pub use rng::{DeriveRng, RngRegistry};
pub use statehash::state_hash;
pub use world::{ChannelSpec, PheromoneField};

/// 库版本
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
