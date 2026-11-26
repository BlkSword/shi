# 势 / Shi — Rust 实现

大规模仿真内核。Python 版用于快速验证假设，Rust 版用于大规模实验。

## 性能

实测（8 核 x86_64，release 模式）：

| 规模 | 耗时 | 吞吐 |
|---|---|---|
| 1,000 个体 × 1,000 tick | 0.018s | 55M agent-ticks/s |
| 10,000 个体 × 1,000 tick | 0.239s | 42M agent-ticks/s |
| 100,000 个体 × 1,000 tick | 2.730s | 37M agent-ticks/s |

对比 Python 版（1,000 个体 × 2,000 tick = 21.75s，即 0.09M agent-ticks/s）：

**约 600 倍加速。**

## 确定性

跨机器验证通过：本地 Windows 和远程 Linux 跑出的状态哈希逐位一致。

    $ shi verify
      基准哈希序列: 10 个
      首个: eb10a3a2a37edf3d
      末个: 83ab2b668071a32d
    [PASS] 20 次运行，哈希序列 100% 一致

## 结构

    rust/
      Cargo.toml          workspace 配置
      shi-core/           仿真内核
        src/lib.rs          模块导出
        src/rng.rs          确定性随机数
        src/world.rs        多通道环境场
        src/agent.rs        阈值个体
        src/engine.rs       主循环
        src/statehash.rs    状态哈希
      shi-cli/            命令行工具
        src/main.rs         bench / verify
      shi-py/             Python 绑定（待实现）

## 构建

    cd rust
    cargo build --release

## 使用

    # 性能基准
    ./target/release/shi bench

    # 确定性验证
    ./target/release/shi verify

## 设计要点

1. **确定性优先**：用 `DeriveRng`（无状态派生）和 `Pcg64Mcg`（可播种），
   禁止 `thread_rng` 等不确定源。
2. **SoA 布局**：个体状态用 `Vec<f64>` 分离存储，便于向量化和缓存友好。
3. **零拷贝哈希**：状态哈希直接遍历 `Vec`，不分配中间缓冲。
4. **无全局状态**：每个个体只感知局部，环境是唯一的共享状态。

## 与 Python 版的关系

- Python 版：快速迭代、验证假设、出图
- Rust 版：大规模实验、生产部署

两者的算法必须逐位一致，通过状态哈希验证。

## 已知限制

- `shi-py`（Python 绑定）暂未启用：PyO3 0.22 不支持 Python 3.14，
  需要升级到 PyO3 0.23+ 或降级 Python。
- 环境场扩散是 O(n) 逐格计算，大规模时可考虑并行化。
