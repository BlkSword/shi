//! 任务模型

use serde::Serialize;

/// 一个待处理的任务
#[derive(Debug, Clone, Serialize)]
pub struct Task {
    /// 任务 id
    pub id: u64,
    /// 需要多少处理能力
    pub demand: f64,
    /// 已完成的处理量
    pub done: f64,
    /// 当前归属的节点（None = 未分配）
    pub owner: Option<usize>,
    /// 是否已完成
    pub finished: bool,
}

impl Task {
    pub fn new(id: u64, demand: f64) -> Self {
        Self {
            id,
            demand,
            done: 0.0,
            owner: None,
            finished: false,
        }
    }

    /// 处理 amount 的量，返回实际完成量
    pub fn process(&mut self, amount: f64) -> f64 {
        if self.finished {
            return 0.0;
        }
        let remaining = self.demand - self.done;
        let actual = amount.min(remaining).max(0.0);
        self.done += actual;
        if self.done >= self.demand - 1e-12 {
            self.done = self.demand;
            self.finished = true;
        }
        actual
    }

    /// 剩余需求比例
    pub fn remaining_ratio(&self) -> f64 {
        if self.demand <= 0.0 {
            return 0.0;
        }
        ((self.demand - self.done) / self.demand).max(0.0)
    }
}

/// 任务配置
#[derive(Debug, Clone)]
pub struct TaskConfig {
    pub n_tasks: usize,
    /// 每个任务的需求量（固定）
    pub demand: f64,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            n_tasks: 1000,
            demand: 10.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_accumulates() {
        let mut t = Task::new(0, 10.0);
        assert_eq!(t.process(3.0), 3.0);
        assert_eq!(t.done, 3.0);
        assert!(!t.finished);
    }

    #[test]
    fn process_clamps_to_remaining() {
        let mut t = Task::new(0, 10.0);
        t.process(8.0);
        assert_eq!(t.process(5.0), 2.0);
        assert!(t.finished);
    }

    #[test]
    fn finished_task_ignores_processing() {
        let mut t = Task::new(0, 1.0);
        t.process(1.0);
        assert_eq!(t.process(1.0), 0.0);
    }

    #[test]
    fn remaining_ratio() {
        let mut t = Task::new(0, 10.0);
        assert_eq!(t.remaining_ratio(), 1.0);
        t.process(5.0);
        assert!((t.remaining_ratio() - 0.5).abs() < 1e-12);
    }
}
