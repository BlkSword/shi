use shi_syssim::{GossipView};

fn main() {
    let n = 100;
    let mut view = GossipView::new(n, 3, 42);
    let loads: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let true_mean: f64 = loads.iter().sum::<f64>() / n as f64;

    println!("true global mean = {:.6}", true_mean);
    for round in [0, 10, 50, 100, 200, 500] {
        // 重新跑
        let mut v = GossipView::new(n, 3, 42);
        for t in 0..round {
            v.gossip_round(&loads, 0.5, 0.0, t, |_, _, _| true);
        }
        let est_mean = v.mean_load_estimate.iter().sum::<f64>() / n as f64;
        let err = v.view_error(&loads);
        println!("round {:>3}: estimate mean = {:.6}, view_error = {:.6}", round, est_mean, err);
    }
}
