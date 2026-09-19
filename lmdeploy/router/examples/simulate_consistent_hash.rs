//! Simulation: Evaluate consistent hash balance for PD disagg decode workers
//!
//! Replicates the router's exact consistent hash algorithm to measure
//! request distribution balance across decode workers with data parallelism.
//!
//! Usage:
//!   cargo run --example simulate_consistent_hash
//!   cargo run --example simulate_consistent_hash -- --num-sessions 500000 --dp-size 4
//!   cargo run --example simulate_consistent_hash -- --trials 100
//!   cargo run --example simulate_consistent_hash -- --mode rendezvous --trials 100

use std::collections::{BTreeMap, HashMap};

use clap::{Parser, ValueEnum};
use uuid::Uuid;

use lmdeploy_router_rs::policies::ConsistentHashPolicy;
use lmdeploy_router_rs::policies::VIRTUAL_NODES_PER_WORKER;

#[derive(Clone, ValueEnum)]
enum Mode {
    /// Ring-based consistent hashing (current router implementation)
    Ring,
    /// Rendezvous hashing (Highest Random Weight)
    Rendezvous,
}

#[derive(Parser)]
#[command(name = "simulate_consistent_hash")]
#[command(about = "Simulate consistent hash distribution for PD disagg decode workers")]
struct Args {
    /// Number of random session UUIDs to simulate
    #[arg(long, default_value_t = 256)]
    num_sessions: usize,

    /// Number of physical decode workers
    #[arg(long, default_value_t = 2)]
    num_workers: usize,

    /// Intra-node data parallel size (GPUs per worker)
    #[arg(long, default_value_t = 4)]
    dp_size: usize,

    /// Virtual nodes per logical worker on the hash ring (only used in ring mode)
    #[arg(long, default_value_t = VIRTUAL_NODES_PER_WORKER)]
    virtual_nodes: u32,

    /// Header name used for hash key (e.g. x-session-id, x-correlation-id)
    #[arg(long, default_value = "x-session-id")]
    header_name: String,

    /// Number of independent trials to run
    #[arg(long, default_value_t = 1)]
    trials: usize,

    /// Hashing mode: ring (consistent hash ring) or rendezvous (HRW)
    #[arg(long, value_enum, default_value_t = Mode::Ring)]
    mode: Mode,
}

/// Stats collected from a single trial
struct TrialStats {
    /// Coefficient of variation for physical workers
    worker_cv: f64,
    /// Imbalance ratio (max/min) for physical workers
    worker_imbalance: f64,
    /// Coefficient of variation for DP ranks
    rank_cv: f64,
    /// Imbalance ratio (max/min) for DP ranks
    rank_imbalance: f64,
}

fn run_single_trial(args: &Args, verbose: bool) -> TrialStats {
    let total_logical = args.num_workers * args.dp_size;

    // Build worker URLs
    let mut worker_urls: Vec<String> = Vec::new();
    for w in 0..args.num_workers {
        for rank in 0..args.dp_size {
            worker_urls.push(format!("http://decode-worker{}:8000@{}", w, rank));
        }
    }

    // Build hash ring (only used in ring mode)
    let mut ring: BTreeMap<u64, String> = BTreeMap::new();
    if matches!(args.mode, Mode::Ring) {
        for url in &worker_urls {
            for i in 0..args.virtual_nodes {
                let virtual_key = format!("{}:{}", url, i);
                let hash_value = ConsistentHashPolicy::fbi_hash(&virtual_key);
                ring.insert(hash_value, url.clone());
            }
        }
    }

    if verbose && matches!(args.mode, Mode::Ring) {
        analyze_ring_distribution(&ring, &worker_urls, args.num_workers, args.dp_size);
    }

    // Generate session UUIDs using proper UUID v4
    let mut session_ids: Vec<String> = Vec::with_capacity(args.num_sessions);
    for _ in 0..args.num_sessions {
        session_ids.push(Uuid::new_v4().to_string());
    }

    // Simulate routing
    let mut per_url_count: HashMap<String, usize> = HashMap::new();
    for url in &worker_urls {
        per_url_count.insert(url.clone(), 0);
    }

    for session_id in &session_ids {
        let hash_key = format!("header:{}:{}", args.header_name, session_id);

        let selected = match args.mode {
            Mode::Ring => {
                let hash_value = ConsistentHashPolicy::fbi_hash(&hash_key);
                ring.range(hash_value..)
                    .next()
                    .or_else(|| ring.iter().next())
                    .map(|(_, url)| url.clone())
                    .unwrap()
            }
            Mode::Rendezvous => {
                // Rendezvous hashing: score each worker, pick highest
                worker_urls
                    .iter()
                    .max_by_key(|url| {
                        let score_key = format!("{}:{}", hash_key, url);
                        ConsistentHashPolicy::fbi_hash(&score_key)
                    })
                    .unwrap()
                    .clone()
            }
        };

        *per_url_count.get_mut(&selected).unwrap() += 1;
    }

    // Aggregate per physical worker
    let mut per_worker_count: Vec<usize> = vec![0; args.num_workers];
    for (w, count) in per_worker_count.iter_mut().enumerate() {
        for rank in 0..args.dp_size {
            let url = format!("http://decode-worker{}:8000@{}", w, rank);
            *count += per_url_count[&url];
        }
    }

    // Physical worker stats
    let mean = args.num_sessions as f64 / args.num_workers as f64;
    let variance: f64 = per_worker_count
        .iter()
        .map(|&c| (c as f64 - mean).powi(2))
        .sum::<f64>()
        / args.num_workers as f64;
    let std_dev = variance.sqrt();
    let worker_cv = std_dev / mean;
    let min_count = *per_worker_count.iter().min().unwrap();
    let max_count = *per_worker_count.iter().max().unwrap();
    let worker_imbalance = max_count as f64 / mean;

    // Per-DP-rank stats
    let rank_counts: Vec<usize> = worker_urls.iter().map(|u| per_url_count[u]).collect();
    let rank_min = *rank_counts.iter().min().unwrap();
    let rank_max = *rank_counts.iter().max().unwrap();
    let rank_mean = args.num_sessions as f64 / total_logical as f64;
    let rank_variance: f64 = rank_counts
        .iter()
        .map(|&c| (c as f64 - rank_mean).powi(2))
        .sum::<f64>()
        / total_logical as f64;
    let rank_std_dev = rank_variance.sqrt();
    let rank_cv = rank_std_dev / rank_mean;
    let rank_imbalance = rank_max as f64 / rank_mean;

    if verbose {
        // Print detailed per-URL distribution
        println!("\n=== Per-DP-Rank Distribution ===");
        println!("{:<45} {:>8} {:>8}", "Worker URL", "Count", "%");
        println!("{}", "-".repeat(65));
        for url in &worker_urls {
            let count = per_url_count[url];
            let pct = 100.0 * count as f64 / args.num_sessions as f64;
            println!("{:<45} {:>8} {:>7.2}%", url, count, pct);
        }

        println!("\n=== Per-Physical-Worker Distribution ===");
        println!("{:<30} {:>8} {:>8}", "Worker", "Count", "%");
        println!("{}", "-".repeat(50));
        for (w, count) in per_worker_count.iter().enumerate() {
            let pct = 100.0 * *count as f64 / args.num_sessions as f64;
            println!("decode-worker{:<17} {:>8} {:>7.2}%", w, count, pct);
        }

        println!("\n=== Balance Statistics (Physical Workers) ===");
        println!("Expected per worker:  {:.1}", mean);
        println!("Min:                  {}", min_count);
        println!("Max:                  {}", max_count);
        println!(
            "Imbalance ratio:      {:.4} (max/expected)",
            worker_imbalance
        );
        println!("Std deviation:        {:.2}", std_dev);
        println!("Coeff of variation:   {:.4} (lower is better)", worker_cv);

        println!("\n=== Balance Statistics (Per DP Rank) ===");
        println!("Expected per rank:    {:.1}", rank_mean);
        println!("Min:                  {}", rank_min);
        println!("Max:                  {}", rank_max);
        println!("Imbalance ratio:      {:.4} (max/expected)", rank_imbalance);
        println!("Std deviation:        {:.2}", rank_std_dev);
        println!("Coeff of variation:   {:.4} (lower is better)", rank_cv);
    }

    TrialStats {
        worker_cv,
        worker_imbalance,
        rank_cv,
        rank_imbalance,
    }
}

fn main() {
    let args = Args::parse();

    let total_logical = args.num_workers * args.dp_size;
    let mode_name = match args.mode {
        Mode::Ring => "ring",
        Mode::Rendezvous => "rendezvous",
    };
    println!("=== Consistent Hash Balance Simulation ===");
    println!("Mode: {}", mode_name);
    println!("Physical decode workers: {}", args.num_workers);
    println!("DP size (GPUs per worker): {}", args.dp_size);
    println!("Logical worker URLs: {}", total_logical);
    if matches!(args.mode, Mode::Ring) {
        println!("Virtual nodes per logical worker: {}", args.virtual_nodes);
        println!(
            "Total virtual nodes on ring: {}",
            total_logical as u32 * args.virtual_nodes
        );
    }
    println!("Sessions to simulate: {}", args.num_sessions);
    println!("Trials: {}", args.trials);
    println!();

    if args.trials == 1 {
        // Single trial: verbose output
        run_single_trial(&args, true);
    } else {
        // Multiple trials: collect stats and aggregate
        let mut all_stats: Vec<TrialStats> = Vec::with_capacity(args.trials);

        for _ in 0..args.trials {
            let stats = run_single_trial(&args, false);
            all_stats.push(stats);
        }

        // Aggregate
        let worker_cvs: Vec<f64> = all_stats.iter().map(|s| s.worker_cv).collect();
        let worker_imbs: Vec<f64> = all_stats.iter().map(|s| s.worker_imbalance).collect();
        let rank_cvs: Vec<f64> = all_stats.iter().map(|s| s.rank_cv).collect();
        let rank_imbs: Vec<f64> = all_stats.iter().map(|s| s.rank_imbalance).collect();

        println!("=== Aggregated Stats over {} Trials ===", args.trials);
        println!();
        print_aggregated("Physical Worker CV", &worker_cvs);
        print_aggregated("Physical Worker Imbalance (max/expected)", &worker_imbs);
        print_aggregated("DP Rank CV", &rank_cvs);
        print_aggregated("DP Rank Imbalance (max/expected)", &rank_imbs);
    }
}

fn print_aggregated(label: &str, values: &[f64]) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let p50 = percentile(&sorted, 50.0);
    let p95 = percentile(&sorted, 95.0);
    let p99 = percentile(&sorted, 99.0);
    let std_dev = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();

    println!("  {}", label);
    println!(
        "    mean={:.4}  std={:.4}  min={:.4}  p50={:.4}  p95={:.4}  p99={:.4}  max={:.4}",
        mean, std_dev, min, p50, p95, p99, max
    );
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let idx = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Compute arc ownership per URL on the ring, returning both totals and per-vnode arcs
/// Analyze how virtual nodes are distributed on the hash ring
fn analyze_ring_distribution(
    ring: &BTreeMap<u64, String>,
    worker_urls: &[String],
    num_workers: usize,
    dp_size: usize,
) {
    println!("=== Hash Ring Virtual Node Analysis ===");

    let mut vnodes_per_url: HashMap<String, usize> = HashMap::new();
    for url in worker_urls {
        vnodes_per_url.insert(url.clone(), 0);
    }
    for url in ring.values() {
        *vnodes_per_url.get_mut(url).unwrap() += 1;
    }

    let mut vnodes_per_worker: Vec<usize> = vec![0; num_workers];
    for (w, count) in vnodes_per_worker.iter_mut().enumerate() {
        for rank in 0..dp_size {
            let url = format!("http://decode-worker{}:8000@{}", w, rank);
            *count += vnodes_per_url[&url];
        }
    }

    let total_vnodes: usize = ring.len();
    println!(
        "Total virtual nodes on ring: {} (expected: {})",
        total_vnodes,
        worker_urls.len() as u32 * VIRTUAL_NODES_PER_WORKER
    );

    for (w, &vnode_count) in vnodes_per_worker.iter().enumerate() {
        let pct = 100.0 * vnode_count as f64 / total_vnodes as f64;
        println!("  decode-worker{}: {} vnodes ({:.1}%)", w, vnode_count, pct);
    }

    let hashes: Vec<u64> = ring.keys().copied().collect();
    let owners: Vec<String> = ring.values().cloned().collect();

    let mut arc_per_worker: Vec<f64> = vec![0.0; num_workers];
    let n = hashes.len();
    for i in 0..n {
        let next = (i + 1) % n;
        let arc = if next > i {
            hashes[next] - hashes[i]
        } else {
            (u64::MAX - hashes[i]) + hashes[next] + 1
        };
        let arc_frac = arc as f64 / u64::MAX as f64;
        let url = &owners[i];
        for (w, arc) in arc_per_worker.iter_mut().enumerate() {
            let prefix = format!("http://decode-worker{}:8000@", w);
            if url.starts_with(&prefix) {
                *arc += arc_frac;
                break;
            }
        }
    }

    println!("\nHash ring arc ownership (fraction of keyspace):");
    for (w, &arc) in arc_per_worker.iter().enumerate() {
        println!("  decode-worker{}: {:.4} ({:.2}%)", w, arc, arc * 100.0);
    }
    let arc_min = arc_per_worker.iter().cloned().fold(f64::INFINITY, f64::min);
    let arc_max = arc_per_worker
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  Arc imbalance ratio: {:.4} (max/min)",
        if arc_min > 0.0 {
            arc_max / arc_min
        } else {
            f64::INFINITY
        }
    );
}
