//! Rendezvous hashing (Highest Random Weight) load balancing policy
//!
//! Routes requests to workers by scoring each worker with a hash of the
//! session key combined with the worker URL, then picking the highest score.
//! This provides better balance than ring-based consistent hashing with
//! zero tuning knobs, while preserving session stickiness and minimal
//! redistribution (~1/n) on worker add/remove.

use std::sync::Arc;

use tracing::info;

use super::get_healthy_worker_indices;
use super::hash_key;
use super::LoadBalancingPolicy;
use super::RequestHeaders;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::ConsistentHashPolicy;

/// Rendezvous hashing (Highest Random Weight) policy
///
/// For each request, scores every healthy worker with
/// `fbi_hash("{hash_key}:{worker_url}")` and picks the one with the
/// highest score. This is deterministic — the same session always maps
/// to the same worker — and gives near-perfect balance across workers.
#[derive(Debug)]
pub struct RendezvousHashPolicy;

impl RendezvousHashPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RendezvousHashPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingPolicy for RendezvousHashPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        let hash_key = hash_key::extract_hash_key(request_text, headers);

        // Score each healthy worker and pick the highest
        let selected_idx = healthy_indices
            .iter()
            .max_by_key(|&&idx| {
                let score_key = format!("{}:{}", hash_key, workers[idx].url());
                ConsistentHashPolicy::fbi_hash(&score_key)
            })
            .copied()
            .unwrap();

        let worker_url = workers[selected_idx].url();
        info!(
            "Rendezvous hash routing: key='{}' -> worker='{}' (index={})",
            hash_key, worker_url, selected_idx
        );

        workers[selected_idx].increment_processed();
        RouterMetrics::record_processed_request(worker_url);
        RouterMetrics::record_policy_decision(self.name(), worker_url);

        Some(selected_idx)
    }

    fn name(&self) -> &'static str {
        "rendezvous_hash"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorker;
    use crate::core::WorkerType;
    use std::collections::HashSet;

    fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.to_string(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn test_rendezvous_hash_consistency() {
        let policy = RendezvousHashPolicy::new();
        let workers = make_workers(&[
            "http://worker1:8000",
            "http://worker2:8000",
            "http://worker3:8000",
        ]);

        let request = r#"{"session_id": "consistent_test"}"#;
        let idx1 = policy.select_worker(&workers, Some(request));
        let idx2 = policy.select_worker(&workers, Some(request));
        let idx3 = policy.select_worker(&workers, Some(request));

        assert_eq!(idx1, idx2);
        assert_eq!(idx2, idx3);
        assert!(idx1.is_some());
    }

    #[test]
    fn test_rendezvous_distribution() {
        let policy = RendezvousHashPolicy::new();
        let workers = make_workers(&[
            "http://worker1:8000",
            "http://worker2:8000",
            "http://worker3:8000",
        ]);

        let mut selected_workers = HashSet::new();
        for i in 0..100 {
            let request = format!(r#"{{"session_id": "session_{}"}}"#, i);
            if let Some(idx) = policy.select_worker(&workers, Some(&request)) {
                selected_workers.insert(idx);
            }
        }

        // All workers should be selected at least once with 100 different sessions
        assert_eq!(selected_workers.len(), 3);
    }

    #[test]
    fn test_rendezvous_with_unhealthy_workers() {
        let policy = RendezvousHashPolicy::new();
        let workers = make_workers(&["http://worker1:8000", "http://worker2:8000"]);

        workers[0].set_healthy(false);

        // Should always select the healthy worker
        for i in 0..10 {
            let request = format!(r#"{{"session_id": "session_{}"}}"#, i);
            assert_eq!(policy.select_worker(&workers, Some(&request)), Some(1));
        }
    }

    #[test]
    fn test_rendezvous_no_healthy_workers() {
        let policy = RendezvousHashPolicy::new();
        let workers = make_workers(&["http://worker1:8000"]);

        workers[0].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, None), None);
    }

    #[test]
    fn test_rendezvous_minimal_redistribution() {
        let policy = RendezvousHashPolicy::new();
        let workers_3 = make_workers(&[
            "http://worker1:8000",
            "http://worker2:8000",
            "http://worker3:8000",
        ]);
        let workers_2 = make_workers(&["http://worker1:8000", "http://worker2:8000"]);

        let mut same_count = 0;
        let total = 100;

        for i in 0..total {
            let request = format!(r#"{{"session_id": "session_{}"}}"#, i);
            let idx_3 = policy.select_worker(&workers_3, Some(&request)).unwrap();
            let idx_2 = policy.select_worker(&workers_2, Some(&request)).unwrap();

            let url_3 = workers_3[idx_3].url();
            let url_2 = workers_2[idx_2].url();
            if url_3 != "http://worker3:8000" && url_3 == url_2 {
                same_count += 1;
            }
        }

        let sessions_not_on_worker3 = (0..total)
            .filter(|i| {
                let request = format!(r#"{{"session_id": "session_{}"}}"#, i);
                let idx = policy.select_worker(&workers_3, Some(&request)).unwrap();
                workers_3[idx].url() != "http://worker3:8000"
            })
            .count();

        // All sessions that were on worker1/worker2 should stay there
        assert_eq!(same_count, sessions_not_on_worker3);
    }

    #[test]
    fn test_rendezvous_header_routing() {
        let policy = RendezvousHashPolicy::new();
        let workers = make_workers(&[
            "http://worker1:8000",
            "http://worker2:8000",
            "http://worker3:8000",
        ]);

        let mut headers = RequestHeaders::new();
        headers.insert("x-session-id".to_string(), "test-session-123".to_string());

        let idx1 = policy.select_worker_with_headers(&workers, None, Some(&headers));
        let idx2 = policy.select_worker_with_headers(&workers, None, Some(&headers));
        assert_eq!(idx1, idx2);
        assert!(idx1.is_some());
    }
}
