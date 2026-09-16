//! Consistent hashing load balancing policy
//!
//! This policy implements consistent hashing to route requests to workers based on
//! session ID or user ID, ensuring that requests from the same user/session are
//! consistently routed to the same worker for better cache locality.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use tracing::debug;
use tracing::info;

use super::get_healthy_worker_indices;
use super::hash_key;
use super::LoadBalancingPolicy;
use super::RequestHeaders;
use crate::core::Worker;
use crate::metrics::RouterMetrics;

/// Number of virtual nodes per physical worker (for better load distribution)
pub const VIRTUAL_NODES_PER_WORKER: u32 = 160;

/// Consistent hashing policy
///
/// Routes requests based on session ID or user ID using consistent hashing,
/// ensuring that requests from the same user/session consistently go to the same worker.
#[derive(Debug)]
pub struct ConsistentHashPolicy {
    /// Hash ring mapping hash values to worker URLs
    hash_ring: RwLock<BTreeMap<u64, String>>,
    /// Current set of workers (for detecting changes)
    current_workers: RwLock<Vec<String>>,
}

impl ConsistentHashPolicy {
    pub fn new() -> Self {
        Self {
            hash_ring: RwLock::new(BTreeMap::new()),
            current_workers: RwLock::new(Vec::new()),
        }
    }

    /// MurmurHash64A implementation from Facebook's mcrouter/lib/fbi/hash.c
    fn murmur_hash_64a(key: &[u8], seed: u32) -> u64 {
        const M: u64 = 0xc6a4a7935bd1e995;
        const R: i32 = 47;

        let mut h = (seed as u64) ^ ((key.len() as u64).wrapping_mul(M));

        // Process 8-byte chunks
        let chunks = key.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut k = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);

            k = k.wrapping_mul(M);
            k ^= k >> R;
            k = k.wrapping_mul(M);

            h ^= k;
            h = h.wrapping_mul(M);
        }

        // Process remaining bytes
        match remainder.len() {
            7 => {
                h ^= (remainder[6] as u64) << 48;
                h ^= (remainder[5] as u64) << 40;
                h ^= (remainder[4] as u64) << 32;
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            6 => {
                h ^= (remainder[5] as u64) << 40;
                h ^= (remainder[4] as u64) << 32;
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            5 => {
                h ^= (remainder[4] as u64) << 32;
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            4 => {
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            3 => {
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            2 => {
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            1 => {
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            _ => {}
        }

        h ^= h >> R;
        h = h.wrapping_mul(M);
        h ^= h >> R;

        h
    }

    /// Optimized MurmurHash64A for rehashing uint64_t keys (from hash.c)
    fn murmur_rehash_64a(k: u64) -> u64 {
        const M: u64 = 0xc6a4a7935bd1e995;
        const R: i32 = 47;
        const SEED: u64 = 4193360111; // From hash.c

        let mut h = SEED ^ (8u64.wrapping_mul(M)); // sizeof(uint64_t) * M

        let mut k = k;
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);

        h ^= k;
        h = h.wrapping_mul(M);

        h ^= h >> R;
        h = h.wrapping_mul(M);
        h ^= h >> R;

        h
    }

    /// FurcHash implementation from Facebook's mcrouter/lib/fbi/hash.c
    /// This is the actual consistent hash function used by Facebook
    fn furc_hash(key: &str, m: u32) -> u32 {
        const MAX_TRIES: u32 = 32;
        const FURC_SHIFT: u32 = 23;
        const FURC_CACHE_SIZE: usize = 1024;

        if m <= 1 {
            return 0;
        }

        let key_bytes = key.as_bytes();
        let mut hash_cache: Vec<u64> = Vec::with_capacity(FURC_CACHE_SIZE);
        let mut old_ord: i32 = -1;

        // Calculate d = ceil(log2(m))
        let mut d = 0;
        while m > (1u32 << d) {
            d += 1;
        }

        let mut a = d;
        for _try in 0..MAX_TRIES {
            // Generate bits until we find a 1 bit or run out of depth
            while !Self::furc_get_bit(key_bytes, a, &mut hash_cache, &mut old_ord) {
                if d == 0 {
                    return 0;
                }
                d -= 1;
                a = d;
            }

            a += FURC_SHIFT;
            let mut num = 1u32;

            // Build the number bit by bit
            for _ in 0..d.saturating_sub(1) {
                num = (num << 1)
                    | Self::furc_get_bit(key_bytes, a, &mut hash_cache, &mut old_ord) as u32;
                a += FURC_SHIFT;
            }

            if num < m {
                return num;
            }
        }

        // Give up and return 0, which is always a legal value
        0
    }

    /// Bit generator function from furc_hash (from hash.c)
    fn furc_get_bit(key: &[u8], idx: u32, hash_cache: &mut Vec<u64>, old_ord: &mut i32) -> bool {
        const SEED: u32 = 4193360111;

        let ord = (idx >> 6) as i32; // idx / 64

        // Extend hash cache if needed
        if *old_ord < ord {
            for n in (*old_ord + 1)..=ord {
                let hash_value = if n == 0 {
                    Self::murmur_hash_64a(key, SEED)
                } else {
                    Self::murmur_rehash_64a(hash_cache[(n - 1) as usize])
                };

                // Ensure cache has enough capacity
                if hash_cache.len() <= n as usize {
                    hash_cache.resize((n as usize) + 1, 0);
                }
                hash_cache[n as usize] = hash_value;
            }
            *old_ord = ord;
        }

        // Extract the bit
        let hash_val = hash_cache[ord as usize];
        let bit_pos = idx & 0x3f; // idx % 64
        (hash_val >> bit_pos) & 0x1 != 0
    }

    /// Facebook-style hash function using furc_hash for consistent hashing
    pub fn fbi_hash(key: &str) -> u64 {
        // Use furc_hash with a large modulus to get good distribution
        // Then expand to u64 for our hash ring
        const LARGE_MODULUS: u32 = (1u32 << 23) - 1; // Max furc_hash modulus

        let furc_result = Self::furc_hash(key, LARGE_MODULUS);

        // Expand the 23-bit furc result to 64-bit by using MurmurHash
        // This gives us the full 64-bit space while preserving consistency
        Self::murmur_hash_64a(
            &furc_result.to_le_bytes(),
            4193360111, // Same seed as furc_hash
        )
    }

    /// Update the hash ring when workers change
    fn update_hash_ring(&self, workers: &[Arc<dyn Worker>]) {
        let worker_urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        // Check if workers have changed
        {
            let current = self.current_workers.read().unwrap();
            if *current == worker_urls {
                return; // No change needed
            }
        }

        // Rebuild hash ring
        let mut new_ring = BTreeMap::new();

        for worker_url in &worker_urls {
            // Create virtual nodes for better distribution
            for i in 0..VIRTUAL_NODES_PER_WORKER {
                let virtual_key = format!("{}:{}", worker_url, i);
                let hash_value = Self::fbi_hash(&virtual_key);
                new_ring.insert(hash_value, worker_url.clone());
            }
        }

        // Update both the ring and current workers
        {
            let mut ring = self.hash_ring.write().unwrap();
            *ring = new_ring;
        }
        {
            let mut current = self.current_workers.write().unwrap();
            *current = worker_urls;
        }

        info!(
            "Updated consistent hash ring with {} workers and {} virtual nodes",
            workers.len(),
            workers.len() as u32 * VIRTUAL_NODES_PER_WORKER
        );
    }

    /// Find the worker for a given hash key using consistent hashing
    fn find_worker_by_hash(&self, hash_key: &str) -> Option<String> {
        let hash_value = Self::fbi_hash(hash_key);

        let ring = self.hash_ring.read().unwrap();
        if ring.is_empty() {
            return None;
        }

        // Find the first worker with hash >= our hash value
        // If none found, wrap around to the first worker (smallest hash)
        let selected_worker = ring
            .range(hash_value..)
            .next()
            .or_else(|| ring.iter().next())
            .map(|(_, worker_url)| worker_url.clone());

        if let Some(ref worker) = selected_worker {
            debug!(
                "Consistent hash: key='{}' hash={:016x} -> worker='{}'",
                hash_key, hash_value, worker
            );
        }

        selected_worker
    }
}

impl LoadBalancingPolicy for ConsistentHashPolicy {
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

        // Update hash ring if needed
        self.update_hash_ring(workers);

        // Extract hash key with priority: headers > body > fallback
        let hash_key = hash_key::extract_hash_key(request_text, headers);

        // DEBUG: Log the request text and extracted hash key
        if let Some(text) = request_text {
            debug!("CONSISTENT_HASH_DEBUG: Request text length: {}", text.len());
        }
        if let Some(hdrs) = headers {
            debug!(
                "CONSISTENT_HASH_DEBUG: Headers available: {:?}",
                hdrs.keys().collect::<Vec<_>>()
            );
        }
        info!("CONSISTENT_HASH_DEBUG: Extracted hash key: {}", hash_key);

        // Find target worker using consistent hashing
        let target_worker_url = match self.find_worker_by_hash(&hash_key) {
            Some(url) => {
                info!(
                    "CONSISTENT_HASH_DEBUG: Hash key '{}' mapped to worker: {}",
                    hash_key, url
                );
                url
            }
            None => {
                // Fallback to first healthy worker if hash ring is empty
                let fallback_idx = healthy_indices[0];
                let worker_url = workers[fallback_idx].url();
                info!(
                    "CONSISTENT_HASH_DEBUG: Hash ring empty, falling back to worker: {}",
                    worker_url
                );
                RouterMetrics::record_processed_request(worker_url);
                RouterMetrics::record_policy_decision(self.name(), worker_url);
                return Some(fallback_idx);
            }
        };

        // Find the worker index that matches our target
        let selected_idx = workers.iter().position(|w| w.url() == target_worker_url);

        debug!(
            "CONSISTENT_HASH_DEBUG: Target worker URL: {}",
            target_worker_url
        );

        match selected_idx {
            Some(idx) => {
                // Verify the worker is healthy
                if workers[idx].is_healthy() && workers[idx].circuit_breaker().can_execute() {
                    let worker_url = workers[idx].url();
                    debug!(
                        "CONSISTENT_HASH_DEBUG: Selected worker at index {}: {}",
                        idx, worker_url
                    );
                    info!(
                        "Consistent hash routing: key='{}' -> worker='{}' (index={})",
                        hash_key, worker_url, idx
                    );

                    // Increment processed counter
                    workers[idx].increment_processed();
                    RouterMetrics::record_processed_request(worker_url);
                    RouterMetrics::record_policy_decision(self.name(), worker_url);

                    Some(idx)
                } else {
                    // Target worker is unhealthy, fall back to first healthy worker
                    debug!(
                        "Target worker '{}' is unhealthy, falling back to healthy worker",
                        workers[idx].url()
                    );
                    let fallback_idx = healthy_indices[0];
                    let worker_url = workers[fallback_idx].url();

                    workers[fallback_idx].increment_processed();
                    RouterMetrics::record_processed_request(worker_url);
                    RouterMetrics::record_policy_decision(self.name(), worker_url);

                    Some(fallback_idx)
                }
            }
            None => {
                // Worker not found in current set, fall back to first healthy worker
                debug!(
                    "Target worker '{}' not found in current worker set, falling back",
                    target_worker_url
                );
                let fallback_idx = healthy_indices[0];
                let worker_url = workers[fallback_idx].url();

                workers[fallback_idx].increment_processed();
                RouterMetrics::record_processed_request(worker_url);
                RouterMetrics::record_policy_decision(self.name(), worker_url);

                Some(fallback_idx)
            }
        }
    }

    fn name(&self) -> &'static str {
        "consistent_hash"
    }

    fn needs_request_text(&self) -> bool {
        true // We need request text to extract session_id/user_id as fallback
    }

    fn needs_headers(&self) -> bool {
        true // We prefer HTTP headers for routing (x-session-id, etc.)
    }

    fn reset(&self) {
        // Clear the hash ring and force rebuild on next request
        {
            let mut ring = self.hash_ring.write().unwrap();
            ring.clear();
        }
        {
            let mut current = self.current_workers.write().unwrap();
            current.clear();
        }
        info!("Consistent hash policy reset - hash ring cleared");
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> {
        // For PD mode, use consistent hashing for both prefill and decode
        // This ensures that the same user's prefill and decode requests
        // go to workers that can share state efficiently

        let prefill_idx =
            self.select_worker_with_headers(prefill_workers, request_text, headers)?;
        let decode_idx = self.select_worker_with_headers(decode_workers, request_text, headers)?;

        Some((prefill_idx, decode_idx))
    }
}

impl Default for ConsistentHashPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorker;
    use crate::core::WorkerType;

    #[test]
    fn test_fbi_hash_consistency() {
        let key = "test_session_123";
        let hash1 = ConsistentHashPolicy::fbi_hash(key);
        let hash2 = ConsistentHashPolicy::fbi_hash(key);
        assert_eq!(hash1, hash2, "Hash function should be deterministic");
    }

    #[test]
    fn test_consistent_hash_selection() {
        let policy = ConsistentHashPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://worker1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://worker2:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://worker3:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Same session should always go to same worker
        let request = r#"{"session_id": "consistent_test"}"#;
        let idx1 = policy.select_worker(&workers, Some(request));
        let idx2 = policy.select_worker(&workers, Some(request));
        let idx3 = policy.select_worker(&workers, Some(request));

        assert_eq!(idx1, idx2);
        assert_eq!(idx2, idx3);
        assert!(idx1.is_some());
    }
}
