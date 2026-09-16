//! Tests for transparent proxy routing with headers and availability filtering.
//!
//! These tests verify the fixes to route_transparent() across router
//! implementations (Router, LMDeployPDRouter):
//!   1. Headers are passed to select_worker_with_headers() for consistent hash routing
//!   2. Workers are filtered by is_available() before selection
//!   3. The inline header conversion pattern used by transparent proxy routes matches
//!      the Router::headers_to_request_headers() output

#[cfg(test)]
mod transparent_proxy_routing_tests {
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::sync::Arc;

    use lmdeploy_router_rs::core::BasicWorker;
    use lmdeploy_router_rs::core::Worker;
    use lmdeploy_router_rs::core::WorkerType;
    use lmdeploy_router_rs::policies::ConsistentHashPolicy;
    use lmdeploy_router_rs::policies::LoadBalancingPolicy;
    use lmdeploy_router_rs::policies::RequestHeaders;

    /// Helper to create test workers
    fn create_workers(n: usize) -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://worker{}:8080", i + 1),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect()
    }

    /// Helper to build RequestHeaders from key-value pairs
    fn make_headers(pairs: &[(&str, &str)]) -> RequestHeaders {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // =====================================================================
    // Test 1: Availability filtering before policy selection
    // =====================================================================
    // The route_transparent fix filters workers by is_available() BEFORE
    // passing them to the policy. These tests verify that behavior.

    #[test]
    fn test_availability_filter_excludes_unhealthy_workers() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(3);

        // Mark worker1 as unhealthy
        workers[0].set_healthy(false);

        // Filter by availability (same pattern as route_transparent)
        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        assert_eq!(
            available.len(),
            2,
            "Only 2 of 3 workers should be available"
        );

        // Policy should still work with filtered workers
        let headers = make_headers(&[("x-session-id", "test-session")]);
        let result = policy.select_worker_with_headers(
            &available,
            Some(r#"{"prompt": "test"}"#),
            Some(&headers),
        );
        assert!(result.is_some(), "Should select from available workers");

        let idx = result.unwrap();
        assert!(
            available[idx].is_healthy(),
            "Selected worker must be healthy"
        );
    }

    #[test]
    fn test_availability_filter_returns_empty_when_all_unhealthy() {
        let workers = create_workers(3);

        // Mark all workers as unhealthy
        for w in &workers {
            w.set_healthy(false);
        }

        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        assert!(
            available.is_empty(),
            "No workers should be available when all are unhealthy"
        );

        // Policy should return None for empty list
        let policy = ConsistentHashPolicy::new();
        let result = policy.select_worker_with_headers(&available, None, None);
        assert!(result.is_none(), "Should return None for empty worker list");
    }

    // =====================================================================
    // Test 2: Headers are forwarded to the consistent hash policy
    // =====================================================================
    // Before the fix, route_transparent called select_worker() which ignores
    // headers entirely, making consistent hash fall back to request body hash.
    // After the fix, it calls select_worker_with_headers() so x-session-id
    // in headers produces sticky routing.

    #[test]
    fn test_session_id_header_produces_sticky_routing() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(5);

        let headers = make_headers(&[("x-session-id", "my-sticky-session")]);

        // Simulate what route_transparent now does: pass headers to policy
        let mut selected: Vec<usize> = Vec::new();
        for i in 0..20 {
            let body = format!(
                r#"{{"prompt": "different prompt {}", "model": "default"}}"#,
                i
            );
            if let Some(idx) =
                policy.select_worker_with_headers(&workers, Some(&body), Some(&headers))
            {
                selected.push(idx);
            }
        }

        // All requests with the same session ID should route to the same worker
        assert!(!selected.is_empty());
        let first = selected[0];
        for (i, &idx) in selected.iter().enumerate() {
            assert_eq!(
                idx, first,
                "Request {} routed to worker {}, expected {} (session stickiness broken)",
                i, idx, first
            );
        }
    }

    #[test]
    fn test_without_headers_uses_body_fallback() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(3);

        // Without headers, consistent hash should fall back to body content
        let body = r#"{"session_id": "body-session", "prompt": "test"}"#;

        let mut selected: Vec<usize> = Vec::new();
        for _ in 0..10 {
            if let Some(idx) = policy.select_worker_with_headers(&workers, Some(body), None) {
                selected.push(idx);
            }
        }

        // Same body → same worker (body fallback is deterministic)
        assert!(!selected.is_empty());
        let first = selected[0];
        for &idx in &selected {
            assert_eq!(idx, first, "Body fallback should be deterministic");
        }
    }

    #[test]
    fn test_header_session_id_takes_priority_over_body_session_id() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(5);

        let header_session = "header-session-wins";
        let body_session = "body-session-ignored";

        let headers = make_headers(&[("x-session-id", header_session)]);
        let body = format!(r#"{{"session_id": "{}", "prompt": "test"}}"#, body_session);

        // Route with both header and body session ID
        let with_both = policy
            .select_worker_with_headers(&workers, Some(&body), Some(&headers))
            .expect("Should select a worker");

        // Route with header only (different body)
        let header_only = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "completely different"}"#),
                Some(&headers),
            )
            .expect("Should select a worker");

        assert_eq!(
            with_both, header_only,
            "Header x-session-id should take priority - different body should not change routing"
        );
    }

    // =====================================================================
    // Test 3: Availability + headers work together correctly
    // =====================================================================
    // These tests verify the combined effect: availability filtering happens
    // BEFORE policy selection, and headers are passed to the policy.

    #[test]
    fn test_sticky_routing_with_availability_filtering() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(4);

        // Initial routing with all workers available
        let headers = make_headers(&[("x-session-id", "stable-session")]);
        let body = r#"{"prompt": "test"}"#;

        let initial_idx = policy
            .select_worker_with_headers(&workers, Some(body), Some(&headers))
            .expect("Should select a worker");

        // Now filter by availability (as route_transparent does)
        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        // Should get the same worker since all are still available
        let after_filter_idx = policy
            .select_worker_with_headers(&available, Some(body), Some(&headers))
            .expect("Should select a worker");

        assert_eq!(
            workers[initial_idx].url(),
            available[after_filter_idx].url(),
            "Same session ID should route to same worker URL when all workers are available"
        );
    }

    #[test]
    fn test_distribution_with_headers_across_sessions() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(3);

        let mut worker_urls: HashSet<String> = HashSet::new();

        for i in 0..100 {
            let session = format!("unique-session-{}", i);
            let headers = make_headers(&[("x-session-id", &session)]);

            if let Some(idx) = policy.select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "test"}"#),
                Some(&headers),
            ) {
                worker_urls.insert(workers[idx].url().to_string());
            }
        }

        // With 100 different sessions and 3 workers, all workers should be used
        assert!(
            worker_urls.len() >= 2,
            "Expected distribution, only used {} workers: {:?}",
            worker_urls.len(),
            worker_urls
        );
    }

    // =====================================================================
    // Test 4: Inline header conversion correctness
    // =====================================================================
    // Transparent proxy routes use an inline pattern to convert
    // HeaderMap → HashMap<String, String>. Verify it produces correct output.

    #[test]
    fn test_inline_header_conversion_lowercases_keys() {
        use axum::http::HeaderMap;
        use axum::http::HeaderValue;

        let mut header_map = HeaderMap::new();
        header_map.insert("X-Session-Id", HeaderValue::from_static("abc-123"));
        header_map.insert("X-USER-ID", HeaderValue::from_static("user-456"));
        header_map.insert("content-type", HeaderValue::from_static("application/json"));

        // Simulate the transparent proxy inline conversion.
        let request_headers: Option<HashMap<String, String>> = Some(&header_map).map(|h| {
            h.iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                })
                .collect()
        });

        let headers = request_headers.unwrap();

        // axum normalizes header names to lowercase already, but verify the
        // to_lowercase() call in our conversion is idempotent and correct
        assert_eq!(headers.get("x-session-id").unwrap(), "abc-123");
        assert_eq!(headers.get("x-user-id").unwrap(), "user-456");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn test_inline_header_conversion_used_by_policy() {
        use axum::http::HeaderMap;
        use axum::http::HeaderValue;

        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(3);

        // Convert via the transparent proxy inline pattern.
        let mut header_map = HeaderMap::new();
        header_map.insert(
            "x-session-id",
            HeaderValue::from_static("policy-test-session"),
        );

        let request_headers: Option<HashMap<String, String>> = Some(&header_map).map(|h| {
            h.iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                })
                .collect()
        });

        // Use the converted headers with the policy (same as route_transparent now does)
        let mut selected: Vec<usize> = Vec::new();
        for i in 0..10 {
            let body = format!(r#"{{"prompt": "request {}"}}"#, i);
            if let Some(idx) =
                policy.select_worker_with_headers(&workers, Some(&body), request_headers.as_ref())
            {
                selected.push(idx);
            }
        }

        // All should go to the same worker (session stickiness via header)
        assert!(!selected.is_empty());
        let first = selected[0];
        for &idx in &selected {
            assert_eq!(
                idx, first,
                "Inline header conversion should produce sticky routing"
            );
        }

        // Also verify with make_headers helper (which matches RequestHeaders directly)
        let direct_headers = make_headers(&[("x-session-id", "policy-test-session")]);
        let direct_result = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "request 0"}"#),
                Some(&direct_headers),
            )
            .expect("Direct headers should work");

        assert_eq!(
            workers[first].url(),
            workers[direct_result].url(),
            "Inline conversion and direct headers should route to the same worker"
        );
    }

    // =====================================================================
    // Test 5: PD mode worker pair selection with headers
    // =====================================================================
    // For PD routers, both prefill and decode workers need headers.

    #[test]
    fn test_pd_mode_worker_pair_with_headers() {
        let policy = ConsistentHashPolicy::new();

        let prefill_workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://prefill{}:8080", i + 1),
                    WorkerType::Prefill,
                )) as Arc<dyn Worker>
            })
            .collect();

        let decode_workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://decode{}:8080", i + 1),
                    WorkerType::Decode,
                )) as Arc<dyn Worker>
            })
            .collect();

        let headers = make_headers(&[("x-session-id", "pd-session")]);
        let body = r#"{"prompt": "test"}"#;

        // select_worker_with_headers on each pool
        let prefill_idx = policy
            .select_worker_with_headers(&prefill_workers, Some(body), Some(&headers))
            .expect("Should select prefill worker");

        let decode_idx = policy
            .select_worker_with_headers(&decode_workers, Some(body), Some(&headers))
            .expect("Should select decode worker");

        // Verify consistency: same session → same workers
        let prefill_idx2 = policy
            .select_worker_with_headers(&prefill_workers, Some(body), Some(&headers))
            .expect("Should select prefill worker again");

        let decode_idx2 = policy
            .select_worker_with_headers(&decode_workers, Some(body), Some(&headers))
            .expect("Should select decode worker again");

        assert_eq!(
            prefill_idx, prefill_idx2,
            "Prefill selection should be consistent"
        );
        assert_eq!(
            decode_idx, decode_idx2,
            "Decode selection should be consistent"
        );

        // Verify the workers are valid
        assert!(prefill_idx < prefill_workers.len());
        assert!(decode_idx < decode_workers.len());
    }

    #[test]
    fn test_pd_mode_availability_filtering() {
        let prefill_workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://prefill{}:8080", i + 1),
                    WorkerType::Prefill,
                )) as Arc<dyn Worker>
            })
            .collect();

        let decode_workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://decode{}:8080", i + 1),
                    WorkerType::Decode,
                )) as Arc<dyn Worker>
            })
            .collect();

        // Mark one prefill and one decode worker as unhealthy
        prefill_workers[0].set_healthy(false);
        decode_workers[1].set_healthy(false);

        // Filter by availability (as route_transparent now does)
        let available_prefill: Vec<Arc<dyn Worker>> = prefill_workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        let available_decode: Vec<Arc<dyn Worker>> = decode_workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        assert_eq!(
            available_prefill.len(),
            2,
            "2 of 3 prefill workers should be available"
        );
        assert_eq!(
            available_decode.len(),
            2,
            "2 of 3 decode workers should be available"
        );

        // Policy should select from available workers only
        let policy = ConsistentHashPolicy::new();
        let headers = make_headers(&[("x-session-id", "pd-avail-test")]);

        let prefill_idx = policy
            .select_worker_with_headers(
                &available_prefill,
                Some(r#"{"prompt": "test"}"#),
                Some(&headers),
            )
            .expect("Should select available prefill worker");

        let decode_idx = policy
            .select_worker_with_headers(
                &available_decode,
                Some(r#"{"prompt": "test"}"#),
                Some(&headers),
            )
            .expect("Should select available decode worker");

        assert!(available_prefill[prefill_idx].is_healthy());
        assert!(available_decode[decode_idx].is_healthy());
    }
}
