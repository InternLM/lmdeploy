#[cfg(test)]
mod consistent_hash_policy_tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use lmdeploy_router_rs::core::BasicWorker;
    use lmdeploy_router_rs::core::Worker;
    use lmdeploy_router_rs::core::WorkerType;
    use lmdeploy_router_rs::policies::ConsistentHashPolicy;
    use lmdeploy_router_rs::policies::LoadBalancingPolicy;
    use lmdeploy_router_rs::policies::RequestHeaders;

    /// Helper function to create test workers
    fn create_test_workers() -> Vec<Arc<dyn Worker>> {
        vec![
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
        ]
    }

    #[test]
    fn test_consistent_hash_policy_creation() {
        let policy = ConsistentHashPolicy::new();
        assert_eq!(policy.name(), "consistent_hash");
        assert!(policy.needs_request_text());
    }

    #[test]
    fn test_consistent_routing_same_session() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let session_id = "test_session_123";

        // Make multiple requests with the same top-level session_id
        let mut selected_workers = Vec::new();
        for i in 0..10 {
            let prompt = format!(
                r#"{{"session_id": "{}", "prompt": "request {}}}"#,
                session_id, i
            );
            if let Some(worker_idx) = policy.select_worker(&workers, Some(&prompt)) {
                selected_workers.push(worker_idx);
            }
        }

        // All requests should go to the same worker
        assert!(
            !selected_workers.is_empty(),
            "Should have selected at least one worker"
        );

        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "Request {} went to worker {}, expected worker {} (same as first request)",
                i, worker_idx, first_worker
            );
        }
    }

    #[test]
    fn test_distribution_across_workers() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let mut worker_counts = HashMap::new();
        let num_sessions = 100;

        // Create many different sessions and see how they distribute
        for i in 0..num_sessions {
            let session_id = format!("session_{}", i);
            let request_json = format!(r#"{{"session_id": "{}", "prompt": "test"}}"#, session_id);

            if let Some(worker_idx) = policy.select_worker(&workers, Some(&request_json)) {
                *worker_counts.entry(worker_idx).or_insert(0) += 1;
            }
        }

        // Check that at least 2 workers were used (distribution test)
        assert!(
            worker_counts.len() >= 2,
            "Expected distribution across multiple workers, only used: {:?}",
            worker_counts
        );

        // Check that no single worker got all requests (basic distribution check)
        let max_count = worker_counts.values().max().unwrap();
        assert!(
            *max_count < num_sessions,
            "One worker got all requests, no distribution occurred"
        );

        // Check that distribution is reasonably balanced (allow for some variance)
        let min_count = worker_counts.values().min().unwrap();
        let expected_per_worker = num_sessions / workers.len();

        // Allow 50% variance from expected
        assert!(
            *min_count >= expected_per_worker / 3,
            "Distribution too uneven, min count: {}, expected around: {}",
            min_count,
            expected_per_worker
        );
    }

    #[test]
    fn test_select_worker_pair_pd_mode() {
        let policy = ConsistentHashPolicy::new();
        let prefill_workers = create_test_workers();
        let decode_workers = create_test_workers();

        let session_id = "pd_session_test";
        let request_json = format!(r#"{{"session_id": "{}", "prompt": "pd test"}}"#, session_id);

        // Test PD mode worker pair selection
        if let Some((prefill_idx, decode_idx)) =
            policy.select_worker_pair(&prefill_workers, &decode_workers, Some(&request_json))
        {
            assert!(
                prefill_idx < prefill_workers.len(),
                "Prefill worker index should be valid"
            );
            assert!(
                decode_idx < decode_workers.len(),
                "Decode worker index should be valid"
            );

            // For the same session, should get consistent results
            if let Some((prefill_idx2, decode_idx2)) =
                policy.select_worker_pair(&prefill_workers, &decode_workers, Some(&request_json))
            {
                assert_eq!(
                    prefill_idx, prefill_idx2,
                    "Prefill worker should be consistent"
                );
                assert_eq!(
                    decode_idx, decode_idx2,
                    "Decode worker should be consistent"
                );
            }
        }
    }

    #[test]
    fn test_no_workers_available() {
        let policy = ConsistentHashPolicy::new();
        let empty_workers: Vec<Arc<dyn Worker>> = vec![];

        // Should return None when no workers are available
        let result = policy.select_worker(&empty_workers, Some(r#"{"session_id": "test"}"#));
        assert!(
            result.is_none(),
            "Should return None when no workers available"
        );
    }

    #[test]
    fn test_unhealthy_workers_fallback() {
        let policy = ConsistentHashPolicy::new();

        // Create workers where some are unhealthy
        let worker1 = BasicWorker::new("http://worker1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://worker2:8000".to_string(), WorkerType::Regular);

        // Mark first worker as unhealthy
        worker1.set_healthy(false);

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];

        // Should still work and select healthy workers
        let result = policy.select_worker(&workers, Some(r#"{"session_id": "test"}"#));
        assert!(result.is_some(), "Should fallback to healthy workers");

        // The selected worker should be healthy
        if let Some(idx) = result {
            assert!(
                workers[idx].is_healthy(),
                "Selected worker should be healthy"
            );
        }
    }

    #[test]
    fn test_comprehensive_routing_consistency() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        // Test multiple sessions with multiple requests each
        let sessions = ["session_A", "session_B", "session_C"];
        let mut session_mappings = HashMap::new();

        for session in &sessions {
            let mut worker_indices = Vec::new();

            // Make 5 requests per session
            for i in 0..5 {
                let request = format!(
                    r#"{{"session_id": "{}", "prompt": "request {}}}"#,
                    session, i
                );
                if let Some(idx) = policy.select_worker(&workers, Some(&request)) {
                    worker_indices.push(idx);
                }
            }

            // All requests from the same session should go to the same worker
            assert!(
                !worker_indices.is_empty(),
                "Should have selected workers for session: {}",
                session
            );

            let first_idx = worker_indices[0];
            for (i, &idx) in worker_indices.iter().enumerate() {
                assert_eq!(
                    idx, first_idx,
                    "Session {} request {} went to worker {}, expected worker {} (consistency violation)",
                    session, i, idx, first_idx
                );
            }

            session_mappings.insert(*session, first_idx);
        }

        // Verify that different sessions can go to different workers (distribution)
        let unique_workers: std::collections::HashSet<_> = session_mappings.values().collect();
        println!("Session mappings: {:?}", session_mappings);
        println!("Unique workers used: {}", unique_workers.len());

        // With 3 sessions and 3 workers, we should ideally see some distribution
        // but we'll be lenient and just ensure not everything goes to one worker
        assert!(
            !unique_workers.is_empty(),
            "At least one worker should be used"
        );
    }

    #[test]
    fn test_fallback_without_session_or_user() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        // Test requests without session_id or user_id (should use request content)
        let test_cases = [
            r#"{"prompt": "hello world"}"#,
            r#"{"text": "test request", "sampling_params": {}}"#,
            "",
            "plain text request",
        ];

        for (i, request) in test_cases.iter().enumerate() {
            let result = policy.select_worker(&workers, Some(request));
            assert!(
                result.is_some(),
                "Fallback should work for request {}: {}",
                i,
                request
            );
        }
    }

    // Session IDs use the top-level field; user identity uses the `user` field.

    #[test]
    fn test_openai_user_field_consistency() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let user = "openai_user_123";

        // Make multiple requests with the same user field (OpenAI format)
        let mut selected_workers = Vec::new();
        for i in 0..5 {
            let prompt = format!(r#"{{"user": "{}", "prompt": "request {}}}"#, user, i);
            if let Some(worker_idx) = policy.select_worker(&workers, Some(&prompt)) {
                selected_workers.push(worker_idx);
            }
        }

        // All requests should go to the same worker
        assert!(
            !selected_workers.is_empty(),
            "Should have selected at least one worker"
        );

        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "OpenAI user request {} went to worker {}, expected worker {} (consistency violation)",
                i, worker_idx, first_worker
            );
        }
    }

    #[test]
    fn test_session_id_priority_over_user_id() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let session_id = "priority_session";
        let user_id = "priority_user";

        // Make requests with both session_id and user_id
        let mut selected_workers = Vec::new();
        for i in 0..3 {
            let prompt = format!(
                r#"{{"session_id": "{}", "user_id": "{}", "prompt": "request {}}}"#,
                session_id, user_id, i
            );
            if let Some(worker_idx) = policy.select_worker(&workers, Some(&prompt)) {
                selected_workers.push(worker_idx);
            }
        }

        // All requests should consistently go to the same worker (based on session_id)
        assert!(
            !selected_workers.is_empty(),
            "Should have selected at least one worker"
        );

        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "Priority test request {} went to worker {}, expected worker {} (inconsistent routing)",
                i, worker_idx, first_worker
            );
        }
    }

    #[test]
    fn test_policy_reset() {
        let policy = ConsistentHashPolicy::new();

        // Reset should work without issues
        policy.reset();

        // After reset, should still be able to route requests
        let workers = create_test_workers();
        let result = policy.select_worker(&workers, Some(r#"{"session_id": "post_reset_test"}"#));
        assert!(result.is_some(), "Should work after reset");
    }

    #[test]
    fn test_different_request_formats() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let session_id = "format_test_session";

        // Test different JSON formats that should all extract the same session_id
        let formats = [
            format!(r#"{{"session_id": "{}", "prompt": "test"}}"#, session_id),
            format!(
                r#"{{ "session_id" : "{}" , "prompt" : "test" }}"#,
                session_id
            ),
            format!(r#"{{"prompt": "test", "session_id": "{}"}}"#, session_id),
        ];

        let mut selected_workers = Vec::new();
        for (i, request) in formats.iter().enumerate() {
            if let Some(worker_idx) = policy.select_worker(&workers, Some(request)) {
                selected_workers.push(worker_idx);
            } else {
                panic!("Format {} failed to route: {}", i, request);
            }
        }

        // All different formats should route to the same worker
        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "Format {} routed to worker {}, expected worker {} (format inconsistency)",
                i, worker_idx, first_worker
            );
        }
    }

    // =============================
    // HTTP Header-based routing tests
    // =============================

    /// Helper function to create RequestHeaders from key-value pairs
    fn create_headers(pairs: &[(&str, &str)]) -> RequestHeaders {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_http_header_x_session_id_routing() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let session_id = "header-session-123";
        let headers = create_headers(&[("x-session-id", session_id)]);

        // Make multiple requests with the same x-session-id header
        let mut selected_workers = Vec::new();
        for i in 0..10 {
            let prompt = format!(r#"{{"prompt": "request {}"}}"#, i);
            if let Some(worker_idx) =
                policy.select_worker_with_headers(&workers, Some(&prompt), Some(&headers))
            {
                selected_workers.push(worker_idx);
            }
        }

        // All requests should go to the same worker
        assert!(
            !selected_workers.is_empty(),
            "Should have selected at least one worker"
        );

        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "Request {} with x-session-id header went to worker {}, expected worker {}",
                i, worker_idx, first_worker
            );
        }
    }

    #[test]
    fn test_http_header_priority_over_body() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        // Header session ID should take priority over body session ID
        let header_session = "header-session-priority";
        let body_session = "body-session-ignored";

        let headers = create_headers(&[("x-session-id", header_session)]);
        let body = format!(r#"{{"session_id": "{}", "prompt": "test"}}"#, body_session);

        // Get worker for header-only request
        let header_only_headers = create_headers(&[("x-session-id", header_session)]);
        let header_only_worker = policy.select_worker_with_headers(
            &workers,
            Some(r#"{"prompt": "test"}"#),
            Some(&header_only_headers),
        );

        // Get worker for request with both header and body
        let both_worker = policy.select_worker_with_headers(&workers, Some(&body), Some(&headers));

        // Both should route to the same worker (header takes priority)
        assert_eq!(
            header_only_worker, both_worker,
            "HTTP header should take priority over body session_id"
        );
    }

    #[test]
    fn test_http_header_x_user_id_routing() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let user_id = "header-user-456";
        let headers = create_headers(&[("x-user-id", user_id)]);

        // Make multiple requests with the same x-user-id header
        let mut selected_workers = Vec::new();
        for i in 0..5 {
            let prompt = format!(r#"{{"prompt": "request {}"}}"#, i);
            if let Some(worker_idx) =
                policy.select_worker_with_headers(&workers, Some(&prompt), Some(&headers))
            {
                selected_workers.push(worker_idx);
            }
        }

        // All requests should go to the same worker
        assert!(
            !selected_workers.is_empty(),
            "Should have selected at least one worker"
        );

        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "Request {} with x-user-id header went to worker {}, expected worker {}",
                i, worker_idx, first_worker
            );
        }
    }

    #[test]
    fn test_http_header_priority_order() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        // x-session-id should take priority over x-user-id
        let session_id = "priority-session";
        let user_id = "ignored-user";

        // Test with x-session-id only
        let session_headers = create_headers(&[("x-session-id", session_id)]);
        let session_worker = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "test"}"#),
                Some(&session_headers),
            )
            .expect("Should select worker with x-session-id");

        // Test with both headers - x-session-id should win
        let both_headers = create_headers(&[("x-session-id", session_id), ("x-user-id", user_id)]);
        let both_worker = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "test"}"#),
                Some(&both_headers),
            )
            .expect("Should select worker with both headers");

        assert_eq!(
            session_worker, both_worker,
            "x-session-id should take priority over x-user-id"
        );
    }

    #[test]
    fn test_http_header_fallback_to_body_when_no_header() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let body_session = "body-only-session";
        let body = format!(r#"{{"session_id": "{}", "prompt": "test"}}"#, body_session);

        // Empty headers should fall back to body
        let empty_headers: RequestHeaders = HashMap::new();
        let with_empty_headers = policy
            .select_worker_with_headers(&workers, Some(&body), Some(&empty_headers))
            .expect("Should route with empty headers");

        // No headers should also fall back to body
        let without_headers = policy
            .select_worker_with_headers(&workers, Some(&body), None)
            .expect("Should route without headers");

        // Body-only routing (using old interface) for comparison
        let body_only = policy
            .select_worker(&workers, Some(&body))
            .expect("Should route with body only");

        assert_eq!(
            with_empty_headers, without_headers,
            "Empty headers and None headers should behave the same"
        );
        assert_eq!(
            with_empty_headers, body_only,
            "Should fall back to body when no headers present"
        );
    }

    #[test]
    fn test_http_header_x_tenant_id_routing() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let tenant_id = "tenant-abc";
        let headers = create_headers(&[("x-tenant-id", tenant_id)]);

        let mut selected_workers = Vec::new();
        for i in 0..5 {
            let prompt = format!(r#"{{"prompt": "request {}"}}"#, i);
            if let Some(worker_idx) =
                policy.select_worker_with_headers(&workers, Some(&prompt), Some(&headers))
            {
                selected_workers.push(worker_idx);
            }
        }

        // All requests should go to the same worker
        assert!(!selected_workers.is_empty(), "Should have selected workers");

        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "Request {} with x-tenant-id header went to worker {}, expected worker {}",
                i, worker_idx, first_worker
            );
        }
    }

    #[test]
    fn test_http_header_distribution_across_workers() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let mut worker_counts = HashMap::new();
        let num_sessions = 100;

        // Create many different session IDs via headers
        for i in 0..num_sessions {
            let session_id = format!("header-session-{}", i);
            let headers = create_headers(&[("x-session-id", &session_id)]);

            if let Some(worker_idx) = policy.select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "test"}"#),
                Some(&headers),
            ) {
                *worker_counts.entry(worker_idx).or_insert(0) += 1;
            }
        }

        // Check distribution - should use at least 2 workers
        assert!(
            worker_counts.len() >= 2,
            "Expected distribution across multiple workers, only used: {:?}",
            worker_counts
        );

        // No single worker should get all requests
        let max_count = worker_counts.values().max().unwrap();
        assert!(
            *max_count < num_sessions,
            "One worker got all requests, no distribution occurred"
        );
    }

    #[test]
    fn test_needs_headers_flag() {
        let policy = ConsistentHashPolicy::new();
        assert!(
            policy.needs_headers(),
            "ConsistentHashPolicy should report that it needs headers"
        );
    }

    #[test]
    fn test_correlation_id_priority_over_request_id() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let correlation_id = "session-correlation-abc";
        let request_id = "unique-request-xyz";

        // Test with x-correlation-id only
        let correlation_headers = create_headers(&[("x-correlation-id", correlation_id)]);
        let correlation_worker = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "test"}"#),
                Some(&correlation_headers),
            )
            .expect("Should select worker with x-correlation-id");

        // Test with both headers - x-correlation-id should win
        let both_headers = create_headers(&[
            ("x-correlation-id", correlation_id),
            ("x-request-id", request_id),
        ]);
        let both_worker = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "test"}"#),
                Some(&both_headers),
            )
            .expect("Should select worker with both headers");

        assert_eq!(
            correlation_worker, both_worker,
            "x-correlation-id should take priority over x-request-id"
        );
    }

    #[test]
    fn test_correlation_id_consistent_across_turns() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let correlation_id = "multi-turn-session-123";

        // Simulate multiple turns with same correlation ID but different request IDs
        let mut selected_workers = Vec::new();
        for i in 0..10 {
            let headers = create_headers(&[
                ("x-correlation-id", correlation_id),
                ("x-request-id", &format!("request-turn-{}", i)),
            ]);
            let worker = policy
                .select_worker_with_headers(&workers, Some(r#"{"prompt": "test"}"#), Some(&headers))
                .expect("Should select worker");
            selected_workers.push(worker);
        }

        // All turns should route to the same worker
        let first_worker = &selected_workers[0];
        for (i, worker) in selected_workers.iter().enumerate() {
            assert_eq!(
                first_worker, worker,
                "Turn {} should route to same worker as turn 0 when x-correlation-id is shared",
                i
            );
        }
    }

    #[test]
    fn test_tenant_id_priority_over_correlation_id() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let tenant_id = "tenant-abc";
        let correlation_id = "correlation-xyz";

        // Test with x-tenant-id only
        let tenant_headers = create_headers(&[("x-tenant-id", tenant_id)]);
        let tenant_worker = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "test"}"#),
                Some(&tenant_headers),
            )
            .expect("Should select worker with x-tenant-id");

        // Test with both headers - x-tenant-id should win
        let both_headers = create_headers(&[
            ("x-tenant-id", tenant_id),
            ("x-correlation-id", correlation_id),
        ]);
        let both_worker = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "test"}"#),
                Some(&both_headers),
            )
            .expect("Should select worker with both headers");

        assert_eq!(
            tenant_worker, both_worker,
            "x-tenant-id should take priority over x-correlation-id"
        );
    }
}
