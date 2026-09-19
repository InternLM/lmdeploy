#[cfg(test)]
mod test_pd_routing {
    use lmdeploy_router_rs::config::{
        CircuitBreakerConfig, ConnectionMode, LMDeployMigrationProtocol, PolicyConfig, RetryConfig,
        RouterConfig, RoutingMode,
    };
    use lmdeploy_router_rs::routers::RouterFactory;

    // ========================================================================
    // Phase 1: Basic PD Components and Router Creation
    // ========================================================================

    #[test]
    fn test_worker_types() {
        use lmdeploy_router_rs::core::{WorkerFactory, WorkerType};

        // Test worker creation for prefill servers
        let prefill_worker = WorkerFactory::create_prefill("http://prefill:8080".to_string());
        assert_eq!(prefill_worker.url(), "http://prefill:8080");
        assert_eq!(prefill_worker.worker_type(), WorkerType::Prefill);

        // Test worker creation for decode servers
        let decode_worker = WorkerFactory::create_decode("http://decode:8080".to_string());
        assert_eq!(decode_worker.url(), "http://decode:8080");
        match decode_worker.worker_type() {
            WorkerType::Decode => (),
            _ => panic!("Expected Decode worker type"),
        }

        // Test regular worker creation
        let regular_worker = WorkerFactory::create_regular("http://regular:8080".to_string());
        assert_eq!(regular_worker.url(), "http://regular:8080");
        match regular_worker.worker_type() {
            WorkerType::Regular => (),
            _ => panic!("Expected Regular worker type"),
        }
    }

    #[tokio::test]
    async fn test_pd_router_configuration() {
        // Test PD router configuration with various policies
        // In the new structure, RoutingMode and PolicyConfig are separate
        let test_cases = vec![
            (
                RoutingMode::LMDeployPrefillDecode {
                    prefill_urls: vec![
                        "http://prefill1:8080".to_string(),
                        "http://prefill2:8080".to_string(),
                    ],
                    decode_urls: vec![
                        "http://decode1:8080".to_string(),
                        "http://decode2:8080".to_string(),
                    ],
                    prefill_policy: None,
                    decode_policy: None,
                    migration_protocol: LMDeployMigrationProtocol::Rdma,
                    rdma_config: None,
                    dummy_prefill: false,
                },
                PolicyConfig::Random,
            ),
            (
                RoutingMode::LMDeployPrefillDecode {
                    prefill_urls: vec!["http://prefill:8080".to_string()],
                    decode_urls: vec!["http://decode:8080".to_string()],
                    prefill_policy: None,
                    decode_policy: None,
                    migration_protocol: LMDeployMigrationProtocol::Rdma,
                    rdma_config: None,
                    dummy_prefill: false,
                },
                PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 5,
                },
            ),
            (
                RoutingMode::LMDeployPrefillDecode {
                    prefill_urls: vec![
                        "http://p1:8080".to_string(),
                        "http://p2:8080".to_string(),
                        "http://p3:8080".to_string(),
                    ],
                    decode_urls: vec!["http://d1:8080".to_string(), "http://d2:8080".to_string()],
                    prefill_policy: None,
                    decode_policy: None,
                    migration_protocol: LMDeployMigrationProtocol::Rdma,
                    rdma_config: None,
                    dummy_prefill: false,
                },
                PolicyConfig::CacheAware {
                    cache_threshold: 0.7,
                    balance_abs_threshold: 20,
                    balance_rel_threshold: 1.2,
                    eviction_interval_secs: 60,
                    max_tree_size: 1000000,
                },
            ),
        ];

        for (mode, policy) in test_cases {
            let config = RouterConfig {
                mode,
                policy,
                host: "127.0.0.1".to_string(),
                port: 3001,
                max_payload_size: 1024 * 1024,
                request_timeout_secs: 60,
                worker_startup_timeout_secs: 10,
                worker_startup_check_interval_secs: 1,
                api_key: None,
                api_key_validation_urls: vec![],
                discovery: None,
                metrics: None,
                log_dir: None,
                log_level: None,
                request_id_headers: None,
                max_concurrent_requests: 64,
                queue_size: 0,
                queue_timeout_secs: 60,
                cors_allowed_origins: vec![],
                retry: RetryConfig::default(),
                circuit_breaker: CircuitBreakerConfig::default(),
                disable_retries: false,
                disable_circuit_breaker: false,
                health_check: lmdeploy_router_rs::config::HealthCheckConfig::default(),
                enable_igw: false,
                rate_limit_tokens_per_second: None,
                connection_mode: ConnectionMode::Http,
                history_backend: lmdeploy_router_rs::config::HistoryBackend::Memory,
                enable_profiling: false,
                profile_timeout_secs: 30,
                kv_connector: lmdeploy_router_rs::config::KvConnector::Nixl,
            };

            // Router creation will fail due to health checks, but config should be valid
            let app_context = lmdeploy_router_rs::server::AppContext::new(
                config.clone(),
                reqwest::Client::new(),
                64,
                None,
                config.api_key_validation_urls.clone(),
            )
            .expect("Failed to create AppContext");
            let app_context = std::sync::Arc::new(app_context);
            let result = RouterFactory::create_router(&app_context).await;
            assert!(result.is_err());
            let error_msg = result.unwrap_err();
            // Error should be about health/timeout, not configuration
            assert!(
                error_msg.contains("healthy") || error_msg.contains("timeout"),
                "Unexpected error: {}",
                error_msg
            );
        }
    }
}
