pub mod config;
pub mod logging;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::collections::HashMap;

pub mod core;
pub mod data_connector;
pub mod metrics;
pub mod middleware;
pub mod otel_http;
pub mod otel_trace;
pub mod policies;
pub mod protocols;
pub mod routers;
pub mod server;
pub mod service_discovery;
pub mod tokenizer;
pub mod tree;
#[cfg(feature = "python")]
use crate::metrics::PrometheusConfig;

#[cfg(feature = "python")]
#[pyclass(eq)]
#[derive(Clone, PartialEq, Debug)]
pub enum PolicyType {
    Random,
    RoundRobin,
    CacheAware,
    PowerOfTwo,
    ConsistentHash,
    RendezvousHash,
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
struct Router {
    host: String,
    port: u16,
    worker_urls: Vec<String>,
    policy: PolicyType,
    worker_startup_timeout_secs: u64,
    worker_startup_check_interval: u64,
    cache_threshold: f32,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    eviction_interval_secs: u64,
    max_tree_size: usize,
    max_payload_size: usize,
    api_key: Option<String>,
    api_key_validation_urls: Vec<String>,
    log_dir: Option<String>,

    log_level: Option<String>,
    service_discovery: bool,
    selector: HashMap<String, String>,
    service_discovery_port: u16,
    service_discovery_namespace: Option<String>,
    prefill_selector: HashMap<String, String>,
    decode_selector: HashMap<String, String>,
    prometheus_port: Option<u16>,
    prometheus_host: Option<String>,
    request_timeout_secs: u64,
    request_id_headers: Option<Vec<String>>,
    prefill_urls: Option<Vec<String>>,
    decode_urls: Option<Vec<String>>,
    prefill_policy: Option<PolicyType>,
    decode_policy: Option<PolicyType>,
    max_concurrent_requests: usize,
    cors_allowed_origins: Vec<String>,
    // Retry configuration
    retry_max_retries: u32,
    retry_initial_backoff_ms: u64,
    retry_max_backoff_ms: u64,
    retry_backoff_multiplier: f32,
    retry_jitter_factor: f32,
    disable_retries: bool,
    // Circuit breaker configuration
    cb_failure_threshold: u32,
    cb_success_threshold: u32,
    cb_timeout_duration_secs: u64,
    cb_window_duration_secs: u64,
    disable_circuit_breaker: bool,
    // Health check configuration
    health_failure_threshold: u32,
    health_success_threshold: u32,
    health_check_timeout_secs: u64,
    health_check_interval_secs: u64,
    health_check_endpoint: String,
    // IGW (Inference Gateway) configuration
    enable_igw: bool,
    queue_size: usize,
    queue_timeout_secs: u64,
    rate_limit_tokens_per_second: Option<usize>,
    // OpenTelemetry tracing
    enable_trace: bool,
    otlp_traces_endpoint: Option<String>,
    // KV connector for PD disaggregation ("nixl" or "mooncake")
    kv_connector: String,
    // lmdeploy PD disaggregation
    lmdeploy_pd_disaggregation: bool,
    lmdeploy_migration_protocol: String,
    lmdeploy_rdma_link_type: String,
    lmdeploy_with_gdr: bool,
    lmdeploy_dummy_prefill: bool,
}

#[cfg(feature = "python")]
impl Router {
    /// Convert PyO3 Router to RouterConfig
    pub fn to_router_config(&self) -> config::ConfigResult<config::RouterConfig> {
        use config::{
            DiscoveryConfig, MetricsConfig, PolicyConfig as ConfigPolicyConfig, RoutingMode,
        };

        // Convert policy helper function
        let convert_policy = |policy: &PolicyType| -> ConfigPolicyConfig {
            match policy {
                PolicyType::Random => ConfigPolicyConfig::Random,
                PolicyType::RoundRobin => ConfigPolicyConfig::RoundRobin,
                PolicyType::CacheAware => ConfigPolicyConfig::CacheAware {
                    cache_threshold: self.cache_threshold,
                    balance_abs_threshold: self.balance_abs_threshold,
                    balance_rel_threshold: self.balance_rel_threshold,
                    eviction_interval_secs: self.eviction_interval_secs,
                    max_tree_size: self.max_tree_size,
                },
                PolicyType::PowerOfTwo => ConfigPolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 5, // Default value
                },
                PolicyType::ConsistentHash => ConfigPolicyConfig::ConsistentHash {
                    virtual_nodes: 160, // Default value
                },
                PolicyType::RendezvousHash => ConfigPolicyConfig::RendezvousHash,
            }
        };

        // Determine routing mode
        let mode = if self.enable_igw {
            // IGW mode - routing mode is not used in IGW, but we need to provide a placeholder
            RoutingMode::Regular {
                worker_urls: vec![],
            }
        } else if self.lmdeploy_pd_disaggregation {
            let migration_protocol = match self
                .lmdeploy_migration_protocol
                .to_ascii_lowercase()
                .as_str()
            {
                "rdma" => config::LMDeployMigrationProtocol::Rdma,
                "nvlink" => config::LMDeployMigrationProtocol::Nvlink,
                other => {
                    return Err(config::ConfigError::ValidationFailed {
                        reason: format!(
                            "Invalid lmdeploy_migration_protocol '{}': expected 'rdma' or 'nvlink'",
                            other
                        ),
                    });
                }
            };
            let rdma_config = if migration_protocol == config::LMDeployMigrationProtocol::Rdma {
                let link_type = match self.lmdeploy_rdma_link_type.to_ascii_lowercase().as_str() {
                    "ib" => config::LMDeployRdmaLinkType::Ib,
                    "roce" => config::LMDeployRdmaLinkType::Roce,
                    other => {
                        return Err(config::ConfigError::ValidationFailed {
                            reason: format!(
                                "Invalid lmdeploy_rdma_link_type '{}': expected 'ib' or 'roce'",
                                other
                            ),
                        });
                    }
                };
                Some(config::LMDeployRdmaConfig {
                    with_gdr: self.lmdeploy_with_gdr,
                    link_type,
                })
            } else {
                None
            };
            RoutingMode::LMDeployPrefillDecode {
                prefill_urls: self.prefill_urls.clone().unwrap_or_default(),
                decode_urls: self.decode_urls.clone().unwrap_or_default(),
                prefill_policy: self.prefill_policy.as_ref().map(convert_policy),
                decode_policy: self.decode_policy.as_ref().map(convert_policy),
                migration_protocol,
                rdma_config,
                dummy_prefill: self.lmdeploy_dummy_prefill,
            }
        } else {
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        // Convert main policy
        let policy = convert_policy(&self.policy);

        // Service discovery configuration
        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: self.selector.clone(),
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
            })
        } else {
            None
        };

        // Metrics configuration
        let metrics = match (self.prometheus_port, self.prometheus_host.as_ref()) {
            (Some(port), Some(host)) => Some(MetricsConfig {
                port,
                host: host.clone(),
            }),
            _ => None,
        };

        Ok(config::RouterConfig {
            mode,
            policy,
            host: self.host.clone(),
            port: self.port,
            connection_mode: config::ConnectionMode::Http,
            max_payload_size: self.max_payload_size,
            request_timeout_secs: self.request_timeout_secs,
            worker_startup_timeout_secs: self.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: self.worker_startup_check_interval,
            api_key: self.api_key.clone(),
            api_key_validation_urls: self.api_key_validation_urls.clone(),
            discovery,
            metrics,
            log_dir: self.log_dir.clone(),
            log_level: self.log_level.clone(),
            request_id_headers: self.request_id_headers.clone(),
            max_concurrent_requests: self.max_concurrent_requests,
            queue_size: self.queue_size,
            queue_timeout_secs: self.queue_timeout_secs,
            rate_limit_tokens_per_second: self.rate_limit_tokens_per_second,
            cors_allowed_origins: self.cors_allowed_origins.clone(),
            retry: config::RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            },
            circuit_breaker: config::CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            },
            disable_retries: self.disable_retries,
            disable_circuit_breaker: self.disable_circuit_breaker,
            health_check: config::HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
            },
            enable_igw: self.enable_igw,
            history_backend: config::HistoryBackend::Memory,
            enable_profiling: false, // Profiling disabled in Python binding by default
            profile_timeout_secs: 10, // Default profiling timeout
            kv_connector: match self.kv_connector.to_ascii_lowercase().as_str() {
                "nixl" => config::KvConnector::Nixl,
                "mooncake" => config::KvConnector::Mooncake,
                "moriio" => config::KvConnector::MoriIO,
                other => {
                    return Err(config::ConfigError::ValidationFailed {
                        reason: format!(
                            "Invalid kv_connector '{}': expected 'nixl', 'mooncake', or 'moriio'",
                            other
                        ),
                    });
                }
            },
        })
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Router {
    #[new]
    #[pyo3(signature = (
        worker_urls,
        policy = PolicyType::RoundRobin,
        host = String::from("127.0.0.1"),
        port = 3001,
        worker_startup_timeout_secs = 600,
        worker_startup_check_interval = 30,
        cache_threshold = 0.3,
        balance_abs_threshold = 64,
        balance_rel_threshold = 1.5,
        eviction_interval_secs = 120,
        max_tree_size = 2usize.pow(26),
        max_payload_size = 512 * 1024 * 1024,  // 512MB default for large batches
        api_key = None,
        api_key_validation_urls = vec![],
        log_dir = None,
        log_level = None,
        service_discovery = false,
        selector = HashMap::new(),
        service_discovery_port = 80,
        service_discovery_namespace = None,
        prefill_selector = HashMap::new(),
        decode_selector = HashMap::new(),
        prometheus_port = None,
        prometheus_host = None,
        request_timeout_secs = 1800,  // Add configurable request timeout
        request_id_headers = None,  // Custom request ID headers
        prefill_urls = None,
        decode_urls = None,
        prefill_policy = None,
        decode_policy = None,
        max_concurrent_requests = 32768,
        cors_allowed_origins = vec![],
        // Retry defaults
        retry_max_retries = 5,
        retry_initial_backoff_ms = 50,
        retry_max_backoff_ms = 30_000,
        retry_backoff_multiplier = 1.5,
        retry_jitter_factor = 0.2,
        disable_retries = false,
        // Circuit breaker defaults
        cb_failure_threshold = 10,
        cb_success_threshold = 3,
        cb_timeout_duration_secs = 60,
        cb_window_duration_secs = 120,
        disable_circuit_breaker = false,
        // Health check defaults
        health_failure_threshold = 3,
        health_success_threshold = 2,
        health_check_timeout_secs = 5,
        health_check_interval_secs = 60,
        health_check_endpoint = String::from("/health"),
        // IGW defaults
        enable_igw = false,
        queue_size = 100,
        queue_timeout_secs = 60,
        rate_limit_tokens_per_second = None,
        // Tracing defaults
        enable_trace = false,
        otlp_traces_endpoint = None,
        // KV connector default (PD disaggregation)
        kv_connector = String::from("nixl"),
        // lmdeploy PD disaggregation defaults
        lmdeploy_pd_disaggregation = false,
        lmdeploy_migration_protocol = String::from("rdma"),
        lmdeploy_rdma_link_type = String::from("roce"),
        lmdeploy_with_gdr = true,
        lmdeploy_dummy_prefill = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        worker_urls: Vec<String>,
        policy: PolicyType,
        host: String,
        port: u16,
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval: u64,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        max_payload_size: usize,
        api_key: Option<String>,
        api_key_validation_urls: Vec<String>,
        log_dir: Option<String>,
        log_level: Option<String>,
        service_discovery: bool,
        selector: HashMap<String, String>,
        service_discovery_port: u16,
        service_discovery_namespace: Option<String>,
        prefill_selector: HashMap<String, String>,
        decode_selector: HashMap<String, String>,
        prometheus_port: Option<u16>,
        prometheus_host: Option<String>,
        request_timeout_secs: u64,
        request_id_headers: Option<Vec<String>>,
        prefill_urls: Option<Vec<String>>,
        decode_urls: Option<Vec<String>>,
        prefill_policy: Option<PolicyType>,
        decode_policy: Option<PolicyType>,
        max_concurrent_requests: usize,
        cors_allowed_origins: Vec<String>,
        retry_max_retries: u32,
        retry_initial_backoff_ms: u64,
        retry_max_backoff_ms: u64,
        retry_backoff_multiplier: f32,
        retry_jitter_factor: f32,
        disable_retries: bool,
        cb_failure_threshold: u32,
        cb_success_threshold: u32,
        cb_timeout_duration_secs: u64,
        cb_window_duration_secs: u64,
        disable_circuit_breaker: bool,
        health_failure_threshold: u32,
        health_success_threshold: u32,
        health_check_timeout_secs: u64,
        health_check_interval_secs: u64,
        health_check_endpoint: String,
        enable_igw: bool,
        queue_size: usize,
        queue_timeout_secs: u64,
        rate_limit_tokens_per_second: Option<usize>,
        enable_trace: bool,
        otlp_traces_endpoint: Option<String>,
        kv_connector: String,
        lmdeploy_pd_disaggregation: bool,
        lmdeploy_migration_protocol: String,
        lmdeploy_rdma_link_type: String,
        lmdeploy_with_gdr: bool,
        lmdeploy_dummy_prefill: bool,
    ) -> PyResult<Self> {
        Ok(Router {
            host,
            port,
            worker_urls,
            policy,
            worker_startup_timeout_secs,
            worker_startup_check_interval,
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
            max_payload_size,
            api_key,
            api_key_validation_urls,
            log_dir,
            log_level,
            service_discovery,
            selector,
            service_discovery_port,
            service_discovery_namespace,
            prefill_selector,
            decode_selector,
            prometheus_port,
            prometheus_host,
            request_timeout_secs,
            request_id_headers,
            prefill_urls,
            decode_urls,
            prefill_policy,
            decode_policy,
            max_concurrent_requests,
            cors_allowed_origins,
            retry_max_retries,
            retry_initial_backoff_ms,
            retry_max_backoff_ms,
            retry_backoff_multiplier,
            retry_jitter_factor,
            disable_retries,
            cb_failure_threshold,
            cb_success_threshold,
            cb_timeout_duration_secs,
            cb_window_duration_secs,
            disable_circuit_breaker,
            health_failure_threshold,
            health_success_threshold,
            health_check_timeout_secs,
            health_check_interval_secs,
            health_check_endpoint,
            enable_igw,
            queue_size,
            queue_timeout_secs,
            rate_limit_tokens_per_second,
            enable_trace,
            otlp_traces_endpoint,
            kv_connector,
            lmdeploy_pd_disaggregation,
            lmdeploy_migration_protocol,
            lmdeploy_rdma_link_type,
            lmdeploy_with_gdr,
            lmdeploy_dummy_prefill,
        })
    }

    fn start(&self) -> PyResult<()> {
        // Convert to RouterConfig and validate
        let router_config = self.to_router_config().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Configuration error: {}", e))
        })?;

        // Validate the configuration
        router_config.validate().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Configuration validation failed: {}",
                e
            ))
        })?;

        // Create service discovery config if enabled
        let service_discovery_config = if self.service_discovery {
            Some(service_discovery::ServiceDiscoveryConfig {
                enabled: true,
                selector: self.selector.clone(),
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
                pd_mode: self.lmdeploy_pd_disaggregation,
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
            })
        } else {
            None
        };

        // Create Prometheus config if enabled
        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port.unwrap_or(29000),
            host: self
                .prometheus_host
                .clone()
                .unwrap_or_else(|| "127.0.0.1".to_string()),
        });

        // Use tokio runtime instead of actix-web System for better compatibility
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Block on the async startup function
        runtime.block_on(async move {
            server::startup(server::ServerConfig {
                host: self.host.clone(),
                port: self.port,
                router_config,
                max_payload_size: self.max_payload_size,
                log_dir: self.log_dir.clone(),
                log_level: self.log_level.clone(),
                service_discovery_config,
                prometheus_config,
                request_timeout_secs: self.request_timeout_secs,
                request_id_headers: self.request_id_headers.clone(),
                trace_config: if self.enable_trace {
                    Some(config::TraceConfig {
                        otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
                        ..Default::default()
                    })
                } else {
                    None
                },
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn lmdeploy_router_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PolicyType>()?;
    m.add_class::<Router>()?;
    Ok(())
}
