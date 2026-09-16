use super::ConfigResult;
use crate::config::validation::ConfigValidator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Routing mode configuration
    pub mode: RoutingMode,
    /// Worker connection mode
    #[serde(default)]
    pub connection_mode: ConnectionMode,
    /// Policy configuration
    pub policy: PolicyConfig,
    /// Server host address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Maximum payload size in bytes
    pub max_payload_size: usize,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Worker startup timeout in seconds
    pub worker_startup_timeout_secs: u64,
    /// Worker health check interval in seconds
    pub worker_startup_check_interval_secs: u64,
    /// The api key used for the authorization with the worker
    pub api_key: Option<String>,
    /// API key validation URLs (if set, incoming requests must validate against them)
    #[serde(default)]
    pub api_key_validation_urls: Vec<String>,
    /// Service discovery configuration (optional)
    pub discovery: Option<DiscoveryConfig>,
    /// Metrics configuration (optional)
    pub metrics: Option<MetricsConfig>,
    /// Log directory (None = stdout only)
    pub log_dir: Option<String>,
    /// Log level (None = info)
    pub log_level: Option<String>,
    /// Custom request ID headers to check (defaults to common headers)
    pub request_id_headers: Option<Vec<String>>,
    /// Maximum concurrent requests allowed (for rate limiting)
    pub max_concurrent_requests: usize,
    /// Queue size for pending requests when max concurrent limit reached (0 = no queue, return 429 immediately)
    pub queue_size: usize,
    /// Maximum time (in seconds) a request can wait in queue before timing out
    pub queue_timeout_secs: u64,
    /// Token bucket refill rate (tokens per second). If not set, defaults to max_concurrent_requests
    pub rate_limit_tokens_per_second: Option<usize>,
    /// CORS allowed origins
    pub cors_allowed_origins: Vec<String>,
    /// Retry configuration
    pub retry: RetryConfig,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Disable retries (overrides retry.max_retries to 1 when true)
    #[serde(default)]
    pub disable_retries: bool,
    /// Disable circuit breaker (overrides circuit_breaker.failure_threshold to u32::MAX when true)
    #[serde(default)]
    pub disable_circuit_breaker: bool,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Enable Inference Gateway mode (false = proxy mode, true = IGW mode)
    #[serde(default)]
    pub enable_igw: bool,
    /// History backend configuration (memory or none, default: memory)
    #[serde(default = "default_history_backend")]
    pub history_backend: HistoryBackend,
    /// Enable profiling calls to backend workers
    #[serde(default)]
    pub enable_profiling: bool,
    /// Profiling timeout in seconds
    #[serde(default = "default_profile_timeout_secs")]
    pub profile_timeout_secs: u64,
    /// KV connector type for PD disaggregation
    #[serde(default)]
    pub kv_connector: KvConnector,
}

fn default_profile_timeout_secs() -> u64 {
    10
}

fn default_history_backend() -> HistoryBackend {
    HistoryBackend::Memory
}

/// History backend configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HistoryBackend {
    /// In-memory storage (default)
    Memory,
    /// No history storage
    None,
}

/// KV connector type for PD disaggregation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum KvConnector {
    /// NIXL pull-based KV transfer (default)
    #[default]
    #[serde(rename = "nixl")]
    #[value(name = "nixl")]
    Nixl,
    /// Mooncake push-based KV transfer
    #[serde(rename = "mooncake")]
    #[value(name = "mooncake")]
    Mooncake,
    /// MoRI-IO KV transfer
    #[serde(rename = "moriio")]
    #[value(name = "moriio")]
    MoriIO,
}

/// Migration protocol for lmdeploy PD disaggregation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum LMDeployMigrationProtocol {
    /// RDMA-based KV migration (default and recommended)
    #[default]
    #[serde(rename = "rdma")]
    #[value(name = "rdma")]
    Rdma,
    /// NVLink-based KV migration
    #[serde(rename = "nvlink")]
    #[value(name = "nvlink")]
    Nvlink,
}

/// RDMA link type used by lmdeploy's migration backend.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum LMDeployRdmaLinkType {
    /// InfiniBand
    #[serde(rename = "IB")]
    #[value(name = "ib")]
    Ib,
    /// RDMA over Converged Ethernet
    #[default]
    #[serde(rename = "RoCE")]
    #[value(name = "roce")]
    Roce,
}

/// RDMA configuration for lmdeploy PD KV migration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LMDeployRdmaConfig {
    /// Whether GPU Direct RDMA is enabled.
    #[serde(default = "default_lmdeploy_with_gdr")]
    pub with_gdr: bool,
    /// RDMA link type (RoCE by default).
    #[serde(default)]
    pub link_type: LMDeployRdmaLinkType,
}

fn default_lmdeploy_with_gdr() -> bool {
    true
}

impl Default for LMDeployRdmaConfig {
    fn default() -> Self {
        Self {
            with_gdr: true,
            link_type: LMDeployRdmaLinkType::Roce,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(tag = "type")]
pub enum ConnectionMode {
    #[default]
    #[serde(rename = "http")]
    Http,
}

/// Routing mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RoutingMode {
    #[serde(rename = "regular")]
    Regular {
        /// List of worker URLs
        worker_urls: Vec<String>,
    },
    #[serde(rename = "openai")]
    OpenAI {
        /// OpenAI-compatible API base(s), provided via worker URLs
        worker_urls: Vec<String>,
    },
    #[serde(rename = "lmdeploy_prefill_decode")]
    LMDeployPrefillDecode {
        /// lmdeploy prefill worker URLs
        prefill_urls: Vec<String>,
        /// lmdeploy decode worker URLs
        decode_urls: Vec<String>,
        /// Optional separate policy for prefill workers
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill_policy: Option<PolicyConfig>,
        /// Optional separate policy for decode workers
        #[serde(skip_serializing_if = "Option::is_none")]
        decode_policy: Option<PolicyConfig>,
        /// KV migration protocol (rdma/nvlink)
        #[serde(default)]
        migration_protocol: LMDeployMigrationProtocol,
        /// Optional RDMA configuration
        #[serde(skip_serializing_if = "Option::is_none")]
        rdma_config: Option<LMDeployRdmaConfig>,
        /// Whether to use dummy prefill (skip actual prefill compute)
        #[serde(default)]
        dummy_prefill: bool,
    },
}

impl RoutingMode {
    pub fn is_pd_mode(&self) -> bool {
        matches!(self, RoutingMode::LMDeployPrefillDecode { .. })
    }

    /// Returns true if this is lmdeploy PD disaggregation mode
    pub fn is_lmdeploy_pd_mode(&self) -> bool {
        matches!(self, RoutingMode::LMDeployPrefillDecode { .. })
    }

    pub fn worker_count(&self) -> usize {
        match self {
            RoutingMode::Regular { worker_urls } => worker_urls.len(),
            RoutingMode::LMDeployPrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => prefill_urls.len() + decode_urls.len(),
            // OpenAI mode represents a single upstream
            RoutingMode::OpenAI { .. } => 1,
        }
    }

    /// Get the effective prefill policy for PD mode
    /// Falls back to the main policy if no specific prefill policy is set
    pub fn get_prefill_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::LMDeployPrefillDecode { prefill_policy, .. } => {
                prefill_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }

    /// Get the effective decode policy for PD mode
    /// Falls back to the main policy if no specific decode policy is set
    pub fn get_decode_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::LMDeployPrefillDecode { decode_policy, .. } => {
                decode_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }
}

/// Policy configuration for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PolicyConfig {
    #[serde(rename = "random")]
    Random,

    #[serde(rename = "round_robin")]
    RoundRobin,

    #[serde(rename = "cache_aware")]
    CacheAware {
        /// Minimum prefix match ratio to use cache-based routing
        cache_threshold: f32,
        /// Absolute load difference threshold for load balancing
        balance_abs_threshold: usize,
        /// Relative load ratio threshold for load balancing
        balance_rel_threshold: f32,
        /// Interval between cache eviction cycles (seconds)
        eviction_interval_secs: u64,
        /// Maximum cache tree size per tenant
        max_tree_size: usize,
    },

    #[serde(rename = "power_of_two")]
    PowerOfTwo {
        /// Interval for load monitoring (seconds)
        load_check_interval_secs: u64,
    },

    #[serde(rename = "consistent_hash")]
    ConsistentHash {
        /// Number of virtual nodes per worker for better distribution
        virtual_nodes: u32,
    },

    #[serde(rename = "rendezvous_hash")]
    RendezvousHash,
}

impl PolicyConfig {
    pub fn name(&self) -> &'static str {
        match self {
            PolicyConfig::Random => "random",
            PolicyConfig::RoundRobin => "round_robin",
            PolicyConfig::CacheAware { .. } => "cache_aware",
            PolicyConfig::PowerOfTwo { .. } => "power_of_two",
            PolicyConfig::ConsistentHash { .. } => "consistent_hash",
            PolicyConfig::RendezvousHash => "rendezvous_hash",
        }
    }
}

/// Service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enable service discovery
    pub enabled: bool,
    /// Kubernetes namespace (None = all namespaces)
    pub namespace: Option<String>,
    /// Service discovery port
    pub port: u16,
    /// Check interval for service discovery
    pub check_interval_secs: u64,
    /// Regular mode selector
    pub selector: HashMap<String, String>,
    /// PD mode prefill selector
    pub prefill_selector: HashMap<String, String>,
    /// PD mode decode selector
    pub decode_selector: HashMap<String, String>,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            namespace: None,
            port: 8000,
            check_interval_secs: 120,
            selector: HashMap::new(),
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
        }
    }
}

/// Retry configuration for request handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds
    pub max_backoff_ms: u64,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f32,
    /// Jitter factor applied to backoff (0.0 - 1.0)
    /// Effective delay D' = D * (1 + U[-j, +j])
    #[serde(default = "default_retry_jitter_factor")]
    pub jitter_factor: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 50,
            max_backoff_ms: 30000,
            backoff_multiplier: 1.5,
            jitter_factor: 0.2,
        }
    }
}

fn default_retry_jitter_factor() -> f32 {
    0.2
}

/// Health check configuration for worker monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
    /// Timeout for health check requests in seconds
    pub timeout_secs: u64,
    /// Interval between health checks in seconds
    pub check_interval_secs: u64,
    /// Health check endpoint path
    pub endpoint: String,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 2,
            timeout_secs: 5,
            check_interval_secs: 60,
            endpoint: "/health".to_string(),
        }
    }
}

/// Circuit breaker configuration for worker reliability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening circuit
    pub failure_threshold: u32,
    /// Number of consecutive successes before closing circuit
    pub success_threshold: u32,
    /// Time before attempting to recover from open state (in seconds)
    pub timeout_duration_secs: u64,
    /// Window duration for failure tracking (in seconds)
    pub window_duration_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 10,
            success_threshold: 3,
            timeout_duration_secs: 60,
            window_duration_secs: 120,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Prometheus metrics port
    pub port: u16,
    /// Prometheus metrics host
    pub host: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "127.0.0.1".to_string(),
        }
    }
}

/// OpenTelemetry tracing configuration.
///
/// Presence of `Some(TraceConfig)` means tracing is enabled;
/// `None` means tracing is disabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    /// OTLP collector endpoint (format: host:port).
    /// When None, the SDK respects OTEL_EXPORTER_OTLP_ENDPOINT.
    #[serde(default)]
    pub otlp_traces_endpoint: Option<String>,
    /// Parent-based trace sampling ratio applied when there is no sampled parent.
    #[serde(default = "TraceConfig::default_sampling_ratio")]
    pub sampling_ratio: f64,
    /// Exact HTTP paths whose server spans should be skipped even when tracing is enabled.
    #[serde(default = "TraceConfig::default_excluded_paths")]
    pub excluded_paths: Vec<String>,
}

impl TraceConfig {
    pub fn default_sampling_ratio() -> f64 {
        1.0
    }

    pub fn default_excluded_paths() -> Vec<String> {
        vec![
            "/health".to_string(),
            "/health_generate".to_string(),
            "/liveness".to_string(),
            "/readiness".to_string(),
        ]
    }
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            otlp_traces_endpoint: None,
            sampling_ratio: Self::default_sampling_ratio(),
            excluded_paths: Self::default_excluded_paths(),
        }
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::Random,
            host: "127.0.0.1".to_string(),
            port: 3001,
            max_payload_size: 536_870_912, // 512MB
            request_timeout_secs: 1800,    // 30 minutes
            worker_startup_timeout_secs: 600,
            worker_startup_check_interval_secs: 30,
            api_key: None,
            api_key_validation_urls: vec![],
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: 32768,
            queue_size: 100,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: HealthCheckConfig::default(),
            enable_igw: false,
            connection_mode: ConnectionMode::Http,
            history_backend: default_history_backend(),
            enable_profiling: false,
            profile_timeout_secs: default_profile_timeout_secs(),
            kv_connector: KvConnector::default(),
        }
    }
}

impl RouterConfig {
    /// Create a new configuration with mode and policy
    pub fn new(mode: RoutingMode, policy: PolicyConfig) -> Self {
        Self {
            mode,
            policy,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> ConfigResult<()> {
        ConfigValidator::validate(self)
    }

    /// Get the routing mode type as a string
    pub fn mode_type(&self) -> &'static str {
        match self.mode {
            RoutingMode::Regular { .. } => "regular",
            RoutingMode::LMDeployPrefillDecode { .. } => "lmdeploy_prefill_decode",
            RoutingMode::OpenAI { .. } => "openai",
        }
    }

    /// Check if service discovery is enabled
    pub fn has_service_discovery(&self) -> bool {
        self.discovery.as_ref().is_some_and(|d| d.enabled)
    }

    /// Check if metrics are enabled
    pub fn has_metrics(&self) -> bool {
        self.metrics.is_some()
    }

    /// Compute the effective retry config considering disable flag
    pub fn effective_retry_config(&self) -> RetryConfig {
        let mut cfg = self.retry.clone();
        if self.disable_retries {
            cfg.max_retries = 1;
        }
        cfg
    }

    /// Compute the effective circuit breaker config considering disable flag
    pub fn effective_circuit_breaker_config(&self) -> CircuitBreakerConfig {
        let mut cfg = self.circuit_breaker.clone();
        if self.disable_circuit_breaker {
            cfg.failure_threshold = u32::MAX;
        }
        cfg
    }

    /// Check if running in IGW (Inference Gateway) mode
    pub fn is_igw_mode(&self) -> bool {
        self.enable_igw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============= RouterConfig Tests =============

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();

        assert!(
            matches!(config.mode, RoutingMode::Regular { worker_urls } if worker_urls.is_empty())
        );
        assert!(matches!(config.policy, PolicyConfig::Random));
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 3001);
        assert_eq!(config.max_payload_size, 536_870_912);
        assert_eq!(config.request_timeout_secs, 1800);
        assert_eq!(config.worker_startup_timeout_secs, 600);
        assert_eq!(config.worker_startup_check_interval_secs, 30);
        assert!(config.discovery.is_none());
        assert!(config.metrics.is_none());
        assert!(config.log_dir.is_none());
        assert!(config.log_level.is_none());
    }

    #[test]
    fn test_router_config_new() {
        let mode = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string(), "http://worker2".to_string()],
        };
        let policy = PolicyConfig::RoundRobin;

        let config = RouterConfig::new(mode, policy);

        match config.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls.len(), 2);
                assert_eq!(worker_urls[0], "http://worker1");
                assert_eq!(worker_urls[1], "http://worker2");
            }
            _ => panic!("Expected Regular mode"),
        }

        assert!(matches!(config.policy, PolicyConfig::RoundRobin));
        // Other fields should be default
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 3001);
    }

    #[test]
    fn test_router_config_serialization() {
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec!["http://worker1".to_string()],
            },
            policy: PolicyConfig::Random,
            host: "0.0.0.0".to_string(),
            port: 8080,
            log_dir: Some("/var/log".to_string()),
            log_level: Some("debug".to_string()),
            ..Default::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.host, deserialized.host);
        assert_eq!(config.port, deserialized.port);
        assert_eq!(config.max_payload_size, deserialized.max_payload_size);
        assert_eq!(config.log_dir, deserialized.log_dir);
        assert_eq!(config.log_level, deserialized.log_level);
        // discovery and metrics are None in Default implementation
        assert!(deserialized.discovery.is_none());
        assert!(deserialized.metrics.is_none());
    }

    // ============= RoutingMode Tests =============

    #[test]
    fn test_routing_mode_is_pd_mode() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        assert!(!regular.is_pd_mode());

        let pd = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://prefill1".to_string()],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::default(),
            rdma_config: None,
            dummy_prefill: false,
        };
        assert!(pd.is_pd_mode());
    }

    #[test]
    fn test_routing_mode_worker_count() {
        let regular = RoutingMode::Regular {
            worker_urls: vec![
                "http://worker1".to_string(),
                "http://worker2".to_string(),
                "http://worker3".to_string(),
            ],
        };
        assert_eq!(regular.worker_count(), 3);

        let pd = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://prefill1".to_string(), "http://prefill2".to_string()],
            decode_urls: vec![
                "http://decode1".to_string(),
                "http://decode2".to_string(),
                "http://decode3".to_string(),
            ],
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::default(),
            rdma_config: None,
            dummy_prefill: false,
        };
        assert_eq!(pd.worker_count(), 5);

        let empty_regular = RoutingMode::Regular {
            worker_urls: vec![],
        };
        assert_eq!(empty_regular.worker_count(), 0);
    }

    #[test]
    fn test_routing_mode_serialization() {
        // Test Regular mode
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        let json = serde_json::to_string(&regular).unwrap();
        assert!(json.contains("\"type\":\"regular\""));
        assert!(json.contains("\"worker_urls\""));

        // Test LMDeployPrefillDecode mode
        let pd = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://prefill1".to_string()],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::default(),
            rdma_config: None,
            dummy_prefill: false,
        };
        let json = serde_json::to_string(&pd).unwrap();
        assert!(json.contains("\"type\":\"lmdeploy_prefill_decode\""));
        assert!(json.contains("\"prefill_urls\""));
        assert!(json.contains("\"decode_urls\""));
    }

    #[test]
    fn test_lmdeploy_prefill_decode_serialization() {
        let lmdeploy_pd = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://prefill1".to_string()],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::Rdma,
            rdma_config: None,
            dummy_prefill: false,
        };
        let json = serde_json::to_string(&lmdeploy_pd).unwrap();
        assert!(json.contains("\"type\":\"lmdeploy_prefill_decode\""));
        assert!(json.contains("\"prefill_urls\""));
        assert!(json.contains("\"decode_urls\""));
        assert!(json.contains("\"migration_protocol\":\"rdma\""));
        assert!(!json.contains("\"rdma_config\""));
        assert!(json.contains("\"dummy_prefill\":false"));

        // Round-trip
        let deserialized: RoutingMode = serde_json::from_str(&json).unwrap();
        assert!(deserialized.is_pd_mode());
        assert!(deserialized.is_lmdeploy_pd_mode());
        assert_eq!(deserialized.worker_count(), 2);

        // Test with rdma config and dummy_prefill=true
        let lmdeploy_pd_rdma = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://p:8000".to_string()],
            decode_urls: vec!["http://d:8000".to_string()],
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::Rdma,
            rdma_config: Some(LMDeployRdmaConfig {
                with_gdr: true,
                link_type: LMDeployRdmaLinkType::Roce,
            }),
            dummy_prefill: true,
        };
        let json_rdma = serde_json::to_string(&lmdeploy_pd_rdma).unwrap();
        assert!(json_rdma.contains("\"migration_protocol\":\"rdma\""));
        assert!(json_rdma.contains("\"rdma_config\""));
        assert!(json_rdma.contains("\"with_gdr\":true"));
        assert!(json_rdma.contains("\"link_type\":\"RoCE\""));
        assert!(json_rdma.contains("\"dummy_prefill\":true"));
    }

    #[test]
    fn test_lmdeploy_migration_protocol_default() {
        assert_eq!(
            LMDeployMigrationProtocol::default(),
            LMDeployMigrationProtocol::Rdma
        );
    }

    #[test]
    fn test_lmdeploy_migration_protocol_serde_variants() {
        // Router configuration uses stable, human-readable lowercase strings.
        assert_eq!(
            serde_json::to_string(&LMDeployMigrationProtocol::Rdma).unwrap(),
            r#""rdma""#
        );
        assert_eq!(
            serde_json::to_string(&LMDeployMigrationProtocol::Nvlink).unwrap(),
            r#""nvlink""#
        );
        // Round-trip deserialization
        let rdma: LMDeployMigrationProtocol = serde_json::from_str(r#""rdma""#).unwrap();
        assert_eq!(rdma, LMDeployMigrationProtocol::Rdma);
        let nvlink: LMDeployMigrationProtocol = serde_json::from_str(r#""nvlink""#).unwrap();
        assert_eq!(nvlink, LMDeployMigrationProtocol::Nvlink);
    }

    #[test]
    fn test_lmdeploy_rdma_config_defaults() {
        let config = LMDeployRdmaConfig::default();
        assert!(config.with_gdr);
        assert_eq!(config.link_type, LMDeployRdmaLinkType::Roce);

        let cfg: LMDeployRdmaConfig = serde_json::from_str(r#"{}"#).unwrap();
        assert!(cfg.with_gdr);
        assert_eq!(cfg.link_type, LMDeployRdmaLinkType::Roce);
    }

    #[test]
    fn test_lmdeploy_pd_mode_type_via_json() {
        // RoutingMode doesn't have a mode_type() method; verify via JSON serialization tag.
        let mode = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://p:8000".to_string()],
            decode_urls: vec!["http://d:8000".to_string()],
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::Rdma,
            rdma_config: None,
            dummy_prefill: false,
        };
        let json = serde_json::to_string(&mode).unwrap();
        assert!(json.contains(r#""type":"lmdeploy_prefill_decode""#));
    }

    #[test]
    fn test_lmdeploy_pd_worker_count_multiple_urls() {
        let mode = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec![
                "http://p1".to_string(),
                "http://p2".to_string(),
                "http://p3".to_string(),
            ],
            decode_urls: vec!["http://d1".to_string(), "http://d2".to_string()],
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::Rdma,
            rdma_config: None,
            dummy_prefill: false,
        };
        assert_eq!(mode.worker_count(), 5);
    }

    #[test]
    fn test_lmdeploy_pd_policy_fallback_to_main() {
        let main_policy = PolicyConfig::RoundRobin;
        let mode = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://p".to_string()],
            decode_urls: vec!["http://d".to_string()],
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::Rdma,
            rdma_config: None,
            dummy_prefill: false,
        };
        // When prefill_policy/decode_policy is None, should fall back to main policy.
        // Compare by name() since PolicyConfig doesn't derive PartialEq.
        let prefill = mode.get_prefill_policy(&main_policy);
        let decode = mode.get_decode_policy(&main_policy);
        assert_eq!(prefill.name(), main_policy.name());
        assert_eq!(decode.name(), main_policy.name());
    }

    #[test]
    fn test_lmdeploy_pd_policy_override() {
        let main_policy = PolicyConfig::RoundRobin;
        let custom_policy = PolicyConfig::Random;
        let mode = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://p".to_string()],
            decode_urls: vec!["http://d".to_string()],
            prefill_policy: Some(custom_policy.clone()),
            decode_policy: Some(custom_policy.clone()),
            migration_protocol: LMDeployMigrationProtocol::Rdma,
            rdma_config: None,
            dummy_prefill: false,
        };
        // Should use custom policy, not main. Compare by name() since PolicyConfig doesn't derive PartialEq.
        let prefill = mode.get_prefill_policy(&main_policy);
        let decode = mode.get_decode_policy(&main_policy);
        assert_eq!(prefill.name(), "random");
        assert_eq!(decode.name(), "random");
    }

    #[test]
    fn test_lmdeploy_pd_deserialize_from_json() {
        let json_str = r#"{
            "type": "lmdeploy_prefill_decode",
            "prefill_urls": ["http://p:23333"],
            "decode_urls": ["http://d:23333"],
            "migration_protocol": "rdma",
            "rdma_config": {"with_gdr": true, "link_type": "RoCE"},
            "dummy_prefill": true
        }"#;
        let mode: RoutingMode = serde_json::from_str(json_str).unwrap();
        assert!(mode.is_lmdeploy_pd_mode());
        assert!(mode.is_pd_mode());
        assert_eq!(mode.worker_count(), 2);
    }

    #[test]
    fn test_lmdeploy_pd_no_rdma_config_when_protocol_is_nvlink() {
        let mode = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://p".to_string()],
            decode_urls: vec!["http://d".to_string()],
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::Nvlink,
            rdma_config: None,
            dummy_prefill: false,
        };
        let json = serde_json::to_string(&mode).unwrap();
        // Option::None fields are skipped by serde by default; rdma_config should be absent.
        assert!(!json.contains("rdma_config"));
    }

    // ============= PolicyConfig Tests =============

    #[test]
    fn test_policy_config_name() {
        assert_eq!(PolicyConfig::Random.name(), "random");
        assert_eq!(PolicyConfig::RoundRobin.name(), "round_robin");

        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.8,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 1000,
        };
        assert_eq!(cache_aware.name(), "cache_aware");

        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        };
        assert_eq!(power_of_two.name(), "power_of_two");
    }

    #[test]
    fn test_policy_config_serialization() {
        // Test Random
        let random = PolicyConfig::Random;
        let json = serde_json::to_string(&random).unwrap();
        assert_eq!(json, r#"{"type":"random"}"#);

        // Test CacheAware with all parameters
        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.8,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 1000,
        };
        let json = serde_json::to_string(&cache_aware).unwrap();
        assert!(json.contains("\"type\":\"cache_aware\""));
        assert!(json.contains("\"cache_threshold\":0.8"));
        assert!(json.contains("\"balance_abs_threshold\":10"));

        // Test PowerOfTwo
        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        };
        let json = serde_json::to_string(&power_of_two).unwrap();
        assert!(json.contains("\"type\":\"power_of_two\""));
        assert!(json.contains("\"load_check_interval_secs\":60"));
    }

    #[test]
    fn test_cache_aware_parameters() {
        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.75,
            balance_abs_threshold: 20,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 600,
            max_tree_size: 5000,
        };

        match cache_aware {
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                assert!((cache_threshold - 0.75).abs() < 0.0001);
                assert_eq!(balance_abs_threshold, 20);
                assert!((balance_rel_threshold - 2.0).abs() < 0.0001);
                assert_eq!(eviction_interval_secs, 600);
                assert_eq!(max_tree_size, 5000);
            }
            _ => panic!("Expected CacheAware"),
        }
    }

    #[test]
    fn test_power_of_two_parameters() {
        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 120,
        };

        match power_of_two {
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            } => {
                assert_eq!(load_check_interval_secs, 120);
            }
            _ => panic!("Expected PowerOfTwo"),
        }
    }

    // ============= DiscoveryConfig Tests =============

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();

        assert!(!config.enabled);
        assert!(config.namespace.is_none());
        assert_eq!(config.port, 8000);
        assert_eq!(config.check_interval_secs, 120);
        assert!(config.selector.is_empty());
        assert!(config.prefill_selector.is_empty());
        assert!(config.decode_selector.is_empty());
    }

    #[test]
    fn test_discovery_config_with_selectors() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "inference".to_string());
        selector.insert("role".to_string(), "worker".to_string());

        let config = DiscoveryConfig {
            enabled: true,
            namespace: Some("default".to_string()),
            port: 9000,
            check_interval_secs: 30,
            selector: selector.clone(),
            prefill_selector: selector.clone(),
            decode_selector: selector.clone(),
        };

        assert!(config.enabled);
        assert_eq!(config.namespace, Some("default".to_string()));
        assert_eq!(config.port, 9000);
        assert_eq!(config.selector.len(), 2);
        assert_eq!(config.selector.get("app"), Some(&"inference".to_string()));
    }

    #[test]
    fn test_discovery_config_namespace() {
        // Test None namespace (all namespaces)
        let config = DiscoveryConfig {
            namespace: None,
            ..Default::default()
        };
        assert!(config.namespace.is_none());

        // Test specific namespace
        let config = DiscoveryConfig {
            namespace: Some("production".to_string()),
            ..Default::default()
        };
        assert_eq!(config.namespace, Some("production".to_string()));
    }

    // ============= MetricsConfig Tests =============

    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();

        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_metrics_config_custom() {
        let config = MetricsConfig {
            port: 9090,
            host: "0.0.0.0".to_string(),
        };

        assert_eq!(config.port, 9090);
        assert_eq!(config.host, "0.0.0.0");
    }

    // ============= RouterConfig Utility Methods Tests =============

    #[test]
    fn test_mode_type() {
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            ..Default::default()
        };
        assert_eq!(config.mode_type(), "regular");

        let config = RouterConfig {
            mode: RoutingMode::LMDeployPrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
                migration_protocol: LMDeployMigrationProtocol::default(),
                rdma_config: None,
                dummy_prefill: false,
            },
            ..Default::default()
        };
        assert_eq!(config.mode_type(), "lmdeploy_prefill_decode");
    }

    #[test]
    fn test_has_service_discovery() {
        let config = RouterConfig::default();
        assert!(!config.has_service_discovery());

        let config = RouterConfig {
            discovery: Some(DiscoveryConfig {
                enabled: false,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(!config.has_service_discovery());

        let config = RouterConfig {
            discovery: Some(DiscoveryConfig {
                enabled: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(config.has_service_discovery());
    }

    #[test]
    fn test_has_metrics() {
        let config = RouterConfig::default();
        assert!(!config.has_metrics());

        let config = RouterConfig {
            metrics: Some(MetricsConfig::default()),
            ..Default::default()
        };
        assert!(config.has_metrics());
    }

    // ============= Edge Cases =============

    #[test]
    fn test_large_worker_lists() {
        let large_urls: Vec<String> = (0..1000).map(|i| format!("http://worker{}", i)).collect();

        let mode = RoutingMode::Regular {
            worker_urls: large_urls.clone(),
        };

        assert_eq!(mode.worker_count(), 1000);

        // Test serialization with large list
        let config = RouterConfig {
            mode,
            ..Default::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        match deserialized.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls.len(), 1000);
            }
            _ => panic!("Expected Regular mode"),
        }
    }

    #[test]
    fn test_unicode_in_config() {
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec!["http://работник1".to_string(), "http://工作者2".to_string()],
            },
            log_dir: Some("/日志/目录".to_string()),
            ..Default::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        match deserialized.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls[0], "http://работник1");
                assert_eq!(worker_urls[1], "http://工作者2");
            }
            _ => panic!("Expected Regular mode"),
        }

        assert_eq!(deserialized.log_dir, Some("/日志/目录".to_string()));
    }

    #[test]
    fn test_empty_string_fields() {
        let config = RouterConfig {
            host: "".to_string(),
            log_dir: Some("".to_string()),
            log_level: Some("".to_string()),
            ..Default::default()
        };

        assert_eq!(config.host, "");
        assert_eq!(config.log_dir, Some("".to_string()));
        assert_eq!(config.log_level, Some("".to_string()));
    }

    // ============= Complex Configuration Tests =============

    #[test]
    fn test_full_pd_mode_config() {
        let config = RouterConfig {
            mode: RoutingMode::LMDeployPrefillDecode {
                prefill_urls: vec![
                    "http://prefill1:8000".to_string(),
                    "http://prefill2:8000".to_string(),
                ],
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: None,
                decode_policy: None,
                migration_protocol: LMDeployMigrationProtocol::default(),
                rdma_config: None,
                dummy_prefill: false,
            },
            policy: PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 30,
            },
            host: "0.0.0.0".to_string(),
            port: 3000,
            max_payload_size: 1048576,
            request_timeout_secs: 120,
            worker_startup_timeout_secs: 60,
            worker_startup_check_interval_secs: 5,
            api_key: None,
            api_key_validation_urls: vec![],
            discovery: Some(DiscoveryConfig {
                enabled: true,
                namespace: Some("router".to_string()),
                ..Default::default()
            }),
            metrics: Some(MetricsConfig {
                port: 9090,
                host: "0.0.0.0".to_string(),
            }),
            log_dir: Some("/var/log/router".to_string()),
            log_level: Some("info".to_string()),
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: HealthCheckConfig::default(),
            enable_igw: false,
            queue_size: 100,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            connection_mode: ConnectionMode::Http,
            history_backend: default_history_backend(),
            enable_profiling: false,
            profile_timeout_secs: default_profile_timeout_secs(),
            kv_connector: KvConnector::default(),
        };

        assert!(config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 4);
        assert_eq!(config.policy.name(), "power_of_two");
        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
    }

    #[test]
    fn test_full_regular_mode_config() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "inference".to_string());

        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![
                    "http://worker1:8000".to_string(),
                    "http://worker2:8000".to_string(),
                    "http://worker3:8000".to_string(),
                ],
            },
            policy: PolicyConfig::CacheAware {
                cache_threshold: 0.9,
                balance_abs_threshold: 5,
                balance_rel_threshold: 1.2,
                eviction_interval_secs: 600,
                max_tree_size: 10000,
            },
            host: "0.0.0.0".to_string(),
            port: 3001,
            max_payload_size: 536870912,
            request_timeout_secs: 300,
            worker_startup_timeout_secs: 180,
            worker_startup_check_interval_secs: 15,
            api_key: None,
            api_key_validation_urls: vec![],
            discovery: Some(DiscoveryConfig {
                enabled: true,
                namespace: None,
                port: 8080,
                check_interval_secs: 45,
                selector,
                ..Default::default()
            }),
            metrics: Some(MetricsConfig::default()),
            log_dir: None,
            log_level: Some("debug".to_string()),
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: HealthCheckConfig::default(),
            enable_igw: false,
            queue_size: 100,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            connection_mode: ConnectionMode::Http,
            history_backend: default_history_backend(),
            enable_profiling: false,
            profile_timeout_secs: default_profile_timeout_secs(),
            kv_connector: KvConnector::default(),
        };

        assert!(!config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 3);
        assert_eq!(config.policy.name(), "cache_aware");
        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
    }

    #[test]
    fn test_config_with_all_options() {
        let mut selectors = HashMap::new();
        selectors.insert("env".to_string(), "prod".to_string());
        selectors.insert("version".to_string(), "v1".to_string());

        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec!["http://worker1".to_string()],
            },
            policy: PolicyConfig::RoundRobin,
            host: "::1".to_string(), // IPv6
            port: 8888,
            max_payload_size: 1024 * 1024 * 512, // 512MB
            request_timeout_secs: 900,
            worker_startup_timeout_secs: 600,
            worker_startup_check_interval_secs: 20,
            api_key: None,
            api_key_validation_urls: vec![],
            discovery: Some(DiscoveryConfig {
                enabled: true,
                namespace: Some("production".to_string()),
                port: 8443,
                check_interval_secs: 120,
                selector: selectors.clone(),
                prefill_selector: selectors.clone(),
                decode_selector: selectors,
            }),
            metrics: Some(MetricsConfig {
                port: 9999,
                host: "::".to_string(), // IPv6 any
            }),
            log_dir: Some("/opt/logs/router".to_string()),
            log_level: Some("trace".to_string()),
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: HealthCheckConfig::default(),
            enable_igw: false,
            queue_size: 100,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            connection_mode: ConnectionMode::Http,
            history_backend: default_history_backend(),
            enable_profiling: false,
            profile_timeout_secs: default_profile_timeout_secs(),
            kv_connector: KvConnector::default(),
        };

        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
        assert_eq!(config.mode_type(), "regular");

        // Test round-trip serialization
        let json = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.host, "::1");
        assert_eq!(deserialized.port, 8888);
        assert_eq!(
            deserialized.discovery.unwrap().namespace,
            Some("production".to_string())
        );
    }

    // ============= Policy Fallback Tests =============

    #[test]
    fn test_pd_policy_fallback_both_specified() {
        // When both prefill and decode policies are specified, they should be used
        let pd = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://prefill1".to_string()],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: Some(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            }),
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
            migration_protocol: LMDeployMigrationProtocol::default(),
            rdma_config: None,
            dummy_prefill: false,
        };

        let main_policy = PolicyConfig::Random;

        // Both specific policies should be used
        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {} // Success
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {} // Success
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_prefill() {
        // When only prefill policy is specified, decode should use main policy
        let pd = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://prefill1".to_string()],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: Some(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            }),
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::default(),
            rdma_config: None,
            dummy_prefill: false,
        };

        let main_policy = PolicyConfig::RoundRobin;

        // Prefill should use specific policy
        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {} // Success
            _ => panic!("Expected CacheAware for prefill"),
        }

        // Decode should fall back to main policy
        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {} // Success
            _ => panic!("Expected RoundRobin for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_decode() {
        // When only decode policy is specified, prefill should use main policy
        let pd = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://prefill1".to_string()],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
            migration_protocol: LMDeployMigrationProtocol::default(),
            rdma_config: None,
            dummy_prefill: false,
        };

        let main_policy = PolicyConfig::Random;

        // Prefill should fall back to main policy
        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::Random => {} // Success
            _ => panic!("Expected Random for prefill"),
        }

        // Decode should use specific policy
        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {} // Success
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_none_specified() {
        // When no specific policies are specified, both should use main policy
        let pd = RoutingMode::LMDeployPrefillDecode {
            prefill_urls: vec!["http://prefill1".to_string()],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::default(),
            rdma_config: None,
            dummy_prefill: false,
        };

        let main_policy = PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 20,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 2000,
        };

        // Both should fall back to main policy
        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.7).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.7).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware for decode"),
        }
    }

    #[test]
    fn test_regular_mode_policy_fallback() {
        // For regular mode, the helper methods should just return the main policy
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };

        let main_policy = PolicyConfig::RoundRobin;

        // Both methods should return main policy for regular mode
        match regular.get_prefill_policy(&main_policy) {
            PolicyConfig::RoundRobin => {} // Success
            _ => panic!("Expected RoundRobin for regular mode"),
        }

        match regular.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {} // Success
            _ => panic!("Expected RoundRobin for regular mode"),
        }
    }
}
