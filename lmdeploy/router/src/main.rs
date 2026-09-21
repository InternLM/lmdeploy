use clap::{ArgAction, Parser, ValueEnum};
use lmdeploy_router_rs::config::{
    CircuitBreakerConfig, ConfigResult, ConnectionMode, DiscoveryConfig, HealthCheckConfig,
    HistoryBackend, KvConnector, LMDeployMigrationProtocol, LMDeployRdmaConfig,
    LMDeployRdmaLinkType, MetricsConfig, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
    TraceConfig,
};
use lmdeploy_router_rs::metrics::PrometheusConfig;
use lmdeploy_router_rs::server::{self, ServerConfig};
use lmdeploy_router_rs::service_discovery::ServiceDiscoveryConfig;
use std::collections::HashMap;

fn parse_prefill_args() -> Vec<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut prefill_entries = Vec::new();
    let mut i = 0;

    while i < args.len() {
        if args[i] == "--prefill" && i + 1 < args.len() {
            prefill_entries.push(args[i + 1].clone());
            i += 2; // Skip --prefill and URL
        } else {
            i += 1;
        }
    }

    prefill_entries
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    #[value(name = "lmdeploy")]
    LMDeploy,
    #[value(name = "openai")]
    Openai,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Backend::LMDeploy => "lmdeploy",
            Backend::Openai => "openai",
        };
        write!(f, "{}", s)
    }
}

#[derive(Parser, Debug)]
#[command(name = "lmdeploy-router")]
#[command(version)]
#[command(about = "LMDeploy Router - High-performance request distribution across worker nodes")]
#[command(long_about = r#"
LMDeploy Router - High-performance request distribution across worker nodes

Usage:
This launcher enables starting a router with individual worker instances. It is useful for
multi-node setups or when you want to start workers and router separately.

Examples:
  # Regular mode
  lmdeploy-router --worker-urls http://worker1:8000 http://worker2:8000

  # LMDeploy dynamic registration (start the router before the API server)
  lmdeploy-router --host 0.0.0.0 --port 30000
  lmdeploy serve api_server MODEL --proxy-url http://router:30000

  # LMDeploy PD dynamic registration (no static URLs required)
  lmdeploy-router --host 0.0.0.0 --port 30000 \
    --lmdeploy-pd-disaggregation --lmdeploy-migration-protocol rdma \
    --lmdeploy-rdma-link-type roce
  lmdeploy serve api_server MODEL --role Prefill --proxy-url http://router:30000
  lmdeploy serve api_server MODEL --role Decode --proxy-url http://router:30000

"#)]
struct CliArgs {
    /// Host address to bind the router server
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port number to bind the router server
    #[arg(long, default_value_t = 30000)]
    port: u16,

    /// Optional worker URLs. May be omitted when LMDeploy servers register via --proxy-url.
    #[arg(long, num_args = 0..)]
    worker_urls: Vec<String>,

    /// Load balancing policy to use
    #[arg(long, default_value = "cache_aware", value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "consistent_hash", "rendezvous_hash"])]
    policy: String,

    /// Decode server URL (can be specified multiple times)
    #[arg(long, action = ArgAction::Append)]
    decode: Vec<String>,

    /// Specific policy for prefill nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "consistent_hash", "rendezvous_hash"])]
    prefill_policy: Option<String>,

    /// Specific policy for decode nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "consistent_hash", "rendezvous_hash"])]
    decode_policy: Option<String>,

    /// Timeout in seconds for worker startup
    #[arg(long, default_value_t = 600)]
    worker_startup_timeout_secs: u64,

    /// Interval in seconds between checks for worker startup
    #[arg(long, default_value_t = 30)]
    worker_startup_check_interval: u64,

    /// Cache threshold (0.0-1.0) for cache-aware routing
    #[arg(long, default_value_t = 0.3)]
    cache_threshold: f32,

    /// Absolute threshold for load balancing
    #[arg(long, default_value_t = 64)]
    balance_abs_threshold: usize,

    /// Relative threshold for load balancing
    #[arg(long, default_value_t = 1.5)]
    balance_rel_threshold: f32,

    /// Interval in seconds between cache eviction operations
    #[arg(long, default_value_t = 120)]
    eviction_interval: u64,

    /// Maximum size of the approximation tree for cache-aware routing
    #[arg(long, default_value_t = 67108864)] // 2^26
    max_tree_size: usize,

    /// Maximum payload size in bytes
    #[arg(long, default_value_t = 536870912)] // 512MB
    max_payload_size: usize,

    /// API key for worker authorization
    #[arg(long)]
    api_key: Option<String>,

    /// API key validation URLs (defaults to env file)
    #[arg(long, num_args = 0..)]
    api_key_validation_urls: Vec<String>,

    /// Backend to route requests to (lmdeploy, openai)
    #[arg(long, value_enum, default_value_t = Backend::LMDeploy, alias = "runtime")]
    backend: Backend,

    /// Directory to store log files
    #[arg(long)]
    log_dir: Option<String>,

    /// Set the logging level
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"])]
    log_level: String,

    /// Enable OpenTelemetry tracing
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Tracing (OpenTelemetry)"
    )]
    enable_trace: bool,

    /// OTLP collector endpoint (format: host:port). If omitted, respects OTEL_EXPORTER_OTLP_ENDPOINT.
    #[arg(long, help_heading = "Tracing (OpenTelemetry)")]
    otlp_traces_endpoint: Option<String>,

    /// Parent-based sampling ratio for OpenTelemetry traces.
    #[arg(long, default_value_t = 1.0, help_heading = "Tracing (OpenTelemetry)")]
    otel_sampling_ratio: f64,

    /// Exact HTTP paths to exclude from OpenTelemetry server spans.
    #[arg(long, num_args = 0.., help_heading = "Tracing (OpenTelemetry)")]
    otel_excluded_paths: Vec<String>,

    /// Enable Kubernetes service discovery
    #[arg(long, default_value_t = false)]
    service_discovery: bool,

    /// Label selector for Kubernetes service discovery (format: key1=value1 key2=value2)
    #[arg(long, num_args = 0..)]
    selector: Vec<String>,

    /// Port to use for discovered worker pods
    #[arg(long, default_value_t = 80)]
    service_discovery_port: u16,

    /// Kubernetes namespace to watch for pods
    #[arg(long)]
    service_discovery_namespace: Option<String>,

    /// Label selector for prefill server pods in PD mode
    #[arg(long, num_args = 0..)]
    prefill_selector: Vec<String>,

    /// Label selector for decode server pods in PD mode
    #[arg(long, num_args = 0..)]
    decode_selector: Vec<String>,

    /// Port to expose Prometheus metrics
    #[arg(long, default_value_t = 29000)]
    prometheus_port: u16,

    /// Host address to bind the Prometheus metrics server
    #[arg(long, default_value = "127.0.0.1")]
    prometheus_host: String,

    /// Custom HTTP headers to check for request IDs
    #[arg(long, num_args = 0..)]
    request_id_headers: Vec<String>,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 1800)]
    request_timeout_secs: u64,

    /// Maximum number of concurrent requests allowed
    #[arg(long, default_value_t = 32768)]
    max_concurrent_requests: usize,

    /// CORS allowed origins
    #[arg(long, num_args = 0..)]
    cors_allowed_origins: Vec<String>,

    // Retry configuration
    /// Maximum number of retries
    #[arg(long, default_value_t = 5)]
    retry_max_retries: u32,

    /// Initial backoff in milliseconds for retries
    #[arg(long, default_value_t = 50)]
    retry_initial_backoff_ms: u64,

    /// Maximum backoff in milliseconds for retries
    #[arg(long, default_value_t = 30000)]
    retry_max_backoff_ms: u64,

    /// Backoff multiplier for exponential backoff
    #[arg(long, default_value_t = 1.5)]
    retry_backoff_multiplier: f32,

    /// Jitter factor for retry backoff
    #[arg(long, default_value_t = 0.2)]
    retry_jitter_factor: f32,

    /// Disable retries
    #[arg(long, default_value_t = false)]
    disable_retries: bool,

    // Circuit breaker configuration
    /// Number of failures before circuit breaker opens
    #[arg(long, default_value_t = 10)]
    cb_failure_threshold: u32,

    /// Number of successes before circuit breaker closes
    #[arg(long, default_value_t = 3)]
    cb_success_threshold: u32,

    /// Timeout duration in seconds for circuit breaker
    #[arg(long, default_value_t = 60)]
    cb_timeout_duration_secs: u64,

    /// Window duration in seconds for circuit breaker
    #[arg(long, default_value_t = 120)]
    cb_window_duration_secs: u64,

    /// Disable circuit breaker
    #[arg(long, default_value_t = false)]
    disable_circuit_breaker: bool,

    // Health check configuration
    /// Number of consecutive health check failures before marking worker unhealthy
    #[arg(long, default_value_t = 3)]
    health_failure_threshold: u32,

    /// Number of consecutive health check successes before marking worker healthy
    #[arg(long, default_value_t = 2)]
    health_success_threshold: u32,

    /// Timeout in seconds for health check requests
    #[arg(long, default_value_t = 5)]
    health_check_timeout_secs: u64,

    /// Interval in seconds between runtime health checks
    #[arg(long, default_value_t = 60)]
    health_check_interval_secs: u64,

    /// Health check endpoint path
    #[arg(long, default_value = "/health")]
    health_check_endpoint: String,

    // IGW (Inference Gateway) configuration
    /// Enable Inference Gateway mode
    #[arg(long, default_value_t = false)]
    enable_igw: bool,

    /// History backend configuration (memory or none)
    #[arg(long, default_value = "memory", value_parser = ["memory", "none"])]
    history_backend: String,

    /// Enable profiling calls to backend workers
    #[arg(long, default_value_t = false)]
    profile: bool,

    /// KV connector type for PD disaggregation (nixl or mooncake)
    #[arg(long, value_enum, default_value_t = KvConnector::Nixl)]
    kv_connector: KvConnector,

    /// Enable lmdeploy PD disaggregation mode
    #[arg(long = "lmdeploy-pd-disaggregation", default_value_t = false)]
    lmdeploy_pd_disaggregation: bool,

    /// Migration protocol for lmdeploy PD
    #[arg(long = "lmdeploy-migration-protocol", value_enum, default_value_t = LMDeployMigrationProtocol::Rdma)]
    lmdeploy_migration_protocol: LMDeployMigrationProtocol,

    /// RDMA link type for lmdeploy PD
    #[arg(long = "lmdeploy-rdma-link-type", value_enum, default_value_t = LMDeployRdmaLinkType::Roce)]
    lmdeploy_rdma_link_type: LMDeployRdmaLinkType,

    /// Disable GPU Direct RDMA for lmdeploy PD
    #[arg(long = "lmdeploy-disable-gdr", default_value_t = false)]
    lmdeploy_disable_gdr: bool,

    /// Use dummy prefill for lmdeploy PD
    #[arg(long = "lmdeploy-dummy-prefill", default_value_t = false)]
    lmdeploy_dummy_prefill: bool,
}

impl CliArgs {
    /// Parse selector strings into HashMap
    fn parse_selector(selector_list: &[String]) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for item in selector_list {
            if let Some(eq_pos) = item.find('=') {
                let key = item[..eq_pos].to_string();
                let value = item[eq_pos + 1..].to_string();
                map.insert(key, value);
            }
        }
        map
    }

    /// Convert policy string to PolicyConfig
    fn parse_policy(&self, policy_str: &str) -> PolicyConfig {
        match policy_str {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval,
                max_tree_size: self.max_tree_size,
            },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5, // Default value
            },
            "consistent_hash" => PolicyConfig::ConsistentHash {
                virtual_nodes: 160, // Default value
            },
            "rendezvous_hash" => PolicyConfig::RendezvousHash,
            _ => PolicyConfig::RoundRobin, // Fallback
        }
    }

    /// Convert CLI arguments to RouterConfig
    fn to_router_config(&self, prefill_urls: Vec<String>) -> ConfigResult<RouterConfig> {
        // Determine routing mode
        let mode = if self.enable_igw {
            // IGW mode - routing mode is not used in IGW, but we need to provide a placeholder
            RoutingMode::Regular {
                worker_urls: vec![],
            }
        } else if matches!(self.backend, Backend::Openai) {
            // OpenAI backend mode - use worker_urls as base(s)
            RoutingMode::OpenAI {
                worker_urls: self.worker_urls.clone(),
            }
        } else if self.lmdeploy_pd_disaggregation {
            // LMDeploy PD disaggregation mode
            let decode_str_urls = self.decode.clone();

            eprintln!("ℹ️  INFO: Using LMDeploy PD disaggregation mode.");
            eprintln!("   Prefill URLs: {:?}", prefill_urls);
            eprintln!("   Decode URLs: {:?}", decode_str_urls);
            eprintln!(
                "   Migration protocol: {:?}",
                self.lmdeploy_migration_protocol
            );
            if self.lmdeploy_migration_protocol == LMDeployMigrationProtocol::Rdma {
                eprintln!("   RDMA link type: {:?}", self.lmdeploy_rdma_link_type);
                eprintln!("   GPU Direct RDMA: {}", !self.lmdeploy_disable_gdr);
            }
            eprintln!("   Dummy prefill: {}", self.lmdeploy_dummy_prefill);

            let rdma_config = if self.lmdeploy_migration_protocol == LMDeployMigrationProtocol::Rdma
            {
                Some(LMDeployRdmaConfig {
                    with_gdr: !self.lmdeploy_disable_gdr,
                    link_type: self.lmdeploy_rdma_link_type,
                })
            } else {
                None
            };

            RoutingMode::LMDeployPrefillDecode {
                prefill_urls,
                decode_urls: decode_str_urls,
                prefill_policy: self.prefill_policy.as_ref().map(|p| self.parse_policy(p)),
                decode_policy: self.decode_policy.as_ref().map(|p| self.parse_policy(p)),
                migration_protocol: self.lmdeploy_migration_protocol,
                rdma_config,
                dummy_prefill: self.lmdeploy_dummy_prefill,
            }
        } else {
            // Regular mode
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        // Main policy
        let policy = self.parse_policy(&self.policy);

        // Service discovery configuration
        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: Self::parse_selector(&self.selector),
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
            })
        } else {
            None
        };

        // Metrics configuration
        let metrics = Some(MetricsConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
        });

        let connection_mode = ConnectionMode::Http;

        let api_key_validation_urls = if !self.api_key_validation_urls.is_empty() {
            self.api_key_validation_urls.clone()
        } else if let Ok(raw_urls) = std::env::var("API_KEY_VALIDATION_URLS") {
            raw_urls
                .split(',')
                .map(|url| url.trim().to_string())
                .filter(|url| !url.is_empty())
                .collect()
        } else {
            Vec::new()
        };

        // Build RouterConfig
        Ok(RouterConfig {
            mode,
            policy,
            connection_mode,
            host: self.host.clone(),
            port: self.port,
            max_payload_size: self.max_payload_size,
            request_timeout_secs: self.request_timeout_secs,
            worker_startup_timeout_secs: self.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: self.worker_startup_check_interval,
            api_key: self.api_key.clone(),
            api_key_validation_urls,
            discovery,
            metrics,
            log_dir: self.log_dir.clone(),
            log_level: Some(self.log_level.clone()),
            request_id_headers: if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            },
            max_concurrent_requests: self.max_concurrent_requests,
            queue_size: 100,        // Default queue size
            queue_timeout_secs: 60, // Default timeout
            cors_allowed_origins: self.cors_allowed_origins.clone(),
            retry: RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            },
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            },
            disable_retries: self.disable_retries,
            disable_circuit_breaker: self.disable_circuit_breaker,
            health_check: HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
            },
            enable_igw: self.enable_igw,
            rate_limit_tokens_per_second: None,
            history_backend: match self.history_backend.as_str() {
                "none" => HistoryBackend::None,
                _ => HistoryBackend::Memory,
            },
            enable_profiling: self.profile,
            profile_timeout_secs: 10, // Default profiling timeout
            kv_connector: self.kv_connector,
        })
    }

    /// Create ServerConfig from CLI args and RouterConfig
    fn to_server_config(&self, router_config: RouterConfig) -> ServerConfig {
        // Create service discovery config if enabled
        let service_discovery_config = if self.service_discovery {
            Some(ServiceDiscoveryConfig {
                enabled: true,
                selector: Self::parse_selector(&self.selector),
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
                // HTTP service discovery supports both PD router implementations.
                pd_mode: self.lmdeploy_pd_disaggregation,
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
            })
        } else {
            None
        };

        // Create Prometheus config
        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
        });

        ServerConfig {
            host: self.host.clone(),
            port: self.port,
            router_config,
            max_payload_size: self.max_payload_size,
            log_dir: self.log_dir.clone(),
            log_level: Some(self.log_level.clone()),
            service_discovery_config,
            prometheus_config,
            request_timeout_secs: self.request_timeout_secs,
            request_id_headers: if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            },
            trace_config: if self.enable_trace {
                Some(TraceConfig {
                    otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
                    sampling_ratio: self.otel_sampling_ratio,
                    excluded_paths: if self.otel_excluded_paths.is_empty() {
                        TraceConfig::default_excluded_paths()
                    } else {
                        self.otel_excluded_paths.clone()
                    },
                })
            } else {
                None
            },
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    println!("DEBUG: Main function started");

    // Parse prefill arguments manually before clap parsing
    println!("DEBUG: Parsing prefill arguments");
    let prefill_urls = parse_prefill_args();
    println!("DEBUG: Prefill URLs parsed: {:?}", prefill_urls);

    // Filter out prefill arguments and their values before passing to clap
    println!("DEBUG: Filtering CLI arguments");
    let mut filtered_args: Vec<String> = Vec::new();
    let raw_args: Vec<String> = std::env::args().collect();
    println!("DEBUG: Raw args: {:?}", raw_args);
    let mut i = 0;

    while i < raw_args.len() {
        if raw_args[i] == "--prefill" && i + 1 < raw_args.len() {
            // Skip --prefill and its URL
            i += 2;
        } else {
            filtered_args.push(raw_args[i].clone());
            i += 1;
        }
    }

    // Parse CLI arguments with clap using filtered args
    println!("DEBUG: Parsing CLI arguments with clap");
    println!("DEBUG: Filtered args: {:?}", filtered_args);
    let cli_args = CliArgs::parse_from(filtered_args);
    println!("DEBUG: CLI args parsed successfully");
    // Print startup info
    println!("LMDeploy Router starting...");
    println!("Host: {}:{}", cli_args.host, cli_args.port);
    let mode_str = if cli_args.enable_igw {
        "IGW (Inference Gateway)".to_string()
    } else if matches!(cli_args.backend, Backend::Openai) {
        "OpenAI Backend".to_string()
    } else if cli_args.lmdeploy_pd_disaggregation {
        "LMDeploy PD Disaggregated".to_string()
    } else {
        format!("Regular ({})", cli_args.backend)
    };
    println!("Mode: {}", mode_str);

    if !cli_args.enable_igw {
        println!("Policy: {}", cli_args.policy);

        if cli_args.lmdeploy_pd_disaggregation {
            println!("LMDeploy Prefill nodes: {:?}", prefill_urls);
            println!("LMDeploy Decode nodes: {:?}", cli_args.decode);
        }
    }

    // Convert to RouterConfig
    println!("DEBUG: Converting to RouterConfig");
    let router_config = cli_args.to_router_config(prefill_urls)?;
    println!("DEBUG: RouterConfig created successfully");

    // Validate configuration
    println!("DEBUG: Validating configuration");
    router_config.validate()?;
    println!("DEBUG: Configuration validated successfully");

    // Create ServerConfig
    println!("DEBUG: Creating ServerConfig");
    println!(
        "DEBUG: CLI host: {}, port: {}",
        cli_args.host, cli_args.port
    );
    let server_config = cli_args.to_server_config(router_config);
    println!(
        "DEBUG: ServerConfig created successfully - host: {}, port: {}",
        server_config.host, server_config.port
    );

    // Create a new runtime for the server (like Python binding does)
    println!("DEBUG: Creating Tokio runtime");
    let runtime = tokio::runtime::Runtime::new()?;
    println!("DEBUG: Tokio runtime created successfully");

    // Block on the async startup function
    println!("DEBUG: Starting server startup function");
    runtime.block_on(async move {
        let result = server::startup(server_config).await;
        // Shut down OTel while the Tokio runtime is still alive so the
        // BatchSpanProcessor can flush its final batch.
        if lmdeploy_router_rs::otel_trace::is_otel_enabled() {
            lmdeploy_router_rs::otel_trace::shutdown_otel();
        }
        result
    })?;

    Ok(())
}
