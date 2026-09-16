use std::path::PathBuf;
use tracing::Level;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::fmt::time::ChronoUtc;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

use crate::config::TraceConfig;

/// Configuration for the logging system
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Log level for the application (default: INFO)
    pub level: Level,
    /// Whether to use json format for logs (default: false)
    pub json_format: bool,
    /// Path to store log files. If None, logs will only go to stdout/stderr
    pub log_dir: Option<String>,
    /// Whether to colorize logs when output is a terminal (default: true)
    pub colorize: bool,
    /// Log file name to use if log_dir is specified (default: "lmdeploy-router")
    pub log_file_name: String,
    /// Custom log targets to filter (default: "lmdeploy_router_rs")
    pub log_targets: Option<Vec<String>>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: Level::INFO,
            json_format: false,
            log_dir: None,
            colorize: true,
            log_file_name: "lmdeploy-router".to_string(),
            log_targets: Some(vec!["lmdeploy_router_rs".to_string()]),
        }
    }
}

/// Guard that keeps the file appender worker thread alive
///
/// This must be kept in scope for the duration of the program
/// to ensure logs are properly written to files
#[allow(dead_code)]
pub struct LogGuard {
    _file_guard: Option<WorkerGuard>,
}

/// Initialize the logging system with the given configuration
///
/// # Arguments
/// * `config` - Configuration for the logging system
///
/// # Returns
/// A LogGuard that must be kept alive for the duration of the program
///
/// # Panics
/// Will not panic, as initialization errors are handled gracefully
pub fn init_logging(config: LoggingConfig, otel_layer_config: Option<TraceConfig>) -> LogGuard {
    // Note: Do NOT call LogTracer::init() here. The tracing_subscriber
    // try_init() method handles LogTracer initialization internally (when the
    // "tracing-log" feature is enabled). Calling it beforehand causes
    // try_init() to fail because log::set_logger() can only be called once.

    // Convert log level to filter string
    let level_filter = match config.level {
        Level::TRACE => "trace",
        Level::DEBUG => "debug",
        Level::INFO => "info",
        Level::WARN => "warn",
        Level::ERROR => "error",
    };

    // Create env filter
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        // Format: <target>=<level>,<target2>=<level2>,...
        let filter_string = if let Some(targets) = &config.log_targets {
            targets
                .iter()
                .enumerate()
                .map(|(i, target)| {
                    if i > 0 {
                        format!(",{}={}", target, level_filter)
                    } else {
                        format!("{}={}", target, level_filter)
                    }
                })
                .collect::<String>()
        } else {
            format!("lmdeploy_router_rs={}", level_filter)
        };

        EnvFilter::new(filter_string)
    });

    // When OTel tracing is enabled, ensure the OTel span target passes through
    // the global EnvFilter regardless of log level. Without this, --log-level warn
    // silently suppresses all info-level OTel spans.
    let env_filter = if otel_layer_config.is_some() {
        env_filter.add_directive(
            "otel_trace=trace"
                .parse()
                .expect("valid EnvFilter directive"),
        )
    } else {
        env_filter
    };

    // Setup stdout/stderr layer
    let mut layers = Vec::new();

    // Standard timestamp format: YYYY-MM-DD HH:MM:SS
    let time_format = "%Y-%m-%d %H:%M:%S".to_string();

    // Configure the console stdout layer
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_ansi(config.colorize)
        .with_file(true)
        .with_line_number(true)
        .with_timer(ChronoUtc::new(time_format.clone()));

    let stdout_layer = if config.json_format {
        stdout_layer.json().flatten_event(true).boxed()
    } else {
        stdout_layer.boxed()
    };

    layers.push(stdout_layer);

    // Create a file appender if log_dir is specified
    let mut file_guard = None;

    if let Some(log_dir) = &config.log_dir {
        let file_name = config.log_file_name.clone();
        let log_dir = PathBuf::from(log_dir);

        // Create log directory if it doesn't exist
        if !log_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&log_dir) {
                eprintln!("Failed to create log directory: {}", e);
                return LogGuard { _file_guard: None };
            }
        }

        let file_appender = RollingFileAppender::new(Rotation::DAILY, log_dir, file_name);

        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        file_guard = Some(guard);

        let file_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false) // Never use ANSI colors in log files
            .with_file(true)
            .with_line_number(true)
            .with_timer(ChronoUtc::new(time_format))
            .with_writer(non_blocking);

        let file_layer = if config.json_format {
            file_layer.json().flatten_event(true).boxed()
        } else {
            file_layer.boxed()
        };

        layers.push(file_layer);
    }

    let mut prepared_otel = None;
    if let Some(trace_config) = otel_layer_config {
        match crate::otel_trace::prepare_otel(&trace_config) {
            Ok(prepared) => {
                layers.push(prepared.layer());
                prepared_otel = Some(prepared);
            }
            Err(e) => {
                eprintln!("Failed to initialize OpenTelemetry: {e}");
            }
        }
    }

    // Initialize the subscriber with all layers
    // Use try_init to handle errors gracefully in case another subscriber is already set
    let subscriber_installed = tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .try_init()
        .is_ok();

    // Only activate OTel (set global propagator + provider, flip ENABLED)
    // after both the layer and subscriber are live
    if let Some(prepared) = prepared_otel {
        if subscriber_installed {
            crate::otel_trace::activate_otel(prepared);
        }
    }

    // Return the guard to keep the file appender worker thread alive
    LogGuard {
        _file_guard: file_guard,
    }
}
