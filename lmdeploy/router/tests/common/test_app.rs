use axum::Router;
use lmdeploy_router_rs::{
    config::RouterConfig,
    otel_trace,
    routers::RouterTrait,
    server::{build_app_with_request_tracing, AppContext, AppState},
};
use reqwest::Client;
use std::sync::Arc;

/// Create a test Axum application using the actual server's build_app function
#[allow(dead_code)]
pub fn create_test_app(
    router: Arc<dyn RouterTrait>,
    client: Client,
    router_config: &RouterConfig,
) -> Router {
    create_test_app_with_tracing(router, client, router_config, otel_trace::is_otel_enabled())
}

/// Create a test Axum application with an explicit request-tracing toggle.
#[allow(dead_code)]
pub fn create_test_app_with_tracing(
    router: Arc<dyn RouterTrait>,
    client: Client,
    router_config: &RouterConfig,
    enable_request_tracing: bool,
) -> Router {
    // Create AppContext
    let app_context = Arc::new(
        AppContext::new(
            router_config.clone(),
            client,
            router_config.max_concurrent_requests,
            router_config.rate_limit_tokens_per_second,
            router_config.api_key_validation_urls.clone(),
        )
        .expect("Failed to create AppContext in test"),
    );

    // Create AppState with the test router and context
    let app_state = Arc::new(AppState {
        router,
        context: app_context,
        concurrency_queue_tx: None,
        router_manager: None,
    });

    // Configure request ID headers (use defaults if not specified)
    let request_id_headers = router_config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    // Use the actual server's build_app function
    build_app_with_request_tracing(
        app_state,
        router_config.max_payload_size,
        request_id_headers,
        router_config.cors_allowed_origins.clone(),
        true, // enable_transparent_proxy
        enable_request_tracing,
    )
}
