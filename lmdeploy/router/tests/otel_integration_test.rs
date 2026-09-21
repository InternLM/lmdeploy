//! Integration tests for OpenTelemetry distributed tracing through the real router.
//!
//! These tests exercise the full request path: middleware -> router -> backend ->
//! header propagation. They use MockWorker as a real TCP backend server and
//! send requests through the actual axum app via tower::ServiceExt::oneshot.

mod common;

use axum::http::{Request, StatusCode};
use common::mock_worker::{self, MockWorker, MockWorkerConfig};
use common::test_app::create_test_app;
use lmdeploy_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    otel_trace,
    routers::RouterFactory,
};
use opentelemetry::{global, propagation::TextMapCompositePropagator, trace::TracerProvider as _};
use opentelemetry_sdk::{
    propagation::{BaggagePropagator, TraceContextPropagator},
    testing::trace::InMemorySpanExporter,
    trace::TracerProvider,
};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

/// Set up OTel with an in-memory exporter for one test section.
fn setup_otel_harness() -> (
    InMemorySpanExporter,
    TracerProvider,
    tracing::subscriber::DefaultGuard,
) {
    let exporter = InMemorySpanExporter::default();
    let provider = TracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let tracer = provider.tracer("test-tracer");

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let subscriber = tracing_subscriber::registry().with(otel_layer);
    let guard = tracing::subscriber::set_default(subscriber);

    global::set_text_map_propagator(TextMapCompositePropagator::new(vec![
        Box::new(TraceContextPropagator::new()),
        Box::new(BaggagePropagator::new()),
    ]));
    otel_trace::mark_otel_enabled();

    (exporter, provider, guard)
}

fn test_router_config(worker_url: &str) -> RouterConfig {
    RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker_url.to_string()],
        },
        policy: PolicyConfig::RoundRobin,
        host: "127.0.0.1".to_string(),
        port: 0,
        ..Default::default()
    }
}

/// All OTel integration tests run sequentially in one function to avoid
/// global state conflicts.
#[tokio::test(flavor = "current_thread")]
async fn test_otel_integration() {
    // Start a real backend server
    let mut worker = MockWorker::new(MockWorkerConfig::default());
    let worker_url = worker.start().await.expect("Failed to start mock worker");
    let port: u16 = worker_url.rsplit(':').next().unwrap().parse().unwrap();
    mock_worker::clear_captured_requests(port);

    let config = test_router_config(&worker_url);
    let ctx = common::create_test_context(config.clone());

    // ==================================================================
    // Section 1: Span attributes on a route through the real app
    // ==================================================================
    {
        let (exporter, provider, _guard) = setup_otel_harness();

        let router = RouterFactory::create_regular_router(std::slice::from_ref(&worker_url), &ctx)
            .await
            .unwrap();
        let app = create_test_app(
            std::sync::Arc::from(router),
            reqwest::Client::new(),
            &config,
        );

        let upstream_trace_id = "baf7651916cd43dd8448eb211c80319c";
        let upstream_span_id = "e7ad6b7169203331";
        let traceparent = format!("00-{upstream_trace_id}-{upstream_span_id}-01");

        let request = Request::builder()
            .method("GET")
            .uri("/health")
            .header("traceparent", &traceparent)
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(response.into_body(), usize::MAX).await;
        tokio::task::yield_now().await;

        provider.force_flush();
        let spans = exporter.get_finished_spans().unwrap();
        assert!(!spans.is_empty(), "Should have at least one exported span");

        // Find the HTTP server span
        let http_span = spans
            .iter()
            .find(|s| s.span_kind == opentelemetry::trace::SpanKind::Server)
            .expect("Should have a SpanKind::Server span");

        // Verify trace_id matches upstream
        let expected_trace_id = opentelemetry::trace::TraceId::from_hex(upstream_trace_id).unwrap();
        assert_eq!(
            http_span.span_context.trace_id(),
            expected_trace_id,
            "Span trace_id should match upstream"
        );

        // Verify parent_span_id matches upstream
        let expected_parent_id = opentelemetry::trace::SpanId::from_hex(upstream_span_id).unwrap();
        assert_eq!(
            http_span.parent_span_id, expected_parent_id,
            "Span parent_span_id should match upstream"
        );

        // Verify span attributes
        let attrs: std::collections::HashMap<String, String> = http_span
            .attributes
            .iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();

        assert_eq!(
            attrs.get("http.request.method").map(|s| s.as_str()),
            Some("GET"),
            "Should record http.request.method"
        );
        assert_eq!(
            attrs.get("url.path").map(|s| s.as_str()),
            Some("/health"),
            "Should record url.path"
        );
        assert!(
            attrs.contains_key("http.response.status_code"),
            "Should record http.response.status_code"
        );
        eprintln!("PASS: Section 1 - Span attributes on health route");
    }

    // ==================================================================
    // Section 2: Trace propagation to backend via chat completions
    // ==================================================================
    {
        let (_exporter, _provider, _guard) = setup_otel_harness();

        mock_worker::clear_captured_requests(port);

        let router = RouterFactory::create_regular_router(std::slice::from_ref(&worker_url), &ctx)
            .await
            .unwrap();
        let app = create_test_app(
            std::sync::Arc::from(router),
            reqwest::Client::new(),
            &config,
        );

        let upstream_trace_id = "aaf7651916cd43dd8448eb211c80319c";
        let upstream_span_id = "c7ad6b7169203331";
        let traceparent = format!("00-{upstream_trace_id}-{upstream_span_id}-01");

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("traceparent", &traceparent)
            .header("baggage", "userId=alice,env=test")
            .body(axum::body::Body::from(
                serde_json::to_string(&serde_json::json!({
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hello"}]
                }))
                .unwrap(),
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(response.into_body(), usize::MAX).await;

        // Verify backend received fresh trace context
        let captured = mock_worker::get_captured_requests(port);
        assert!(
            !captured.is_empty(),
            "Backend should have received at least one request"
        );

        let backend_req = &captured[0];

        // The backend should have traceparent with the SAME trace_id but a
        // DIFFERENT span_id (the router's span, not the upstream's)
        let backend_tp = backend_req
            .headers
            .get("traceparent")
            .expect("Backend should receive traceparent header");
        let parts: Vec<&str> = backend_tp.split('-').collect();
        assert_eq!(parts.len(), 4, "traceparent should have 4 parts");
        assert_eq!(
            parts[1], upstream_trace_id,
            "trace_id should be preserved from upstream"
        );
        assert_ne!(
            parts[2], upstream_span_id,
            "span_id should be the router's own span, not the upstream's"
        );

        // Baggage should be propagated
        let backend_baggage = backend_req
            .headers
            .get("baggage")
            .expect("Backend should receive baggage header");
        assert!(
            backend_baggage.contains("userId=alice"),
            "baggage should contain userId=alice, got: {backend_baggage}"
        );
        eprintln!("PASS: Section 2 - Trace propagation to backend");
    }

    // Clean up
    worker.stop().await;
    otel_trace::shutdown_otel();
}
