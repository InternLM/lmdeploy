mod common;

use std::sync::Arc;

use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue, Request, StatusCode},
};
use common::{
    create_test_context,
    mock_worker::{MockWorker, MockWorkerConfig},
    test_app::create_test_app_with_tracing,
};
use lmdeploy_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    middleware::RequestSpan,
    otel_trace,
    routers::{header_utils, RouterFactory},
};
use reqwest::Client;
use tower::ServiceExt;
use tower_http::trace::MakeSpan;

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

#[test]
fn request_span_is_none_when_otel_disabled() {
    otel_trace::shutdown_otel();

    let request = Request::builder()
        .method("GET")
        .uri("/health")
        .header(
            "traceparent",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        )
        .body(())
        .unwrap();

    let mut request_span = RequestSpan;
    let span = request_span.make_span(&request);

    assert!(
        span.is_none(),
        "RequestSpan should return Span::none() when OTel is disabled"
    );
}

#[test]
fn inject_trace_context_http_is_noop_when_otel_disabled() {
    otel_trace::shutdown_otel();

    let mut headers = HeaderMap::new();
    otel_trace::inject_trace_context_http(&mut headers);

    assert!(
        headers.is_empty(),
        "No trace context should be injected when OTel is disabled"
    );
}

#[test]
fn propagate_trace_headers_passively_forwards_only_w3c_headers_when_otel_disabled() {
    otel_trace::shutdown_otel();

    let mut incoming = HeaderMap::new();
    incoming.insert(
        "traceparent",
        HeaderValue::from_static("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"),
    );
    incoming.insert("tracestate", HeaderValue::from_static("vendor=value"));
    incoming.insert("baggage", HeaderValue::from_static("userId=alice"));
    incoming.insert("x-extra-header", HeaderValue::from_static("ignore-me"));

    let request = header_utils::propagate_trace_headers(
        reqwest::Client::new().post("http://example.com"),
        Some(&incoming),
    )
    .build()
    .unwrap();

    assert_eq!(
        request.headers().get("traceparent"),
        incoming.get("traceparent"),
        "traceparent should be forwarded unchanged"
    );
    assert_eq!(
        request.headers().get("tracestate"),
        incoming.get("tracestate"),
        "tracestate should be forwarded unchanged"
    );
    assert_eq!(
        request.headers().get("baggage"),
        incoming.get("baggage"),
        "baggage should be forwarded unchanged"
    );
    assert!(
        request.headers().get("x-extra-header").is_none(),
        "Non-trace headers should not be forwarded by propagate_trace_headers"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn app_routes_requests_without_trace_layer_when_otel_disabled() {
    otel_trace::shutdown_otel();

    let mut worker = MockWorker::new(MockWorkerConfig::default());
    let worker_url = worker.start().await.expect("Failed to start mock worker");

    let config = test_router_config(&worker_url);
    let ctx = create_test_context(config.clone());
    let router = RouterFactory::create_regular_router(std::slice::from_ref(&worker_url), &ctx)
        .await
        .expect("Failed to create router");
    let app = create_test_app_with_tracing(Arc::from(router), Client::new(), &config, false);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/liveness")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("Request should succeed with tracing disabled");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response.headers().contains_key("x-request-id"),
        "RequestIdLayer should still run when request tracing is disabled"
    );

    worker.stop().await;
}
