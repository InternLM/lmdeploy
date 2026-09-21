mod common;

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, HeaderValue, StatusCode},
};
use common::mock_openai_server::MockOpenAIServer;
use lmdeploy_router_rs::{
    middleware::RequestSpan,
    otel_trace,
    protocols::spec::ChatCompletionRequest,
    routers::{openai_router::OpenAIRouter, RouterTrait},
};
use opentelemetry::{global, propagation::TextMapCompositePropagator, trace::TracerProvider as _};
use opentelemetry_sdk::{
    propagation::{BaggagePropagator, TraceContextPropagator},
    testing::trace::InMemorySpanExporter,
    trace::TracerProvider,
};
use tower_http::trace::MakeSpan;
use tracing_subscriber::layer::SubscriberExt;

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

fn create_chat_request() -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32
    }))
    .unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn test_openai_router_propagates_trace_context() {
    let (_exporter, _provider, _guard) = setup_otel_harness();

    let mock_server = MockOpenAIServer::new().await;
    mock_server.clear_captured_requests();

    let router = OpenAIRouter::new(mock_server.base_url(), None)
        .await
        .expect("OpenAI router should be created");

    let upstream_trace_id = "daf7651916cd43dd8448eb211c80319c";
    let upstream_span_id = "a7ad6b7169203331";
    let traceparent = format!("00-{upstream_trace_id}-{upstream_span_id}-01");

    let mut headers = HeaderMap::new();
    headers.insert("traceparent", HeaderValue::from_str(&traceparent).unwrap());
    headers.insert("baggage", HeaderValue::from_static("userId=alice,env=test"));

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("traceparent", &traceparent)
        .header("baggage", "userId=alice,env=test")
        .body(Body::empty())
        .unwrap();
    let mut request_span = RequestSpan;
    let span = request_span.make_span(&request);
    assert!(
        !span.is_none(),
        "OpenAI chat requests should create a real server span when OTel is enabled"
    );

    let response = {
        let _enter = span.enter();
        router
            .route_chat(Some(&headers), &create_chat_request(), None)
            .await
    };
    assert_eq!(response.status(), StatusCode::OK);
    let _ = axum::body::to_bytes(response.into_body(), usize::MAX).await;

    let captured = mock_server.captured_requests();
    assert!(
        !captured.is_empty(),
        "Mock OpenAI server should capture at least one request"
    );
    let backend_request = &captured[0];
    assert_eq!(
        backend_request.path, "/v1/chat/completions",
        "Trace propagation test should hit the chat completions endpoint"
    );

    let backend_traceparent = backend_request
        .headers
        .get("traceparent")
        .expect("OpenAI upstream should receive traceparent");
    let parts: Vec<&str> = backend_traceparent.split('-').collect();
    assert_eq!(parts.len(), 4, "traceparent should have 4 parts");
    assert_eq!(
        parts[1], upstream_trace_id,
        "OpenAI upstream trace_id should stay on the upstream trace"
    );
    assert_ne!(
        parts[2], upstream_span_id,
        "OpenAI upstream span_id should be a child span, not the original caller span"
    );
    let baggage = backend_request
        .headers
        .get("baggage")
        .expect("OpenAI upstream should receive baggage");
    assert!(
        baggage.contains("userId=alice") && baggage.contains("env=test"),
        "OpenAI upstream should receive both baggage items, got: {baggage}"
    );

    otel_trace::shutdown_otel();
}
