use axum::http::Request;
use lmdeploy_router_rs::{
    config::TraceConfig,
    middleware::RequestSpan,
    otel_trace::{self, PreparedOtel},
};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::{testing::trace::InMemorySpanExporter, trace::TracerProvider};
use tower_http::trace::MakeSpan;
use tracing_subscriber::layer::SubscriberExt;

#[test]
fn test_trace_config_controls() {
    let invalid_config = TraceConfig {
        sampling_ratio: 1.5,
        ..Default::default()
    };
    assert!(
        otel_trace::prepare_otel(&invalid_config).is_err(),
        "sampling ratios outside [0.0, 1.0] should be rejected"
    );

    let exporter = InMemorySpanExporter::default();
    let provider = TracerProvider::builder()
        .with_simple_exporter(exporter)
        .build();
    let tracer = provider.tracer("test-tracer");
    let prepared =
        PreparedOtel::from_parts_with_excluded_paths(provider, tracer, vec!["/health".to_string()]);

    let subscriber = tracing_subscriber::registry().with(prepared.layer());
    let _guard = tracing::subscriber::set_default(subscriber);
    otel_trace::activate_otel(prepared);

    let mut request_span = RequestSpan;

    let excluded_request = Request::builder()
        .method("GET")
        .uri("/health")
        .body(())
        .unwrap();
    let excluded_span = request_span.make_span(&excluded_request);
    assert!(
        excluded_span.is_none(),
        "Excluded paths should not create OTel spans"
    );

    let traced_request = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(())
        .unwrap();
    let traced_span = request_span.make_span(&traced_request);
    assert!(
        !traced_span.is_none(),
        "Non-excluded paths should still create OTel spans"
    );

    otel_trace::shutdown_otel();
}
