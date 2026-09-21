//! Integration tests for OpenTelemetry distributed tracing.
//!
//! These tests verify that:
//! 1. Inbound `traceparent` is extracted into the router span and the router
//!    span becomes a child of the upstream caller's span.
//! 2. `inject_trace_context_http` (the real crate helper) correctly injects
//!    the router's span context into outgoing backend requests.
//! 3. Inbound `baggage` survives the enabled path via the composite propagator.
//!
//! NOTE: These tests share global OTel state (propagator, ENABLED flag) and
//! must run sequentially. They are combined into a single test function to
//! avoid parallel execution issues with `set_text_map_propagator` and the
//! global ENABLED flag.

use axum::http::{HeaderMap, HeaderValue};
use lmdeploy_router_rs::otel_trace;
use opentelemetry::{
    global,
    propagation::TextMapCompositePropagator,
    trace::{SpanId, TraceId, TracerProvider as _},
};
use opentelemetry_sdk::{
    propagation::{BaggagePropagator, TraceContextPropagator},
    testing::trace::InMemorySpanExporter,
    trace::TracerProvider,
};
use tracing::info_span;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::layer::SubscriberExt;

/// Helper: set up a test OTel environment with in-memory exporter.
fn setup_test_otel() -> (
    InMemorySpanExporter,
    TracerProvider,
    tracing::subscriber::DefaultGuard,
) {
    let exporter = InMemorySpanExporter::default();
    let provider = TracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let tracer = provider.tracer("test-tracer");

    // Use the same composite propagator as production code
    global::set_text_map_propagator(TextMapCompositePropagator::new(vec![
        Box::new(TraceContextPropagator::new()),
        Box::new(BaggagePropagator::new()),
    ]));

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let subscriber = tracing_subscriber::registry().with(otel_layer);
    let guard = tracing::subscriber::set_default(subscriber);

    // Enable the flag so real code paths (make_span, inject_trace_context_http) activate
    otel_trace::mark_otel_enabled();

    (exporter, provider, guard)
}

/// All OTel tests run sequentially in one function to avoid global state
/// conflicts (propagator, ENABLED flag, tracer provider).
#[test]
fn test_otel_distributed_tracing() {
    // =====================================================================
    // Section 1: End-to-end trace context propagation using real crate helpers
    // =====================================================================
    {
        let (exporter, provider, _guard) = setup_test_otel();

        let upstream_trace_id = "0af7651916cd43dd8448eb211c80319c";
        let upstream_span_id = "b7ad6b7169203331";
        let traceparent = format!("00-{}-{}-01", upstream_trace_id, upstream_span_id);

        let mut incoming_headers = HeaderMap::new();
        incoming_headers.insert("traceparent", HeaderValue::from_str(&traceparent).unwrap());

        // Extract parent context from headers (same as make_span does)
        let parent_cx = global::get_text_map_propagator(|propagator| {
            propagator.extract(&opentelemetry_http::HeaderExtractor(&incoming_headers))
        });

        let span = info_span!(
            target: "otel_trace",
            "http_request",
            method = "POST",
            uri = "/v1/chat/completions"
        );
        span.set_parent(parent_cx);

        // Outbound injection using the REAL crate helper
        let outgoing_headers = {
            let _enter = span.enter();
            let mut headers = HeaderMap::new();
            otel_trace::inject_trace_context_http(&mut headers);
            headers
        };

        drop(span);
        provider.force_flush();

        // Verify inbound context was correctly linked
        let spans = exporter.get_finished_spans().unwrap();
        assert!(
            !spans.is_empty(),
            "Expected at least one span to be exported"
        );

        let exported_span = &spans[0];
        let expected_trace_id = TraceId::from_hex(upstream_trace_id).unwrap();
        assert_eq!(
            exported_span.span_context.trace_id(),
            expected_trace_id,
            "Span's trace_id should match the incoming traceparent's trace_id"
        );

        let expected_parent_id = SpanId::from_hex(upstream_span_id).unwrap();
        assert_eq!(
            exported_span.parent_span_id, expected_parent_id,
            "Span's parent_span_id should match the incoming traceparent's span_id"
        );

        // Verify outbound context was correctly injected
        let outgoing_tp = outgoing_headers
            .get("traceparent")
            .expect("Expected outgoing traceparent header")
            .to_str()
            .unwrap();

        let parts: Vec<&str> = outgoing_tp.split('-').collect();
        assert_eq!(parts.len(), 4, "traceparent should have 4 parts");
        assert_eq!(parts[0], "00", "version should be 00");
        assert_eq!(
            parts[1], upstream_trace_id,
            "trace_id in outgoing request should match the original upstream trace_id"
        );
        assert_ne!(
            parts[2], upstream_span_id,
            "outgoing span_id should be the router's own span, not the upstream's"
        );
        assert_eq!(parts[3], "01", "trace flags should indicate sampled");

        eprintln!("PASS: Section 1 - End-to-end trace context propagation");
    }

    // =====================================================================
    // Section 2: RequestSpan::make_span exercises parent extraction path
    // =====================================================================
    {
        use axum::http::Request;
        use lmdeploy_router_rs::middleware::RequestSpan;
        use tower_http::trace::MakeSpan;

        let (exporter, provider, _guard) = setup_test_otel();

        let upstream_trace_id = "2af7651916cd43dd8448eb211c80319c";
        let upstream_span_id = "d7ad6b7169203331";
        let traceparent = format!("00-{}-{}-01", upstream_trace_id, upstream_span_id);

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("traceparent", &traceparent)
            .body(())
            .unwrap();

        let mut request_span = RequestSpan;
        let span = request_span.make_span(&request);

        {
            let _enter = span.enter();
        }
        drop(span);
        provider.force_flush();

        let spans = exporter.get_finished_spans().unwrap();
        assert!(!spans.is_empty(), "Expected at least one span");

        // Verify parent context was extracted (this was previously skipped!)
        let exported_span = &spans[0];
        let expected_trace_id = TraceId::from_hex(upstream_trace_id).unwrap();
        assert_eq!(
            exported_span.span_context.trace_id(),
            expected_trace_id,
            "make_span should extract the upstream trace_id from traceparent"
        );

        let expected_parent_id = SpanId::from_hex(upstream_span_id).unwrap();
        assert_eq!(
            exported_span.parent_span_id, expected_parent_id,
            "make_span should set the upstream span as parent"
        );

        eprintln!("PASS: Section 2 - RequestSpan::make_span parent extraction");
    }

    // =====================================================================
    // Section 3: Baggage survives the enabled path
    // =====================================================================
    {
        let (_exporter, _provider, _guard) = setup_test_otel();

        let mut incoming_headers = HeaderMap::new();
        incoming_headers.insert(
            "traceparent",
            HeaderValue::from_static("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"),
        );
        incoming_headers.insert(
            "baggage",
            HeaderValue::from_static("userId=alice,serverNode=DF%2028"),
        );

        let parent_cx = global::get_text_map_propagator(|propagator| {
            propagator.extract(&opentelemetry_http::HeaderExtractor(&incoming_headers))
        });

        let span = info_span!(target: "otel_trace", "http_request");
        span.set_parent(parent_cx);

        let outgoing_headers = {
            let _enter = span.enter();
            let mut headers = HeaderMap::new();
            otel_trace::inject_trace_context_http(&mut headers);
            headers
        };

        let baggage = outgoing_headers
            .get("baggage")
            .expect("baggage header should be propagated when tracing is enabled")
            .to_str()
            .unwrap();

        assert!(
            baggage.contains("userId=alice"),
            "baggage should contain userId=alice, got: {baggage}"
        );

        eprintln!("PASS: Section 3 - Baggage survives enabled path");
    }

    // =====================================================================
    // Section 4: Client spans can be exported with proper kind and attrs
    // =====================================================================
    {
        let (exporter, provider, _guard) = setup_test_otel();

        let span = info_span!(
            target: "otel_trace",
            "http_client_request",
            otel.kind = "client",
            otel.name = "POST /v1/chat/completions",
            http.request.method = "POST",
            http.route = "/v1/chat/completions",
            url.full = "http://127.0.0.1:3000/v1/chat/completions",
        );

        {
            let _enter = span.enter();
        }
        drop(span);
        provider.force_flush();

        let spans = exporter.get_finished_spans().unwrap();
        let client_span = spans
            .iter()
            .find(|span| span.span_kind == opentelemetry::trace::SpanKind::Client)
            .expect("Expected a client span to be exported");

        let attrs: std::collections::HashMap<String, String> = client_span
            .attributes
            .iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();

        assert_eq!(
            attrs.get("http.request.method").map(|value| value.as_str()),
            Some("POST"),
            "Client span should record the HTTP method"
        );
        assert_eq!(
            attrs.get("http.route").map(|value| value.as_str()),
            Some("/v1/chat/completions"),
            "Client span should record the logical route"
        );
        assert_eq!(
            attrs.get("url.full").map(|value| value.as_str()),
            Some("http://127.0.0.1:3000/v1/chat/completions"),
            "Client span should record the backend URL"
        );

        eprintln!("PASS: Section 4 - Client span kind and attrs");
    }

    // Clean up global state
    otel_trace::shutdown_otel();
}
