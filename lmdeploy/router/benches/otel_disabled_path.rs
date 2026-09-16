use axum::http::{HeaderMap, HeaderValue, Request};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use lmdeploy_router_rs::{middleware::RequestSpan, otel_trace, routers::header_utils};
use tower_http::trace::MakeSpan;

fn bench_request_span_disabled(c: &mut Criterion) {
    otel_trace::shutdown_otel();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(
            "traceparent",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        )
        .body(())
        .unwrap();
    let mut request_span = RequestSpan;

    c.bench_function("otel_request_span_disabled", |b| {
        b.iter(|| {
            let span = request_span.make_span(black_box(&request));
            black_box(span.is_none());
        });
    });
}

fn bench_trace_header_forwarding_disabled(c: &mut Criterion) {
    otel_trace::shutdown_otel();

    let client = reqwest::Client::new();
    let mut incoming = HeaderMap::new();
    incoming.insert(
        "traceparent",
        HeaderValue::from_static("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"),
    );
    incoming.insert("tracestate", HeaderValue::from_static("vendor=value"));
    incoming.insert("baggage", HeaderValue::from_static("userId=alice"));

    for index in 0..32 {
        let name = format!("x-noise-{index}");
        incoming.insert(
            axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_static("1"),
        );
    }

    c.bench_function("otel_trace_header_forwarding_disabled", |b| {
        b.iter_batched(
            || client.post("http://example.com"),
            |request_builder| {
                let request = header_utils::propagate_trace_headers(
                    request_builder,
                    Some(black_box(&incoming)),
                )
                .build()
                .unwrap();
                black_box(request.headers().len());
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    otel_disabled_path_benches,
    bench_request_span_disabled,
    bench_trace_header_forwarding_disabled
);
criterion_main!(otel_disabled_path_benches);
