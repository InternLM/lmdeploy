use axum::http::HeaderMap;
use tracing::{field::Empty, info_span, Instrument, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use url::Url;

pub const TRACE_HEADER_NAMES: &[&str] = &["traceparent", "tracestate", "baggage"];

#[derive(Clone, Copy, Debug)]
pub struct ClientRequestOptions<'a> {
    pub method: &'a str,
    pub url: &'a str,
    pub route: Option<&'a str>,
    pub request_phase: Option<&'a str>,
}

pub struct PreparedClientRequest {
    request: reqwest::RequestBuilder,
    span: Span,
}

impl PreparedClientRequest {
    pub async fn send(self) -> Result<reqwest::Response, reqwest::Error> {
        let Self { request, span } = self;

        let response = if span.is_none() {
            request.send().await
        } else {
            request.send().instrument(span.clone()).await
        };

        match response {
            Ok(response) => {
                if !span.is_disabled() {
                    let status = response.status();
                    span.record("http.response.status_code", status.as_u16());

                    if status.is_server_error() {
                        span.set_status(opentelemetry::trace::Status::error(format!(
                            "HTTP {}",
                            status.as_u16()
                        )));
                    }
                }

                Ok(response)
            }
            Err(error) => {
                if !span.is_disabled() {
                    let message = error.to_string();
                    span.record("error", message.as_str());
                    span.set_status(opentelemetry::trace::Status::error(message));
                }

                Err(error)
            }
        }
    }
}

pub fn prepare_client_request(
    mut request: reqwest::RequestBuilder,
    incoming_headers: Option<&HeaderMap>,
    options: ClientRequestOptions<'_>,
) -> PreparedClientRequest {
    let span = if crate::otel_trace::is_otel_enabled() {
        let span = info_span!(
            target: "otel_trace",
            "http_client_request",
            otel.kind = "client",
            otel.name = %format_args!(
                "{} {}",
                options.method,
                options.route.unwrap_or(options.url)
            ),
            http.request.method = %options.method,
            http.route = Empty,
            url.full = %options.url,
            server.address = Empty,
            server.port = Empty,
            inference.request_phase = Empty,
            http.response.status_code = Empty,
            error = Empty,
        );

        if let Some(route) = options.route {
            span.record("http.route", route);
        }
        if let Some(request_phase) = options.request_phase {
            span.record("inference.request_phase", request_phase);
        }
        record_server_address(&span, options.url);

        {
            let _enter = span.enter();
            request = propagate_trace_headers(request, incoming_headers);
        }
        span
    } else {
        request = propagate_trace_headers(request, incoming_headers);
        Span::none()
    };

    PreparedClientRequest { request, span }
}

pub async fn send_client_request(
    request: reqwest::RequestBuilder,
    incoming_headers: Option<&HeaderMap>,
    options: ClientRequestOptions<'_>,
) -> Result<reqwest::Response, reqwest::Error> {
    prepare_client_request(request, incoming_headers, options)
        .send()
        .await
}

pub fn propagate_trace_headers(
    mut request: reqwest::RequestBuilder,
    headers: Option<&HeaderMap>,
) -> reqwest::RequestBuilder {
    if crate::otel_trace::is_otel_enabled() {
        let mut trace_headers = HeaderMap::new();
        crate::otel_trace::inject_trace_context_http(&mut trace_headers);
        for (name, value) in trace_headers.iter() {
            request = request.header(name, value);
        }
        return request;
    }

    if let Some(headers) = headers {
        for &name in TRACE_HEADER_NAMES {
            if let Some(value) = headers.get(name) {
                request = request.header(name, value);
            }
        }
    }

    request
}

fn record_server_address(span: &Span, url: &str) {
    let Ok(parsed) = Url::parse(url) else {
        return;
    };

    if let Some(host) = parsed.host_str() {
        span.record("server.address", host);
    }

    if let Some(port) = parsed.port_or_known_default() {
        span.record("server.port", port);
    }
}
