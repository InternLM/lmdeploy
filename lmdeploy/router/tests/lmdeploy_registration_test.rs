use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    Router,
};
use lmdeploy_router_rs::{
    config::{LMDeployMigrationProtocol, RouterConfig, RoutingMode},
    routers::{http::lmdeploy_pd_router::LMDeployPDRouter, http::router::Router as HttpRouter},
    server::{build_app_with_request_tracing, AppContext, AppState},
};
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;

async fn regular_app() -> (Router, Arc<AppContext>) {
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: Vec::new(),
        },
        ..RouterConfig::default()
    };
    let context = Arc::new(
        AppContext::new(
            config.clone(),
            Client::new(),
            config.max_concurrent_requests,
            config.rate_limit_tokens_per_second,
            Vec::new(),
        )
        .unwrap(),
    );
    let router = Arc::new(HttpRouter::new(Vec::new(), &context).await.unwrap());
    let state = Arc::new(AppState {
        router,
        context: context.clone(),
        concurrency_queue_tx: None,
        router_manager: None,
    });
    (
        build_app_with_request_tracing(state, 1024 * 1024, Vec::new(), Vec::new(), true, false),
        context,
    )
}

async fn lmdeploy_pd_app() -> (Router, Arc<AppContext>) {
    let config = RouterConfig {
        mode: RoutingMode::LMDeployPrefillDecode {
            prefill_urls: Vec::new(),
            decode_urls: Vec::new(),
            prefill_policy: None,
            decode_policy: None,
            migration_protocol: LMDeployMigrationProtocol::Rdma,
            rdma_config: None,
            dummy_prefill: false,
        },
        ..RouterConfig::default()
    };
    let context = Arc::new(
        AppContext::new(
            config.clone(),
            Client::new(),
            config.max_concurrent_requests,
            config.rate_limit_tokens_per_second,
            Vec::new(),
        )
        .unwrap(),
    );
    let router = Arc::new(
        LMDeployPDRouter::new(
            Vec::new(),
            Vec::new(),
            LMDeployMigrationProtocol::Rdma,
            None,
            false,
            &context,
        )
        .await
        .unwrap(),
    );
    let state = Arc::new(AppState {
        router,
        context: context.clone(),
        concurrency_queue_tx: None,
        router_manager: None,
    });
    (
        build_app_with_request_tracing(state, 1024 * 1024, Vec::new(), Vec::new(), true, false),
        context,
    )
}

async fn json_request(app: Router, method: &str, uri: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = serde_json::from_slice(&body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&body).into_owned()));
    (status, body)
}

#[tokio::test]
async fn lmdeploy_hybrid_registration_is_immediate_and_idempotent() {
    let (app, context) = regular_app().await;
    let url = "http://127.0.0.1:49151";
    let node = json!({
        "url": url,
        "status": {"models": ["test-model"], "role": 1}
    });

    let (status, body) = json_request(app.clone(), "POST", "/nodes/add", node.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!("Added successfully"));
    assert_eq!(context.worker_registry.get_all().len(), 1);

    let (status, _) = json_request(app.clone(), "POST", "/nodes/add", node).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(context.worker_registry.get_all().len(), 1);

    let (status, body) = json_request(app.clone(), "GET", "/nodes/status", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body[url]["role"], 1);
    assert_eq!(body[url]["models"], json!(["test-model"]));
    assert_eq!(body[url]["unfinished"], 0);
    assert_eq!(body[url]["latency"], json!([]));
    assert!(body[url]["speed"].is_null());

    let (status, body) = json_request(app, "POST", "/nodes/remove", json!({"url": url})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!("Deleted successfully"));
    assert!(context.worker_registry.get_all().is_empty());
}

#[tokio::test]
async fn lmdeploy_pd_registration_dispatches_prefill_and_decode_roles() {
    let (app, context) = lmdeploy_pd_app().await;
    let prefill = "http://127.0.0.1:49152";
    let decode = "http://127.0.0.1:49153";

    for (url, role) in [(prefill, 2), (decode, 3)] {
        let (status, body) = json_request(
            app.clone(),
            "POST",
            "/nodes/add",
            json!({
                "url": url,
                "status": {"models": ["test-model"], "role": role}
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!("Added successfully"));
    }

    assert_eq!(context.worker_registry.get_prefill_workers().len(), 1);
    assert_eq!(context.worker_registry.get_decode_workers().len(), 1);

    let (status, _) = json_request(
        app,
        "POST",
        "/nodes/add",
        json!({
            "url": "http://127.0.0.1:49154",
            "status": {"models": ["test-model"], "role": 1}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
