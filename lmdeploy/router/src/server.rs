use crate::{
    config::{HistoryBackend, LMDeployMigrationProtocol, RouterConfig, TraceConfig},
    core::{WorkerRegistry, WorkerType},
    data_connector::{MemoryResponseStorage, NoOpResponseStorage, SharedResponseStorage},
    logging::{self, LoggingConfig},
    metrics::{self, PrometheusConfig},
    middleware::{self, QueuedRequest, TokenBucket},
    policies::PolicyRegistry,
    protocols::{
        spec::{ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest},
        worker_spec::{WorkerApiResponse, WorkerConfigRequest, WorkerErrorResponse},
    },
    routers::{
        factory::LMDeployPDRouterParams,
        router_manager::{RouterId, RouterManager},
        RouterFactory, RouterTrait,
    },
    service_discovery::{start_service_discovery, ServiceDiscoveryConfig},
};
use axum::{
    extract::{DefaultBodyLimit, Path, Query, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    serve, Json, Router,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{HashMap, VecDeque},
    sync::atomic::{AtomicBool, Ordering},
    sync::Arc,
    time::Duration,
};
use tokio::{
    net::TcpListener,
    signal, spawn,
    sync::{Mutex, RwLock},
};
use tracing::{error, info, warn, Level};

#[derive(Clone)]
pub struct AppContext {
    pub client: Client,
    pub router_config: RouterConfig,
    pub rate_limiter: Arc<TokenBucket>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub response_storage: SharedResponseStorage,
    pub api_key_cache: Arc<RwLock<HashMap<String, bool>>>,
    pub api_key_validation_urls: Arc<Vec<String>>,
    pub(crate) lmdeploy_nodes: Arc<Mutex<HashMap<String, LMDeployNodeStatus>>>,
}

impl AppContext {
    pub fn new(
        router_config: RouterConfig,
        client: Client,
        max_concurrent_requests: usize,
        rate_limit_tokens_per_second: Option<usize>,
        api_key_validation_urls: Vec<String>,
    ) -> Result<Self, String> {
        let rate_limit_tokens = rate_limit_tokens_per_second.unwrap_or(max_concurrent_requests);
        let rate_limiter = Arc::new(TokenBucket::new(max_concurrent_requests, rate_limit_tokens));

        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

        let router_manager = None;

        // Initialize response storage based on configuration
        let response_storage: SharedResponseStorage = match router_config.history_backend {
            HistoryBackend::Memory => Arc::new(MemoryResponseStorage::new()),
            HistoryBackend::None => Arc::new(NoOpResponseStorage::new()),
        };

        Ok(Self {
            client,
            router_config,
            rate_limiter,
            worker_registry,
            policy_registry,
            router_manager,
            response_storage,
            api_key_cache: Arc::new(RwLock::new(HashMap::new())),
            api_key_validation_urls: Arc::new(api_key_validation_urls),
            lmdeploy_nodes: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

const LMDEPLOY_ROLE_HYBRID: u8 = 1;
const LMDEPLOY_ROLE_PREFILL: u8 = 2;
const LMDEPLOY_ROLE_DECODE: u8 = 3;

fn default_lmdeploy_role() -> u8 {
    LMDEPLOY_ROLE_HYBRID
}

/// Wire-compatible with lmdeploy.serve.proxy.proxy.Status.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct LMDeployNodeStatus {
    #[serde(default = "default_lmdeploy_role")]
    role: u8,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    unfinished: usize,
    #[serde(default)]
    latency: VecDeque<f64>,
    #[serde(default)]
    speed: Option<i64>,
}

impl Default for LMDeployNodeStatus {
    fn default() -> Self {
        Self {
            role: LMDEPLOY_ROLE_HYBRID,
            models: Vec::new(),
            unfinished: 0,
            latency: VecDeque::new(),
            speed: None,
        }
    }
}

/// Wire-compatible with lmdeploy.serve.proxy.proxy.Node.
#[derive(Debug, Deserialize)]
struct LMDeployNode {
    url: String,
    #[serde(default)]
    status: Option<LMDeployNodeStatus>,
}

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<dyn RouterTrait>,
    pub context: Arc<AppContext>,
    pub concurrency_queue_tx: Option<tokio::sync::mpsc::Sender<QueuedRequest>>,
    pub router_manager: Option<Arc<RouterManager>>,
}

// Fallback handler for unmatched routes
async fn sink_handler() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

/// Transparent proxy handler for unmatched routes
/// Routes requests through the router's route_transparent method
async fn transparent_proxy_handler(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let headers = req.headers().clone();

    // Check authorization
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    // Extract path and method
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    // Read body
    let body_bytes = match axum::body::to_bytes(req.into_body(), usize::MAX).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("Failed to read request body: {}", e),
            )
                .into_response()
        }
    };

    // Parse body as JSON
    let body_json: serde_json::Value = if body_bytes.is_empty() {
        serde_json::Value::Null
    } else {
        match serde_json::from_slice(&body_bytes) {
            Ok(json) => json,
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("Invalid JSON body: {}", e))
                    .into_response()
            }
        }
    };

    // Route through transparent proxy
    state
        .router
        .route_transparent(Some(&headers), &path, &method, body_json)
        .await
}

// Health check endpoints
async fn liveness(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let headers = req.headers().clone();
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.liveness()
}

async fn readiness(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let headers = req.headers().clone();
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.readiness()
}

async fn health(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let headers = req.headers().clone();
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.health(req).await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let headers = req.headers().clone();
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.health_generate(req).await
}

async fn get_server_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let headers = req.headers().clone();
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.get_server_info(req).await
}

async fn v1_models(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let headers = req.headers().clone();
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.get_models(req).await
}

async fn get_model_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let headers = req.headers().clone();
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.get_model_info(req).await
}

// Generation endpoints
// The RouterTrait now accepts optional headers and typed body directly
async fn generate(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<GenerateRequest>,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state
        .router
        .route_generate(Some(&headers), &body, None)
        .await
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.route_chat(Some(&headers), &body, None).await
}

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<CompletionRequest>,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state
        .router
        .route_completion(Some(&headers), &body, None)
        .await
}

async fn v1_responses(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state
        .router
        .route_transparent(Some(&headers), "/v1/responses", &http::Method::POST, body)
        .await
}

async fn v1_embeddings(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state
        .router
        .route_embeddings(Some(&headers), &body, None)
        .await
}

async fn v1_responses_get(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state
        .router
        .get_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_cancel(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state
        .router
        .cancel_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_delete(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    // Python server does not support this yet
    state
        .router
        .delete_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_list_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    // Python server does not support this yet
    state
        .router
        .list_response_input_items(Some(&headers), &response_id)
        .await
}

const AUTH_FAILURE_MESSAGE: &str =
    "You must provide a valid API key. Obtain one from http://helmholtz.cloud";

// ---------- Worker management endpoints (Legacy) ----------

#[derive(Deserialize)]
struct UrlQuery {
    url: String,
}

async fn authorize_request(
    state: &Arc<AppState>,
    headers: &http::HeaderMap,
) -> Result<(), Box<Response>> {
    let validation_urls = state.context.api_key_validation_urls.as_ref();
    if validation_urls.is_empty() {
        return Ok(());
    }

    let auth_header = headers
        .get(http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();

    let token = auth_header
        .strip_prefix("Bearer ")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            Box::new((StatusCode::UNAUTHORIZED, AUTH_FAILURE_MESSAGE).into_response())
        })?;

    if let Some(valid) = state.context.api_key_cache.read().await.get(token).copied() {
        if valid {
            return Ok(());
        }
        return Err(Box::new(
            (StatusCode::UNAUTHORIZED, AUTH_FAILURE_MESSAGE).into_response(),
        ));
    }

    let mut validated = false;
    for url in validation_urls {
        match state
            .context
            .client
            .get(url)
            .header(http::header::AUTHORIZATION, format!("Bearer {token}"))
            .send()
            .await
        {
            Ok(response) if response.status() == StatusCode::OK => {
                validated = true;
                break;
            }
            Ok(_) => {
                continue;
            }
            Err(err) => {
                warn!("Failed to validate API key against {url}: {err}");
            }
        }
    }

    state
        .context
        .api_key_cache
        .write()
        .await
        .insert(token.to_string(), validated);

    if validated {
        Ok(())
    } else {
        Err(Box::new(
            (StatusCode::UNAUTHORIZED, AUTH_FAILURE_MESSAGE).into_response(),
        ))
    }
}

fn validate_lmdeploy_role_for_router(state: &AppState, role: u8) -> Result<(), String> {
    if state
        .router
        .as_any()
        .is::<crate::routers::http::lmdeploy_pd_router::LMDeployPDRouter>()
    {
        return match role {
            LMDEPLOY_ROLE_PREFILL | LMDEPLOY_ROLE_DECODE => Ok(()),
            LMDEPLOY_ROLE_HYBRID => Err(
                "LMDeploy PD mode accepts only Prefill (2) or Decode (3) registrations".to_string(),
            ),
            _ => Err(format!(
                "Invalid LMDeploy engine role {role}; expected Hybrid=1, Prefill=2, or Decode=3"
            )),
        };
    }

    if state
        .router
        .as_any()
        .is::<crate::routers::http::router::Router>()
    {
        return match role {
            LMDEPLOY_ROLE_HYBRID => Ok(()),
            LMDEPLOY_ROLE_PREFILL | LMDEPLOY_ROLE_DECODE => {
                Err("Regular mode accepts only Hybrid (1) LMDeploy registrations".to_string())
            }
            _ => Err(format!(
                "Invalid LMDeploy engine role {role}; expected Hybrid=1, Prefill=2, or Decode=3"
            )),
        };
    }

    Err("LMDeploy node registration is supported only in regular or LMDeploy PD mode".to_string())
}

fn registered_lmdeploy_role(state: &AppState, node_url: &str) -> Option<u8> {
    let dp_prefix = format!("{}@", node_url);
    state
        .context
        .worker_registry
        .get_all()
        .into_iter()
        .find(|worker| worker.url() == node_url || worker.url().starts_with(&dp_prefix))
        .map(|worker| match worker.worker_type() {
            WorkerType::Regular => LMDEPLOY_ROLE_HYBRID,
            WorkerType::Prefill => LMDEPLOY_ROLE_PREFILL,
            WorkerType::Decode => LMDEPLOY_ROLE_DECODE,
        })
}

fn register_lmdeploy_worker_unchecked(
    state: &AppState,
    node_url: &str,
    role: u8,
) -> Result<(), String> {
    validate_lmdeploy_role_for_router(state, role)?;

    if let Some(router) = state
        .router
        .as_any()
        .downcast_ref::<crate::routers::http::lmdeploy_pd_router::LMDeployPDRouter>(
    ) {
        return match role {
            LMDEPLOY_ROLE_PREFILL => router
                .register_prefill_server_unchecked(node_url.to_string())
                .map(|_| ())
                .map_err(|error| error.to_string()),
            LMDEPLOY_ROLE_DECODE => router
                .register_decode_server_unchecked(node_url.to_string())
                .map(|_| ())
                .map_err(|error| error.to_string()),
            _ => unreachable!("role was validated above"),
        };
    }

    let router = state
        .router
        .as_any()
        .downcast_ref::<crate::routers::http::router::Router>()
        .expect("regular router type was validated above");
    router.register_worker_unchecked(node_url).map(|_| ())
}

async fn remove_lmdeploy_worker(state: &AppState, node_url: &str, role: u8) -> Result<(), String> {
    validate_lmdeploy_role_for_router(state, role)?;

    if let Some(router) = state
        .router
        .as_any()
        .downcast_ref::<crate::routers::http::lmdeploy_pd_router::LMDeployPDRouter>(
    ) {
        return match role {
            LMDEPLOY_ROLE_PREFILL => router
                .remove_prefill_server(node_url)
                .await
                .map(|_| ())
                .map_err(|error| error.to_string()),
            LMDEPLOY_ROLE_DECODE => router
                .remove_decode_server(node_url)
                .await
                .map(|_| ())
                .map_err(|error| error.to_string()),
            _ => unreachable!("role was validated above"),
        };
    }

    let router = state
        .router
        .as_any()
        .downcast_ref::<crate::routers::http::router::Router>()
        .expect("regular router type was validated above");
    router.remove_worker(node_url);
    Ok(())
}

/// LMDeploy-compatible dynamic registration endpoint. This endpoint is
/// intentionally unauthenticated: `lmdeploy serve api_server --proxy-url`
/// does not send an Authorization header.
async fn add_lmdeploy_node(
    State(state): State<Arc<AppState>>,
    Json(node): Json<LMDeployNode>,
) -> Response {
    if node.url.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json("Failed to add, please check the input url."),
        )
            .into_response();
    }

    let status = node.status.unwrap_or_default();
    if let Err(error) = validate_lmdeploy_role_for_router(&state, status.role) {
        return (StatusCode::BAD_REQUEST, error).into_response();
    }

    // Serialize registration/removal for a URL. This also makes repeated
    // startup registration idempotent instead of surfacing WorkerAlreadyExists.
    let mut nodes = state.context.lmdeploy_nodes.lock().await;
    let current_role = registered_lmdeploy_role(&state, &node.url);

    if let Some(old_role) = current_role {
        if old_role != status.role {
            if let Err(error) = remove_lmdeploy_worker(&state, &node.url, old_role).await {
                return (StatusCode::BAD_REQUEST, error).into_response();
            }
            if let Err(error) = register_lmdeploy_worker_unchecked(&state, &node.url, status.role) {
                let rollback = register_lmdeploy_worker_unchecked(&state, &node.url, old_role);
                let message = match rollback {
                    Ok(()) => error,
                    Err(rollback_error) => {
                        format!("{error}; failed to restore previous role: {rollback_error}")
                    }
                };
                return (StatusCode::BAD_REQUEST, message).into_response();
            }
        }
    } else if let Err(error) = register_lmdeploy_worker_unchecked(&state, &node.url, status.role) {
        return (StatusCode::BAD_REQUEST, error).into_response();
    }

    info!(
        "LMDeploy node registered: url={}, role={}, models={:?}",
        node.url, status.role, status.models
    );
    nodes.insert(node.url, status);
    Json("Added successfully").into_response()
}

async fn remove_lmdeploy_node(
    State(state): State<Arc<AppState>>,
    Json(node): Json<LMDeployNode>,
) -> Response {
    let mut nodes = state.context.lmdeploy_nodes.lock().await;
    let role = registered_lmdeploy_role(&state, &node.url)
        .or_else(|| nodes.get(&node.url).map(|status| status.role))
        .or_else(|| node.status.map(|status| status.role));

    if let Some(role) = role {
        // Native LMDeploy treats removal as idempotent. A missing worker can
        // race with health-based/manual removal, so it is still a success.
        if let Err(error) = remove_lmdeploy_worker(&state, &node.url, role).await {
            if registered_lmdeploy_role(&state, &node.url).is_some() {
                return (StatusCode::BAD_REQUEST, error).into_response();
            }
        }
    }

    nodes.remove(&node.url);
    info!("LMDeploy node removed: url={}", node.url);
    Json("Deleted successfully").into_response()
}

async fn lmdeploy_node_status(State(state): State<Arc<AppState>>) -> Response {
    let dynamic_nodes = state.context.lmdeploy_nodes.lock().await.clone();
    let mut nodes = dynamic_nodes;

    for worker in state.context.worker_registry.get_all() {
        let role = match worker.worker_type() {
            WorkerType::Regular => LMDEPLOY_ROLE_HYBRID,
            WorkerType::Prefill => LMDEPLOY_ROLE_PREFILL,
            WorkerType::Decode => LMDEPLOY_ROLE_DECODE,
        };
        nodes
            .entry(worker.url().to_string())
            .or_insert_with(|| LMDeployNodeStatus {
                role,
                ..LMDeployNodeStatus::default()
            });
    }

    Json(nodes).into_response()
}

async fn add_worker(
    State(state): State<Arc<AppState>>,
    Query(UrlQuery { url }): Query<UrlQuery>,
    headers: http::HeaderMap,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    match state.router.add_worker(&url).await {
        Ok(message) => (StatusCode::OK, message).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, error).into_response(),
    }
}

async fn list_workers(State(state): State<Arc<AppState>>, headers: http::HeaderMap) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    let worker_list = state.router.get_worker_urls();
    Json(serde_json::json!({ "urls": worker_list })).into_response()
}

async fn remove_worker(
    State(state): State<Arc<AppState>>,
    Query(UrlQuery { url }): Query<UrlQuery>,
    headers: http::HeaderMap,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.remove_worker(&url);
    (
        StatusCode::OK,
        format!("Successfully removed worker: {url}"),
    )
        .into_response()
}

async fn flush_cache(State(state): State<Arc<AppState>>, headers: http::HeaderMap) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.flush_cache().await
}

async fn get_loads(State(state): State<Arc<AppState>>, headers: http::HeaderMap) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    state.router.get_worker_loads().await
}

// ---------- Worker management endpoints (RESTful) ----------

/// POST /workers - Add a new worker with full configuration
async fn create_worker(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(config): Json<WorkerConfigRequest>,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    // Check if we have a RouterManager (enable_igw=true)
    if let Some(router_manager) = &state.router_manager {
        // Call RouterManager's add_worker method directly with the full config
        match router_manager.add_worker(config).await {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err(error) => (StatusCode::BAD_REQUEST, Json(error)).into_response(),
        }
    } else {
        // In single router mode, use the router's add_worker with basic config
        match state.router.add_worker(&config.url).await {
            Ok(message) => {
                let response = WorkerApiResponse {
                    success: true,
                    message,
                    worker: None,
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(error) => {
                let error_response = WorkerErrorResponse {
                    error,
                    code: "ADD_WORKER_FAILED".to_string(),
                };
                (StatusCode::BAD_REQUEST, Json(error_response)).into_response()
            }
        }
    }
}

async fn list_workers_rest(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    if let Some(router_manager) = &state.router_manager {
        let response = router_manager.list_workers();
        Json(response).into_response()
    } else {
        // In single router mode, get detailed worker info from registry
        let workers = state.context.worker_registry.get_all();
        let response = serde_json::json!({
            "workers": workers.iter().map(|worker| {
                let worker_info = serde_json::json!({
                    "url": worker.url(),
                    "model_id": worker.model_id(),
                    "worker_type": match worker.worker_type() {
                        WorkerType::Regular => "regular",
                        WorkerType::Prefill => "prefill",
                        WorkerType::Decode => "decode",
                    },
                    "is_healthy": worker.is_healthy(),
                    "load": worker.load(),
                    "connection_mode": format!("{:?}", worker.connection_mode()),
                    "priority": worker.priority(),
                    "cost": worker.cost(),
                });

                worker_info
            }).collect::<Vec<_>>(),
            "total": workers.len(),
            "stats": {
                "prefill_count": state.context.worker_registry.get_prefill_workers().len(),
                "decode_count": state.context.worker_registry.get_decode_workers().len(),
                "regular_count": state.context.worker_registry.get_by_type(&WorkerType::Regular).len(),
            }
        });
        Json(response).into_response()
    }
}

/// GET /workers/{url} - Get specific worker info
async fn get_worker(
    State(state): State<Arc<AppState>>,
    Path(url): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    if let Some(router_manager) = &state.router_manager {
        if let Some(worker) = router_manager.get_worker(&url) {
            Json(worker).into_response()
        } else {
            let error = WorkerErrorResponse {
                error: format!("Worker {url} not found"),
                code: "WORKER_NOT_FOUND".to_string(),
            };
            (StatusCode::NOT_FOUND, Json(error)).into_response()
        }
    } else {
        let workers = state.router.get_worker_urls();
        if workers.contains(&url) {
            Json(json!({
                "url": url,
                "model_id": "unknown",
                "is_healthy": true
            }))
            .into_response()
        } else {
            let error = WorkerErrorResponse {
                error: format!("Worker {url} not found"),
                code: "WORKER_NOT_FOUND".to_string(),
            };
            (StatusCode::NOT_FOUND, Json(error)).into_response()
        }
    }
}

/// DELETE /workers/{url} - Remove a worker
async fn delete_worker(
    State(state): State<Arc<AppState>>,
    Path(url): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    if let Err(response) = authorize_request(&state, &headers).await {
        return *response;
    }

    if let Some(router_manager) = &state.router_manager {
        match router_manager.remove_worker_from_registry(&url) {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err(error) => (StatusCode::BAD_REQUEST, Json(error)).into_response(),
        }
    } else {
        // In single router mode, use router's remove_worker
        state.router.remove_worker(&url);
        let response = WorkerApiResponse {
            success: true,
            message: format!("Worker {url} removed successfully"),
            worker: None,
        };
        (StatusCode::OK, Json(response)).into_response()
    }
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub router_config: RouterConfig,
    pub max_payload_size: usize,
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub service_discovery_config: Option<ServiceDiscoveryConfig>,
    pub prometheus_config: Option<PrometheusConfig>,
    pub request_timeout_secs: u64,
    pub request_id_headers: Option<Vec<String>>,
    pub trace_config: Option<TraceConfig>,
}

/// Build the Axum application with all routes and middleware using the
/// current runtime OpenTelemetry state.
pub fn build_app(
    app_state: Arc<AppState>,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
    enable_transparent_proxy: bool,
) -> Router {
    build_app_with_request_tracing(
        app_state,
        max_payload_size,
        request_id_headers,
        cors_allowed_origins,
        enable_transparent_proxy,
        crate::otel_trace::is_otel_enabled(),
    )
}

/// Build the Axum application with an explicit request-tracing toggle.
pub fn build_app_with_request_tracing(
    app_state: Arc<AppState>,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
    enable_transparent_proxy: bool,
    enable_request_tracing: bool,
) -> Router {
    // Create routes
    let protected_routes = Router::new()
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/v1/responses", post(v1_responses))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/v1/responses/{response_id}", get(v1_responses_get))
        .route(
            "/v1/responses/{response_id}/cancel",
            post(v1_responses_cancel),
        )
        .route("/v1/responses/{response_id}", delete(v1_responses_delete))
        .route(
            "/v1/responses/{response_id}/input",
            get(v1_responses_list_input_items),
        )
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ));

    let public_routes = Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/v1/models", get(v1_models))
        .route("/get_model_info", get(get_model_info))
        .route("/get_server_info", get(get_server_info))
        .route("/nodes/status", get(lmdeploy_node_status))
        .route("/nodes/add", post(add_lmdeploy_node))
        .route("/nodes/remove", post(remove_lmdeploy_node));

    let admin_routes = Router::new()
        .route("/add_worker", post(add_worker))
        .route("/remove_worker", post(remove_worker))
        .route("/list_workers", get(list_workers))
        .route("/flush_cache", post(flush_cache))
        .route("/get_loads", get(get_loads));

    // Worker management routes
    let worker_routes = Router::new()
        .route("/workers", post(create_worker))
        .route("/workers", get(list_workers_rest))
        .route("/workers/{url}", get(get_worker))
        .route("/workers/{url}", delete(delete_worker));

    // Build base app with all routes and middleware
    let base_app = Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .merge(admin_routes)
        .merge(worker_routes)
        // Request body size limiting
        .layer(DefaultBodyLimit::max(max_payload_size))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ));

    let base_app = if enable_request_tracing {
        base_app.layer(middleware::create_logging_layer())
    } else {
        base_app
    };

    let base_app = base_app
        .layer(middleware::RequestIdLayer::new(request_id_headers))
        .layer(create_cors_layer(cors_allowed_origins));

    // Choose fallback based on transparent proxy mode
    if enable_transparent_proxy {
        base_app
            .fallback(transparent_proxy_handler)
            .with_state(app_state)
    } else {
        base_app.fallback(sink_handler).with_state(app_state)
    }
}

pub async fn startup(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("DEBUG: Server startup function called");

    // Only initialize logging if not already done (for Python bindings support)
    static LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

    println!("DEBUG: Initializing logging");
    let _log_guard = if !LOGGING_INITIALIZED.swap(true, Ordering::SeqCst) {
        Some(logging::init_logging(
            LoggingConfig {
                level: config
                    .log_level
                    .as_deref()
                    .and_then(|s| match s.to_uppercase().parse::<Level>() {
                        Ok(l) => Some(l),
                        Err(_) => {
                            warn!("Invalid log level string: '{s}'. Defaulting to INFO.");
                            None
                        }
                    })
                    .unwrap_or(Level::INFO),
                json_format: false,
                log_dir: config.log_dir.clone(),
                colorize: true,
                log_file_name: "lmdeploy-router".to_string(),
                log_targets: None,
            },
            config.trace_config.clone(),
        ))
    } else {
        if config.trace_config.is_some() && !crate::otel_trace::is_otel_enabled() {
            warn!(
                "Tracing was requested after logging was already initialized; \
                 the existing subscriber will continue without enabling OpenTelemetry"
            );
        }
        None
    };
    println!("DEBUG: Logging initialized");

    // Initialize prometheus metrics exporter
    println!("DEBUG: Initializing Prometheus metrics");
    if let Some(prometheus_config) = config.prometheus_config {
        metrics::start_prometheus(prometheus_config);
    }
    println!("DEBUG: Prometheus metrics initialized");

    info!(
        "Starting router on {}:{} | mode: {:?} | policy: {:?} | kv_connector: {:?} | max_payload: {}MB",
        config.host,
        config.port,
        config.router_config.mode,
        config.router_config.policy,
        config.router_config.kv_connector,
        config.max_payload_size / (1024 * 1024)
    );

    println!("DEBUG: Creating HTTP client");
    let client = Client::builder()
        .pool_idle_timeout(Some(Duration::from_secs(50)))
        .pool_max_idle_per_host(500)
        .timeout(Duration::from_secs(config.request_timeout_secs))
        .connect_timeout(Duration::from_secs(10))
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .build()
        .expect("Failed to create HTTP client");
    println!("DEBUG: HTTP client created");

    // Create the application context with all dependencies
    println!("DEBUG: Creating AppContext");
    let app_context = AppContext::new(
        config.router_config.clone(),
        client.clone(),
        config.router_config.max_concurrent_requests,
        config.router_config.rate_limit_tokens_per_second,
        config.router_config.api_key_validation_urls.clone(),
    )?;
    println!("DEBUG: AppContext created");

    let app_context = Arc::new(app_context);

    // Create the appropriate router based on enable_igw flag
    let (router, router_manager): (Arc<dyn RouterTrait>, Option<Arc<RouterManager>>) =
        if config.router_config.enable_igw {
            info!("Multi-router mode enabled (enable_igw=true)");

            // Create RouterManager with shared registries from AppContext
            let router_manager = Arc::new(RouterManager::new(
                config.router_config.clone(),
                client.clone(),
                app_context.worker_registry.clone(),
                app_context.policy_registry.clone(),
            ));

            // 1. HTTP Regular Router
            match RouterFactory::create_regular_router(
                &[], // Empty worker list - workers added later
                &app_context,
            )
            .await
            {
                Ok(http_regular) => {
                    info!("Created HTTP Regular router");
                    router_manager.register_router(
                        RouterId::new("http-regular".to_string()),
                        Arc::from(http_regular),
                    );
                }
                Err(e) => {
                    warn!("Failed to create HTTP Regular router: {e}");
                }
            }

            // 2. HTTP LMDeploy PD Router
            match RouterFactory::create_lmdeploy_pd_router(LMDeployPDRouterParams {
                prefill_urls: &[],
                decode_urls: &[],
                migration_protocol: LMDeployMigrationProtocol::default(),
                rdma_config: None,
                dummy_prefill: false,
                prefill_policy_config: None,
                decode_policy_config: None,
                main_policy_config: &config.router_config.policy,
                ctx: &app_context,
            })
            .await
            {
                Ok(http_pd) => {
                    info!("Created HTTP LMDeploy PD router");
                    router_manager
                        .register_router(RouterId::new("http-pd".to_string()), Arc::from(http_pd));
                }
                Err(e) => {
                    warn!("Failed to create HTTP PD router: {e}");
                }
            }

            info!(
                "RouterManager initialized with {} routers",
                router_manager.router_count()
            );
            (
                router_manager.clone() as Arc<dyn RouterTrait>,
                Some(router_manager),
            )
        } else {
            info!("Single router mode (enable_igw=false)");
            // Create single router with the context
            (
                Arc::from(RouterFactory::create_router(&app_context).await?),
                None,
            )
        };

    // Start health checker for all workers in the registry
    let _health_checker = app_context
        .worker_registry
        .start_health_checker(config.router_config.health_check.check_interval_secs);
    info!(
        "Started health checker for workers with {}s interval",
        config.router_config.health_check.check_interval_secs
    );

    // Set up concurrency limiter with queue if configured
    let (limiter, processor) = middleware::ConcurrencyLimiter::new(
        app_context.rate_limiter.clone(),
        config.router_config.queue_size,
        Duration::from_secs(config.router_config.queue_timeout_secs),
    );

    // Start queue processor if enabled
    if let Some(processor) = processor {
        tokio::spawn(processor.run());
        info!(
            "Started request queue with size: {}, timeout: {}s",
            config.router_config.queue_size, config.router_config.queue_timeout_secs
        );
    }

    // Create app state with router and context
    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        concurrency_queue_tx: limiter.queue_tx.clone(),
        router_manager,
    });
    let router_arc = Arc::clone(&app_state.router);

    // Start the service discovery if enabled
    if let Some(service_discovery_config) = config.service_discovery_config {
        if service_discovery_config.enabled {
            match start_service_discovery(service_discovery_config, router_arc).await {
                Ok(handle) => {
                    info!("Service discovery started");
                    // Spawn a task to handle the service discovery thread
                    spawn(async move {
                        if let Err(e) = handle.await {
                            error!("Service discovery task failed: {:?}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to start service discovery: {e}");
                    warn!("Continuing without service discovery");
                }
            }
        }
    }

    info!(
        "Router ready | workers: {:?}",
        app_state.router.get_worker_urls()
    );

    let request_id_headers = config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    // Build the application
    // Enable transparent proxy for all routing modes
    let enable_transparent_proxy = true;

    let app = build_app_with_request_tracing(
        app_state,
        config.max_payload_size,
        request_id_headers,
        config.router_config.cors_allowed_origins.clone(),
        enable_transparent_proxy,
        crate::otel_trace::is_otel_enabled(),
    );

    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr).await?;
    info!("Starting server on {}", addr);
    serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    Ok(())
}

// Graceful shutdown handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        },
        _ = terminate => {
            info!("Received terminate signal, starting graceful shutdown");
        },
    }
}

// CORS Layer Creation
fn create_cors_layer(allowed_origins: Vec<String>) -> tower_http::cors::CorsLayer {
    use tower_http::cors::Any;

    let cors = if allowed_origins.is_empty() {
        // Allow all origins if none specified
        tower_http::cors::CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .expose_headers(Any)
    } else {
        // Restrict to specific origins
        let origins: Vec<http::HeaderValue> = allowed_origins
            .into_iter()
            .filter_map(|origin| origin.parse().ok())
            .collect();

        tower_http::cors::CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([http::Method::GET, http::Method::POST, http::Method::OPTIONS])
            .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
            .expose_headers([http::header::HeaderName::from_static("x-request-id")])
    };

    cors.max_age(Duration::from_secs(3600))
}
