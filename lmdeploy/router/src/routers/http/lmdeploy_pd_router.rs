// lmdeploy PD (Prefill-Decode) Router Implementation
// Extends PdRouterBase to handle lmdeploy-specific two-stage processing.
//
// LMDeploy PD protocol differs from the legacy PD protocol:
// - Prefill request sets with_cache=true, preserve_cache=true, max_tokens=1
// - Prefill response carries top-level id, cache_block_ids, remote_token_ids
// - P2P RDMA connection established via /distserve/p2p_initialize + /distserve/p2p_connect
// - Decode request carries migration_request field (not kv_transfer_params)
use super::pd_router::PdRouterBase;
use super::pd_types::{error_chain, PDRouterError};
use crate::config::{LMDeployMigrationProtocol, LMDeployRdmaConfig, LMDeployRdmaLinkType};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::otel_http::{self, ClientRequestOptions};
use crate::policies::PolicyRegistry;
use crate::routers::{header_utils, RouterTrait, WorkerManagement};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::OnceCell;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

type P2pKey = (String, String);
type P2pConnectionCell = Arc<OnceCell<()>>;

/// lmdeploy PD Router that extends PdRouterBase with lmdeploy-specific request handling
#[derive(Debug)]
pub struct LMDeployPDRouter {
    /// Underlying PD router for worker management, health, models
    pd_router: PdRouterBase,
    /// HTTP client for P2P setup calls (/distserve/*)
    http_client: reqwest::Client,
    /// Policy registry for load balancing
    policy_registry: Arc<PolicyRegistry>,
    /// KV migration protocol (rdma/nvlink)
    migration_protocol: LMDeployMigrationProtocol,
    /// Optional RDMA configuration
    rdma_config: Option<LMDeployRdmaConfig>,
    /// Whether to use dummy prefill
    dummy_prefill: bool,
    /// Per-pair, single-flight P2P connection initialization.
    p2p_pool: Arc<DashMap<P2pKey, P2pConnectionCell>>,
}

/// Releases a preserved prefill session if the request exits before a successful
/// decode response has completed. The asynchronous cleanup is spawned from Drop
/// so cancellation and streaming-client disconnects are covered as well.
#[derive(Debug)]
struct PrefillCacheGuard {
    client: reqwest::Client,
    prefill_base_url: String,
    session_id: i64,
    completed: Arc<AtomicBool>,
}

impl PrefillCacheGuard {
    fn new(client: reqwest::Client, prefill_base_url: String, session_id: i64) -> Self {
        Self {
            client,
            prefill_base_url,
            session_id,
            completed: Arc::new(AtomicBool::new(false)),
        }
    }

    fn mark_completed(&self) {
        self.completed.store(true, Ordering::Release);
    }
}

impl Drop for PrefillCacheGuard {
    fn drop(&mut self) {
        if self.completed.load(Ordering::Acquire) {
            return;
        }

        let client = self.client.clone();
        let prefill_base_url = self.prefill_base_url.clone();
        let session_id = self.session_id;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let url = format!("{}/distserve/free_cache", prefill_base_url);
                let result = client
                    .post(&url)
                    .json(&json!({
                        "remote_engine_id": prefill_base_url,
                        "remote_session_id": session_id,
                    }))
                    .send()
                    .await;
                match result {
                    Ok(response) if response.status().is_success() => {
                        debug!("Released preserved prefill session {}", session_id);
                    }
                    Ok(response) => {
                        warn!(
                            "Failed to release prefill session {}: status {}",
                            session_id,
                            response.status()
                        );
                    }
                    Err(error) => {
                        warn!(
                            "Failed to release prefill session {} at {}: {}",
                            session_id, url, error
                        );
                    }
                }
            });
        }
    }
}

impl LMDeployPDRouter {
    /// Create a new LMDeployPDRouter with static prefill/decode URLs
    pub async fn new(
        prefill_urls: Vec<String>,
        decode_urls: Vec<String>,
        migration_protocol: LMDeployMigrationProtocol,
        rdma_config: Option<LMDeployRdmaConfig>,
        dummy_prefill: bool,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        info!(
            "LMDeployPDRouter::new with {} prefill, {} decode, protocol={:?}, dummy_prefill={}",
            prefill_urls.len(),
            decode_urls.len(),
            migration_protocol,
            dummy_prefill
        );

        let pd_router = PdRouterBase::new(prefill_urls, decode_urls, ctx).await?;

        // Initialize policies with workers from registry
        let prefill_workers = pd_router.worker_registry.get_prefill_workers();
        let decode_workers = pd_router.worker_registry.get_decode_workers();
        let prefill_policy = ctx.policy_registry.get_prefill_policy();
        let decode_policy = ctx.policy_registry.get_decode_policy();
        if prefill_policy.requires_initialization() {
            info!("Initializing prefill policy with workers.");
            prefill_policy.init_workers(&prefill_workers);
        }
        if decode_policy.requires_initialization() {
            info!("Initializing decode policy with workers.");
            decode_policy.init_workers(&decode_workers);
        }

        let rdma_config = match migration_protocol {
            LMDeployMigrationProtocol::Rdma => Some(rdma_config.unwrap_or_default()),
            LMDeployMigrationProtocol::Nvlink => None,
        };

        Ok(Self {
            pd_router,
            http_client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(120))
                .build()
                .map_err(|e| format!("Failed to build LMDeploy control-plane client: {e}"))?,
            policy_registry: ctx.policy_registry.clone(),
            migration_protocol,
            rdma_config,
            dummy_prefill,
            p2p_pool: Arc::new(DashMap::new()),
        })
    }

    /// LMDeploy uses `enum.Enum` with `enum.auto()`. Pydantic's JSON mode
    /// therefore serializes the wire values as integers, not enum names.
    fn protocol_wire_value(&self) -> u8 {
        match self.migration_protocol {
            LMDeployMigrationProtocol::Rdma => 2,
            LMDeployMigrationProtocol::Nvlink => 3,
        }
    }

    fn rdma_config_json(&self) -> Value {
        match &self.rdma_config {
            Some(cfg) => json!({
                "with_gdr": cfg.with_gdr,
                "link_type": match cfg.link_type {
                    LMDeployRdmaLinkType::Ib => 1,
                    LMDeployRdmaLinkType::Roce => 2,
                },
            }),
            None => Value::Null,
        }
    }

    /// Prepare prefill request: force max_tokens=1, stream=false,
    /// with_cache=true, preserve_cache=true (lmdeploy PD-specific fields).
    fn prepare_prefill_request(mut request: Value, _path: &str) -> Value {
        request["max_tokens"] = json!(1);
        if request.get("max_completion_tokens").is_some() {
            request["max_completion_tokens"] = json!(1);
        }
        // lmdeploy uses min_new_tokens (not min_tokens)
        if let Some(min_new_tokens) = request.get("min_new_tokens").and_then(|v| v.as_u64()) {
            if min_new_tokens > 1 {
                request["min_new_tokens"] = json!(1);
            }
        }
        // Force non-streaming for prefill to get JSON response with cache metadata
        request["stream"] = json!(false);
        if let Some(obj) = request.as_object_mut() {
            obj.remove("stream_options");
            // Never forward client-supplied control-plane state to prefill.
            obj.remove("migration_request");
        }
        // lmdeploy PD-specific cache fields
        request["with_cache"] = json!(true);
        request["preserve_cache"] = json!(true);
        request
    }

    /// Build and strictly validate migration_request from a prefill response.
    fn build_migration_request(
        &self,
        prefill_base_url: &str,
        prefill_response: &Value,
    ) -> Result<Value, String> {
        Self::build_migration_request_static(
            prefill_base_url,
            prefill_response,
            self.protocol_wire_value(),
            self.dummy_prefill,
        )
    }

    /// Static helper to build migration_request from prefill response.
    /// Extracted for unit testing without requiring a full LMDeployPDRouter instance.
    fn build_migration_request_static(
        prefill_base_url: &str,
        prefill_response: &Value,
        protocol: u8,
        dummy_prefill: bool,
    ) -> Result<Value, String> {
        if dummy_prefill {
            return Ok(json!({
                "protocol": protocol,
                "remote_engine_id": "dummy:dummy",
                "remote_session_id": 0,
                "remote_token_id": 0,
                "remote_block_ids": [],
                "is_dummy_prefill": true,
            }));
        }

        let session_id = Self::extract_session_id(prefill_response)
            .ok_or_else(|| "Prefill response is missing a numeric remote session id".to_string())?;

        let remote_block_ids: Vec<i64> = prefill_response
            .get("cache_block_ids")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "Prefill response is missing cache_block_ids".to_string())?
            .iter()
            .map(|value| {
                value
                    .as_i64()
                    .ok_or_else(|| "cache_block_ids must contain only integers".to_string())
            })
            .collect::<Result<_, _>>()?;
        if remote_block_ids.is_empty() {
            return Err("Prefill response contains no cache_block_ids".to_string());
        }

        let remote_token_id: i64 = prefill_response
            .get("remote_token_ids")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.last())
            .and_then(|x| x.as_i64())
            .ok_or_else(|| {
                "Prefill response is missing a final integer remote_token_id".to_string()
            })?;

        Ok(json!({
            "protocol": protocol,
            "remote_engine_id": prefill_base_url,
            "remote_session_id": session_id,
            "remote_token_id": remote_token_id,
            "remote_block_ids": remote_block_ids,
            "is_dummy_prefill": dummy_prefill,
        }))
    }

    fn extract_session_id(prefill_response: &Value) -> Option<i64> {
        let parse_session_id = |value: &Value| {
            value
                .as_i64()
                .or_else(|| value.as_str().and_then(|text| text.parse::<i64>().ok()))
        };

        prefill_response
            .get("remote_session_id")
            .and_then(parse_session_id)
            .or_else(|| prefill_response.get("id").and_then(parse_session_id))
    }

    /// Ensure P2P RDMA connection between prefill and decode is established.
    /// Cached per (prefill_base_url, decode_base_url) pair.
    async fn ensure_p2p_connection(
        &self,
        prefill_base_url: &str,
        decode_base_url: &str,
    ) -> Result<(), String> {
        let key = (prefill_base_url.to_string(), decode_base_url.to_string());
        let cell = self
            .p2p_pool
            .entry(key.clone())
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();

        let result = cell
            .get_or_try_init(|| async {
                self.establish_p2p_connection(prefill_base_url, decode_base_url)
                    .await
            })
            .await
            .map(|_| ());

        if result.is_err() {
            // Remove only the cell that failed. Another request may already have
            // replaced it with a new initialization attempt.
            if let Entry::Occupied(entry) = self.p2p_pool.entry(key) {
                if Arc::ptr_eq(entry.get(), &cell) {
                    entry.remove();
                }
            }
        }
        result
    }

    async fn establish_p2p_connection(
        &self,
        prefill_base_url: &str,
        decode_base_url: &str,
    ) -> Result<(), String> {
        info!(
            "Establishing P2P connection: prefill={}, decode={}, protocol={:?}",
            prefill_base_url, decode_base_url, self.migration_protocol
        );

        let result = async {
            let (prefill_engine_info, decode_engine_info) = tokio::try_join!(
                self.fetch_engine_info(prefill_base_url),
                self.fetch_engine_info(decode_base_url),
            )?;

            let prefill_tp_size = Self::engine_tp_size(&prefill_engine_info, prefill_base_url)?;
            let decode_tp_size = Self::engine_tp_size(&decode_engine_info, decode_base_url)?;
            if prefill_tp_size != decode_tp_size {
                return Err(format!(
                    "LMDeploy PD peers must use the same tp_size: prefill={}, decode={}",
                    prefill_tp_size, decode_tp_size
                ));
            }

            let prefill_endpoints = self
                .p2p_initialize(
                    prefill_base_url,
                    prefill_engine_info.clone(),
                    decode_base_url,
                    decode_engine_info.clone(),
                )
                .await?;

            let decode_endpoints = self
                .p2p_initialize(
                    decode_base_url,
                    decode_engine_info,
                    prefill_base_url,
                    prefill_engine_info,
                )
                .await?;

            self.p2p_connect(decode_base_url, prefill_base_url, &prefill_endpoints)
                .await?;
            self.p2p_connect(prefill_base_url, decode_base_url, &decode_endpoints)
                .await?;
            Ok(())
        }
        .await;

        if result.is_err() {
            self.drop_p2p_connection(prefill_base_url, decode_base_url)
                .await;
        } else {
            info!(
                "P2P connection established successfully: prefill={}, decode={}",
                prefill_base_url, decode_base_url
            );
        }
        result
    }

    /// GET {base_url}/distserve/engine_info -> DistServeEngineConfig JSON
    async fn fetch_engine_info(&self, base_url: &str) -> Result<Value, String> {
        let url = format!("{}/distserve/engine_info", base_url);
        let resp = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("GET {} failed: {}", url, e))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("GET {} returned status {}: {}", url, status, body));
        }
        let value = resp
            .json::<Value>()
            .await
            .map_err(|e| format!("Failed to parse engine_info from {}: {}", url, e))?;
        Self::parse_engine_info_value(value)
    }

    fn parse_engine_info_value(value: Value) -> Result<Value, String> {
        let value = match value {
            Value::String(encoded) => serde_json::from_str(&encoded)
                .map_err(|e| format!("engine_info contained invalid nested JSON: {e}"))?,
            value => value,
        };
        if !value.is_object() {
            return Err("engine_info must decode to a JSON object".to_string());
        }
        Ok(value)
    }

    fn engine_tp_size(engine_info: &Value, base_url: &str) -> Result<u64, String> {
        engine_info
            .get("tp_size")
            .and_then(Value::as_u64)
            .filter(|size| *size > 0)
            .ok_or_else(|| format!("engine_info from {base_url} has no positive integer tp_size"))
    }

    /// POST {local_url}/distserve/p2p_initialize with DistServeInitRequest
    async fn p2p_initialize(
        &self,
        local_url: &str,
        local_engine_config: Value,
        remote_url: &str,
        remote_engine_config: Value,
    ) -> Result<Value, String> {
        let url = format!("{}/distserve/p2p_initialize", local_url);
        let body = json!({
            "local_engine_id": local_url,
            "local_engine_config": local_engine_config,
            "remote_engine_id": remote_url,
            "remote_engine_config": remote_engine_config,
            "protocol": self.protocol_wire_value(),
            "rdma_config": self.rdma_config_json(),
            "nvlink_config": if self.migration_protocol == LMDeployMigrationProtocol::Nvlink {
                json!({})
            } else {
                Value::Null
            },
        });
        let resp = self
            .http_client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("POST {} failed: {}", url, e))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("POST {} returned status {}: {}", url, status, body));
        }
        let value = resp.json::<Value>().await.map_err(|e| {
            format!(
                "Failed to parse p2p_initialize response from {}: {}",
                url, e
            )
        })?;
        Self::validate_connection_response(&value, true)?;
        Ok(value)
    }

    /// POST {local_url}/distserve/p2p_connect with DistServeConnectionRequest
    async fn p2p_connect(
        &self,
        local_url: &str,
        remote_url: &str,
        remote_endpoints: &Value,
    ) -> Result<(), String> {
        let url = format!("{}/distserve/p2p_connect", local_url);
        let remote_engine_endpoint_info = remote_endpoints
            .get("engine_endpoint_info")
            .cloned()
            .ok_or_else(|| "p2p_initialize response is missing engine_endpoint_info".to_string())?;
        let remote_kvtransfer_endpoint_info = remote_endpoints
            .get("kvtransfer_endpoint_info")
            .cloned()
            .filter(Value::is_array)
            .ok_or_else(|| {
                "p2p_initialize response is missing kvtransfer_endpoint_info".to_string()
            })?;
        let body = json!({
            "protocol": self.protocol_wire_value(),
            "remote_engine_id": remote_url,
            "remote_engine_endpoint_info": remote_engine_endpoint_info,
            "remote_kvtransfer_endpoint_info": remote_kvtransfer_endpoint_info,
        });
        let resp = self
            .http_client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("POST {} failed: {}", url, e))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("POST {} returned status {}: {}", url, status, body));
        }
        let value = resp
            .json::<Value>()
            .await
            .map_err(|e| format!("Failed to parse p2p_connect response from {}: {}", url, e))?;
        Self::validate_connection_response(&value, false)?;
        Ok(())
    }

    fn validate_connection_response(value: &Value, require_endpoints: bool) -> Result<(), String> {
        if value.get("status").and_then(Value::as_u64) != Some(1) {
            return Err(format!("LMDeploy connection operation failed: {value}"));
        }
        if require_endpoints
            && (value.get("engine_endpoint_info").is_none()
                || !value
                    .get("kvtransfer_endpoint_info")
                    .is_some_and(Value::is_array))
        {
            return Err(format!(
                "LMDeploy initialization response is missing endpoint information: {value}"
            ));
        }
        Ok(())
    }

    async fn drop_p2p_connection(&self, prefill_base_url: &str, decode_base_url: &str) {
        let requests = [
            (prefill_base_url, decode_base_url),
            (decode_base_url, prefill_base_url),
        ];
        for (local, remote) in requests {
            let url = format!("{}/distserve/p2p_drop_connect", local);
            if let Err(error) = self
                .http_client
                .post(&url)
                .json(&json!({
                    "engine_id": local,
                    "remote_engine_id": remote,
                }))
                .send()
                .await
            {
                debug!("Best-effort P2P drop failed for {}: {}", url, error);
            }
        }
    }

    async fn invalidate_p2p_connection(&self, prefill_base_url: &str, decode_base_url: &str) {
        self.p2p_pool
            .remove(&(prefill_base_url.to_string(), decode_base_url.to_string()));
        self.drop_p2p_connection(prefill_base_url, decode_base_url)
            .await;
    }

    fn invalidate_worker_connections(&self, worker_url: &str) {
        self.p2p_pool
            .retain(|key, _| key.0 != worker_url && key.1 != worker_url);
    }

    async fn invalidate_worker_connections_and_drop(&self, worker_url: &str) {
        let keys: Vec<P2pKey> = self
            .p2p_pool
            .iter()
            .filter(|entry| {
                let (prefill, decode) = entry.key();
                prefill == worker_url || decode == worker_url
            })
            .map(|entry| entry.key().clone())
            .collect();
        for (prefill, decode) in keys {
            self.invalidate_p2p_connection(&prefill, &decode).await;
        }
    }

    /// Process lmdeploy two-stage request: prefill -> decode
    async fn process_lmd_two_stage_request(
        &self,
        original_request: Value,
        prefill_worker: Option<Arc<dyn Worker>>,
        decode_worker: Arc<dyn Worker>,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<Response, PDRouterError> {
        let start_time = Instant::now();
        let request_id = format!("lmd-{}", Uuid::new_v4());
        let decode_base_url = decode_worker.url().to_string();
        let decode_url = decode_worker.endpoint_url(path);

        debug!(
            "LMD prefill={}, decode={}, path={}, dummy_prefill={}",
            prefill_worker
                .as_ref()
                .map(|worker| worker.url())
                .unwrap_or("dummy:dummy"),
            decode_worker.url(),
            path,
            self.dummy_prefill
        );

        let (prefill_base_url, migration_request, cache_guard) = if self.dummy_prefill {
            let migration_request = self
                .build_migration_request("dummy:dummy", &json!({}))
                .map_err(|message| PDRouterError::NetworkError { message })?;
            ("dummy:dummy".to_string(), migration_request, None)
        } else {
            let prefill_worker = prefill_worker.ok_or_else(|| PDRouterError::NetworkError {
                message: "LMDeploy PD request has no prefill worker".to_string(),
            })?;
            let prefill_base_url = prefill_worker.url().to_string();

            // Establish the transport before preserving a prefill cache. This avoids
            // retaining KV blocks when the control-plane handshake itself fails.
            if let Err(message) = self
                .ensure_p2p_connection(&prefill_base_url, &decode_base_url)
                .await
            {
                RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                return Err(PDRouterError::NetworkError {
                    message: format!(
                        "P2P connection setup failed for ({}, {}): {}",
                        prefill_base_url, decode_base_url, message
                    ),
                });
            }

            prefill_worker.increment_load();

            // Stage 1: Prefill
            let prefill_request = Self::prepare_prefill_request(original_request.clone(), path);
            let prefill_url = prefill_worker.endpoint_url(path);

            debug!("LMD Stage 1 - Prefill: {}", prefill_url);

            let prefill_request_builder = self
                .pd_router
                .client
                .post(&prefill_url)
                .header("Content-Type", "application/json")
                .header(
                    "Authorization",
                    format!(
                        "Bearer {}",
                        std::env::var("OPENAI_API_KEY").unwrap_or_default()
                    ),
                )
                .header("X-Request-Id", &request_id);

            let prefill_response = match otel_http::send_client_request(
                prefill_request_builder.json(&prefill_request),
                headers,
                ClientRequestOptions {
                    method: "POST",
                    url: &prefill_url,
                    route: Some(path),
                    request_phase: Some("prefill"),
                },
            )
            .await
            {
                Ok(response) => response,
                Err(error) => {
                    prefill_worker.decrement_load();
                    RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                    return Err(PDRouterError::NetworkError {
                        message: format!(
                            "Prefill request failed to {}: {}",
                            prefill_url,
                            error_chain(&error)
                        ),
                    });
                }
            };

            let prefill_status = prefill_response.status();
            let prefill_bytes = match prefill_response.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => {
                    prefill_worker.decrement_load();
                    RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                    return Err(PDRouterError::NetworkError {
                        message: format!(
                            "Failed to read prefill response from {}: {}",
                            prefill_url,
                            error_chain(&error)
                        ),
                    });
                }
            };
            prefill_worker.decrement_load();

            if !prefill_status.is_success() {
                RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                return Err(PDRouterError::NetworkError {
                    message: format!(
                        "Prefill server {} returned {}: {}",
                        prefill_url,
                        prefill_status,
                        String::from_utf8_lossy(&prefill_bytes)
                    ),
                });
            }

            let prefill_response_json: Value =
                serde_json::from_slice(&prefill_bytes).map_err(|e| {
                    RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                    PDRouterError::NetworkError {
                        message: format!("Failed to parse prefill response as JSON: {e}"),
                    }
                })?;

            let session_id = Self::extract_session_id(&prefill_response_json).ok_or_else(|| {
                RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                PDRouterError::NetworkError {
                    message: "Prefill response is missing a numeric id".to_string(),
                }
            })?;
            let cache_guard = PrefillCacheGuard::new(
                self.http_client.clone(),
                prefill_base_url.clone(),
                session_id,
            );
            let migration_request = self
                .build_migration_request(&prefill_base_url, &prefill_response_json)
                .map_err(|message| {
                    RouterMetrics::record_pd_prefill_error(&prefill_base_url);
                    PDRouterError::NetworkError { message }
                })?;

            (prefill_base_url, migration_request, Some(cache_guard))
        };

        // Stage 2: Decode
        decode_worker.increment_load();
        let mut decode_request = original_request.clone();
        if let Some(object) = decode_request.as_object_mut() {
            object.remove("with_cache");
            object.remove("preserve_cache");
            object.remove("migration_request");
            object.insert("migration_request".to_string(), migration_request);
        }

        debug!("LMD Stage 2 - Decode: {}", decode_url);

        let decode_request_builder = self
            .pd_router
            .client
            .post(&decode_url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!(
                    "Bearer {}",
                    std::env::var("OPENAI_API_KEY").unwrap_or_default()
                ),
            )
            .header("X-Request-Id", &request_id);

        let decode_response = match otel_http::send_client_request(
            decode_request_builder.json(&decode_request),
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &decode_url,
                route: Some(path),
                request_phase: Some("decode"),
            },
        )
        .await
        {
            Ok(resp) => resp,
            Err(e) => {
                decode_worker.decrement_load();
                if !self.dummy_prefill {
                    self.invalidate_p2p_connection(&prefill_base_url, &decode_base_url)
                        .await;
                }
                let full_error = error_chain(&e);
                let duration = start_time.elapsed();
                RouterMetrics::record_pd_decode_error(&decode_base_url);
                RouterMetrics::record_pd_request(path);
                RouterMetrics::record_pd_request_duration(path, duration);
                RouterMetrics::record_pd_prefill_request(&prefill_base_url);
                return Err(PDRouterError::NetworkError {
                    message: format!("Decode request failed to {}: {}", decode_url, full_error),
                });
            }
        };

        decode_worker.decrement_load();
        let status = decode_response.status();

        if !status.is_success() && !self.dummy_prefill {
            self.invalidate_p2p_connection(&prefill_base_url, &decode_base_url)
                .await;
        }

        let duration = start_time.elapsed();
        RouterMetrics::record_pd_request(path);
        RouterMetrics::record_pd_request_duration(path, duration);
        RouterMetrics::record_pd_prefill_request(&prefill_base_url);
        RouterMetrics::record_pd_decode_request(&decode_base_url);

        if !status.is_success() {
            RouterMetrics::record_pd_decode_error(&decode_base_url);
        }

        // Determine streaming vs full body (no logprobs merge for lmdeploy v1)
        let is_streaming = original_request
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if is_streaming {
            let mut response_builder = axum::http::Response::builder().status(status);
            let mut decode_headers =
                header_utils::preserve_response_headers(decode_response.headers());
            decode_headers.remove(axum::http::header::CONTENT_LENGTH);
            for (name, value) in decode_headers.iter() {
                response_builder = response_builder.header(name, value);
            }
            let stream = Box::pin(decode_response.bytes_stream());
            let body = match (status.is_success(), cache_guard) {
                (true, Some(guard)) => {
                    let guarded_stream = futures_util::stream::unfold(
                        (stream, guard),
                        |(mut stream, guard)| async move {
                            match stream.next().await {
                                Some(item) => {
                                    // LMDeploy sends the prefill cache-free message over
                                    // ZMQ immediately after migration and before yielding
                                    // the first decode output. Once a successful body chunk
                                    // arrives, HTTP fallback cleanup would be a double free.
                                    if item.is_ok() {
                                        guard.mark_completed();
                                    }
                                    Some((item, (stream, guard)))
                                }
                                // If no successful body chunk was observed, dropping the
                                // guard retains the fallback cleanup behavior.
                                None => None,
                            }
                        },
                    );
                    axum::body::Body::from_stream(guarded_stream)
                }
                (_, guard) => {
                    drop(guard);
                    axum::body::Body::from_stream(stream)
                }
            };
            response_builder
                .body(body)
                .map_err(|e| PDRouterError::NetworkError {
                    message: format!("Failed to build streaming response: {}", e),
                })
        } else {
            let decode_headers = decode_response.headers().clone();
            let body = decode_response
                .bytes()
                .await
                .map_err(|e| PDRouterError::NetworkError {
                    message: format!("Failed to read decode response: {}", e),
                })?;
            if status.is_success() {
                if let Some(guard) = cache_guard.as_ref() {
                    guard.mark_completed();
                }
            }
            drop(cache_guard);
            let mut response_builder = axum::http::Response::builder().status(status);
            for (name, value) in decode_headers.iter() {
                response_builder = response_builder.header(name, value);
            }
            response_builder
                .body(axum::body::Body::from(body))
                .map_err(|e| PDRouterError::NetworkError {
                    message: format!("Failed to build response: {}", e),
                })
        }
    }

    /// Process lmdeploy request: select PD pair via policy, then run two-stage
    async fn process_lmd_request(
        &self,
        headers: Option<&HeaderMap>,
        request_json: Value,
        path: &str,
        _model_id: Option<&str>,
    ) -> Response {
        let prefill_workers = self.pd_router.worker_registry.get_prefill_workers();
        let decode_workers = self.pd_router.worker_registry.get_decode_workers();

        if (!self.dummy_prefill && prefill_workers.is_empty()) || decode_workers.is_empty() {
            RouterMetrics::record_pd_error("server_selection");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "No workers available: {} prefill, {} decode",
                    prefill_workers.len(),
                    decode_workers.len()
                ),
            )
                .into_response();
        }

        let request_text = serde_json::to_string(&request_json).ok();
        let request_str = request_text.as_deref();
        let request_headers: Option<HashMap<String, String>> = headers.map(|h| {
            h.iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                })
                .collect()
        });

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        let prefill_worker = if self.dummy_prefill {
            None
        } else {
            let prefill_idx = match prefill_policy.select_worker_with_headers(
                &prefill_workers,
                request_str,
                request_headers.as_ref(),
            ) {
                Some(idx) => idx,
                None => {
                    RouterMetrics::record_pd_error("server_selection");
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Prefill policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };
            Some(prefill_workers[prefill_idx].clone())
        };

        let decode_idx = match decode_policy.select_worker_with_headers(
            &decode_workers,
            request_str,
            request_headers.as_ref(),
        ) {
            Some(idx) => idx,
            None => {
                RouterMetrics::record_pd_error("server_selection");
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Decode policy failed to select a worker".to_string(),
                )
                    .into_response();
            }
        };

        let decode_worker = decode_workers[decode_idx].clone();

        if let Some(worker) = prefill_worker.as_ref() {
            info!(
                "LMD routing: prefill={} [policy:{}], decode={} [policy:{}]",
                worker.url(),
                prefill_policy.name(),
                decode_worker.url(),
                decode_policy.name()
            );
        } else {
            info!(
                "LMD routing: dummy prefill, decode={} [policy:{}]",
                decode_worker.url(),
                decode_policy.name()
            );
        }

        match self
            .process_lmd_two_stage_request(
                request_json,
                prefill_worker,
                decode_worker,
                path,
                headers,
            )
            .await
        {
            Ok(response) => response,
            Err(e) => {
                error!("LMD two-stage processing failed: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request processing failed: {}", e),
                )
                    .into_response()
            }
        }
    }

    pub async fn add_prefill_server(&self, url: String) -> Result<String, PDRouterError> {
        self.pd_router.add_prefill_server(url).await
    }

    pub async fn add_decode_server(&self, url: String) -> Result<String, PDRouterError> {
        self.pd_router.add_decode_server(url).await
    }

    pub fn register_prefill_server_unchecked(&self, url: String) -> Result<String, PDRouterError> {
        self.pd_router.register_prefill_server_unchecked(url)
    }

    pub fn register_decode_server_unchecked(&self, url: String) -> Result<String, PDRouterError> {
        self.pd_router.register_decode_server_unchecked(url)
    }

    pub async fn remove_prefill_server(&self, url: &str) -> Result<String, PDRouterError> {
        self.invalidate_worker_connections_and_drop(url).await;
        self.pd_router.remove_prefill_server(url).await
    }

    pub async fn remove_decode_server(&self, url: &str) -> Result<String, PDRouterError> {
        self.invalidate_worker_connections_and_drop(url).await;
        self.pd_router.remove_decode_server(url).await
    }
}

#[async_trait]
impl RouterTrait for LMDeployPDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, req: Request<Body>) -> Response {
        self.pd_router.health(req).await
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // lmdeploy has no /health_generate endpoint
        (
            StatusCode::NOT_IMPLEMENTED,
            "health_generate not supported for lmdeploy PD router",
        )
            .into_response()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // lmdeploy has no /get_server_info endpoint
        (
            StatusCode::NOT_IMPLEMENTED,
            "get_server_info not supported for lmdeploy PD router",
        )
            .into_response()
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.pd_router.get_models(req).await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.pd_router.get_model_info(req).await
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "LMDeploy /generate does not expose the cache migration fields required for PD",
        )
            .into_response()
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let request_json = match serde_json::to_value(body) {
            Ok(json) => json,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
                    .into_response()
            }
        };
        self.process_lmd_request(headers, request_json, "/v1/chat/completions", model_id)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let request_json = match serde_json::to_value(body) {
            Ok(json) => json,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
                    .into_response()
            }
        };
        self.process_lmd_request(headers, request_json, "/v1/completions", model_id)
            .await
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "LMDeploy /v1/responses does not expose the cache migration fields required for PD",
        )
            .into_response()
    }

    async fn get_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses retrieve not supported for lmdeploy PD router",
        )
            .into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses cancel not supported for lmdeploy PD router",
        )
            .into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Embeddings not supported for lmdeploy PD router",
        )
            .into_response()
    }

    async fn flush_cache(&self) -> Response {
        // lmdeploy has POST /terminate but requires --allow-terminate-by-client
        (
            StatusCode::NOT_IMPLEMENTED,
            "flush_cache not supported for lmdeploy PD router",
        )
            .into_response()
    }

    async fn get_worker_loads(&self) -> Response {
        self.pd_router.get_worker_loads().await
    }

    fn router_type(&self) -> &'static str {
        "lmdeploy-pd"
    }

    fn is_pd_mode(&self) -> bool {
        true
    }

    fn readiness(&self) -> Response {
        self.pd_router.readiness()
    }

    async fn route_transparent(
        &self,
        headers: Option<&HeaderMap>,
        path: &str,
        method: &Method,
        body: serde_json::Value,
    ) -> Response {
        if *method != Method::POST {
            return (
                StatusCode::METHOD_NOT_ALLOWED,
                "Only POST requests are supported for transparent proxy",
            )
                .into_response();
        }
        if !matches!(path, "/v1/chat/completions" | "/v1/completions") {
            return (
                StatusCode::NOT_IMPLEMENTED,
                format!("Endpoint {path} is not supported by the LMDeploy PD protocol"),
            )
                .into_response();
        }
        self.process_lmd_request(headers, body, path, None).await
    }
}

#[async_trait]
impl WorkerManagement for LMDeployPDRouter {
    async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        self.pd_router.add_worker(worker_url).await
    }

    fn remove_worker(&self, worker_url: &str) {
        self.invalidate_worker_connections(worker_url);
        self.pd_router.remove_worker(worker_url);
    }

    fn get_worker_urls(&self) -> Vec<String> {
        self.pd_router.get_worker_urls()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_prefill_request_sets_max_tokens_1() {
        let request = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
            "stream": true
        });
        let result = LMDeployPDRouter::prepare_prefill_request(request, "/v1/chat/completions");
        assert_eq!(result["max_tokens"], 1);
        assert_eq!(result["stream"], false);
        assert_eq!(result["with_cache"], true);
        assert_eq!(result["preserve_cache"], true);
    }

    #[test]
    fn test_prepare_prefill_request_sets_max_completion_tokens_1() {
        let request = json!({
            "model": "test",
            "max_tokens": 512,
            "max_completion_tokens": 256
        });
        let result = LMDeployPDRouter::prepare_prefill_request(request, "/v1/chat/completions");
        assert_eq!(result["max_tokens"], 1);
        assert_eq!(result["max_completion_tokens"], 1);
    }

    #[test]
    fn test_prepare_prefill_request_clamps_min_new_tokens() {
        let request = json!({
            "model": "test",
            "max_tokens": 512,
            "min_new_tokens": 100
        });
        let result = LMDeployPDRouter::prepare_prefill_request(request, "/v1/chat/completions");
        assert_eq!(result["min_new_tokens"], 1);
    }

    #[test]
    fn test_prepare_prefill_request_removes_stream_options() {
        let request = json!({
            "model": "test",
            "max_tokens": 512,
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let result = LMDeployPDRouter::prepare_prefill_request(request, "/v1/chat/completions");
        assert_eq!(result["stream"], false);
        assert!(result.get("stream_options").is_none());
    }

    #[test]
    fn test_protocol_config_serialization() {
        assert_eq!(
            serde_json::to_value(LMDeployMigrationProtocol::Rdma).unwrap(),
            serde_json::json!("rdma")
        );
        assert_eq!(
            serde_json::to_value(LMDeployMigrationProtocol::Nvlink).unwrap(),
            serde_json::json!("nvlink")
        );
    }

    #[test]
    fn test_migration_protocol_default_is_rdma() {
        assert_eq!(
            LMDeployMigrationProtocol::default(),
            LMDeployMigrationProtocol::Rdma
        );
    }

    #[test]
    fn test_build_migration_request_extracts_fields() {
        let prefill_response = json!({
            "id": 12345,
            "cache_block_ids": [1, 2, 3],
            "remote_token_ids": [10, 20, 30]
        });
        let result = LMDeployPDRouter::build_migration_request_static(
            "http://prefill:23333",
            &prefill_response,
            2,
            false,
        );
        let req = result.unwrap();
        assert_eq!(req["protocol"], 2);
        assert_eq!(req["remote_engine_id"], "http://prefill:23333");
        assert_eq!(req["remote_session_id"], 12345);
        assert_eq!(req["remote_token_id"], 30);
        assert_eq!(req["remote_block_ids"], json!([1, 2, 3]));
        assert_eq!(req["is_dummy_prefill"], false);
    }

    #[test]
    fn test_build_migration_request_prefers_remote_session_id() {
        let prefill_response = json!({
            "id": "chatcmpl-not-a-number",
            "remote_session_id": 67890,
            "cache_block_ids": [5],
            "remote_token_ids": [42]
        });
        let result = LMDeployPDRouter::build_migration_request_static(
            "http://p:8000",
            &prefill_response,
            2,
            false,
        );
        let req = result.unwrap();
        assert_eq!(req["remote_session_id"], 67890);
    }

    #[test]
    fn test_build_migration_request_parses_string_id() {
        let prefill_response = json!({
            "id": "67890",
            "cache_block_ids": [5],
            "remote_token_ids": [42]
        });
        let result = LMDeployPDRouter::build_migration_request_static(
            "http://p:8000",
            &prefill_response,
            2,
            false,
        );
        let req = result.unwrap();
        assert_eq!(req["remote_session_id"], 67890);
        assert_eq!(req["remote_token_id"], 42);
        assert_eq!(req["remote_block_ids"], json!([5]));
        assert_eq!(req["is_dummy_prefill"], false);
    }

    #[test]
    fn test_build_migration_request_rejects_missing_id() {
        let prefill_response = json!({
            "cache_block_ids": [1, 2],
            "remote_token_ids": [10]
        });
        let result = LMDeployPDRouter::build_migration_request_static(
            "http://p:8000",
            &prefill_response,
            2,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_migration_request_rejects_non_numeric_id() {
        let prefill_response = json!({
            "id": "not-a-number",
            "cache_block_ids": [1],
            "remote_token_ids": [10]
        });
        let result = LMDeployPDRouter::build_migration_request_static(
            "http://p:8000",
            &prefill_response,
            2,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_migration_request_rejects_missing_cache_block_ids() {
        let prefill_response = json!({
            "id": 100,
            "remote_token_ids": [7]
        });
        let result = LMDeployPDRouter::build_migration_request_static(
            "http://p:8000",
            &prefill_response,
            2,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_migration_request_rejects_empty_remote_token_ids() {
        let prefill_response = json!({
            "id": 200,
            "cache_block_ids": [1, 2],
            "remote_token_ids": []
        });
        let result = LMDeployPDRouter::build_migration_request_static(
            "http://p:8000",
            &prefill_response,
            2,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_dummy_migration_request_skips_prefill_metadata() {
        let request =
            LMDeployPDRouter::build_migration_request_static("unused", &json!({}), 2, true)
                .unwrap();
        assert_eq!(request["protocol"], 2);
        assert_eq!(request["remote_engine_id"], "dummy:dummy");
        assert_eq!(request["remote_session_id"], 0);
        assert_eq!(request["remote_block_ids"], json!([]));
        assert_eq!(request["is_dummy_prefill"], true);
    }

    #[test]
    fn test_engine_info_nested_json_is_decoded() {
        let value = Value::String(r#"{"tp_size":8,"dp_size":1}"#.to_string());
        let decoded = LMDeployPDRouter::parse_engine_info_value(value).unwrap();
        assert_eq!(decoded["tp_size"], 8);
        assert_eq!(decoded["dp_size"], 1);
    }

    #[test]
    fn test_connection_response_requires_success_and_endpoints() {
        let response = json!({
            "status": 1,
            "engine_endpoint_info": {"zmq_address": "tcp://host:1234"},
            "kvtransfer_endpoint_info": [],
        });
        assert!(LMDeployPDRouter::validate_connection_response(&response, true).is_ok());
        assert!(
            LMDeployPDRouter::validate_connection_response(&json!({"status": 2}), false).is_err()
        );
    }

    #[test]
    fn test_prepare_prefill_request_sets_with_cache_and_preserve_cache() {
        let request = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
        });
        let result = LMDeployPDRouter::prepare_prefill_request(request, "/v1/chat/completions");
        assert_eq!(result["with_cache"], true);
        assert_eq!(result["preserve_cache"], true);
        assert_eq!(result["max_tokens"], 1);
        assert_eq!(result["stream"], false);
    }

    #[test]
    fn test_prepare_prefill_request_preserves_token_io_fields() {
        let request = json!({
            "model": "test-model",
            "messages": [],
            "input_ids": [151644, 8948, 198],
            "return_token_ids": true,
            "do_preprocess": false,
            "max_tokens": 32,
        });
        let result = LMDeployPDRouter::prepare_prefill_request(request, "/v1/chat/completions");
        assert_eq!(result["input_ids"], json!([151644, 8948, 198]));
        assert_eq!(result["return_token_ids"], true);
        assert_eq!(result["do_preprocess"], false);
        assert_eq!(result["max_tokens"], 1);
    }

    #[test]
    fn test_prepare_prefill_request_handles_generate_path() {
        let request = json!({
            "model": "test-model",
            "prompt": "hello",
            "max_tokens": 256,
            "stream": true,
        });
        let result = LMDeployPDRouter::prepare_prefill_request(request, "/generate");
        assert_eq!(result["max_tokens"], 1);
        assert_eq!(result["stream"], false);
        assert_eq!(result["with_cache"], true);
        assert_eq!(result["preserve_cache"], true);
    }
}
