// PD (Prefill-Decode) Router Implementation
// This module handles routing for disaggregated prefill-decode systems
use super::pd_types::PDRouterError;
use crate::core::{
    BasicWorker, CircuitBreakerConfig, HealthConfig, Worker, WorkerFactory, WorkerRegistry,
    WorkerType,
};
use crate::policies::{LoadBalancingPolicy, PolicyRegistry};
use crate::routers::header_utils;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone)]
pub struct PdRouterBase {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub worker_startup_timeout_secs: u64,
    pub worker_startup_check_interval_secs: u64,
    pub worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    pub load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    pub client: Client,
    pub circuit_breaker_config: CircuitBreakerConfig,
}

impl PdRouterBase {
    // Private helper method to perform health check on a new server
    async fn wait_for_server_health(&self, url: &str) -> Result<(), PDRouterError> {
        crate::routers::http::router::Router::wait_for_healthy_workers(
            &[url.to_string()],
            self.worker_startup_timeout_secs,
            self.worker_startup_check_interval_secs,
        )
        .await
        .map_err(|_| PDRouterError::HealthCheckFailed {
            url: url.to_string(),
        })
    }

    // Generic helper for processing all workers with an endpoint
    async fn process_workers(
        &self,
        worker_type_enum: WorkerType,
        worker_type: &str,
        endpoint: &str,
    ) -> (Vec<String>, Vec<String>) {
        let mut results = Vec::new();
        let mut errors = Vec::new();

        // Get workers from registry based on type
        let workers = self.worker_registry.get_by_type(&worker_type_enum);
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        // Process each worker
        for worker_url in urls {
            let url = format!("{}/{}", worker_url, endpoint);
            match self.client.post(&url).send().await {
                Ok(res) if res.status().is_success() => {
                    results.push(format!("{} {}: OK", worker_type, worker_url));
                }
                Ok(res) => {
                    errors.push(format!(
                        "{} {} returned status: {}",
                        worker_type,
                        worker_url,
                        res.status()
                    ));
                }
                Err(e) => {
                    errors.push(format!("{} {} error: {}", worker_type, worker_url, e));
                }
            }
        }

        (results, errors)
    }

    // Helper to get prefill worker URLs
    fn get_prefill_worker_urls(&self) -> Vec<String> {
        self.worker_registry
            .get_prefill_workers()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    // Helper to get decode worker URLs
    fn get_decode_worker_urls(&self) -> Vec<String> {
        self.worker_registry
            .get_decode_workers()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    /// Start profiling on a backend server
    pub async fn start_profiling(&self, worker_url: &str) {
        let url = format!("{}/start_profile", worker_url);
        match self.client.post(&url).send().await {
            Ok(res) if res.status().is_success() => {
                info!("Started profiling on {}", worker_url);
            }
            Ok(res) => {
                warn!(
                    "Failed to start profiling on {}: status {}",
                    worker_url,
                    res.status()
                );
            }
            Err(e) => {
                warn!("Error starting profiling on {}: {}", worker_url, e);
            }
        }
    }

    /// Stop profiling on a backend server
    pub async fn stop_profiling(&self, worker_url: &str) {
        let url = format!("{}/stop_profile", worker_url);
        match self.client.post(&url).send().await {
            Ok(res) if res.status().is_success() => {
                info!("Stopped profiling on {}", worker_url);
            }
            Ok(res) => {
                warn!(
                    "Failed to stop profiling on {}: status {}",
                    worker_url,
                    res.status()
                );
            }
            Err(e) => {
                warn!("Error stopping profiling on {}: {}", worker_url, e);
            }
        }
    }

    // Helper for proxying requests to the first prefill worker
    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No prefill servers available".to_string(),
            )
                .into_response()
        }
    }

    // Generic helper for proxying to a specific worker
    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = format!("{}/{}", worker_url, endpoint);
        let mut request_builder = self.client.get(&url);

        // Add headers if provided
        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(axum::body::Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response()
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                (status, format!("{} server returned status: ", res.status())).into_response()
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to proxy request: {}", e),
                )
                    .into_response()
            }
        }
    }

    pub async fn add_prefill_server(&self, url: String) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        self.wait_for_server_health(&url).await?;

        self.register_prefill_server_unchecked(url)
    }

    /// Register a prefill server without probing it first. This is used by
    /// LMDeploy's startup-time `/nodes/add` callback, before the API server is
    /// able to answer `/health`.
    pub fn register_prefill_server_unchecked(&self, url: String) -> Result<String, PDRouterError> {
        if self.worker_registry.get_by_url(&url).is_some() {
            return Err(PDRouterError::WorkerAlreadyExists { url: url.clone() });
        }
        let worker_arc: Arc<dyn Worker> = Arc::from(WorkerFactory::create_prefill_with_config(
            url.clone(),
            self.circuit_breaker_config.clone(),
        ));
        self.register_and_notify(worker_arc);
        info!("Added prefill server: {}", url);

        Ok(format!("Successfully added prefill server: {}", url))
    }

    pub async fn add_decode_server(&self, url: String) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        self.wait_for_server_health(&url).await?;

        self.register_decode_server_unchecked(url)
    }

    /// Register a decode server without probing it first. See
    /// `register_prefill_server_unchecked` for why startup-time registration
    /// must not synchronously perform a health check.
    pub fn register_decode_server_unchecked(&self, url: String) -> Result<String, PDRouterError> {
        if self.worker_registry.get_by_url(&url).is_some() {
            return Err(PDRouterError::WorkerAlreadyExists { url: url.clone() });
        }
        let worker_arc: Arc<dyn Worker> = Arc::from(WorkerFactory::create_decode_with_config(
            url.clone(),
            self.circuit_breaker_config.clone(),
        ));
        self.register_and_notify(worker_arc);
        info!("Added decode server: {}", url);

        Ok(format!("Successfully added decode server: {}", url))
    }

    /// Register a worker and notify the policy registry.
    fn register_and_notify(&self, worker_arc: Arc<dyn Worker>) {
        self.worker_registry.register(worker_arc.clone());
        let model_id = worker_arc.model_id();
        let policy = self.policy_registry.on_worker_added(model_id, None);
        if policy.name() == "cache_aware" {
            if let Some(cache_aware) = policy
                .as_any()
                .downcast_ref::<crate::policies::CacheAwarePolicy>()
            {
                let model_workers = self.worker_registry.get_by_model_fast(model_id);
                cache_aware.init_workers(&model_workers);
            }
        }
    }

    pub async fn remove_prefill_server(&self, url: &str) -> Result<String, PDRouterError> {
        let model_id = match self.worker_registry.get_by_url(url) {
            Some(worker) => worker.model_id().to_string(),
            None => {
                return Err(PDRouterError::WorkerNotFound {
                    url: url.to_string(),
                });
            }
        };

        let removed = self.worker_registry.remove_by_url(url);

        if removed.is_some() {
            self.policy_registry.on_worker_removed(&model_id);
            if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    if let Some(cache_aware) = policy
                        .as_any()
                        .downcast_ref::<crate::policies::CacheAwarePolicy>()
                    {
                        cache_aware.remove_worker_by_url(url);
                    }
                }
            }
        }

        if removed.is_some() {
            info!("Removed prefill server: {}", url);
            Ok(format!("Successfully removed prefill server: {}", url))
        } else {
            Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            })
        }
    }

    pub async fn remove_decode_server(&self, url: &str) -> Result<String, PDRouterError> {
        let model_id = match self.worker_registry.get_by_url(url) {
            Some(worker) => worker.model_id().to_string(),
            None => {
                return Err(PDRouterError::WorkerNotFound {
                    url: url.to_string(),
                });
            }
        };

        let removed = self.worker_registry.remove_by_url(url);

        if removed.is_some() {
            // Notify PolicyRegistry about the removed worker
            self.policy_registry.on_worker_removed(&model_id);

            // Get the policy for this model to update cache-aware if needed
            if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    if let Some(cache_aware) = policy
                        .as_any()
                        .downcast_ref::<crate::policies::CacheAwarePolicy>()
                    {
                        cache_aware.remove_worker_by_url(url);
                    }
                }
            }
        }

        if removed.is_some() {
            info!("Removed decode server: {}", url);
            Ok(format!("Successfully removed decode server: {}", url))
        } else {
            Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        prefill_urls: Vec<String>,
        decode_urls: Vec<String>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        let mut prefill_workers_urls = vec![];
        let mut decode_workers_urls = vec![];
        let health_config = HealthConfig {
            timeout_secs: ctx.router_config.health_check.timeout_secs,
            check_interval_secs: ctx.router_config.health_check.check_interval_secs,
            endpoint: ctx.router_config.health_check.endpoint.clone(),
            failure_threshold: ctx.router_config.health_check.failure_threshold,
            success_threshold: ctx.router_config.health_check.success_threshold,
        };

        // Register prefill workers in the registry
        for url in prefill_urls {
            prefill_workers_urls.push(url.clone());
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorker::new(url, WorkerType::Prefill)
                    .with_circuit_breaker_config(core_cb_config.clone())
                    .with_health_config(health_config.clone()),
            );
            ctx.worker_registry.register(worker);
        }

        // Register decode workers in the registry
        for url in decode_urls {
            decode_workers_urls.push(url.clone());
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorker::new(url, WorkerType::Decode)
                    .with_circuit_breaker_config(core_cb_config.clone())
                    .with_health_config(health_config.clone()),
            );
            ctx.worker_registry.register(worker);
        }

        // Get all workers from registry for health check
        let all_workers = ctx.worker_registry.get_all();
        let all_urls: Vec<String> = all_workers
            .iter()
            .map(|worker| worker.url().to_string())
            .collect();
        // At least one prefill and one decode are up
        if !prefill_workers_urls.is_empty() {
            crate::routers::http::router::Router::wait_for_healthy_workers(
                &prefill_workers_urls,
                ctx.router_config.worker_startup_timeout_secs,
                ctx.router_config.worker_startup_check_interval_secs,
            )
            .await?;
        }

        if !decode_workers_urls.is_empty() {
            crate::routers::http::router::Router::wait_for_healthy_workers(
                &decode_workers_urls,
                ctx.router_config.worker_startup_timeout_secs,
                ctx.router_config.worker_startup_check_interval_secs,
            )
            .await?;
        }

        // Initialize cache-aware policies with workers from registry
        // Note: We need to get workers by type and convert to Box<dyn Worker> for CacheAwarePolicy
        // This is a temporary workaround until CacheAwarePolicy is updated to work with Arc<dyn Worker>
        // TODO: Update CacheAwarePolicy to accept Arc<dyn Worker> instead of Box<dyn Worker>

        // Set up background load monitoring for power-of-two selection
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        // Get policies from registry to check if we need load monitoring
        let prefill_policy = ctx.policy_registry.get_prefill_policy();
        let decode_policy = ctx.policy_registry.get_decode_policy();

        let load_monitor_handle =
            if prefill_policy.name() == "power_of_two" || decode_policy.name() == "power_of_two" {
                let monitor_urls = all_urls.clone();
                let monitor_interval = ctx.router_config.worker_startup_check_interval_secs;
                let monitor_client = ctx.client.clone();
                let prefill_policy_clone = Arc::clone(&prefill_policy);
                let decode_policy_clone = Arc::clone(&decode_policy);

                Some(Arc::new(tokio::spawn(async move {
                    Self::monitor_worker_loads_with_client(
                        monitor_urls,
                        tx,
                        monitor_interval,
                        monitor_client,
                        prefill_policy_clone,
                        decode_policy_clone,
                    )
                    .await;
                })))
            } else {
                None
            };

        // Note: Health checking is now handled centrally by RouterManager
        // Individual routers no longer need to manage health checkers

        Ok(PdRouterBase {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            worker_startup_timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: ctx
                .router_config
                .worker_startup_check_interval_secs,
            worker_loads,
            load_monitor_handle,
            client: ctx.client.clone(),
            circuit_breaker_config: core_cb_config,
        })
    }

    // Select a pair of prefill and decode servers considering circuit breaker state
    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        // Get workers from registry - filter by model if provided
        let prefill_workers = if let Some(model) = model_id {
            // Get model-specific workers and filter for prefill type
            self.worker_registry
                .get_by_model_fast(model)
                .into_iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill))
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = model_id {
            // Get model-specific workers and filter for decode type
            self.worker_registry
                .get_by_model_fast(model)
                .into_iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        // Select workers using helper function
        // Use separate policies for prefill and decode to avoid counter conflicts
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        let prefill = Self::pick_worker_by_policy_arc(
            &prefill_workers,
            &*prefill_policy,
            request_text,
            "prefill",
        )?;

        let decode = Self::pick_worker_by_policy_arc(
            &decode_workers,
            &*decode_policy,
            request_text,
            "decode",
        )?;

        Ok((prefill, decode))
    }

    // Helper function to select a worker using the policy (Arc version)
    fn pick_worker_by_policy_arc(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        // Check if we have any workers
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        // Filter available workers (healthy + circuit breaker not open)
        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        // Let policy select from available workers (no conversion needed now!)
        let selected_idx = policy
            .select_worker(&available_workers, request_text)
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        // Return the selected Arc worker
        Ok(available_workers[selected_idx].clone())
    }

    // Background task to monitor worker loads with shared client
    async fn monitor_worker_loads_with_client(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
        client: Client,
        prefill_policy: Arc<dyn LoadBalancingPolicy>,
        decode_policy: Arc<dyn LoadBalancingPolicy>,
    ) {
        loop {
            let mut loads = HashMap::new();

            let futures: Vec<_> = worker_urls
                .iter()
                .map(|url| {
                    let client = client.clone();
                    let url = url.clone();
                    async move {
                        let load = get_worker_load(&client, &url).await.unwrap_or(0);
                        (url, load)
                    }
                })
                .collect();

            let results = futures_util::future::join_all(futures).await;

            for (url, load) in results {
                loads.insert(url, load);
            }

            debug!("Worker loads updated: {:?}", loads);

            // Update both policies with current loads
            prefill_policy.update_loads(&loads);
            decode_policy.update_loads(&loads);

            // Check if receiver is still active
            if tx.send(loads).is_err() {
                info!("Load monitor receiver dropped, shutting down monitor task");
                break;
            }

            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        }
    }
}

// Helper functions

async fn get_worker_load(client: &Client, worker_url: &str) -> Option<isize> {
    match client.get(format!("{}/get_load", worker_url)).send().await {
        Ok(res) if res.status().is_success() => match res.bytes().await {
            Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(data) => data
                    .get("load")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as isize),
                Err(e) => {
                    debug!("Failed to parse load response from {}: {}", worker_url, e);
                    None
                }
            },
            Err(e) => {
                debug!("Failed to read load response from {}: {}", worker_url, e);
                None
            }
        },
        Ok(res) => {
            debug!(
                "Worker {} returned non-success status: {}",
                worker_url,
                res.status()
            );
            None
        }
        Err(e) => {
            debug!("Failed to get load from {}: {}", worker_url, e);
            None
        }
    }
}

impl PdRouterBase {
    pub async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        // For PD router, we don't support adding workers via this generic method
        Err(
            "PD router requires specific add_prefill_server or add_decode_server methods"
                .to_string(),
        )
    }

    pub fn remove_worker(&self, worker_url: &str) {
        // Remove from registry
        if let Some(worker) = self.worker_registry.remove_by_url(worker_url) {
            match worker.worker_type() {
                WorkerType::Prefill => {
                    info!("Removed prefill worker: {}", worker_url);
                }
                WorkerType::Decode => {
                    info!("Removed decode worker: {}", worker_url);
                }
                _ => {
                    info!("Removed worker: {}", worker_url);
                }
            }
        }
    }

    pub fn get_worker_urls(&self) -> Vec<String> {
        self.worker_registry.get_all_urls()
    }
}

impl PdRouterBase {
    pub async fn health(&self, _req: Request<Body>) -> Response {
        // This is a server readiness check - checking if we have healthy workers
        // Workers handle their own health checks in the background
        let mut all_healthy = true;
        let mut unhealthy_servers = Vec::new();

        // Check all workers
        for worker in self.worker_registry.get_all() {
            if !worker.is_healthy() {
                all_healthy = false;
                let worker_type = match worker.worker_type() {
                    WorkerType::Prefill => "Prefill",
                    WorkerType::Decode => "Decode",
                    _ => "Worker",
                };
                unhealthy_servers.push(format!("{}: {}", worker_type, worker.url()));
            }
        }

        if all_healthy {
            (StatusCode::OK, "All servers healthy").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Unhealthy servers: {:?}", unhealthy_servers),
            )
                .into_response()
        }
    }

    pub async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Test model generation capability by selecting a random pair and testing them
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode) = match self.select_pd_pair(None, None).await {
            Ok(pair) => pair,
            Err(e) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("No healthy worker pair available: {}", e),
                )
                    .into_response();
            }
        };

        let (prefill_result, decode_result) = tokio::join!(
            self.client
                .get(format!("{}/health_generate", prefill.url()))
                .send(),
            self.client
                .get(format!("{}/health_generate", decode.url()))
                .send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Health generate failed: {:?}", errors),
            )
                .into_response()
        }
    }

    pub async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("get_server_info", None)
            .await
    }

    pub async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    pub async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("get_model_info", Some(headers))
            .await
    }

    pub async fn get_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses retrieve endpoint not implemented for PD router",
        )
            .into_response()
    }

    pub async fn cancel_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses cancel endpoint not implemented for PD router",
        )
            .into_response()
    }

    pub async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Embeddings endpoint not implemented for PD router",
        )
            .into_response()
    }

    pub async fn flush_cache(&self) -> Response {
        // Process both prefill and decode workers
        let (prefill_results, prefill_errors) = self
            .process_workers(WorkerType::Prefill, "Prefill", "flush_cache")
            .await;
        let (decode_results, decode_errors) = self
            .process_workers(WorkerType::Decode, "Decode", "flush_cache")
            .await;

        // Combine results and errors
        let mut results = prefill_results;
        results.extend(decode_results);
        let mut errors = prefill_errors;
        errors.extend(decode_errors);

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!("Cache flushed successfully: {:?}", results),
            )
                .into_response()
        } else {
            (
                StatusCode::PARTIAL_CONTENT,
                format!(
                    "Partial success. Results: {:?}, Errors: {:?}",
                    results, errors
                ),
            )
                .into_response()
        }
    }

    pub async fn get_worker_loads(&self) -> Response {
        let mut loads = HashMap::new();
        let mut errors = Vec::new();

        // Process prefill workers
        let prefill_urls = self.get_prefill_worker_urls();
        for worker_url in prefill_urls {
            match get_worker_load(&self.client, &worker_url).await {
                Some(load) => {
                    loads.insert(format!("prefill_{}", worker_url), load);
                }
                None => {
                    errors.push(format!("Failed to get load from prefill {}", worker_url));
                }
            }
        }

        // Process decode workers
        let decode_urls = self.get_decode_worker_urls();
        for worker_url in decode_urls {
            match get_worker_load(&self.client, &worker_url).await {
                Some(load) => {
                    loads.insert(format!("decode_{}", worker_url), load);
                }
                None => {
                    errors.push(format!("Failed to get load from decode {}", worker_url));
                }
            }
        }

        let response_data = serde_json::json!({
            "loads": loads,
            "errors": errors
        });

        (StatusCode::OK, Json(response_data)).into_response()
    }

    pub fn readiness(&self) -> Response {
        // PD router is ready if it has at least one healthy prefill AND one healthy decode worker
        let prefill_workers = self.worker_registry.get_prefill_workers();
        let decode_workers = self.worker_registry.get_decode_workers();

        let healthy_prefill_count = prefill_workers.iter().filter(|w| w.is_healthy()).count();

        let healthy_decode_count = decode_workers.iter().filter(|w| w.is_healthy()).count();

        let total_prefill = prefill_workers.len();
        let total_decode = decode_workers.len();

        if healthy_prefill_count > 0 && healthy_decode_count > 0 {
            Json(json!({
                "status": "ready",
                "prefill": {
                    "healthy": healthy_prefill_count,
                    "total": total_prefill
                },
                "decode": {
                    "healthy": healthy_decode_count,
                    "total": total_decode
                }
            }))
            .into_response()
        } else {
            let mut reasons = Vec::new();
            if healthy_prefill_count == 0 {
                reasons.push("no healthy prefill workers");
            }
            if healthy_decode_count == 0 {
                reasons.push("no healthy decode workers");
            }

            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "not_ready",
                    "reason": reasons.join(", "),
                    "prefill": {
                        "healthy": healthy_prefill_count,
                        "total": total_prefill
                    },
                    "decode": {
                        "healthy": healthy_decode_count,
                        "total": total_decode
                    }
                })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, Worker, WorkerType};

    fn create_test_pd_router() -> PdRouterBase {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        PdRouterBase {
            worker_registry,
            policy_registry,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            client: Client::new(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorker::new(url, worker_type);
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    // ============= Worker Management Tests =============

    #[tokio::test]
    async fn test_add_prefill_server_already_exists() {
        let router = create_test_pd_router();

        // Add a worker first
        let worker = create_test_worker(
            "http://localhost:8000".to_string(),
            WorkerType::Prefill,
            true,
        );
        router.worker_registry.register(Arc::from(worker));

        // Try to add the same URL again - this would fail during health check in real scenario
        // For unit test, we test the duplicate check logic
        let exists = router
            .worker_registry
            .get_by_url("http://localhost:8000")
            .is_some();
        assert!(exists);
    }

    #[tokio::test]
    async fn test_remove_prefill_server_success() {
        let router = create_test_pd_router();

        // Add servers first
        let worker1 = create_test_worker("http://worker1".to_string(), WorkerType::Prefill, true);
        let worker2 = create_test_worker("http://worker2".to_string(), WorkerType::Prefill, true);

        router.worker_registry.register(Arc::from(worker1));
        router.worker_registry.register(Arc::from(worker2));

        // Remove one
        let result = router.remove_prefill_server("http://worker1").await;

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Successfully removed"));

        let workers = router.worker_registry.get_prefill_workers();
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].url(), "http://worker2");
    }

    #[tokio::test]
    async fn test_remove_prefill_server_not_found() {
        let router = create_test_pd_router();

        let result = router.remove_prefill_server("http://nonexistent").await;

        assert!(result.is_err());
        match result.unwrap_err() {
            PDRouterError::WorkerNotFound { url } => {
                assert_eq!(url, "http://nonexistent");
            }
            _ => panic!("Expected WorkerNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_remove_decode_server_success() {
        let router = create_test_pd_router();

        // Add server first
        let worker = create_test_worker("http://decode1".to_string(), WorkerType::Decode, true);
        router.worker_registry.register(Arc::from(worker));

        let result = router.remove_decode_server("http://decode1").await;

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Successfully removed"));

        let workers = router.worker_registry.get_decode_workers();
        assert_eq!(workers.len(), 0);
    }

    // ============= Lock Error Handling Tests =============

    #[test]
    fn test_registry_operations() {
        let router = create_test_pd_router();

        // Test registry operations
        let workers = router.worker_registry.get_all();
        assert_eq!(workers.len(), 0);

        // Add a worker
        let worker = create_test_worker("http://test".to_string(), WorkerType::Prefill, true);
        router.worker_registry.register(Arc::from(worker));

        let workers = router.worker_registry.get_all();
        assert_eq!(workers.len(), 1);

        let prefill_workers = router.worker_registry.get_prefill_workers();
        assert_eq!(prefill_workers.len(), 1);
    }

    // ============= Worker Selection Tests =============

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        // Add mix of healthy and unhealthy workers
        let healthy_worker =
            create_test_worker("http://healthy".to_string(), WorkerType::Prefill, true);
        let unhealthy_worker =
            create_test_worker("http://unhealthy".to_string(), WorkerType::Prefill, false);
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router.select_pd_pair(None, None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        // Should select the healthy worker
        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_pair(None, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    // ============= Health Endpoints Tests =============

    #[tokio::test]
    async fn test_health_endpoints() {
        let router = create_test_pd_router();

        // Add healthy workers - create_test_worker returns Box<dyn Worker>, convert to Arc
        let prefill_worker = create_test_worker(
            "http://localhost:8000".to_string(),
            WorkerType::Prefill,
            true,
        );
        let decode_worker = create_test_worker(
            "http://localhost:8001".to_string(),
            WorkerType::Decode,
            true,
        );

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        // Test health endpoint
        let http_req = axum::http::Request::builder()
            .body(axum::body::Body::empty())
            .unwrap();
        let response = router.health(http_req).await;

        assert_eq!(response.status(), 200);

        // Test readiness endpoint
        let response = router.readiness();
        assert_eq!(response.status(), 200);
    }

    // ============= Load Monitoring Tests =============

    #[tokio::test]
    async fn test_load_monitor_updates() {
        let power_of_two_policy = Arc::new(crate::policies::PowerOfTwoPolicy::new());
        let mut router = create_test_pd_router();
        // Set power_of_two policies in the registry
        router
            .policy_registry
            .set_prefill_policy(power_of_two_policy.clone());
        router
            .policy_registry
            .set_decode_policy(power_of_two_policy);

        // Create load channel
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        router.worker_loads = Arc::new(rx);

        // Simulate load updates
        let mut loads = HashMap::new();
        loads.insert("http://worker1".to_string(), 10);
        loads.insert("http://worker2".to_string(), 5);

        let _ = tx.send(loads.clone());

        // Router should receive updates
        let received = router.worker_loads.borrow().clone();
        assert_eq!(received.get("http://worker1"), Some(&10));
        assert_eq!(received.get("http://worker2"), Some(&5));
    }

    // ============= Concurrent Operations Tests =============

    #[tokio::test]
    async fn test_concurrent_worker_operations() {
        let router = Arc::new(create_test_pd_router());

        let mut handles = vec![];

        // Spawn tasks to add workers
        for i in 0..5 {
            let router_clone = Arc::clone(&router);
            let url = format!("http://worker{}", i);
            let handle = tokio::spawn(async move {
                let worker = create_test_worker(url, WorkerType::Prefill, true);
                router_clone.worker_registry.register(Arc::from(worker));
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            let _ = handle.await;
        }

        // Check final state
        let workers = router.worker_registry.get_prefill_workers();
        assert_eq!(workers.len(), 5);
    }
}
