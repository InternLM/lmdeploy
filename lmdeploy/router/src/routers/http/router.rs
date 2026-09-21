use crate::config::types::RetryConfig;
use crate::core::{
    is_retryable_status, BasicWorker, CircuitBreakerConfig, HealthConfig, RetryExecutor, Worker,
    WorkerRegistry, WorkerType,
};
use crate::metrics::RouterMetrics;
use crate::otel_http::{self, ClientRequestOptions};
use crate::policies::{LoadBalancingPolicy, PolicyRegistry};
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest, GenerationRequest,
    ResponsesRequest,
};
use crate::routers::header_utils;
use crate::routers::{RouterTrait, WorkerManagement};
use axum::{
    body::Body,
    extract::Request,
    http::{
        header::CONTENT_LENGTH, header::CONTENT_TYPE, HeaderMap, HeaderValue, Method, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

/// Regular router that uses injected load balancing policies
#[derive(Debug)]
pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: Client,
    worker_startup_timeout_secs: u64,
    worker_startup_check_interval_secs: u64,
    api_key: Option<String>,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    _worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    _load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
}

impl Router {
    /// Create a new router with injected policy and client
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        worker_urls: Vec<String>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Update active workers gauge
        RouterMetrics::set_active_workers(worker_urls.len());

        // Wait for workers to be healthy (skip if empty - for service discovery mode)
        if !worker_urls.is_empty() {
            Self::wait_for_healthy_workers(
                &worker_urls,
                ctx.router_config.worker_startup_timeout_secs,
                ctx.router_config.worker_startup_check_interval_secs,
            )
            .await?;
        }

        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Register workers in the registry
        // In IGW mode, we need to fetch model info from workers
        let health_config = HealthConfig {
            timeout_secs: ctx.router_config.health_check.timeout_secs,
            check_interval_secs: ctx.router_config.health_check.check_interval_secs,
            endpoint: ctx.router_config.health_check.endpoint.clone(),
            failure_threshold: ctx.router_config.health_check.failure_threshold,
            success_threshold: ctx.router_config.health_check.success_threshold,
        };
        for url in &worker_urls {
            // TODO: In IGW mode, fetch model_id from worker's /get_model_info endpoint
            // For now, create worker without model_id
            let worker_arc: Arc<dyn Worker> = Arc::new(
                BasicWorker::new(url.clone(), WorkerType::Regular)
                    .with_circuit_breaker_config(core_cb_config.clone())
                    .with_health_config(health_config.clone()),
            );
            ctx.worker_registry.register(worker_arc.clone());

            // Notify PolicyRegistry about the new worker
            let model_id = worker_arc.model_id();
            let policy = ctx.policy_registry.on_worker_added(model_id, None);

            // If this is a cache-aware policy and it's the first worker for this model,
            // initialize it with the worker
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy
                    .as_any()
                    .downcast_ref::<crate::policies::CacheAwarePolicy>()
                {
                    let worker_dyn: Arc<dyn Worker> = worker_arc.clone();
                    cache_aware.init_workers(std::slice::from_ref(&worker_dyn));
                }
            }
        }

        // Setup load monitoring for PowerOfTwo policy
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        // Check if default policy is power_of_two for load monitoring
        let default_policy = ctx.policy_registry.get_default_policy();
        let load_monitor_handle = if default_policy.name() == "power_of_two" {
            let monitor_urls = worker_urls.clone();
            let monitor_interval = ctx.router_config.worker_startup_check_interval_secs;
            let policy_clone = default_policy.clone();
            let client_clone = ctx.client.clone();

            Some(Arc::new(tokio::spawn(async move {
                Self::monitor_worker_loads(
                    monitor_urls,
                    tx,
                    monitor_interval,
                    policy_clone,
                    client_clone,
                )
                .await;
            })))
        } else {
            None
        };

        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            policy_registry: ctx.policy_registry.clone(),
            client: ctx.client.clone(),
            worker_startup_timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: ctx
                .router_config
                .worker_startup_check_interval_secs,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            circuit_breaker_config: core_cb_config,
            _worker_loads: worker_loads,
            _load_monitor_handle: load_monitor_handle,
        })
    }

    /// Get the current list of worker URLs
    pub fn get_worker_urls(&self) -> Vec<String> {
        self.worker_registry.get_all_urls()
    }

    /// Get worker URLs for a specific model
    pub fn get_worker_urls_for_model(&self, model_id: Option<&str>) -> Vec<String> {
        let workers = match model_id {
            Some(model) => self.worker_registry.get_by_model_fast(model),
            None => self.worker_registry.get_all(),
        };
        workers.iter().map(|w| w.url().to_string()).collect()
    }

    pub async fn wait_for_healthy_workers(
        worker_urls: &[String],
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval_secs: u64,
    ) -> Result<(), String> {
        if worker_urls.is_empty() {
            return Err(
                "Timeout waiting for workers to become healthy: no workers provided".to_string(),
            );
        }

        // Perform health check asynchronously
        Self::wait_for_healthy_workers_async(
            worker_urls,
            worker_startup_timeout_secs,
            worker_startup_check_interval_secs,
        )
        .await
    }

    async fn wait_for_healthy_workers_async(
        worker_urls: &[String],
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval_secs: u64,
    ) -> Result<(), String> {
        let unique_hosts_vec: Vec<String> = worker_urls.to_vec();

        info!(
            "Waiting for {} workers to become healthy (timeout: {}s)",
            unique_hosts_vec.len(),
            worker_startup_timeout_secs
        );

        let start_time = std::time::Instant::now();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        loop {
            if start_time.elapsed() > Duration::from_secs(worker_startup_timeout_secs) {
                error!(
                    "Timeout {}s waiting for hosts {:?} to become healthy. Set --worker-startup-timeout-secs to a larger value",
                    worker_startup_timeout_secs, unique_hosts_vec
                );
                return Err(format!(
                    "Timeout {}s waiting for hosts {:?} to become healthy. Set --worker-startup-timeout-secs to a larger value",
                    worker_startup_timeout_secs, unique_hosts_vec
                ));
            }

            let mut health_checks = Vec::new();
            for worker_url in &unique_hosts_vec {
                let client_clone = client.clone();
                let url_clone = worker_url.clone();

                let check_health = tokio::spawn(async move {
                    let health_url = format!("{}/health", url_clone);
                    match client_clone.get(&health_url).send().await {
                        Ok(res) => {
                            if res.status().is_success() {
                                None
                            } else {
                                Some((url_clone, format!("status: {}", res.status())))
                            }
                        }
                        Err(_) => Some((url_clone, "not ready".to_string())),
                    }
                });

                health_checks.push(check_health);
            }

            // Wait for all health checks to complete
            let results = futures::future::join_all(health_checks).await;

            let mut unhealthy_hosts = Vec::new();
            let mut healthy_host_count = 0;

            for result in results {
                match result {
                    Ok(None) => {
                        healthy_host_count += 1;
                        // Host is healthy
                    }
                    Ok(Some((url, reason))) => {
                        unhealthy_hosts.push((url, reason));
                    }
                    Err(e) => {
                        unhealthy_hosts.push(("unknown".to_string(), format!("task error: {}", e)));
                    }
                }
            }

            if healthy_host_count > 0 {
                info!(
                    "{} out of {} workers are healthy",
                    healthy_host_count,
                    unique_hosts_vec.len()
                );
                return Ok(());
            } else {
                debug!(
                    "Waiting for at least 1 of {} workers to become healthy ({} unhealthy: {:?})",
                    unique_hosts_vec.len(),
                    unhealthy_hosts.len(),
                    unhealthy_hosts
                );
                tokio::time::sleep(Duration::from_secs(worker_startup_check_interval_secs)).await;
            }
        }
    }

    fn select_first_worker(&self) -> Result<String, String> {
        let workers = self.worker_registry.get_all();
        if workers.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(workers[0].url().to_string())
        }
    }

    #[allow(dead_code)]
    fn select_first_worker_for_model(&self, model_id: Option<&str>) -> Result<String, String> {
        let workers = match model_id {
            Some(model) => self.worker_registry.get_by_model_fast(model),
            None => self.worker_registry.get_all(),
        };
        if workers.is_empty() {
            Err(format!(
                "No workers are available for model: {:?}",
                model_id
            ))
        } else {
            Ok(workers[0].url().to_string())
        }
    }

    pub async fn send_health_check(&self, worker_url: &str) -> Response {
        let request_builder = self.client.get(format!("{}/health", worker_url));

        let response = match request_builder.send().await {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                match res.bytes().await {
                    Ok(body) => (status, body).into_response(),
                    Err(e) => {
                        error!(
                            worker_url = %worker_url,
                            error = %e,
                            "Failed to read health response body"
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response()
                    }
                }
            }
            Err(e) => {
                error!(
                    worker_url = %worker_url,
                    error = %e,
                    "Failed to send health request to worker"
                );
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to send request to worker {}: {}", worker_url, e),
                )
                    .into_response()
            }
        };

        // Don't record metrics for health checks
        response
    }

    // Helper method to proxy GET requests to the first available worker
    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let incoming_headers = req.headers();
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let url = format!("{}/{}", worker_url, endpoint);
                let route_name = format!("/{}", endpoint);
                let mut request_builder = self.client.get(&url);
                for (name, value) in headers {
                    let name_lc = name.to_lowercase();
                    if name_lc != "content-type"
                        && name_lc != "content-length"
                        && !header_utils::TRACE_HEADER_NAMES.contains(&name_lc.as_str())
                    {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match otel_http::send_client_request(
                    request_builder,
                    Some(incoming_headers),
                    ClientRequestOptions {
                        method: "GET",
                        url: &url,
                        route: Some(&route_name),
                        request_phase: None,
                    },
                )
                .await
                {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(axum::body::Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Failed to read response: {}", e),
                            )
                                .into_response(),
                        }
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Request failed: {}", e),
                    )
                        .into_response(),
                }
            }
            Err(e) => (StatusCode::SERVICE_UNAVAILABLE, e).into_response(),
        }
    }

    /// Convert axum HeaderMap to policy RequestHeaders (HashMap<String, String>)
    fn headers_to_request_headers(
        headers: Option<&HeaderMap>,
    ) -> Option<crate::policies::RequestHeaders> {
        headers.map(|h| {
            h.iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                })
                .collect()
        })
    }

    /// Select worker for a specific model considering circuit breaker state
    fn select_worker_for_model(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Option<Arc<dyn Worker>> {
        // Get workers for the specified model (O(1) lookup if model_id is provided)
        let workers = match model_id {
            Some(model) => self.worker_registry.get_by_model_fast(model),
            None => self.worker_registry.get_all(),
        };

        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Convert headers for policies that need them (e.g., consistent_hash)
        let request_headers = Self::headers_to_request_headers(headers);

        let idx = policy.select_worker_with_headers(&available, text, request_headers.as_ref())?;
        Some(available[idx].clone())
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &str,
        model_id: Option<&str>,
    ) -> Response {
        let start = Instant::now();
        let is_stream = typed_req.is_stream();
        let text = typed_req.extract_text_for_routing();

        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // operation per attempt
            |_: u32| async {
                let worker = match self.select_worker_for_model(model_id, Some(&text), headers) {
                    Some(w) => w,
                    None => {
                        RouterMetrics::record_request_error(route, "no_available_workers");
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            "No available workers (all circuits open or unhealthy)",
                        )
                            .into_response();
                    }
                };

                // Optional load tracking for cache-aware policy
                // Get the policy for this model to check if it's cache-aware
                let policy = match model_id {
                    Some(model) => self.policy_registry.get_policy_or_default(model),
                    None => self.policy_registry.get_default_policy(),
                };

                let load_incremented = if policy.name() == "cache_aware" {
                    worker.increment_load();
                    RouterMetrics::set_running_requests(worker.url(), worker.load());
                    true
                } else {
                    false
                };

                // Keep a clone for potential cleanup on retry
                let worker_for_cleanup = if load_incremented {
                    Some(worker.clone())
                } else {
                    None
                };

                let response = self
                    .send_typed_request(
                        headers,
                        typed_req,
                        route,
                        worker.url(),
                        is_stream,
                        load_incremented,
                    )
                    .await;

                // Client errors (4xx) are not worker failures - only server errors (5xx)
                // should count against the circuit breaker.
                let status = response.status();
                worker.record_outcome(status.is_success() || status.is_client_error());

                // For retryable failures, we need to decrement load since send_typed_request
                // won't have done it (it only decrements on success or non-retryable failures)
                if is_retryable_status(response.status()) && load_incremented {
                    if let Some(cleanup_worker) = worker_for_cleanup {
                        cleanup_worker.decrement_load();
                        RouterMetrics::set_running_requests(
                            cleanup_worker.url(),
                            cleanup_worker.load(),
                        );
                    }
                }

                response
            },
            // should_retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // on_backoff hook
            |delay, attempt| {
                RouterMetrics::record_retry(route);
                RouterMetrics::record_retry_backoff_duration(delay, attempt);
            },
            // on_exhausted hook
            || RouterMetrics::record_retries_exhausted(route),
        )
        .await;

        if response.status().is_success() {
            let duration = start.elapsed();
            RouterMetrics::record_request(route);
            RouterMetrics::record_generate_duration(duration);
        } else if !is_retryable_status(response.status()) {
            RouterMetrics::record_request_error(route, "non_retryable_error");
        }

        response
    }

    // Generic simple routing for GET/POST without JSON body
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
    ) -> Response {
        // TODO: currently the backend is using in-memory state management, so this implementation has to fan out to all workers.
        // Eventually, we need to have router to manage the chat history with a proper database, will update this implementation accordingly.
        let worker_urls = self.get_worker_urls();
        if worker_urls.is_empty() {
            return (StatusCode::SERVICE_UNAVAILABLE, "No available workers").into_response();
        }

        let mut last_response: Option<Response> = None;
        for worker_url in worker_urls {
            let url = format!("{}/{}", worker_url, endpoint);

            let route_name = format!("/{}", endpoint);
            let method_name = method.as_str().to_string();
            let mut request_builder = match method.clone() {
                Method::GET => self.client.get(&url),
                Method::POST => self.client.post(&url),
                _ => {
                    return (
                        StatusCode::METHOD_NOT_ALLOWED,
                        "Unsupported method for simple routing",
                    )
                        .into_response()
                }
            };

            if let Some(hdrs) = headers {
                for (name, value) in hdrs {
                    let name_lc = name.as_str().to_lowercase();
                    if name_lc != "content-type"
                        && name_lc != "content-length"
                        && !header_utils::TRACE_HEADER_NAMES.contains(&name_lc.as_str())
                    {
                        request_builder = request_builder.header(name, value);
                    }
                }
            }

            match otel_http::send_client_request(
                request_builder,
                headers,
                ClientRequestOptions {
                    method: &method_name,
                    url: &url,
                    route: Some(&route_name),
                    request_phase: None,
                },
            )
            .await
            {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    let response_headers = header_utils::preserve_response_headers(res.headers());
                    match res.bytes().await {
                        Ok(body) => {
                            let mut response = Response::new(axum::body::Body::from(body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;
                            if status.is_success() {
                                return response;
                            }
                            last_response = Some(response);
                        }
                        Err(e) => {
                            last_response = Some(
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    format!("Failed to read response: {}", e),
                                )
                                    .into_response(),
                            );
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Request failed: {}", e),
                        )
                            .into_response(),
                    );
                }
            }
        }

        last_response
            .unwrap_or_else(|| (StatusCode::BAD_GATEWAY, "No worker response").into_response())
    }

    // Route a GET request with provided headers to a specific endpoint
    async fn route_get_request(&self, headers: Option<&HeaderMap>, endpoint: &str) -> Response {
        self.route_simple_request(headers, endpoint, Method::GET)
            .await
    }

    // Route a POST request with empty body to a specific endpoint
    async fn route_post_empty_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
    ) -> Response {
        self.route_simple_request(headers, endpoint, Method::POST)
            .await
    }

    // Send typed request directly without conversion
    async fn send_typed_request<T: serde::Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &str,
        worker_url: &str,
        is_stream: bool,
        load_incremented: bool, // Whether load was incremented for this request
    ) -> Response {
        let request_url = format!("{}{}", worker_url, route);
        let mut request_builder = self.client.post(&request_url).json(typed_req);

        // Copy all headers from original request if provided, skipping
        // Content-Type/Content-Length (.json() sets them) and trace headers
        // (propagate_trace_headers below injects fresh context).
        if let Some(headers) = headers {
            for (name, value) in headers {
                if *name != CONTENT_TYPE
                    && *name != CONTENT_LENGTH
                    && !header_utils::TRACE_HEADER_NAMES
                        .iter()
                        .any(|&th| name.as_str().eq_ignore_ascii_case(th))
                {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        let res = match otel_http::send_client_request(
            request_builder,
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &request_url,
                route: Some(route),
                request_phase: Some("inference"),
            },
        )
        .await
        {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker_url, route, e
                );

                // Decrement load on error if it was incremented
                if load_incremented {
                    if let Some(worker) = self.worker_registry.get_by_url(worker_url) {
                        worker.decrement_load();
                        RouterMetrics::set_running_requests(worker_url, worker.load());
                    }
                }

                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request failed: {}", e),
                )
                    .into_response();
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, preserve headers
            let response_headers = header_utils::preserve_response_headers(res.headers());

            let response = match res.bytes().await {
                Ok(body) => {
                    let mut response = Response::new(axum::body::Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    // IMPORTANT: Decrement load on error before returning
                    if load_incremented {
                        if let Some(worker) = self.worker_registry.get_by_url(worker_url) {
                            worker.decrement_load();
                            RouterMetrics::set_running_requests(worker_url, worker.load());
                        }
                    }

                    let error_msg = format!("Failed to get response body: {}", e);
                    (StatusCode::INTERNAL_SERVER_ERROR, error_msg).into_response()
                }
            };

            // Decrement load counter for non-streaming requests if it was incremented
            if load_incremented {
                if let Some(worker) = self.worker_registry.get_by_url(worker_url) {
                    worker.decrement_load();
                    RouterMetrics::set_running_requests(worker_url, worker.load());
                }
            }

            response
        } else if load_incremented {
            // For streaming with load tracking, we need to manually decrement when done
            let registry = Arc::clone(&self.worker_registry);
            let worker_url = worker_url.to_string();

            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            let stream = res.bytes_stream();
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

            // Spawn task to forward stream and detect completion
            tokio::spawn(async move {
                let mut stream = stream;
                let mut decremented = false;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            // Check for stream end marker
                            if bytes
                                .as_ref()
                                .windows(12)
                                .any(|window| window == b"data: [DONE]")
                            {
                                if let Some(worker) = registry.get_by_url(&worker_url) {
                                    worker.decrement_load();
                                    RouterMetrics::set_running_requests(&worker_url, worker.load());
                                    decremented = true;
                                }
                            }
                            if tx.send(Ok(bytes)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Stream error: {}", e)));
                            break;
                        }
                    }
                }
                if !decremented {
                    if let Some(worker) = registry.get_by_url(&worker_url) {
                        worker.decrement_load();
                        RouterMetrics::set_running_requests(&worker_url, worker.load());
                    }
                }
            });

            let stream = UnboundedReceiverStream::new(rx);
            let body = Body::from_stream(stream);

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;
            response
        } else {
            // For requests without load tracking, just stream
            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            let stream = res.bytes_stream();
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

            // Spawn task to forward stream
            tokio::spawn(async move {
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok(bytes)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Stream error: {}", e)));
                            break;
                        }
                    }
                }
            });

            let stream = UnboundedReceiverStream::new(rx);
            let body = Body::from_stream(stream);

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;
            response
        }
    }

    pub async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        let start_time = std::time::Instant::now();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.worker_startup_timeout_secs))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        loop {
            if start_time.elapsed() > Duration::from_secs(self.worker_startup_timeout_secs) {
                error!(
                    "Timeout {}s waiting for worker {} to become healthy. Set --worker-startup-timeout-secs to a larger value",
                    self.worker_startup_timeout_secs, worker_url
                );
                return Err(format!(
                    "Timeout {}s waiting for worker {} to become healthy. Set --worker-startup-timeout-secs to a larger value",
                    self.worker_startup_timeout_secs, worker_url
                ));
            }

            match client.get(format!("{}/health", worker_url)).send().await {
                Ok(res) => {
                    if res.status().is_success() {
                        return self.register_worker_unchecked(worker_url);
                    } else {
                        debug!(
                            "Worker {} health check pending - status: {}",
                            worker_url,
                            res.status()
                        );
                        // if the url does not have http or https prefix, warn users
                        if !worker_url.starts_with("http://") && !worker_url.starts_with("https://")
                        {
                            warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                        }

                        tokio::time::sleep(Duration::from_secs(
                            self.worker_startup_check_interval_secs,
                        ))
                        .await;
                        continue;
                    }
                }
                Err(e) => {
                    debug!("Worker {} health check pending - error: {}", worker_url, e);

                    // if the url does not have http or https prefix, warn users
                    if !worker_url.starts_with("http://") && !worker_url.starts_with("https://") {
                        warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                    }

                    tokio::time::sleep(Duration::from_secs(
                        self.worker_startup_check_interval_secs,
                    ))
                    .await;
                    continue;
                }
            }
        }
    }

    /// Register a worker without waiting for its HTTP server to become healthy.
    ///
    /// LMDeploy registers from FastAPI's startup callback, before uvicorn starts
    /// accepting requests. Performing a synchronous health check from
    /// `/nodes/add` would therefore deadlock registration until the router's
    /// startup timeout expires. The central health checker takes over after the
    /// worker has been registered.
    pub fn register_worker_unchecked(&self, worker_url: &str) -> Result<String, String> {
        if self.worker_registry.get_by_url(worker_url).is_some() {
            return Err(format!("Worker {} already exists", worker_url));
        }

        let worker_arc: Arc<dyn Worker> = Arc::new(
            BasicWorker::new(worker_url.to_string(), WorkerType::Regular)
                .with_circuit_breaker_config(self.circuit_breaker_config.clone()),
        );
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
        info!(
            "Registered worker without startup health check: {}",
            worker_url
        );

        RouterMetrics::set_active_workers(self.worker_registry.get_all().len());
        Ok(format!("Successfully added worker: {}", worker_url))
    }

    pub fn remove_worker(&self, worker_url: &str) {
        // Get the worker first to extract model_id
        let model_id = if let Some(worker) = self.worker_registry.get_by_url(worker_url) {
            worker.model_id().to_string()
        } else {
            warn!("Worker {} not found, skipping removal", worker_url);
            return;
        };

        if self.worker_registry.remove_by_url(worker_url).is_some() {
            info!("Removed worker: {}", worker_url);

            // Notify PolicyRegistry about the removed worker
            self.policy_registry.on_worker_removed(&model_id);

            RouterMetrics::set_active_workers(self.worker_registry.get_all().len());
        }

        // If the model is using cache aware policy, remove the worker from the tree
        if let Some(policy) = self.policy_registry.get_policy(&model_id) {
            if let Some(cache_aware) = policy
                .as_any()
                .downcast_ref::<crate::policies::CacheAwarePolicy>()
            {
                cache_aware.remove_worker_by_url(worker_url);
                info!("Removed worker from cache-aware tree: {}", worker_url);
            }
        }
    }

    async fn get_worker_load(&self, worker_url: &str) -> Option<isize> {
        match self
            .client
            .get(format!("{}/get_load", worker_url))
            .send()
            .await
        {
            Ok(res) if res.status().is_success() => match res.bytes().await {
                Ok(bytes) => match serde_json::from_slice::<serde_json::Value>(&bytes) {
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

    // Background task to monitor worker loads
    async fn monitor_worker_loads(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
        policy: Arc<dyn LoadBalancingPolicy>,
        client: Client,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

        loop {
            interval.tick().await;

            let mut loads = HashMap::new();
            for url in &worker_urls {
                if let Some(load) = Self::get_worker_load_static(&client, url).await {
                    loads.insert(url.clone(), load);
                }
            }

            if !loads.is_empty() {
                // Update policy with new loads
                policy.update_loads(&loads);

                // Send to watchers
                if let Err(e) = tx.send(loads) {
                    error!("Failed to send load update: {}", e);
                }
            }
        }
    }

    // Static version of get_worker_load for use in monitoring task
    async fn get_worker_load_static(client: &reqwest::Client, worker_url: &str) -> Option<isize> {
        match client.get(format!("{}/get_load", worker_url)).send().await {
            Ok(res) if res.status().is_success() => match res.bytes().await {
                Ok(bytes) => match serde_json::from_slice::<serde_json::Value>(&bytes) {
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
}

use async_trait::async_trait;

#[async_trait]
impl WorkerManagement for Router {
    async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        Router::add_worker(self, worker_url).await
    }

    fn remove_worker(&self, worker_url: &str) {
        Router::remove_worker(self, worker_url)
    }

    fn get_worker_urls(&self) -> Vec<String> {
        Router::get_worker_urls(self)
    }
}

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        let workers = self.worker_registry.get_all();
        let unhealthy_servers: Vec<_> = workers
            .iter()
            .filter(|w| !w.is_healthy())
            .map(|w| w.url().to_string())
            .collect();

        if unhealthy_servers.is_empty() {
            (StatusCode::OK, "All servers healthy").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Unhealthy servers: {:?}", unhealthy_servers),
            )
                .into_response()
        }
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_server_info").await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "v1/models").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/generate", model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/chat/completions", model_id)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/completions", model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/responses", model_id)
            .await
    }

    async fn get_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}", response_id);
        self.route_get_request(headers, &endpoint).await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}/cancel", response_id);
        self.route_post_empty_request(headers, &endpoint).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Record embeddings-specific metrics in addition to general request metrics
        let start = Instant::now();
        let res = self
            .route_typed_request(headers, body, "/v1/embeddings", model_id)
            .await;

        // Embedding specific metrics
        if res.status().is_success() {
            RouterMetrics::record_embeddings_request();
            RouterMetrics::record_embeddings_duration(start.elapsed());
        } else {
            let error_type = format!("http_{}", res.status().as_u16());
            RouterMetrics::record_embeddings_error(&error_type);
        }

        res
    }

    async fn flush_cache(&self) -> Response {
        // Get all worker URLs
        let worker_urls = self.get_worker_urls();

        // Send requests to all workers concurrently without headers
        let mut tasks = Vec::new();
        for worker_url in &worker_urls {
            let request_builder = self.client.post(format!("{}/flush_cache", worker_url));
            tasks.push(request_builder.send());
        }

        // Wait for all responses
        let results = futures_util::future::join_all(tasks).await;

        // Check if all succeeded
        let all_success = results.iter().all(|r| {
            r.as_ref()
                .map(|res| res.status().is_success())
                .unwrap_or(false)
        });

        if all_success {
            (StatusCode::OK, "Cache flushed on all servers").into_response()
        } else {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Cache flush failed on one or more servers",
            )
                .into_response()
        }
    }

    async fn get_worker_loads(&self) -> Response {
        let urls = self.get_worker_urls();
        let mut loads = Vec::new();

        // Get loads from all workers
        for url in &urls {
            let load = self.get_worker_load(url).await.unwrap_or(-1);
            loads.push(serde_json::json!({
                "worker": url,
                "load": load
            }));
        }

        Json(serde_json::json!({
            "workers": loads
        }))
        .into_response()
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }

    fn readiness(&self) -> Response {
        // Regular router is ready if it has at least one healthy worker
        let workers = self.worker_registry.get_all();
        let healthy_count = workers.iter().filter(|w| w.is_healthy()).count();
        let total_workers = workers.len();

        if healthy_count > 0 {
            Json(serde_json::json!({
                "status": "ready",
                "healthy_workers": healthy_count,
                "total_workers": total_workers
            }))
            .into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "not_ready",
                    "reason": "no healthy workers available",
                    "total_workers": total_workers
                })),
            )
                .into_response()
        }
    }

    /// Route a transparent proxy request to a backend worker
    /// Forwards the request as-is to a selected worker
    async fn route_transparent(
        &self,
        headers: Option<&HeaderMap>,
        path: &str,
        method: &Method,
        body: serde_json::Value,
    ) -> Response {
        debug!("Transparent proxy: routing {} {} to backend", method, path);

        // Select a worker (filter by availability like select_worker_for_model)
        let all_workers = self.worker_registry.get_all();
        let workers: Vec<Arc<dyn Worker>> = all_workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if workers.is_empty() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "No available workers".to_string(),
            )
                .into_response();
        }

        let policy = self.policy_registry.get_default_policy();
        let request_text = serde_json::to_string(&body).ok();
        let request_headers = Self::headers_to_request_headers(headers);
        let worker_idx = match policy.select_worker_with_headers(
            &workers,
            request_text.as_deref(),
            request_headers.as_ref(),
        ) {
            Some(idx) => idx,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Failed to select a worker".to_string(),
                )
                    .into_response();
            }
        };

        let worker: &dyn Worker = workers[worker_idx].as_ref();
        let url = worker.endpoint_url(path);

        debug!("Transparent proxy: forwarding to {}", url);

        // Build the request
        let mut request_builder = match *method {
            Method::GET => self.client.get(&url),
            Method::POST => self.client.post(&url),
            Method::PUT => self.client.put(&url),
            Method::DELETE => self.client.delete(&url),
            Method::PATCH => self.client.patch(&url),
            Method::HEAD => self.client.head(&url),
            _ => {
                return (
                    StatusCode::METHOD_NOT_ALLOWED,
                    format!("Method {} not supported", method),
                )
                    .into_response();
            }
        };

        // Add JSON body if not null/empty
        if !body.is_null() {
            request_builder = request_builder.json(&body);
        }

        // Add authorization if configured
        if let Some(ref key) = self.api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", key));
        }

        // Forward original request headers (e.g. anthropic-version, x-api-key)
        // so protocol-specific backends receive the metadata they require.
        // Skip content-type/content-length (set by .json(&body)), host and
        // hop-by-hop headers, and trace headers (handled by send_client_request).
        if let Some(hdrs) = headers {
            for (name, value) in hdrs {
                let name_lc = name.as_str().to_lowercase();
                if name_lc != "content-type"
                    && name_lc != "content-length"
                    && name_lc != "host"
                    && name_lc != "connection"
                    && name_lc != "keep-alive"
                    && name_lc != "proxy-authenticate"
                    && name_lc != "proxy-authorization"
                    && name_lc != "te"
                    && name_lc != "trailers"
                    && name_lc != "transfer-encoding"
                    && name_lc != "upgrade"
                    && !header_utils::TRACE_HEADER_NAMES.contains(&name_lc.as_str())
                {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        // Send request
        match otel_http::send_client_request(
            request_builder,
            headers,
            ClientRequestOptions {
                method: method.as_str(),
                url: &url,
                route: Some(path),
                request_phase: Some("inference"),
            },
        )
        .await
        {
            Ok(response) => {
                let status = response.status();
                let headers = response.headers().clone();

                // Stream the response body
                let body = Body::from_stream(response.bytes_stream());
                let mut response_builder = Response::builder().status(status.as_u16());

                for (name, value) in headers.iter() {
                    if name != "transfer-encoding" && name != "content-length" {
                        response_builder = response_builder.header(name, value);
                    }
                }

                match response_builder.body(body) {
                    Ok(response) => response,
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to build response: {}", e),
                    )
                        .into_response(),
                }
            }
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                format!("Backend request failed: {}", e),
            )
                .into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_regular_router() -> Router {
        // Create registries
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));

        // Register test workers
        let worker1 = BasicWorker::new("http://worker1:8080".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://worker2:8080".to_string(), WorkerType::Regular);
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));

        let (_, rx) = tokio::sync::watch::channel(HashMap::new());
        Router {
            worker_registry,
            policy_registry,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            api_key: None,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            _worker_loads: Arc::new(rx),
            _load_monitor_handle: None,
        }
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let urls = router.get_worker_urls();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();
        // DashMap doesn't guarantee order, so just check we get one of the workers
        assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_empty_list() {
        // Empty list will return error immediately
        let result = Router::wait_for_healthy_workers(&[], 1, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no workers provided"));
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_invalid_urls() {
        // This test will timeout quickly since the URLs are invalid
        let result =
            Router::wait_for_healthy_workers(&["http://nonexistent:8080".to_string()], 1, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Timeout"));
    }

    // =============================
    // Tests for transparent proxy header/availability fixes
    // =============================

    /// Create a test router with ConsistentHash policy instead of RoundRobin
    fn create_test_consistent_hash_router() -> Router {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::ConsistentHash { virtual_nodes: 100 },
        ));

        let worker1 = BasicWorker::new("http://worker1:8080".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://worker2:8080".to_string(), WorkerType::Regular);
        let worker3 = BasicWorker::new("http://worker3:8080".to_string(), WorkerType::Regular);
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));
        worker_registry.register(Arc::new(worker3));

        let (_, rx) = tokio::sync::watch::channel(HashMap::new());
        Router {
            worker_registry,
            policy_registry,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            api_key: None,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            _worker_loads: Arc::new(rx),
            _load_monitor_handle: None,
        }
    }

    #[test]
    fn test_headers_to_request_headers_basic() {
        // Test that headers_to_request_headers correctly converts HeaderMap to HashMap
        let mut header_map = HeaderMap::new();
        header_map.insert("x-session-id", HeaderValue::from_static("session-123"));
        header_map.insert("content-type", HeaderValue::from_static("application/json"));
        header_map.insert("X-Custom-Header", HeaderValue::from_static("custom-value"));

        let result = Router::headers_to_request_headers(Some(&header_map));
        assert!(result.is_some());
        let headers = result.unwrap();

        // All keys should be lowercased
        assert_eq!(headers.get("x-session-id").unwrap(), "session-123");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
        assert_eq!(headers.get("x-custom-header").unwrap(), "custom-value");
    }

    #[test]
    fn test_headers_to_request_headers_none() {
        // Test that None headers produce None output
        let result = Router::headers_to_request_headers(None);
        assert!(result.is_none());
    }

    #[test]
    fn test_headers_to_request_headers_empty() {
        // Test that empty HeaderMap produces empty HashMap
        let header_map = HeaderMap::new();
        let result = Router::headers_to_request_headers(Some(&header_map));
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_select_worker_for_model_with_consistent_hash_uses_headers() {
        // Verify that select_worker_for_model passes headers through to the policy,
        // producing consistent routing for the same session ID
        let router = create_test_consistent_hash_router();

        let mut header_map = HeaderMap::new();
        header_map.insert("x-session-id", HeaderValue::from_static("sticky-session-1"));

        // Make multiple selections with the same headers - should all pick the same worker
        let mut selected_urls: Vec<String> = Vec::new();
        for _ in 0..10 {
            let worker = router
                .select_worker_for_model(None, Some(r#"{"prompt": "test"}"#), Some(&header_map))
                .expect("Should select a worker");
            selected_urls.push(worker.url().to_string());
        }

        // All selections should go to the same worker (sticky routing)
        let first = &selected_urls[0];
        for (i, url) in selected_urls.iter().enumerate() {
            assert_eq!(
                url, first,
                "Request {} routed to {}, expected {} (session stickiness broken)",
                i, url, first
            );
        }
    }

    #[test]
    fn test_select_worker_for_model_filters_unavailable_workers() {
        // Verify that select_worker_for_model skips unhealthy workers
        let router = create_test_consistent_hash_router();

        // Mark worker1 and worker2 as unhealthy, leaving only worker3
        let all_workers = router.worker_registry.get_all();
        for w in &all_workers {
            if w.url() == "http://worker1:8080" || w.url() == "http://worker2:8080" {
                w.set_healthy(false);
            }
        }

        let worker = router
            .select_worker_for_model(None, Some(r#"{"prompt": "test"}"#), None)
            .expect("Should select the remaining healthy worker");

        assert_eq!(
            worker.url(),
            "http://worker3:8080",
            "Should only select the healthy worker"
        );
    }

    #[test]
    fn test_select_worker_for_model_returns_none_when_all_unavailable() {
        // Verify that when all workers are unhealthy, None is returned
        let router = create_test_consistent_hash_router();

        // Mark all workers as unhealthy
        let all_workers = router.worker_registry.get_all();
        for w in &all_workers {
            w.set_healthy(false);
        }

        let result = router.select_worker_for_model(None, Some(r#"{"prompt": "test"}"#), None);
        assert!(
            result.is_none(),
            "Should return None when all workers are unavailable"
        );
    }

    #[test]
    fn test_consistent_hash_different_sessions_can_route_differently() {
        // Verify that different session IDs can route to different workers
        let router = create_test_consistent_hash_router();

        let mut worker_urls_seen = std::collections::HashSet::new();
        for i in 0..50 {
            let mut header_map = HeaderMap::new();
            let session_id = format!("session-{}", i);
            header_map.insert("x-session-id", HeaderValue::from_str(&session_id).unwrap());

            if let Some(worker) = router.select_worker_for_model(
                None,
                Some(r#"{"prompt": "test"}"#),
                Some(&header_map),
            ) {
                worker_urls_seen.insert(worker.url().to_string());
            }
        }

        // With 50 different sessions and 3 workers, we should see at least 2 workers used
        assert!(
            worker_urls_seen.len() >= 2,
            "Expected distribution across workers, only used: {:?}",
            worker_urls_seen
        );
    }

    #[test]
    fn test_inline_header_conversion_matches_headers_to_request_headers() {
        // Verify that the inline header conversion pattern used by transparent
        // proxy routes matches Router::headers_to_request_headers.
        let mut header_map = HeaderMap::new();
        header_map.insert("X-Session-Id", HeaderValue::from_static("session-abc"));
        header_map.insert("Content-Type", HeaderValue::from_static("application/json"));
        header_map.insert("x-user-id", HeaderValue::from_static("user-42"));

        // Method 1: Router::headers_to_request_headers (used in router.rs)
        let method1 = Router::headers_to_request_headers(Some(&header_map)).unwrap();

        // Method 2: Inline conversion used by transparent proxy routes.
        let method2: HashMap<String, String> = header_map
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|v| (name.as_str().to_lowercase(), v.to_string()))
            })
            .collect();

        assert_eq!(
            method1, method2,
            "Both header conversion methods should produce identical results"
        );
    }

    /// Helper: start a minimal mock server that responds 200 on /health.
    async fn start_healthy_mock_server() -> (String, tokio::task::JoinHandle<()>) {
        use axum::{routing::get, Router as AxumRouter};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = AxumRouter::new().route("/health", get(|| async { "ok" }));
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        (format!("http://{}", addr), handle)
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_all_healthy() {
        let (url, _handle) = start_healthy_mock_server().await;
        let result = Router::wait_for_healthy_workers(&[url], 5, 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_partial_health() {
        // One healthy server + one unreachable URL.
        // The new behaviour succeeds when at least one host is healthy.
        let (healthy_url, _handle) = start_healthy_mock_server().await;
        let unreachable_url = "http://127.0.0.1:1".to_string(); // port 1 is unreachable

        let result = Router::wait_for_healthy_workers(&[healthy_url, unreachable_url], 5, 1).await;
        assert!(result.is_ok());
    }

    /// Helper: start a mock server that returns 503 on /health for a given
    /// duration, then switches to 200. Simulates a worker with a slow startup.
    async fn start_delayed_healthy_mock_server(
        delay: std::time::Duration,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use axum::{extract::State, http::StatusCode, routing::get, Router as AxumRouter};
        use std::sync::Arc;
        use tokio::net::TcpListener;

        let start = std::time::Instant::now();
        let ready_after = Arc::new(delay);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let app = AxumRouter::new()
            .route(
                "/health",
                get(
                    move |State((start, ready_after)): State<(
                        std::time::Instant,
                        Arc<std::time::Duration>,
                    )>| async move {
                        if start.elapsed() >= *ready_after {
                            StatusCode::OK
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE
                        }
                    },
                ),
            )
            .with_state((start, ready_after));

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        (format!("http://{}", addr), handle)
    }

    #[tokio::test]
    async fn test_delayed_worker_becomes_routable_via_health_checker() {
        use crate::core::{BasicWorker, HealthConfig, WorkerRegistry, WorkerType};
        use std::sync::Arc;

        // Two workers: one immediately healthy, one delayed (503 for 2s, then 200).
        let (healthy_url, _h1) = start_healthy_mock_server().await;
        let (delayed_url, _h2) =
            start_delayed_healthy_mock_server(std::time::Duration::from_secs(2)).await;

        // Verify the delayed worker is genuinely unhealthy right now.
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{}/health", delayed_url))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 503);

        // ── Step 1: wait_for_healthy_workers (mirrors PdRouterBase::new startup) ──
        // This succeeds because the healthy worker responds immediately,
        // even though the delayed worker is still returning 503.
        let result =
            Router::wait_for_healthy_workers(&[healthy_url.clone(), delayed_url.clone()], 10, 1)
                .await;
        assert!(
            result.is_ok(),
            "Startup should succeed with at least one healthy worker"
        );

        // ── Step 2: register workers in the registry (mirrors PdRouterBase::new) ──
        let registry = Arc::new(WorkerRegistry::new());

        let healthy_worker = Arc::new(
            BasicWorker::new(healthy_url, WorkerType::Decode).with_health_config(HealthConfig {
                timeout_secs: 2,
                check_interval_secs: 1,
                endpoint: "/health".to_string(),
                failure_threshold: 3,
                success_threshold: 1,
            }),
        );
        registry.register(healthy_worker);

        let delayed_worker = Arc::new(
            BasicWorker::new(delayed_url, WorkerType::Decode).with_health_config(HealthConfig {
                timeout_secs: 2,
                check_interval_secs: 1,
                endpoint: "/health".to_string(),
                failure_threshold: 3,
                success_threshold: 1,
            }),
        );
        delayed_worker.set_healthy(false); // starts unhealthy
        registry.register(delayed_worker.clone());

        // Only the immediately-healthy worker should be available for routing.
        let healthy = registry.get_workers_filtered(None, None, None, true);
        assert_eq!(
            healthy.len(),
            1,
            "Only 1 worker should be healthy initially, got {}",
            healthy.len()
        );

        // ── Step 3: start background health checker (mirrors PdRouterBase::new) ──
        let health_checker = registry.start_health_checker(1);

        // ── Step 4: wait for delayed worker to recover ──
        // The mock server switches to 200 at t≈2s. With a 1s check interval
        // and success_threshold=1, the worker should be healthy by t≈3-4s.
        tokio::time::sleep(std::time::Duration::from_secs(4)).await;

        // Both workers should now be available for routing.
        let healthy = registry.get_workers_filtered(None, None, None, true);
        assert_eq!(
            healthy.len(),
            2,
            "Both workers should be healthy after recovery, got {}",
            healthy.len()
        );
        assert!(
            delayed_worker.is_healthy(),
            "Delayed worker should have transitioned to healthy via health checker"
        );

        health_checker.shutdown().await;
    }
}
