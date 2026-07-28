// Copyright (c) OpenMMLab. All rights reserved.

//! Axum entry point linked into the `_turbomind` Python extension.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures::{StreamExt, stream};
use lmdeploy_parser::{AssistantEvent, ParserConfig, ResponseParser, SUPPORTED_TOOL_PARSERS};
use lmdeploy_serving::engine::{Engine, GenerateRequest};
use lmdeploy_serving::protocol::{
    ChatCompletionRequest, CompletionRequest, ErrorBody, ModelCard, ModelList, OneOrMany,
    OpenAiError, PplInput, PplRequest, PplResponse, Usage,
};
use lmdeploy_serving::{ChatTemplate, Tokenizer};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tracing::info;
use turbomind_runtime::GenerationConfig;

pub const SERVER_ABI_VERSION: u32 = 1;

fn default_host() -> String {
    "0.0.0.0".into()
}

fn default_port() -> u16 {
    23333
}

fn default_max_new_tokens() -> i32 {
    512
}

fn default_log_level() -> String {
    "WARNING".into()
}

#[cfg(any(feature = "ffi", test))]
fn tracing_level_directive(log_level: &str) -> anyhow::Result<&'static str> {
    match log_level.to_ascii_uppercase().as_str() {
        "NOTSET" | "TRACE" => Ok("trace"),
        "DEBUG" => Ok("debug"),
        "INFO" => Ok("info"),
        "WARN" | "WARNING" => Ok("warn"),
        "ERROR" | "CRITICAL" | "FATAL" => Ok("error"),
        _ => anyhow::bail!("invalid log level: {log_level}"),
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub server_name: String,
    #[serde(default = "default_port")]
    pub server_port: u16,
    pub model_dir: String,
    pub model_name: String,
    #[serde(default = "default_max_new_tokens")]
    pub default_max_new_tokens: i32,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub reasoning_parser: Option<String>,
    #[serde(default)]
    pub tool_call_parser: Option<String>,
}

#[derive(Default)]
struct ServerMetrics {
    requests: AtomicU64,
    failed_requests: AtomicU64,
    prompt_tokens: AtomicU64,
    generated_tokens: AtomicU64,
}

pub struct AppState {
    config: ServerConfig,
    engine: Arc<dyn Engine>,
    tokenizer: Tokenizer,
    chat_template: ChatTemplate,
    metrics: ServerMetrics,
    next_session_id: AtomicU64,
}

impl AppState {
    pub fn new(config: ServerConfig, engine: Arc<dyn Engine>) -> anyhow::Result<Self> {
        if config.model_name.is_empty() {
            anyhow::bail!("model_name must not be empty");
        }
        if config.default_max_new_tokens < 1 {
            anyhow::bail!("default_max_new_tokens must be greater than zero");
        }
        if config.tool_call_parser.as_deref() == Some("gpt-oss") {
            anyhow::bail!("gpt-oss/Harmony parsing is not supported by rust_api_server");
        }
        if let Some(parser) = config.tool_call_parser.as_deref()
            && !SUPPORTED_TOOL_PARSERS.contains(&parser)
        {
            anyhow::bail!("unsupported Rust tool-call parser: {parser}");
        }
        if let Some(parser) = config.reasoning_parser.as_deref()
            && ![
                "default",
                "deepseek-v3",
                "deepseek-v32",
                "deepseek-v3.2",
                "deepseek-v4",
                "qwen-qwq",
                "intern-s1",
                "deepseek-r1",
            ]
            .contains(&parser)
        {
            anyhow::bail!("unsupported Rust reasoning parser: {parser}");
        }
        Ok(Self {
            tokenizer: Tokenizer::from_model_dir(&config.model_dir)?,
            chat_template: ChatTemplate::from_model_dir(&config.model_dir)?,
            config,
            engine,
            metrics: ServerMetrics::default(),
            next_session_id: AtomicU64::new(1),
        })
    }

    fn session_id(&self, requested: Option<u64>) -> u64 {
        requested.unwrap_or_else(|| self.next_session_id.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    kind: &'static str,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
        }
    }

    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: "invalid API key".into(),
            kind: "authentication_error",
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: "server_error",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let code = self.status.as_u16() as i32;
        (
            self.status,
            Json(ErrorBody {
                error: OpenAiError {
                    message: self.message,
                    kind: self.kind.into(),
                    code,
                    param: None,
                },
            }),
        )
            .into_response()
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/get_ppl", post(get_ppl))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ModelList>, ApiError> {
    authorize(&state, &headers)?;
    Ok(Json(ModelList {
        object: "list".into(),
        data: vec![ModelCard {
            id: state.config.model_name.clone(),
            object: "model".into(),
            owned_by: "lmdeploy".into(),
        }],
    }))
}

async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let requests = state.metrics.requests.load(Ordering::Relaxed);
    let failures = state.metrics.failed_requests.load(Ordering::Relaxed);
    let prompt_tokens = state.metrics.prompt_tokens.load(Ordering::Relaxed);
    let generated_tokens = state.metrics.generated_tokens.load(Ordering::Relaxed);
    let schedule = state
        .engine
        .schedule_metrics()
        .ok()
        .flatten()
        .unwrap_or_default();
    let body = format!(
        concat!(
            "# TYPE lmdeploy_requests_total counter\n",
            "lmdeploy_requests_total {}\n",
            "# TYPE lmdeploy_request_failures_total counter\n",
            "lmdeploy_request_failures_total {}\n",
            "# TYPE lmdeploy_prompt_tokens_total counter\n",
            "lmdeploy_prompt_tokens_total {}\n",
            "# TYPE lmdeploy_generated_tokens_total counter\n",
            "lmdeploy_generated_tokens_total {}\n",
            "# TYPE lmdeploy_turbomind_active_sequences gauge\n",
            "lmdeploy_turbomind_active_sequences {}\n",
            "# TYPE lmdeploy_turbomind_waiting_sequences gauge\n",
            "lmdeploy_turbomind_waiting_sequences {}\n",
            "# TYPE lmdeploy_turbomind_free_blocks gauge\n",
            "lmdeploy_turbomind_free_blocks {}\n"
        ),
        requests,
        failures,
        prompt_tokens,
        generated_tokens,
        schedule.active_sequences,
        schedule.waiting_sequences,
        schedule.free_blocks
    );
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    authorize(&state, &headers)?;
    validate_model(&state, &request.model)?;
    if request.n != 1 {
        return Err(ApiError::bad_request(
            "rust_api_server currently requires n=1",
        ));
    }
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    let prompt = state
        .chat_template
        .render(
            &request.messages,
            request.tools.as_deref(),
            request.chat_template_kwargs.as_ref(),
        )
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let input_ids = state
        .tokenizer
        .encode(&prompt, false)
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let prompt_tokens = input_ids.len();
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(state.config.default_max_new_tokens);
    let generation = build_generation(
        &state,
        max_tokens,
        request.min_new_tokens.unwrap_or(0),
        request.temperature,
        request.top_p,
        request.min_p,
        request.top_k,
        request.repetition_penalty,
        request.seed,
        request.stop,
        request.ignore_eos,
        request
            .top_logprobs
            .filter(|_| request.logprobs)
            .unwrap_or(0),
    )?;
    let generation_stream = state
        .engine
        .generate(GenerateRequest {
            input_ids,
            session_id: state.session_id(request.session_id),
            generation,
        })
        .await
        .map_err(|error| engine_error(&state, error))?;
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_timestamp();
    let skip_special_tokens = request.skip_special_tokens.unwrap_or(true);
    if request.stream {
        return Ok(chat_stream_response(
            state,
            generation_stream,
            request_id,
            created,
            request.model,
            prompt_tokens,
            max_tokens,
            skip_special_tokens,
            request
                .stream_options
                .is_some_and(|options| options.include_usage),
        ));
    }

    let (text, reasoning, tool_calls, output_ids, cancelled) = collect_chat(
        &state,
        generation_stream,
        skip_special_tokens,
        parser_config(
            state.config.reasoning_parser.as_deref(),
            state.config.tool_call_parser.as_deref(),
            request.chat_template_kwargs.as_ref(),
        ),
    )
    .await?;
    record_tokens(&state, prompt_tokens, output_ids.len());
    let finish_reason = finish_reason(
        cancelled,
        output_ids.len(),
        max_tokens,
        !tool_calls.is_empty(),
    );
    let tool_calls: Vec<_> = tool_calls
        .into_iter()
        .map(|call| {
            json!({
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            })
        })
        .collect();
    Ok(Json(json!({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
                "reasoning_content": reasoning,
                "tool_calls": (!tool_calls.is_empty()).then_some(tool_calls),
            },
            "finish_reason": finish_reason,
        }],
        "usage": Usage::new(prompt_tokens, output_ids.len()),
        "output_ids": request.return_token_ids.then_some(output_ids),
    }))
    .into_response())
}

async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    authorize(&state, &headers)?;
    validate_model(&state, &request.model)?;
    if request.n != 1 {
        return Err(ApiError::bad_request(
            "rust_api_server currently requires n=1",
        ));
    }
    let prompts = request.prompt.clone().into_vec();
    if prompts.is_empty() {
        return Err(ApiError::bad_request("prompt must not be empty"));
    }
    if request.stream && prompts.len() != 1 {
        return Err(ApiError::bad_request(
            "streaming completions currently require exactly one prompt",
        ));
    }
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(16);
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_timestamp();
    let skip_special_tokens = request.skip_special_tokens.unwrap_or(true);
    let mut prepared = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let input_ids = state
            .tokenizer
            .encode(&prompt, true)
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        let generation = build_generation(
            &state,
            max_tokens,
            0,
            request.temperature,
            request.top_p,
            request.min_p,
            request.top_k,
            request.repetition_penalty,
            request.seed,
            request.stop.clone(),
            request.ignore_eos,
            request.logprobs.unwrap_or(0),
        )?;
        let output = state
            .engine
            .generate(GenerateRequest {
                input_ids: input_ids.clone(),
                session_id: state.session_id(request.session_id),
                generation,
            })
            .await
            .map_err(|error| engine_error(&state, error))?;
        prepared.push((prompt, input_ids.len(), output));
    }

    if request.stream {
        let (prompt, prompt_tokens, output) = prepared.pop().expect("one prepared prompt");
        return Ok(completion_stream_response(
            state,
            output,
            request_id,
            created,
            request.model,
            prompt,
            prompt_tokens,
            max_tokens,
            skip_special_tokens,
            request.echo,
            request
                .stream_options
                .is_some_and(|options| options.include_usage),
        ));
    }

    let mut choices = Vec::with_capacity(prepared.len());
    let mut total_prompt = 0;
    let mut total_completion = 0;
    for (index, (prompt, prompt_tokens, output)) in prepared.into_iter().enumerate() {
        let (generated, output_ids, cancelled) =
            collect_text(&state, output, skip_special_tokens).await?;
        total_prompt += prompt_tokens;
        total_completion += output_ids.len();
        choices.push(json!({
            "index": index,
            "text": if request.echo { format!("{prompt}{generated}") } else { generated },
            "logprobs": Value::Null,
            "finish_reason": finish_reason(cancelled, output_ids.len(), max_tokens, false),
        }));
    }
    record_tokens(&state, total_prompt, total_completion);
    Ok(Json(json!({
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": choices,
        "usage": Usage::new(total_prompt, total_completion),
    }))
    .into_response())
}

async fn get_ppl(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<PplRequest>,
) -> Result<Json<PplResponse>, ApiError> {
    authorize(&state, &headers)?;
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    let input_ids = match request.input {
        PplInput::Text(text) => state
            .tokenizer
            .encode(&text, true)
            .map_err(|error| ApiError::bad_request(error.to_string()))?,
        PplInput::TokenIds(ids) => ids,
    };
    if input_ids.len() < 2 {
        return Err(ApiError::bad_request(
            "input must have at least 2 tokens to compute ppl",
        ));
    }
    let prompt_tokens = input_ids.len();
    let ppl = state
        .engine
        .ppl(input_ids, state.session_id(None))
        .await
        .map_err(|error| engine_error(&state, error))?;
    state
        .metrics
        .prompt_tokens
        .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
    Ok(Json(PplResponse { ppl }))
}

#[allow(clippy::too_many_arguments)]
fn build_generation(
    state: &AppState,
    max_new_tokens: i32,
    min_new_tokens: i32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    top_k: Option<i32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
    stop: Option<OneOrMany<String>>,
    ignore_eos: bool,
    output_logprobs: i32,
) -> Result<GenerationConfig, ApiError> {
    if max_new_tokens < 1 {
        return Err(ApiError::bad_request(
            "max_tokens must be greater than zero",
        ));
    }
    if min_new_tokens < 0 || min_new_tokens > max_new_tokens {
        return Err(ApiError::bad_request(
            "min_new_tokens must be between zero and max_tokens",
        ));
    }
    // TurboMind's eos_ids are used to suppress EOS before min_new_tokens;
    // actual termination is driven by stop_ids. Keep both in sync with the
    // existing Python TurboMind engine unless the caller explicitly ignores
    // EOS.
    let eos_ids = if ignore_eos {
        Vec::new()
    } else {
        state.tokenizer.eos_ids().to_vec()
    };
    let mut stop_ids = eos_ids.clone();
    for stop in stop.map(OneOrMany::into_vec).unwrap_or_default() {
        let ids = state
            .tokenizer
            .encode(&stop, false)
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        if ids.len() != 1 {
            return Err(ApiError::bad_request(format!(
                "stop string {stop:?} must encode to exactly one token for TurboMind"
            )));
        }
        stop_ids.push(ids[0]);
    }
    Ok(GenerationConfig {
        max_new_tokens,
        min_new_tokens,
        top_k: top_k.unwrap_or(1),
        top_p: top_p.unwrap_or(1.0),
        min_p: min_p.unwrap_or(0.0),
        temperature: temperature.unwrap_or(1.0),
        repetition_penalty: repetition_penalty.unwrap_or(1.0),
        random_seed: seed.unwrap_or(0),
        output_logprobs,
        eos_ids,
        stop_ids,
        ..Default::default()
    })
}

async fn collect_text(
    state: &AppState,
    mut output: lmdeploy_serving::GenerateStream,
    skip_special_tokens: bool,
) -> Result<(String, Vec<i32>, bool), ApiError> {
    let mut ids = Vec::new();
    let mut cancelled = false;
    while let Some(chunk) = output.next().await {
        let chunk = chunk.map_err(|error| engine_error(state, error))?;
        ids.extend(chunk.token_ids);
        cancelled |= chunk.cancelled;
    }
    let text = state
        .tokenizer
        .decode(&ids, skip_special_tokens)
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok((text, ids, cancelled))
}

async fn collect_chat(
    state: &AppState,
    output: lmdeploy_serving::GenerateStream,
    skip_special_tokens: bool,
    parser_config: ParserConfig,
) -> Result<(String, Option<String>, Vec<ParsedToolCall>, Vec<i32>, bool), ApiError> {
    let (decoded, ids, cancelled) = collect_text(state, output, skip_special_tokens).await?;
    let mut parser = ResponseParser::new(parser_config);
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for event in parser.push(&decoded).into_iter().chain(parser.finish()) {
        match event {
            AssistantEvent::Content { text } => content.push_str(&text),
            AssistantEvent::Reasoning { text } => reasoning.push_str(&text),
            AssistantEvent::ToolStart { index, id, name } => tool_calls.push(ParsedToolCall {
                index,
                id,
                name,
                arguments: String::new(),
            }),
            AssistantEvent::ToolArguments { index, arguments } => {
                if let Some(call) = tool_calls.iter_mut().find(|call| call.index == index) {
                    call.arguments.push_str(&arguments);
                }
            }
            AssistantEvent::ToolEnd { .. } => {}
        }
    }
    Ok((
        content,
        (!reasoning.is_empty()).then_some(reasoning),
        tool_calls,
        ids,
        cancelled,
    ))
}

struct ParsedToolCall {
    index: usize,
    id: String,
    name: String,
    arguments: String,
}

#[allow(clippy::too_many_arguments)]
fn chat_stream_response(
    state: Arc<AppState>,
    output: lmdeploy_serving::GenerateStream,
    request_id: String,
    created: u64,
    model: String,
    prompt_tokens: usize,
    max_tokens: i32,
    skip_special_tokens: bool,
    include_usage: bool,
) -> Response {
    struct StreamState {
        state: Arc<AppState>,
        output: lmdeploy_serving::GenerateStream,
        parser: ResponseParser,
        ids: Vec<i32>,
        decoded: String,
        request_id: String,
        created: u64,
        model: String,
        prompt_tokens: usize,
        max_tokens: i32,
        skip_special_tokens: bool,
        include_usage: bool,
        tool_emitted: bool,
        first: bool,
        done: bool,
    }
    let parser_config = parser_config(
        state.config.reasoning_parser.as_deref(),
        state.config.tool_call_parser.as_deref(),
        None,
    );
    let stream_state = StreamState {
        state,
        output,
        parser: ResponseParser::new(parser_config),
        ids: Vec::new(),
        decoded: String::new(),
        request_id,
        created,
        model,
        prompt_tokens,
        max_tokens,
        skip_special_tokens,
        include_usage,
        tool_emitted: false,
        first: true,
        done: false,
    };
    let events = stream::unfold(stream_state, |mut stream_state| async move {
        if stream_state.done {
            return None;
        }
        if stream_state.first {
            stream_state.first = false;
            let chunk = chat_sse_json(
                &stream_state.request_id,
                stream_state.created,
                &stream_state.model,
                json!({"role": "assistant", "content": ""}),
                None,
                None,
            );
            return Some((Ok::<_, Infallible>(Bytes::from(chunk)), stream_state));
        }
        match stream_state.output.next().await {
            Some(Ok(chunk)) => {
                stream_state.ids.extend(chunk.token_ids);
                let decoded = match stream_state
                    .state
                    .tokenizer
                    .decode(&stream_state.ids, stream_state.skip_special_tokens)
                {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        stream_state.done = true;
                        return Some((Ok(Bytes::from(sse_error(error.to_string()))), stream_state));
                    }
                };
                let delta = decoded
                    .strip_prefix(&stream_state.decoded)
                    .unwrap_or(&decoded)
                    .to_owned();
                stream_state.decoded = decoded;
                let mut content = String::new();
                let mut reasoning = String::new();
                let mut tool_calls = Vec::new();
                let mut events = stream_state.parser.push(&delta);
                if chunk.finished || chunk.cancelled {
                    events.extend(stream_state.parser.finish());
                }
                for event in events {
                    match event {
                        AssistantEvent::Content { text } => content.push_str(&text),
                        AssistantEvent::Reasoning { text } => reasoning.push_str(&text),
                        AssistantEvent::ToolStart { index, id, name } => {
                            stream_state.tool_emitted = true;
                            tool_calls.push(json!({
                                "index": index,
                                "id": id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }));
                        }
                        AssistantEvent::ToolArguments { index, arguments } => {
                            tool_calls.push(json!({
                                "index": index,
                                "function": {"arguments": arguments},
                            }));
                        }
                        AssistantEvent::ToolEnd { .. } => {}
                    }
                }
                let delta = json!({
                    "content": (!content.is_empty()).then_some(content),
                    "reasoning_content": (!reasoning.is_empty()).then_some(reasoning),
                    "tool_calls": (!tool_calls.is_empty()).then_some(tool_calls),
                });
                let finish = (chunk.finished || chunk.cancelled).then(|| {
                    finish_reason(
                        chunk.cancelled,
                        stream_state.ids.len(),
                        stream_state.max_tokens,
                        stream_state.tool_emitted,
                    )
                });
                let data = chat_sse_json(
                    &stream_state.request_id,
                    stream_state.created,
                    &stream_state.model,
                    delta,
                    finish.flatten(),
                    None,
                );
                if chunk.finished || chunk.cancelled {
                    record_tokens(
                        &stream_state.state,
                        stream_state.prompt_tokens,
                        stream_state.ids.len(),
                    );
                    stream_state.done = true;
                    let usage = if stream_state.include_usage {
                        format!(
                            "data: {}\n\n",
                            json!({
                                "id": stream_state.request_id,
                                "object": "chat.completion.chunk",
                                "created": stream_state.created,
                                "model": stream_state.model,
                                "choices": [],
                                "usage": Usage::new(stream_state.prompt_tokens, stream_state.ids.len()),
                            })
                        )
                    } else {
                        String::new()
                    };
                    return Some((
                        Ok(Bytes::from(format!("{data}{usage}data: [DONE]\n\n"))),
                        stream_state,
                    ));
                }
                Some((Ok(Bytes::from(data)), stream_state))
            }
            Some(Err(error)) => {
                stream_state
                    .state
                    .metrics
                    .failed_requests
                    .fetch_add(1, Ordering::Relaxed);
                stream_state.done = true;
                Some((Ok(Bytes::from(sse_error(error.to_string()))), stream_state))
            }
            None => {
                stream_state.done = true;
                Some((
                    Ok(Bytes::from(sse_error(
                        "generation stream ended unexpectedly",
                    ))),
                    stream_state,
                ))
            }
        }
    });
    sse_response(events)
}

#[allow(clippy::too_many_arguments)]
fn completion_stream_response(
    state: Arc<AppState>,
    output: lmdeploy_serving::GenerateStream,
    request_id: String,
    created: u64,
    model: String,
    prompt: String,
    prompt_tokens: usize,
    max_tokens: i32,
    skip_special_tokens: bool,
    echo: bool,
    include_usage: bool,
) -> Response {
    struct StreamState {
        state: Arc<AppState>,
        output: lmdeploy_serving::GenerateStream,
        ids: Vec<i32>,
        decoded: String,
        request_id: String,
        created: u64,
        model: String,
        prompt: String,
        prompt_tokens: usize,
        max_tokens: i32,
        skip_special_tokens: bool,
        echo: bool,
        include_usage: bool,
        first: bool,
        done: bool,
    }
    let stream_state = StreamState {
        state,
        output,
        ids: Vec::new(),
        decoded: String::new(),
        request_id,
        created,
        model,
        prompt,
        prompt_tokens,
        max_tokens,
        skip_special_tokens,
        echo,
        include_usage,
        first: true,
        done: false,
    };
    let events = stream::unfold(stream_state, |mut stream_state| async move {
        if stream_state.done {
            return None;
        }
        if stream_state.first && stream_state.echo {
            stream_state.first = false;
            let data = completion_sse_json(
                &stream_state.request_id,
                stream_state.created,
                &stream_state.model,
                &stream_state.prompt,
                None,
                None,
            );
            return Some((Ok::<_, Infallible>(Bytes::from(data)), stream_state));
        }
        stream_state.first = false;
        match stream_state.output.next().await {
            Some(Ok(chunk)) => {
                stream_state.ids.extend(chunk.token_ids);
                let decoded = match stream_state
                    .state
                    .tokenizer
                    .decode(&stream_state.ids, stream_state.skip_special_tokens)
                {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        stream_state.done = true;
                        return Some((Ok(Bytes::from(sse_error(error.to_string()))), stream_state));
                    }
                };
                let delta = decoded
                    .strip_prefix(&stream_state.decoded)
                    .unwrap_or(&decoded)
                    .to_owned();
                stream_state.decoded = decoded;
                let finish = (chunk.finished || chunk.cancelled)
                    .then(|| {
                        finish_reason(
                            chunk.cancelled,
                            stream_state.ids.len(),
                            stream_state.max_tokens,
                            false,
                        )
                    })
                    .flatten();
                let data = completion_sse_json(
                    &stream_state.request_id,
                    stream_state.created,
                    &stream_state.model,
                    &delta,
                    finish,
                    None,
                );
                if chunk.finished || chunk.cancelled {
                    record_tokens(
                        &stream_state.state,
                        stream_state.prompt_tokens,
                        stream_state.ids.len(),
                    );
                    stream_state.done = true;
                    let usage = if stream_state.include_usage {
                        completion_sse_json(
                            &stream_state.request_id,
                            stream_state.created,
                            &stream_state.model,
                            "",
                            None,
                            Some(Usage::new(
                                stream_state.prompt_tokens,
                                stream_state.ids.len(),
                            )),
                        )
                    } else {
                        String::new()
                    };
                    return Some((
                        Ok(Bytes::from(format!("{data}{usage}data: [DONE]\n\n"))),
                        stream_state,
                    ));
                }
                Some((Ok(Bytes::from(data)), stream_state))
            }
            Some(Err(error)) => {
                stream_state
                    .state
                    .metrics
                    .failed_requests
                    .fetch_add(1, Ordering::Relaxed);
                stream_state.done = true;
                Some((Ok(Bytes::from(sse_error(error.to_string()))), stream_state))
            }
            None => {
                stream_state.done = true;
                Some((
                    Ok(Bytes::from(sse_error(
                        "generation stream ended unexpectedly",
                    ))),
                    stream_state,
                ))
            }
        }
    });
    sse_response(events)
}

fn chat_sse_json(
    request_id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Option<&'static str>,
    usage: Option<Usage>,
) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            "usage": usage,
        })
    )
}

fn completion_sse_json(
    request_id: &str,
    created: u64,
    model: &str,
    text: &str,
    finish_reason: Option<&'static str>,
    usage: Option<Usage>,
) -> String {
    let choices = if usage.is_some() {
        Vec::new()
    } else {
        vec![
            json!({"index": 0, "text": text, "logprobs": Value::Null, "finish_reason": finish_reason}),
        ]
    };
    format!(
        "data: {}\n\n",
        json!({
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": choices,
            "usage": usage,
        })
    )
}

fn sse_response<S>(events: S) -> Response
where
    S: futures::Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
{
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(events))
        .expect("valid SSE response")
}

fn sse_error(message: impl Into<String>) -> String {
    format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"error": {"message": message.into(), "type": "server_error"}})
    )
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    if state.config.api_keys.is_empty() {
        return Ok(());
    }
    let supplied = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if supplied.is_some_and(|key| {
        state
            .config
            .api_keys
            .iter()
            .any(|candidate| candidate == key)
    }) {
        Ok(())
    } else {
        Err(ApiError::unauthorized())
    }
}

fn validate_model(state: &AppState, model: &str) -> Result<(), ApiError> {
    if model == state.config.model_name {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "model {model:?} is not served; available model is {:?}",
            state.config.model_name
        )))
    }
}

fn parser_config(
    name: Option<&str>,
    tool_parser: Option<&str>,
    kwargs: Option<&std::collections::HashMap<String, Value>>,
) -> ParserConfig {
    let enabled = name.is_some();
    let explicit_thinking = kwargs
        .and_then(|kwargs| {
            kwargs
                .get("thinking")
                .or_else(|| kwargs.get("enable_thinking"))
        })
        .and_then(Value::as_bool);
    let starts_in_reasoning = match name {
        None => false,
        Some("deepseek-v3" | "deepseek-v32" | "deepseek-v3.2" | "deepseek-v4") => {
            explicit_thinking.unwrap_or(false)
        }
        Some(_) => true,
    };
    ParserConfig {
        reasoning_open_tag: enabled.then(|| "<think>".into()),
        reasoning_close_tag: enabled.then(|| "</think>".into()),
        starts_in_reasoning,
        tool_parser: tool_parser.map(str::to_owned),
    }
}

fn finish_reason(
    cancelled: bool,
    generated: usize,
    max_tokens: i32,
    has_tools: bool,
) -> Option<&'static str> {
    if cancelled {
        Some("abort")
    } else if has_tools {
        Some("tool_calls")
    } else if generated >= max_tokens as usize {
        Some("length")
    } else {
        Some("stop")
    }
}

fn engine_error(state: &AppState, error: lmdeploy_serving::engine::Error) -> ApiError {
    state
        .metrics
        .failed_requests
        .fetch_add(1, Ordering::Relaxed);
    ApiError::internal(error.to_string())
}

fn record_tokens(state: &AppState, prompt: usize, generated: usize) {
    state
        .metrics
        .prompt_tokens
        .fetch_add(prompt as u64, Ordering::Relaxed);
    state
        .metrics
        .generated_tokens
        .fetch_add(generated as u64, Ordering::Relaxed);
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub async fn serve(state: Arc<AppState>) -> anyhow::Result<()> {
    let address: SocketAddr =
        format!("{}:{}", state.config.server_name, state.config.server_port).parse()?;
    let listener = TcpListener::bind(address).await?;
    info!(%address, model = %state.config.model_name, "rust_api_server listening");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(feature = "ffi")]
mod ffi;

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use async_trait::async_trait;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use lmdeploy_serving::engine::{
        Error as EngineError, GenerateChunk, GenerateStream, Result as EngineResult,
        ScheduleMetrics,
    };
    use tower::ServiceExt;

    use super::*;

    struct MockEngine;

    #[async_trait]
    impl Engine for MockEngine {
        async fn generate(&self, _request: GenerateRequest) -> EngineResult<GenerateStream> {
            Ok(Box::pin(stream::iter([
                Ok(GenerateChunk {
                    token_ids: vec![2],
                    finished: false,
                    cancelled: false,
                }),
                Ok(GenerateChunk {
                    token_ids: vec![3],
                    finished: true,
                    cancelled: false,
                }),
            ])))
        }

        async fn ppl(&self, _input_ids: Vec<i32>, _session_id: u64) -> EngineResult<f32> {
            Ok(1.25)
        }

        fn schedule_metrics(&self) -> EngineResult<Option<ScheduleMetrics>> {
            Ok(Some(ScheduleMetrics {
                active_sequences: 2,
                free_blocks: 17,
                ..Default::default()
            }))
        }
    }

    struct ModelFixture {
        path: PathBuf,
    }

    impl ModelFixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "lmdeploy-rust-api-test-{}",
                uuid::Uuid::new_v4().simple()
            ));
            fs::create_dir(&path).unwrap();
            write(
                &path.join("tokenizer.json"),
                r#"{
                    "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
                    "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
                    "post_processor":null,"decoder":null,
                    "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2,"!":3},"unk_token":"[UNK]"}
                }"#,
            );
            write(
                &path.join("tokenizer_config.json"),
                r#"{"chat_template":"{{ messages[0].content }}","eos_token":"!"}"#,
            );
            Self { path }
        }
    }

    impl Drop for ModelFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write(path: &Path, content: &str) {
        fs::write(path, content).unwrap();
    }

    fn test_state(fixture: &ModelFixture) -> AppState {
        AppState::new(
            ServerConfig {
                server_name: "127.0.0.1".into(),
                server_port: 23333,
                model_dir: fixture.path.to_string_lossy().into_owned(),
                model_name: "test-model".into(),
                default_max_new_tokens: 8,
                log_level: "WARNING".into(),
                api_keys: Vec::new(),
                reasoning_parser: None,
                tool_call_parser: None,
            },
            Arc::new(MockEngine),
        )
        .unwrap()
    }

    fn test_router(fixture: &ModelFixture) -> Router {
        router(Arc::new(test_state(fixture)))
    }

    async fn body_text(response: Response) -> String {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    async fn request(app: Router, method: &str, uri: &str, body: Option<Value>) -> Response {
        let mut builder = Request::builder().method(method).uri(uri);
        let body = if let Some(body) = body {
            builder = builder.header(header::CONTENT_TYPE, "application/json");
            Body::from(body.to_string())
        } else {
            Body::empty()
        };
        app.oneshot(builder.body(body).unwrap()).await.unwrap()
    }

    #[tokio::test]
    async fn only_initial_routes_are_registered() {
        let fixture = ModelFixture::new();
        let response = request(test_router(&fixture), "GET", "/health", None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let response = request(
            test_router(&fixture),
            "POST",
            "/v1/responses",
            Some(json!({})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let response = request(
            test_router(&fixture),
            "POST",
            "/v1/messages",
            Some(json!({})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn log_levels_map_to_rust_tracing_directives() {
        assert_eq!(tracing_level_directive("NOTSET").unwrap(), "trace");
        assert_eq!(tracing_level_directive("debug").unwrap(), "debug");
        assert_eq!(tracing_level_directive("INFO").unwrap(), "info");
        assert_eq!(tracing_level_directive("WARNING").unwrap(), "warn");
        assert_eq!(tracing_level_directive("CRITICAL").unwrap(), "error");
        assert!(tracing_level_directive("invalid").is_err());
    }

    #[tokio::test]
    async fn models_metrics_and_ppl_contracts() {
        let fixture = ModelFixture::new();
        let response = request(test_router(&fixture), "GET", "/v1/models", None).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(body_text(response).await.contains("test-model"));

        let response = request(test_router(&fixture), "GET", "/metrics", None).await;
        let metrics = body_text(response).await;
        assert!(metrics.contains("lmdeploy_turbomind_active_sequences 2"));
        assert!(metrics.contains("lmdeploy_turbomind_free_blocks 17"));

        let response = request(
            test_router(&fixture),
            "POST",
            "/get_ppl",
            Some(json!({"input": [1, 2]})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(body_text(response).await.contains("1.25"));
    }

    #[tokio::test]
    async fn chat_and_completion_support_json_and_sse() {
        let fixture = ModelFixture::new();
        let response = request(
            test_router(&fixture),
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}]
            })),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_text(response).await;
        assert!(body.contains("chat.completion"));
        assert!(body.contains("world !"));

        let response = request(
            test_router(&fixture),
            "POST",
            "/v1/completions",
            Some(json!({"model": "test-model", "prompt": "hello", "stream": true})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        let body = body_text(response).await;
        assert!(body.contains("text_completion"));
        assert!(body.contains("data: [DONE]"));
    }

    #[test]
    fn engine_error_is_sendable() {
        fn assert_send<T: Send>() {}
        assert_send::<EngineError>();
    }

    #[test]
    fn eos_ids_are_also_native_stop_ids() {
        let fixture = ModelFixture::new();
        let state = test_state(&fixture);
        let generation = build_generation(
            &state, 8, 0, None, None, None, None, None, None, None, false, 0,
        )
        .unwrap();
        assert_eq!(generation.eos_ids, vec![3]);
        assert_eq!(generation.stop_ids, vec![3]);

        let ignored = build_generation(
            &state, 8, 0, None, None, None, None, None, None, None, true, 0,
        )
        .unwrap();
        assert!(ignored.eos_ids.is_empty());
        assert!(ignored.stop_ids.is_empty());
    }
}
