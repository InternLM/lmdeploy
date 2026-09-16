use crate::utils::json;
use bytes::Bytes;
use http_body_util::Full;
use hyper::StatusCode;
use hyper::body::Incoming;
use hyper::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Request = hyper::Request<Incoming>;
pub type Response = hyper::Response<Full<Bytes>>;

// Server failed to start up due to bad config
#[derive(Debug, thiserror::Error)]
#[error("Configuration error: {0}")]
pub struct ConfigurationError(String);

// Request-specific errors
#[derive(Debug)]
pub enum AppError {
    BadRequest(String),
    NotFound(String),
    MethodNotAllowed(String),
    InternalError(String),
}

impl AppError {
    pub fn to_response(self) -> Response {
        let (status, message) = match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::MethodNotAllowed(msg) => (StatusCode::METHOD_NOT_ALLOWED, msg),
            AppError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = serde_json::json!({
            "error": message,
            "status": status.as_u16()
        });

        hyper::Response::builder()
            .status(status)
            .header(CONTENT_TYPE, "application/json")
            .body(Full::new(Bytes::from(body.to_string())))
            .unwrap()
    }
}

impl From<reqwest::Error> for AppError {
    fn from(error: reqwest::Error) -> Self {
        AppError::BadRequest(format!("Bad request: {}", error))
    }
}

impl From<json::RequireError> for ConfigurationError {
    fn from(error: json::RequireError) -> Self {
        ConfigurationError(format!("Failed to configure router: {}", error))
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub rest: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    #[serde(flatten)]
    pub rest: HashMap<String, serde_json::Value>,
}
