// Copyright (c) OpenMMLab. All rights reserved.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

fn default_one() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    #[serde(default = "model_object")]
    pub object: String,
    #[serde(default = "lmdeploy_owner")]
    pub owned_by: String,
}

fn model_object() -> String {
    "model".into()
}

fn lmdeploy_owner() -> String {
    "lmdeploy".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    #[serde(default = "list_object")]
    pub object: String,
    pub data: Vec<ModelCard>,
}

fn list_object() -> String {
    "list".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: Value,
}

fn function_type() -> String {
    "function".into()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub max_completion_tokens: Option<i32>,
    #[serde(default)]
    pub max_tokens: Option<i32>,
    #[serde(default)]
    pub min_new_tokens: Option<i32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<OneOrMany<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default = "default_one")]
    pub n: usize,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
    #[serde(default)]
    pub session_id: Option<u64>,
    #[serde(default)]
    pub logprobs: bool,
    #[serde(default)]
    pub top_logprobs: Option<i32>,
    #[serde(default)]
    pub return_token_ids: bool,
    #[serde(default)]
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: OneOrMany<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub max_completion_tokens: Option<i32>,
    #[serde(default)]
    pub max_tokens: Option<i32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<OneOrMany<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default = "default_one")]
    pub n: usize,
    #[serde(default)]
    pub echo: bool,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
    #[serde(default)]
    pub session_id: Option<u64>,
    #[serde(default)]
    pub logprobs: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PplRequest {
    pub input: PplInput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PplInput {
    Text(String),
    TokenIds(Vec<i32>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PplResponse {
    pub ppl: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub error: OpenAiError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiError {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub code: i32,
    pub param: Option<String>,
}
