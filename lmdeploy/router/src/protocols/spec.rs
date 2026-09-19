use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// # Protocol Specifications
//
// This module contains all protocol definitions for OpenAI-compatible APIs.
//
// ## Table of Contents
//
// 1. **OPENAI SPEC - Chat Completions API**
//    - Message Types
//    - Response Format Types
//    - Tool/Function Types
//    - Streaming Delta Types
//    - Request/Response structures
//
// 2. **OPENAI SPEC - Completions API**
//    - Request/Response structures
//    - Streaming support
//
// 3. **OPENAI SPEC - Responses API**
//    - Tool Definitions
//    - Reasoning Configuration
//    - Input/Output Items
//    - Service Tier & Tool Choice
//    - Request/Response structures
//
// 4. **OPENAI SPEC - Common**
//    - Shared Request Components
//    - Tool Choice Types
//    - Usage Tracking
//    - Logprobs Types
//    - Error Response Types
//
// 5. **LMDeploy SPEC - GENERATE API**
//    - Generate Parameters
//    - Sampling Parameters
//    - Request/Response structures
//
// 7. **OPENAI SPEC - Embeddings API**
//    - Request structures
//
// 8. **COMMON**
//    - GenerationRequest trait
//    - StringOrArray & LoRAPath types
//    - Helper functions

// ==================================================================
// =            OPENAI SPEC - Chat Completions API                  =
// ==================================================================

// ============= Message Types =============

// Note: We implement Deserialize manually below to dispatch based on the "role" field.
// This fixes a bug where serde's untagged enum matching would incorrectly match Assistant
// messages as System messages (because System only requires role + content, serde tries
// it first and silently ignores reasoning and other Assistant-specific fields).
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ChatMessage {
    System {
        role: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        role: String, // "user"
        content: UserMessageContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        role: String, // "assistant"
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        function_call: Option<FunctionCallResponse>,
        /// Reasoning content for reasoning models. The custom Deserialize impl
        /// below also accepts the deprecated `reasoning_content` alias.
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    Tool {
        role: String, // "tool"
        content: Value,
        tool_call_id: String,
    },
    Function {
        role: String, // "function"
        content: String,
        name: String,
    },
}

impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        let value = Value::deserialize(deserializer)?;
        let role = value
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or_else(|| D::Error::custom("missing role field"))?;

        match role {
            "assistant" => Ok(ChatMessage::Assistant {
                role: role.to_string(),
                content: value.get("content").and_then(|c| {
                    if c.is_null() {
                        None
                    } else {
                        c.as_str().map(String::from)
                    }
                }),
                name: value.get("name").and_then(|n| {
                    if n.is_null() {
                        None
                    } else {
                        n.as_str().map(String::from)
                    }
                }),
                tool_calls: value.get("tool_calls").and_then(|tc| {
                    if tc.is_null() {
                        None
                    } else {
                        serde_json::from_value(tc.clone()).ok()
                    }
                }),
                function_call: value.get("function_call").and_then(|fc| {
                    if fc.is_null() {
                        None
                    } else {
                        serde_json::from_value(fc.clone()).ok()
                    }
                }),
                // `reasoning` is the canonical field. Prefer a usable canonical
                // string when both keys are present; otherwise fall back to the
                // deprecated `reasoning_content` alias, including when `reasoning`
                // is null or not a string. Serialization remains canonical.
                reasoning: value
                    .get("reasoning")
                    .and_then(Value::as_str)
                    .or_else(|| value.get("reasoning_content").and_then(Value::as_str))
                    .map(String::from),
            }),
            "system" => Ok(ChatMessage::System {
                role: role.to_string(),
                content: value
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string(),
                name: value.get("name").and_then(|n| {
                    if n.is_null() {
                        None
                    } else {
                        n.as_str().map(String::from)
                    }
                }),
            }),
            "user" => {
                let content = value
                    .get("content")
                    .map(|c| {
                        serde_json::from_value(c.clone())
                            .unwrap_or(UserMessageContent::Text(String::new()))
                    })
                    .unwrap_or(UserMessageContent::Text(String::new()));
                Ok(ChatMessage::User {
                    role: role.to_string(),
                    content,
                    name: value.get("name").and_then(|n| {
                        if n.is_null() {
                            None
                        } else {
                            n.as_str().map(String::from)
                        }
                    }),
                })
            }
            "tool" => Ok(ChatMessage::Tool {
                role: role.to_string(),
                content: value
                    .get("content")
                    .cloned()
                    .unwrap_or_else(|| Value::String(String::new())),
                tool_call_id: value
                    .get("tool_call_id")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
            }),
            "function" => Ok(ChatMessage::Function {
                role: role.to_string(),
                content: value
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string(),
                name: value
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string(),
            }),
            _ => Err(D::Error::custom(format!("unknown role: {}", role))),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct StructuredOutputsParams {
    /// JSON schema for structured output (mutually exclusive with other constraints)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<serde_json::Value>,

    /// Regex pattern for structured output (mutually exclusive with other constraints)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,

    /// List of choices for structured output (mutually exclusive with other constraints)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice: Option<Vec<String>>,

    /// Grammar for structured output (mutually exclusive with other constraints)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,

    /// JSON object mode (mutually exclusive with other constraints)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_object: Option<bool>,

    /// Structural tag (mutually exclusive with other constraints)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structural_tag: Option<String>,

    /// Disable fallback to non-structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_fallback: Option<bool>,

    /// Disable any whitespace in structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_any_whitespace: Option<bool>,

    /// Disable additional properties in JSON schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_additional_properties: Option<bool>,

    /// Whitespace pattern for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub whitespace_pattern: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum UserMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "auto", "low", or "high"
}

// ============= Response Format Types =============

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonSchemaFormat {
    pub name: String,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// ============= Streaming Delta Types =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCallDelta>,
    /// Reasoning content delta for reasoning models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ============= Request =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    /// ID of the model to use (optional, LMDeploy supports requests without model)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// A list of messages comprising the conversation so far
    pub messages: Vec<ChatMessage>,

    /// Pre-tokenized input for LMDeploy token-in-token-out requests.
    /// This is used when `messages` is empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<Vec<i32>>,

    /// What sampling temperature to use, between 0 and 2
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// How many chat completion choices to generate for each input message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// If set, partial message deltas will be sent
    #[serde(default)]
    pub stream: bool,

    /// Options for streaming response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Up to 4 sequences where the API will stop generating further tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,

    /// The maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// An upper bound for the number of tokens that can be generated for a completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Modify the likelihood of specified tokens appearing in the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// A unique identifier representing your end-user
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// If specified, our system will make a best effort to sample deterministically
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Whether to return log probabilities of the output tokens
    #[serde(default)]
    pub logprobs: bool,

    /// An integer between 0 and 20 specifying the number of most likely tokens to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// An object specifying the format that the model must output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// A list of tools the model may call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Controls which (if any) tool is called by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Whether to enable parallel function calling during tool use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Deprecated: use tools instead
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<Function>>,

    /// Deprecated: use tool_choice instead
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,

    // ============= Generation Extensions =============
    /// Top-k sampling parameter (-1 to disable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Min-p nucleus sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,

    /// Minimum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,

    /// Repetition penalty for reducing repetitive text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    /// Regex constraint for output generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,

    /// EBNF grammar constraint for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ebnf: Option<String>,

    /// Specific token IDs to use as stop conditions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<i32>>,

    /// Skip trimming stop tokens from output
    #[serde(default)]
    pub no_stop_trim: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default)]
    pub ignore_eos: bool,

    /// Add generation prompt to the chat template
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,

    /// Continue generating from final assistant message
    #[serde(default)]
    pub continue_final_message: bool,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    // ============= Generation Extensions =============
    /// Separate reasoning content from final answer (O1-style models)
    #[serde(default = "default_true")]
    pub separate_reasoning: bool,

    /// Stream reasoning tokens during generation
    #[serde(default = "default_true")]
    pub stream_reasoning: bool,

    /// Chat template kwargs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<HashMap<String, serde_json::Value>>,

    /// Echo back the prompt in addition to the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,

    /// Reasoning effort level for reasoning models (low, medium, high)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,

    /// Whether to include reasoning in the response
    #[serde(default = "default_true")]
    pub include_reasoning: bool,
    /// Structured outputs parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_outputs: Option<StructuredOutputsParams>,

    /// Additional fields passed through transparently to the backend
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
}

impl GenerationRequest for ChatCompletionRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn extract_text_for_routing(&self) -> String {
        // LMDeploy fallback: when messages is empty, input_ids is the active input.
        // Use the typed field directly so consistent_hash/cache_aware receive a
        // deterministic routing key without reparsing flattened JSON.
        if let Some(input_ids) = self.input_ids.as_ref().filter(|ids| !ids.is_empty()) {
            return input_ids
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(" ");
        }

        // Return empty string if no routing key found - random/roundrobin/power_of_two
        // policies work without a key; consistent_hash/cache_aware will treat all as same bucket.
        String::new()
    }
}

// ============= Regular Response =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>, // "stop", "length", "tool_calls", "content_filter", "function_call"
    /// Information about which stop condition was matched
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<serde_json::Value>, // Can be string or integer
    /// Generated token IDs (LMDeploy extension, enabled by `return_token_ids`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_ids: Option<Vec<i32>>,
    /// Preserve LMDeploy extensions such as `output_token_logprobs` and
    /// `routed_experts`, as well as future backend-specific choice fields.
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
}

// ============= Streaming Response =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: String, // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    /// Generated token IDs for this stream chunk (LMDeploy extension).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_ids: Option<Vec<i32>>,
    /// Preserve LMDeploy extensions such as `output_token_logprobs` and
    /// `routed_experts`, as well as future backend-specific choice fields.
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
}

// ==================================================================
// =            OPENAI SPEC - Completions API                       =
// ==================================================================
// Completions API request types (v1/completions) - DEPRECATED but still supported

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionRequest {
    /// ID of the model to use (optional, LMDeploy supports requests without model)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// The prompt(s) to generate completions for
    /// Supports: str, list[str], list[int], list[list[int]]
    pub prompt: PromptInput,

    /// The suffix that comes after a completion of inserted text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    /// The maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// What sampling temperature to use, between 0 and 2
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature (nucleus sampling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// How many completions to generate for each prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Whether to stream back partial progress
    #[serde(default)]
    pub stream: bool,

    /// Options for streaming response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Include the log probabilities on the logprobs most likely tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,

    /// Echo back the prompt in addition to the completion
    #[serde(default)]
    pub echo: bool,

    /// Up to 4 sequences where the API will stop generating further tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Generates best_of completions server-side and returns the "best"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,

    /// Modify the likelihood of specified tokens appearing in the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// A unique identifier representing your end-user
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// If specified, our system will make a best effort to sample deterministically
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    // ============= Generation Extensions =============
    /// Top-k sampling parameter (-1 to disable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Min-p nucleus sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,

    /// Minimum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,

    /// Repetition penalty for reducing repetitive text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    /// Regex constraint for output generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,

    /// EBNF grammar constraint for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ebnf: Option<String>,

    /// JSON schema constraint for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<String>,

    /// Specific token IDs to use as stop conditions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<i32>>,

    /// Skip trimming stop tokens from output
    #[serde(default)]
    pub no_stop_trim: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default)]
    pub ignore_eos: bool,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Additional fields passed through transparently to the backend
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
}

impl GenerationRequest for CompletionRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn extract_text_for_routing(&self) -> String {
        if let Some(user) = self.user.as_deref() {
            return format!("user:{user}");
        }
        self.prompt.extract_text_for_routing()
    }
}

// ============= Regular Response =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String, // "text_completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>, // "stop", "length", "content_filter", etc.
    /// Information about which stop condition was matched
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<serde_json::Value>, // Can be string or integer
}

// ============= Streaming Response =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionStreamResponse {
    pub id: String,
    pub object: String, // "text_completion"
    pub created: u64,
    pub choices: Vec<CompletionStreamChoice>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionStreamChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
}

// ==================================================================
// =            OPENAI SPEC - Responses API                         =
// ==================================================================

// ============= Tool Definitions =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseTool {
    #[serde(rename = "type")]
    pub r#type: ResponseToolType,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseToolType {
    WebSearchPreview,
    CodeInterpreter,
}

// ============= Reasoning Configuration =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseReasoningParam {
    #[serde(default = "default_reasoning_effort")]
    pub effort: Option<ReasoningEffort>,
}

fn default_reasoning_effort() -> Option<ReasoningEffort> {
    Some(ReasoningEffort::Medium)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

// ============= Input/Output Items =============

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseInputOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        summary: Vec<String>,
        content: Vec<ResponseReasoningContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_tool_call")]
    FunctionToolCall {
        id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<ChatLogProbs>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseReasoningContent {
    #[serde(rename = "reasoning_text")]
    ReasoningText { text: String },
}

// ============= Output Items for Response =============

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        status: String,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        summary: Vec<String>,
        content: Vec<ResponseReasoningContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_tool_call")]
    FunctionToolCall {
        id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        status: String,
    },
}

// ============= Service Tier =============

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ServiceTier {
    #[default]
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

// ============= Truncation =============

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum Truncation {
    Auto,
    #[default]
    Disabled,
}

// ============= Response Status =============

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

// ============= Include Fields =============

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IncludeField {
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
    #[serde(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputImageUrl,
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageUrl,
    #[serde(rename = "message.output_text.logprobs")]
    MessageOutputTextLogprobs,
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
}

// ============= Usage Info =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptTokenUsageInfo {
    pub cached_tokens: u32,
}

// ============= Response Usage Format =============

/// OpenAI Responses API usage format (different from standard UsageInfo)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u32,
}

impl UsageInfo {
    /// Convert to OpenAI Responses API format
    pub fn to_response_usage(&self) -> ResponseUsage {
        ResponseUsage {
            input_tokens: self.prompt_tokens,
            output_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            input_tokens_details: self.prompt_tokens_details.as_ref().map(|details| {
                InputTokensDetails {
                    cached_tokens: details.cached_tokens,
                }
            }),
            output_tokens_details: self.reasoning_tokens.map(|tokens| OutputTokensDetails {
                reasoning_tokens: tokens,
            }),
        }
    }
}

impl From<UsageInfo> for ResponseUsage {
    fn from(usage: UsageInfo) -> Self {
        usage.to_response_usage()
    }
}

impl ResponseUsage {
    /// Convert back to standard UsageInfo format
    pub fn to_usage_info(&self) -> UsageInfo {
        UsageInfo {
            prompt_tokens: self.input_tokens,
            completion_tokens: self.output_tokens,
            total_tokens: self.total_tokens,
            reasoning_tokens: self
                .output_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens),
            prompt_tokens_details: self.input_tokens_details.as_ref().map(|details| {
                PromptTokenUsageInfo {
                    cached_tokens: details.cached_tokens,
                }
            }),
        }
    }
}

fn generate_request_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesRequest {
    // ============= Core OpenAI API fields =============
    /// Run the request in the background
    #[serde(default)]
    pub background: bool,

    /// Fields to include in the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeField>>,

    /// Input content - can be string or structured items
    pub input: ResponseInput,

    /// System instructions for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Maximum number of output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Maximum number of tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,

    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// Model to use (optional, LMDeploy supports requests without model)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Whether to enable parallel tool calls
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,

    /// ID of previous response to continue from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Reasoning configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoningParam>,

    /// Service tier
    #[serde(default)]
    pub service_tier: ServiceTier,

    /// Whether to store the response
    #[serde(default = "default_true")]
    pub store: bool,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Tool choice behavior
    #[serde(default)]
    pub tool_choice: ToolChoice,

    /// Available tools
    #[serde(default)]
    pub tools: Vec<ResponseTool>,

    /// Number of top logprobs to return
    #[serde(default)]
    pub top_logprobs: u32,

    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Truncation behavior
    #[serde(default)]
    pub truncation: Truncation,

    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    // ============= Generation Extensions =============
    /// Request ID
    #[serde(default = "generate_request_id")]
    pub request_id: String,

    /// Request priority
    #[serde(default)]
    pub priority: i32,

    /// Frequency penalty
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Presence penalty
    #[serde(default)]
    pub presence_penalty: f32,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,

    /// Top-k sampling parameter
    #[serde(default = "default_top_k")]
    pub top_k: i32,

    /// Min-p sampling parameter
    #[serde(default)]
    pub min_p: f32,

    /// Repetition penalty
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Items(Vec<ResponseInputOutputItem>),
}

fn default_top_k() -> i32 {
    -1
}

fn default_repetition_penalty() -> f32 {
    1.0
}

impl ResponsesRequest {
    /// Default sampling parameters
    const DEFAULT_TEMPERATURE: f32 = 0.7;
    const DEFAULT_TOP_P: f32 = 1.0;

    /// Convert to sampling parameters for generation
    pub fn to_sampling_params(
        &self,
        default_max_tokens: u32,
        default_params: Option<HashMap<String, serde_json::Value>>,
    ) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        // Use max_output_tokens if available
        let max_tokens = if let Some(max_output) = self.max_output_tokens {
            std::cmp::min(max_output, default_max_tokens)
        } else {
            default_max_tokens
        };

        // Avoid exceeding context length by minus 1 token
        let max_tokens = max_tokens.saturating_sub(1);

        // Temperature
        let temperature = self.temperature.unwrap_or_else(|| {
            default_params
                .as_ref()
                .and_then(|p| p.get("temperature"))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(Self::DEFAULT_TEMPERATURE)
        });

        // Top-p
        let top_p = self.top_p.unwrap_or_else(|| {
            default_params
                .as_ref()
                .and_then(|p| p.get("top_p"))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(Self::DEFAULT_TOP_P)
        });

        params.insert(
            "max_new_tokens".to_string(),
            serde_json::Value::Number(serde_json::Number::from(max_tokens)),
        );
        params.insert(
            "temperature".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(temperature as f64).unwrap()),
        );
        params.insert(
            "top_p".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(top_p as f64).unwrap()),
        );
        params.insert(
            "frequency_penalty".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.frequency_penalty as f64).unwrap(),
            ),
        );
        params.insert(
            "presence_penalty".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.presence_penalty as f64).unwrap(),
            ),
        );
        params.insert(
            "top_k".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.top_k)),
        );
        params.insert(
            "min_p".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(self.min_p as f64).unwrap()),
        );
        params.insert(
            "repetition_penalty".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.repetition_penalty as f64).unwrap(),
            ),
        );

        if let Some(ref stop) = self.stop {
            match serde_json::to_value(stop) {
                Ok(value) => params.insert("stop".to_string(), value),
                Err(_) => params.insert("stop".to_string(), serde_json::Value::Null),
            };
        }

        // Apply any additional default parameters
        if let Some(default_params) = default_params {
            for (key, value) in default_params {
                params.entry(key).or_insert(value);
            }
        }

        params
    }
}

impl GenerationRequest for ResponsesRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn extract_text_for_routing(&self) -> String {
        match &self.input {
            ResponseInput::Text(text) => text.clone(),
            ResponseInput::Items(items) => items
                .iter()
                .filter_map(|item| match item {
                    ResponseInputOutputItem::Message { content, .. } => {
                        let texts: Vec<String> = content
                            .iter()
                            .map(|part| match part {
                                ResponseContentPart::OutputText { text, .. } => text.clone(),
                            })
                            .collect();
                        if texts.is_empty() {
                            None
                        } else {
                            Some(texts.join(" "))
                        }
                    }
                    ResponseInputOutputItem::Reasoning { content, .. } => {
                        let texts: Vec<String> = content
                            .iter()
                            .map(|part| match part {
                                ResponseReasoningContent::ReasoningText { text } => text.clone(),
                            })
                            .collect();
                        if texts.is_empty() {
                            None
                        } else {
                            Some(texts.join(" "))
                        }
                    }
                    ResponseInputOutputItem::FunctionToolCall { arguments, .. } => {
                        Some(arguments.clone())
                    }
                })
                .collect::<Vec<String>>()
                .join(" "),
        }
    }
}

fn generate_response_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs() as i64
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesResponse {
    /// Response ID
    #[serde(default = "generate_response_id")]
    pub id: String,

    /// Object type
    #[serde(default = "default_object_type")]
    pub object: String,

    /// Creation timestamp
    #[serde(default = "current_timestamp")]
    pub created_at: i64,

    /// Model name
    pub model: String,

    /// Output items
    #[serde(default)]
    pub output: Vec<ResponseOutputItem>,

    /// Response status
    pub status: ResponseStatus,

    /// Usage information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,

    /// Whether parallel tool calls are enabled
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,

    /// Tool choice setting
    #[serde(default = "default_tool_choice")]
    pub tool_choice: String,

    /// Available tools
    #[serde(default)]
    pub tools: Vec<ResponseTool>,
}

fn default_object_type() -> String {
    "response".to_string()
}

fn default_tool_choice() -> String {
    "auto".to_string()
}

impl ResponsesResponse {
    /// Create a response from a request
    #[allow(clippy::too_many_arguments)]
    pub fn from_request(
        request: &ResponsesRequest,
        _sampling_params: &HashMap<String, serde_json::Value>,
        model_name: String,
        created_time: i64,
        output: Vec<ResponseOutputItem>,
        status: ResponseStatus,
        usage: Option<UsageInfo>,
    ) -> Self {
        Self {
            id: request.request_id.clone(),
            object: "response".to_string(),
            created_at: created_time,
            model: model_name,
            output,
            status,
            usage,
            parallel_tool_calls: request.parallel_tool_calls,
            tool_choice: match &request.tool_choice {
                ToolChoice::Value(ToolChoiceValue::Auto) => "auto".to_string(),
                ToolChoice::Value(ToolChoiceValue::Required) => "required".to_string(),
                ToolChoice::Value(ToolChoiceValue::None) => "none".to_string(),
                ToolChoice::Function { .. } => "function".to_string(),
            },
            tools: request.tools.clone(),
        }
    }

    /// Create a new response with default values
    pub fn new(request_id: String, model: String, status: ResponseStatus) -> Self {
        Self {
            id: request_id,
            object: "response".to_string(),
            created_at: current_timestamp(),
            model,
            output: Vec::new(),
            status,
            usage: None,
            parallel_tool_calls: true,
            tool_choice: "auto".to_string(),
            tools: Vec::new(),
        }
    }

    /// Add an output item to the response
    pub fn add_output(&mut self, item: ResponseOutputItem) {
        self.output.push(item);
    }

    /// Set the usage information
    pub fn set_usage(&mut self, usage: UsageInfo) {
        self.usage = Some(usage);
    }

    /// Update the status
    pub fn set_status(&mut self, status: ResponseStatus) {
        self.status = status;
    }

    /// Check if the response is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.status, ResponseStatus::Completed)
    }

    /// Check if the response is in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(self.status, ResponseStatus::InProgress)
    }

    /// Check if the response failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, ResponseStatus::Failed)
    }

    /// Check if the response was cancelled
    pub fn is_cancelled(&self) -> bool {
        matches!(self.status, ResponseStatus::Cancelled)
    }

    /// Check if the response is queued
    pub fn is_queued(&self) -> bool {
        matches!(self.status, ResponseStatus::Queued)
    }

    /// Convert usage to OpenAI Responses API format
    pub fn usage_in_response_format(&self) -> Option<ResponseUsage> {
        self.usage.as_ref().map(|usage| usage.to_response_usage())
    }

    /// Get the response as a JSON value with usage in response format
    pub fn to_response_format(&self) -> serde_json::Value {
        let mut response = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);

        // Convert usage to response format if present
        if let Some(usage) = &self.usage {
            if let Ok(usage_value) = serde_json::to_value(usage.to_response_usage()) {
                response["usage"] = usage_value;
            }
        }

        response
    }
}

// ============= Helper Functions =============

impl ResponseOutputItem {
    /// Create a new message output item
    pub fn new_message(
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        status: String,
    ) -> Self {
        Self::Message {
            id,
            role,
            content,
            status,
        }
    }

    /// Create a new reasoning output item
    pub fn new_reasoning(
        id: String,
        summary: Vec<String>,
        content: Vec<ResponseReasoningContent>,
        status: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            content,
            status,
        }
    }

    /// Create a new function tool call output item
    pub fn new_function_tool_call(
        id: String,
        name: String,
        arguments: String,
        output: Option<String>,
        status: String,
    ) -> Self {
        Self::FunctionToolCall {
            id,
            name,
            arguments,
            output,
            status,
        }
    }
}

impl ResponseContentPart {
    /// Create a new text content part
    pub fn new_text(
        text: String,
        annotations: Vec<String>,
        logprobs: Option<ChatLogProbs>,
    ) -> Self {
        Self::OutputText {
            text,
            annotations,
            logprobs,
        }
    }
}

impl ResponseReasoningContent {
    /// Create a new reasoning text content
    pub fn new_reasoning_text(text: String) -> Self {
        Self::ReasoningText { text }
    }
}

impl UsageInfo {
    /// Create a new usage info with token counts
    pub fn new(prompt_tokens: u32, completion_tokens: u32, reasoning_tokens: Option<u32>) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            reasoning_tokens,
            prompt_tokens_details: None,
        }
    }

    /// Create usage info with cached token details
    pub fn new_with_cached(
        prompt_tokens: u32,
        completion_tokens: u32,
        reasoning_tokens: Option<u32>,
        cached_tokens: u32,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            reasoning_tokens,
            prompt_tokens_details: Some(PromptTokenUsageInfo { cached_tokens }),
        }
    }
}

// ==================================================================
// =            OPENAI SPEC - Common                                =
// ==================================================================

// ============= Shared Request Components =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

// ============= Tool Choice Types =============

/// Tool choice value for simple string options
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceValue {
    Auto,
    Required,
    None,
}

/// Tool choice for both Chat Completion and Responses APIs
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Value(ToolChoiceValue),
    Function {
        #[serde(rename = "type")]
        tool_type: String, // "function"
        function: FunctionChoice,
    },
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Value(ToolChoiceValue::Auto)
    }
}

/// Function choice specification for ToolChoice::Function
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionChoice {
    pub name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: Function,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value, // JSON Schema
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: FunctionCallResponse,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum FunctionCall {
    None,
    Auto,
    Function { name: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallResponse {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<String>, // JSON string
}

// ============= Usage Tracking =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<u32>,
}

// ============= Logprobs Types =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
    pub text_offset: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatLogProbs {
    pub content: Option<Vec<ChatLogProbsContent>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatLogProbsContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogProb>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

// ==================================================================
// =            LMDeploy SPEC - GENERATE API                      =
// ==================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputIds {
    Single(Vec<i32>),
    Batch(Vec<Vec<i32>>),
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GenerateParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoder_input_details: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub do_sample: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_full_text: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub watermark: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SamplingParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_special_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ebnf: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_stop_trim: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// The prompt to generate from (OpenAI style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<StringOrArray>,

    /// Text input retained for clients that use compact prompt form
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Input IDs for tokenized input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<InputIds>,

    /// Generation parameters
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<GenerateParameters>,

    /// Sampling parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<SamplingParams>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Whether to return logprobs
    #[serde(default)]
    pub return_logprob: bool,

    /// Request ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<String>,

    /// Pass-through fields (e.g. max_tokens, temperature, stop, stop_token_ids,
    /// session_id, repetition_penalty, top_k, top_p, min_p, ignore_eos, etc.)
    /// for backends like lmdeploy whose /generate endpoint accepts these as
    /// top-level fields. Without this flatten map, unknown fields would be
    /// silently dropped during deserialization and lost when re-serializing
    /// the request to the backend.
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
}

/// LMDeploy `/generate` response metadata.
///
/// LMDeploy evolves this object independently of the router, so known token
/// accounting fields are typed while additional fields are retained verbatim.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateResponseMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_token_logprobs: Option<Vec<(f32, i32)>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routed_experts: Option<Value>,
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
}

/// LMDeploy `/generate` token-in-token-out response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text: String,
    pub output_ids: Vec<i32>,
    pub meta_info: GenerateResponseMeta,
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
}

impl GenerationRequest for GenerateRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        // Generate requests typically don't have a model field
        None
    }

    fn extract_text_for_routing(&self) -> String {
        // Check fields in priority order: text, prompt, inputs
        if let Some(ref text) = self.text {
            return text.clone();
        }

        if let Some(ref prompt) = self.prompt {
            return match prompt {
                StringOrArray::String(s) => s.clone(),
                StringOrArray::Array(v) => v.join(" "),
            };
        }

        if let Some(ref input_ids) = self.input_ids {
            return match input_ids {
                InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| id.to_string())
                    .collect::<Vec<String>>()
                    .join(" "),
                InputIds::Batch(batches) => batches
                    .iter()
                    .flat_map(|batch| batch.iter().map(|&id| id.to_string()))
                    .collect::<Vec<String>>()
                    .join(" "),
            };
        }

        // No text input found
        String::new()
    }
}

// ==================================================================
// =            OPENAI SPEC - Embeddings API                        =
// ==================================================================

/// Embeddings request compatible with OpenAI API
/// We intentionally keep fields flexible to pass through to workers.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingRequest {
    /// ID of the model to use (optional; some backends omit it)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Input can be a string, array of strings, tokens, or batch inputs
    pub input: serde_json::Value,

    /// Optional encoding format (e.g., "float", "base64")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,

    /// Optional user identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Optional number of dimensions for the embedding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,

    /// Request ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<String>,
}

impl GenerationRequest for EmbeddingRequest {
    fn is_stream(&self) -> bool {
        // Embeddings are non-streaming
        false
    }

    fn get_model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn extract_text_for_routing(&self) -> String {
        // Best effort: extract text content for routing decisions
        match &self.input {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            _ => String::new(),
        }
    }
}

// ==================================================================
// =            COMMON                                              =
// ==================================================================

/// Helper function for serde default value
pub fn default_true() -> bool {
    true
}

/// Common trait for all generation requests across different APIs
pub trait GenerationRequest: Send + Sync {
    /// Check if the request is for streaming
    fn is_stream(&self) -> bool;

    /// Get the model name if specified
    fn get_model(&self) -> Option<&str>;

    /// Extract text content for routing decisions
    fn extract_text_for_routing(&self) -> String;
}

/// Helper type for string or array of strings
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}
impl StringOrArray {
    /// Get the number of items in the StringOrArray
    pub fn len(&self) -> usize {
        match self {
            StringOrArray::String(_) => 1,
            StringOrArray::Array(arr) => arr.len(),
        }
    }

    /// Check if the StringOrArray is empty
    pub fn is_empty(&self) -> bool {
        match self {
            StringOrArray::String(s) => s.is_empty(),
            StringOrArray::Array(arr) => arr.is_empty(),
        }
    }

    /// Convert to a vector of strings
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            StringOrArray::String(s) => vec![s.clone()],
            StringOrArray::Array(arr) => arr.clone(),
        }
    }
}

/// Prompt input type supporting both text and token IDs
/// Compatible with the OpenAI prompt field format
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum PromptInput {
    /// Batch of token ID sequences: list[list[int]]
    /// This must come first due to serde untagged matching order
    IntBatch(Vec<Vec<i32>>),
    /// Array of strings: list[str]
    StringArray(Vec<String>),
    /// Single token ID sequence: list[int]
    IntArray(Vec<i32>),
    /// Single string: str
    String(String),
}

impl PromptInput {
    /// Get the number of items in the PromptInput
    pub fn len(&self) -> usize {
        match self {
            PromptInput::String(_) => 1,
            PromptInput::StringArray(arr) => arr.len(),
            PromptInput::IntArray(_) => 1,
            PromptInput::IntBatch(batch) => batch.len(),
        }
    }

    /// Check if the PromptInput is empty
    pub fn is_empty(&self) -> bool {
        match self {
            PromptInput::String(s) => s.is_empty(),
            PromptInput::StringArray(arr) => arr.is_empty(),
            PromptInput::IntArray(arr) => arr.is_empty(),
            PromptInput::IntBatch(batch) => batch.is_empty(),
        }
    }

    /// Extract text representation for routing decisions
    /// For token IDs, converts to a string representation
    pub fn extract_text_for_routing(&self) -> String {
        match self {
            PromptInput::String(s) => s.clone(),
            PromptInput::StringArray(arr) => arr.join(" "),
            PromptInput::IntArray(ids) => {
                // Convert token IDs to string representation for routing
                // Format: "token_ids:<count>" to indicate this is a token-based prompt
                format!("token_ids:{}", ids.len())
            }
            PromptInput::IntBatch(batches) => {
                // For batches, use total token count
                let total_tokens: usize = batches.iter().map(|b| b.len()).sum();
                format!("token_ids_batch:{}:{}", batches.len(), total_tokens)
            }
        }
    }

    /// Check if this prompt is token-based (not text)
    pub fn is_token_based(&self) -> bool {
        matches!(self, PromptInput::IntArray(_) | PromptInput::IntBatch(_))
    }

    /// Get the total token count if token-based, otherwise estimate from text
    pub fn estimated_token_count(&self) -> usize {
        match self {
            PromptInput::String(s) => {
                // Rough estimate: ~4 chars per token on average
                s.len() / 4
            }
            PromptInput::StringArray(arr) => {
                // Sum estimated tokens for all strings
                arr.iter().map(|s| s.len() / 4).sum()
            }
            PromptInput::IntArray(ids) => ids.len(),
            PromptInput::IntBatch(batches) => batches.iter().map(|b| b.len()).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================================================================
    // =            EMBEDDINGS REQUEST TESTS                             =
    // ==================================================================

    #[test]
    fn test_embedding_request_serialization_string_input() {
        let req = EmbeddingRequest {
            model: Some("test-emb".to_string()),
            input: serde_json::Value::String("hello".to_string()),
            encoding_format: Some("float".to_string()),
            user: Some("user-1".to_string()),
            dimensions: Some(128),
            rid: Some("rid-123".to_string()),
        };

        let serialized = serde_json::to_string(&req).unwrap();
        let deserialized: EmbeddingRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.model, req.model);
        assert_eq!(deserialized.input, req.input);
        assert_eq!(deserialized.encoding_format, req.encoding_format);
        assert_eq!(deserialized.user, req.user);
        assert_eq!(deserialized.dimensions, req.dimensions);
        assert_eq!(deserialized.rid, req.rid);
    }

    #[test]
    fn test_embedding_request_serialization_array_input() {
        let req = EmbeddingRequest {
            model: Some("test-emb".to_string()),
            input: serde_json::json!(["a", "b", "c"]),
            encoding_format: None,
            user: None,
            dimensions: None,
            rid: None,
        };

        let serialized = serde_json::to_string(&req).unwrap();
        let de: EmbeddingRequest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(de.model, req.model);
        assert_eq!(de.input, req.input);
    }

    #[test]
    fn test_embedding_generation_request_trait_string() {
        let req = EmbeddingRequest {
            model: Some("emb-model".to_string()),
            input: serde_json::Value::String("hello".to_string()),
            encoding_format: None,
            user: None,
            dimensions: None,
            rid: None,
        };
        assert!(!req.is_stream());
        assert_eq!(req.get_model(), Some("emb-model"));
        assert_eq!(req.extract_text_for_routing(), "hello");
    }

    #[test]
    fn test_embedding_generation_request_trait_array() {
        let req = EmbeddingRequest {
            model: Some("emb-model".to_string()),
            input: serde_json::json!(["hello", "world"]),
            encoding_format: None,
            user: None,
            dimensions: None,
            rid: None,
        };
        assert_eq!(req.extract_text_for_routing(), "hello world");
    }

    #[test]
    fn test_embedding_generation_request_trait_non_text() {
        let req = EmbeddingRequest {
            model: Some("emb-model".to_string()),
            input: serde_json::json!({"tokens": [1, 2, 3]}),
            encoding_format: None,
            user: None,
            dimensions: None,
            rid: None,
        };
        assert_eq!(req.extract_text_for_routing(), "");
    }

    #[test]
    fn test_embedding_generation_request_trait_mixed_array_ignores_nested() {
        let req = EmbeddingRequest {
            model: Some("emb-model".to_string()),
            input: serde_json::json!(["a", ["b", "c"], 123, {"k": "v"}]),
            encoding_format: None,
            user: None,
            dimensions: None,
            rid: None,
        };
        // Only top-level string elements are extracted
        assert_eq!(req.extract_text_for_routing(), "a");
    }

    // ==================================================================
    // =            OPTIONAL MODEL FIELD TESTS                           =
    // ==================================================================

    #[test]
    fn test_chat_completion_request_without_model() {
        let json = r#"{
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(request.model.is_none());
        assert_eq!(request.get_model(), None);
    }

    #[test]
    fn test_chat_completion_request_with_model() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model.as_deref(), Some("gpt-4"));
        assert_eq!(request.get_model(), Some("gpt-4"));
    }

    #[test]
    fn test_chat_completion_request_with_null_model() {
        let json = r#"{
            "model": null,
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(request.model.is_none());
        assert_eq!(request.get_model(), None);
    }

    #[test]
    fn test_chat_completion_request_without_model_roundtrip() {
        let json = r#"{
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&request).unwrap();
        // model should not appear in serialized output when None
        assert!(!serialized.contains("\"model\""));

        let roundtrip: ChatCompletionRequest = serde_json::from_str(&serialized).unwrap();
        assert!(roundtrip.model.is_none());
    }

    #[test]
    fn test_chat_completion_request_without_model_validates() {
        use crate::protocols::validation::ValidatableRequest;

        let json = r#"{
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(request.model.is_none());
        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_completion_request_without_model() {
        let json = r#"{
            "prompt": "Hello, world!"
        }"#;

        let request: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(request.model.is_none());
        assert_eq!(request.get_model(), None);
    }

    #[test]
    fn test_completion_request_with_model() {
        let json = r#"{
            "model": "text-davinci-003",
            "prompt": "Hello, world!"
        }"#;

        let request: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model.as_deref(), Some("text-davinci-003"));
        assert_eq!(request.get_model(), Some("text-davinci-003"));
    }

    #[test]
    fn test_completion_request_without_model_roundtrip() {
        let json = r#"{
            "prompt": "Hello, world!"
        }"#;

        let request: CompletionRequest = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&request).unwrap();
        assert!(!serialized.contains("\"model\""));

        let roundtrip: CompletionRequest = serde_json::from_str(&serialized).unwrap();
        assert!(roundtrip.model.is_none());
    }

    #[test]
    fn test_embedding_request_without_model() {
        let json = r#"{
            "input": "Hello, world!"
        }"#;

        let request: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert!(request.model.is_none());
        assert_eq!(request.get_model(), None);
    }

    #[test]
    fn test_embedding_request_with_model() {
        let json = r#"{
            "model": "text-embedding-ada-002",
            "input": "Hello, world!"
        }"#;

        let request: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model.as_deref(), Some("text-embedding-ada-002"));
        assert_eq!(request.get_model(), Some("text-embedding-ada-002"));
    }

    #[test]
    fn test_embedding_request_without_model_roundtrip() {
        let json = r#"{
            "input": "Hello, world!"
        }"#;

        let request: EmbeddingRequest = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&request).unwrap();
        assert!(!serialized.contains("\"model\""));

        let roundtrip: EmbeddingRequest = serde_json::from_str(&serialized).unwrap();
        assert!(roundtrip.model.is_none());
    }

    // ==================================================================
    // =            STRUCTURED OUTPUTS PARAMS TESTS                      =
    // ==================================================================

    #[test]
    fn test_structured_outputs_params_default() {
        let params = StructuredOutputsParams::default();

        assert!(params.json.is_none());
        assert!(params.regex.is_none());
        assert!(params.choice.is_none());
        assert!(params.grammar.is_none());
        assert!(params.json_object.is_none());
        assert!(params.structural_tag.is_none());
        assert!(params.disable_fallback.is_none());
        assert!(params.disable_any_whitespace.is_none());
        assert!(params.disable_additional_properties.is_none());
        assert!(params.whitespace_pattern.is_none());
    }

    #[test]
    fn test_structured_outputs_params_with_json_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let params = StructuredOutputsParams {
            json: Some(schema.clone()),
            ..Default::default()
        };

        assert_eq!(params.json, Some(schema));
        assert!(params.regex.is_none());
    }

    #[test]
    fn test_structured_outputs_params_with_regex() {
        let params = StructuredOutputsParams {
            regex: Some(r"^\d{3}-\d{2}-\d{4}$".to_string()),
            ..Default::default()
        };

        assert!(params.json.is_none());
        assert_eq!(params.regex, Some(r"^\d{3}-\d{2}-\d{4}$".to_string()));
    }

    #[test]
    fn test_structured_outputs_params_with_choice() {
        let choices = vec!["yes".to_string(), "no".to_string(), "maybe".to_string()];

        let params = StructuredOutputsParams {
            choice: Some(choices.clone()),
            ..Default::default()
        };

        assert_eq!(params.choice, Some(choices));
    }

    #[test]
    fn test_structured_outputs_params_with_grammar() {
        let grammar = r#"
            root ::= sentence
            sentence ::= subject " " verb " " object
            subject ::= "I" | "You" | "They"
            verb ::= "eat" | "drink" | "see"
            object ::= "apple" | "water" | "sky"
        "#
        .to_string();

        let params = StructuredOutputsParams {
            grammar: Some(grammar.clone()),
            ..Default::default()
        };

        assert_eq!(params.grammar, Some(grammar));
    }

    #[test]
    fn test_structured_outputs_params_with_json_object_mode() {
        let params = StructuredOutputsParams {
            json_object: Some(true),
            ..Default::default()
        };

        assert_eq!(params.json_object, Some(true));
    }

    #[test]
    fn test_structured_outputs_params_with_structural_tag() {
        let params = StructuredOutputsParams {
            structural_tag: Some("<output>".to_string()),
            ..Default::default()
        };

        assert_eq!(params.structural_tag, Some("<output>".to_string()));
    }

    #[test]
    fn test_structured_outputs_params_with_all_flags() {
        let params = StructuredOutputsParams {
            disable_fallback: Some(true),
            disable_any_whitespace: Some(true),
            disable_additional_properties: Some(false),
            whitespace_pattern: Some(r"\s*".to_string()),
            ..Default::default()
        };

        assert_eq!(params.disable_fallback, Some(true));
        assert_eq!(params.disable_any_whitespace, Some(true));
        assert_eq!(params.disable_additional_properties, Some(false));
        assert_eq!(params.whitespace_pattern, Some(r"\s*".to_string()));
    }

    #[test]
    fn test_structured_outputs_params_serialization_empty() {
        let params = StructuredOutputsParams::default();

        let serialized = serde_json::to_string(&params).unwrap();
        // All fields should be skipped when None
        assert_eq!(serialized, "{}");
    }

    #[test]
    fn test_structured_outputs_params_deserialization_partial() {
        let json = r#"{
            "regex": "^[a-z]+$",
            "disable_fallback": true
        }"#;

        let params: StructuredOutputsParams = serde_json::from_str(json).unwrap();

        assert!(params.json.is_none());
        assert_eq!(params.regex, Some("^[a-z]+$".to_string()));
        assert!(params.choice.is_none());
        assert!(params.grammar.is_none());
        assert!(params.json_object.is_none());
        assert!(params.structural_tag.is_none());
        assert_eq!(params.disable_fallback, Some(true));
        assert!(params.disable_any_whitespace.is_none());
        assert!(params.disable_additional_properties.is_none());
        assert!(params.whitespace_pattern.is_none());
    }

    #[test]
    fn test_structured_outputs_params_complex_json_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "email": {"type": "string", "format": "email"},
                        "age": {"type": "integer", "minimum": 0, "maximum": 150}
                    },
                    "required": ["name", "email"]
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "value": {"type": "number"}
                        }
                    }
                }
            },
            "required": ["user"]
        });

        let params = StructuredOutputsParams {
            json: Some(schema.clone()),
            disable_additional_properties: Some(true),
            ..Default::default()
        };

        let serialized = serde_json::to_string(&params).unwrap();
        let deserialized: StructuredOutputsParams = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.json, Some(schema));
        assert_eq!(deserialized.disable_additional_properties, Some(true));
    }

    #[test]
    fn test_structured_outputs_params_empty_choice_array() {
        let params = StructuredOutputsParams {
            choice: Some(vec![]),
            ..Default::default()
        };

        let serialized = serde_json::to_string(&params).unwrap();
        let deserialized: StructuredOutputsParams = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.choice, Some(vec![]));
    }

    #[test]
    fn test_structured_outputs_params_in_chat_completion_request() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Generate JSON"}],
            "structured_outputs": {
                "json": {"type": "object"},
                "disable_fallback": true
            }
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        assert!(request.structured_outputs.is_some());
        let structured_outputs = request.structured_outputs.unwrap();
        assert!(structured_outputs.json.is_some());
        assert_eq!(structured_outputs.disable_fallback, Some(true));
    }

    #[test]
    fn test_structured_outputs_params_in_chat_completion_request_none() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        assert!(request.structured_outputs.is_none());
    }

    // ==================================================================
    // =            CHAT MESSAGE DESERIALIZATION TESTS                   =
    // ==================================================================

    #[test]
    fn test_chat_message_assistant_with_reasoning() {
        let json = r#"{
            "role": "assistant",
            "content": "Hello there!",
            "reasoning": "Let me think about how to greet the user..."
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();

        match message {
            ChatMessage::Assistant {
                content, reasoning, ..
            } => {
                assert_eq!(content.as_ref().unwrap(), "Hello there!");
                assert_eq!(
                    reasoning.as_ref().unwrap(),
                    "Let me think about how to greet the user..."
                );
            }
            other => panic!(
                "Expected Assistant message but got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn test_chat_message_assistant_with_null_content_and_reasoning() {
        let json = r#"{
            "role": "assistant",
            "content": null,
            "reasoning": "Deep thinking in progress..."
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();

        match message {
            ChatMessage::Assistant {
                content, reasoning, ..
            } => {
                assert!(content.is_none());
                assert_eq!(reasoning.as_ref().unwrap(), "Deep thinking in progress...");
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_chat_message_assistant_with_tool_calls() {
        let json = r#"{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Boston\"}"
                }
            }]
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();

        match message {
            ChatMessage::Assistant { tool_calls, .. } => {
                let calls = tool_calls.unwrap();
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_abc123");
                assert_eq!(calls[0].function.name, "get_weather");
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_chat_message_system() {
        let json = r#"{
            "role": "system",
            "content": "You are a helpful assistant."
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();

        match message {
            ChatMessage::System { content, .. } => {
                assert_eq!(content, "You are a helpful assistant.");
            }
            _ => panic!("Expected System message"),
        }
    }

    #[test]
    fn test_chat_message_user_text() {
        let json = r#"{
            "role": "user",
            "content": "Hello!"
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();

        match message {
            ChatMessage::User { content, .. } => match content {
                UserMessageContent::Text(text) => assert_eq!(text, "Hello!"),
                _ => panic!("Expected Text content"),
            },
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_chat_message_tool() {
        let json = r#"{
            "role": "tool",
            "content": "Tool result here",
            "tool_call_id": "call_123"
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();

        match message {
            ChatMessage::Tool {
                content,
                tool_call_id,
                ..
            } => {
                assert_eq!(content, Value::String("Tool result here".to_string()));
                assert_eq!(tool_call_id, "call_123");
            }
            _ => panic!("Expected Tool message"),
        }
    }

    #[test]
    fn test_chat_message_tool_preserves_array_content() {
        let json = r#"{
            "role": "tool",
            "content": [{"type": "text", "text": "1 files found: ['wi_example.docx']"}],
            "tool_call_id": "call_456"
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();

        match message {
            ChatMessage::Tool {
                content,
                tool_call_id,
                ..
            } => {
                let expected = serde_json::json!([
                    {"type": "text", "text": "1 files found: ['wi_example.docx']"}
                ]);
                assert_eq!(content, expected);
                assert_eq!(tool_call_id, "call_456");
            }
            _ => panic!("Expected Tool message"),
        }
    }

    #[test]
    fn test_chat_message_function() {
        let json = r#"{
            "role": "function",
            "content": "Function result",
            "name": "my_function"
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();

        match message {
            ChatMessage::Function { content, name, .. } => {
                assert_eq!(content, "Function result");
                assert_eq!(name, "my_function");
            }
            _ => panic!("Expected Function message"),
        }
    }

    #[test]
    fn test_chat_message_roundtrip_serialization() {
        let original = ChatMessage::Assistant {
            role: "assistant".to_string(),
            content: Some("Hello!".to_string()),
            name: None,
            tool_calls: None,
            function_call: None,
            reasoning: Some("Thinking...".to_string()),
        };

        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: ChatMessage = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            ChatMessage::Assistant {
                content, reasoning, ..
            } => {
                assert_eq!(content.as_ref().unwrap(), "Hello!");
                assert_eq!(reasoning.as_ref().unwrap(), "Thinking...");
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_chat_message_assistant_reasoning_content_alias_roundtrip() {
        // Regression: assistant turns using the deprecated `reasoning_content`
        // alias must survive forwarding and normalize to canonical `reasoning`.
        let json = r#"{
            "role": "assistant",
            "content": "The answer is 42.",
            "reasoning_content": "First I considered X, then Y..."
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();
        match &message {
            ChatMessage::Assistant { reasoning, .. } => {
                assert_eq!(
                    reasoning.as_deref(),
                    Some("First I considered X, then Y...")
                );
            }
            _ => panic!("Expected Assistant message"),
        }

        // Re-serialization emits only the canonical field.
        let serialized = serde_json::to_value(&message).unwrap();
        assert_eq!(
            serialized.get("reasoning").and_then(|v| v.as_str()),
            Some("First I considered X, then Y..."),
        );
        assert!(serialized.get("reasoning_content").is_none());

        let reparsed: ChatMessage = serde_json::from_value(serialized).unwrap();
        match reparsed {
            ChatMessage::Assistant { reasoning, .. } => {
                assert_eq!(
                    reasoning.as_deref(),
                    Some("First I considered X, then Y...")
                );
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_chat_message_assistant_reasoning_alias_precedence() {
        let cases = [
            (
                r#"{
                    "role": "assistant",
                    "reasoning": "canonical",
                    "reasoning_content": "deprecated alias"
                }"#,
                "canonical",
            ),
            (
                r#"{
                    "role": "assistant",
                    "reasoning": null,
                    "reasoning_content": "deprecated alias"
                }"#,
                "deprecated alias",
            ),
            (
                r#"{
                    "role": "assistant",
                    "reasoning": 42,
                    "reasoning_content": "deprecated alias"
                }"#,
                "deprecated alias",
            ),
        ];

        for (json, expected) in cases {
            let message: ChatMessage = serde_json::from_str(json).unwrap();
            match message {
                ChatMessage::Assistant { reasoning, .. } => {
                    assert_eq!(reasoning.as_deref(), Some(expected));
                }
                _ => panic!("Expected Assistant message"),
            }
        }
    }

    // ===== lmdeploy token-in-token-out routing tests =====

    #[test]
    fn test_chat_completion_extract_routing_with_input_ids() {
        // Build a ChatCompletionRequest via JSON deserialization (lmdeploy-style:
        // empty messages + input_ids fallback).
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [],
            "input_ids": [151644, 8948, 198, 2610],
        });
        let req: ChatCompletionRequest =
            serde_json::from_value(body).expect("deserialize chat request");
        assert_eq!(req.extract_text_for_routing(), "151644 8948 198 2610");
    }

    #[test]
    fn test_chat_completion_extract_routing_ignores_wrapped_session_id() {
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [],
            "input_ids": [1, 2, 3],
            "session_id": "sess-abc"
        });
        let req: ChatCompletionRequest =
            serde_json::from_value(body).expect("deserialize chat request");
        assert_eq!(req.extract_text_for_routing(), "1 2 3");
    }

    #[test]
    fn test_chat_completion_extract_routing_empty_when_no_keys() {
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        });
        let req: ChatCompletionRequest =
            serde_json::from_value(body).expect("deserialize chat request");
        assert_eq!(req.extract_text_for_routing(), "");
    }

    #[test]
    fn test_chat_completion_rejects_non_lmdeploy_input_ids_shapes() {
        for input_ids in [
            serde_json::json!([[10, 20], [30, 40]]),
            serde_json::json!([10, "invalid", 20]),
            serde_json::json!("not-an-array"),
        ] {
            let body = serde_json::json!({
                "model": "test-model",
                "messages": [],
                "input_ids": input_ids,
            });
            assert!(serde_json::from_value::<ChatCompletionRequest>(body).is_err());
        }
    }

    #[test]
    fn test_chat_completion_preserves_typed_input_ids_on_serialize() {
        // Ensure input_ids survives round-trip serialization (forwarded to backend).
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [],
            "input_ids": [1, 2, 3],
            "max_tokens": 16,
        });
        let req: ChatCompletionRequest =
            serde_json::from_value(body).expect("deserialize chat request");
        assert_eq!(req.input_ids, Some(vec![1, 2, 3]));
        assert!(!req.other.contains_key("input_ids"));
        let out = serde_json::to_value(&req).expect("serialize chat request");
        assert_eq!(out["input_ids"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_chat_completion_preserves_all_lmdeploy_token_input_fields() {
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [],
            "input_ids": [1, 2, 3],
            "image_data": ["data:image/png;base64,AAAA"],
            "do_preprocess": false,
            "return_token_ids": true,
        });
        let req: ChatCompletionRequest =
            serde_json::from_value(body.clone()).expect("deserialize chat request");
        let out = serde_json::to_value(&req).expect("serialize chat request");
        for field in [
            "input_ids",
            "image_data",
            "do_preprocess",
            "return_token_ids",
        ] {
            assert_eq!(out[field], body[field], "field {field} must survive");
        }
    }

    #[test]
    fn test_generate_request_preserves_lmdeploy_token_input_fields() {
        let body = serde_json::json!({
            "input_ids": [151644, 8948, 198],
            "session_id": 42,
            "max_tokens": 16,
            "return_logprob": true,
            "stream": false,
        });
        let req: GenerateRequest =
            serde_json::from_value(body.clone()).expect("deserialize generate request");
        assert_eq!(req.extract_text_for_routing(), "151644 8948 198");

        let out = serde_json::to_value(&req).expect("serialize generate request");
        for field in ["input_ids", "session_id", "max_tokens", "return_logprob"] {
            assert_eq!(out[field], body[field], "field {field} must survive");
        }
    }

    #[test]
    fn test_lmdeploy_chat_response_preserves_output_ids() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
                "output_ids": [100, 101],
                "output_token_logprobs": [[-0.1, 100]],
                "routed_experts": [[[1, 2]]]
            }],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5
            }
        });
        let response: ChatCompletionResponse =
            serde_json::from_value(body.clone()).expect("deserialize chat response");
        assert_eq!(response.choices[0].output_ids, Some(vec![100, 101]));
        assert!(response.choices[0]
            .other
            .contains_key("output_token_logprobs"));
        assert!(response.choices[0].other.contains_key("routed_experts"));

        let out = serde_json::to_value(response).expect("serialize chat response");
        assert_eq!(
            out["choices"][0]["output_ids"],
            body["choices"][0]["output_ids"]
        );
        assert_eq!(
            out["choices"][0]["output_token_logprobs"],
            body["choices"][0]["output_token_logprobs"]
        );
    }

    #[test]
    fn test_lmdeploy_chat_stream_response_preserves_output_ids() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "ok"},
                "finish_reason": null,
                "output_ids": [100, 101]
            }]
        });
        let response: ChatCompletionStreamResponse =
            serde_json::from_value(body).expect("deserialize chat stream response");
        assert_eq!(response.choices[0].output_ids, Some(vec![100, 101]));
    }

    #[test]
    fn test_lmdeploy_generate_response_preserves_output_ids() {
        let body = serde_json::json!({
            "text": "ok",
            "output_ids": [100, 101],
            "meta_info": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 100]],
                "first_token_latency": 0.01
            }
        });
        let response: GenerateResponse =
            serde_json::from_value(body.clone()).expect("deserialize generate response");
        assert_eq!(response.output_ids, vec![100, 101]);
        assert_eq!(response.meta_info.prompt_tokens, Some(3));
        assert!(response.meta_info.other.contains_key("first_token_latency"));

        let out = serde_json::to_value(response).expect("serialize generate response");
        assert_eq!(out["output_ids"], body["output_ids"]);
        assert_eq!(out["meta_info"]["first_token_latency"], 0.01);
    }
}
