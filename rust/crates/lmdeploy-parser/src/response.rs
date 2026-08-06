// Copyright (c) OpenMMLab. All rights reserved.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use uuid::Uuid;

pub const SUPPORTED_TOOL_PARSERS: &[&str] = &[
    "qwen",
    "qwen2d5",
    "qwen3",
    "qwen3coder",
    "llama3",
    "internlm",
    "intern-s1",
    "interns2-preview",
    "glm47",
    "deepseek-v32",
    "deepseek-v3.2",
    "deepseek-v4",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantEvent {
    Content {
        text: String,
    },
    Reasoning {
        text: String,
    },
    ToolStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolArguments {
        index: usize,
        arguments: String,
    },
    ToolEnd {
        index: usize,
    },
}

#[derive(Debug, Clone, Default)]
pub struct ParserConfig {
    pub reasoning_open_tag: Option<String>,
    pub reasoning_close_tag: Option<String>,
    pub starts_in_reasoning: bool,
    pub tool_parser: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Content,
    Reasoning,
    Tool,
}

#[derive(Debug, Clone, Copy)]
enum ToolFormat {
    Json,
    QwenXml,
    GlmXml,
    Dsml,
}

#[derive(Debug, Clone)]
struct ToolProfile {
    open: String,
    close: Option<String>,
    format: ToolFormat,
}

/// Protocol-neutral incremental response parser.
///
/// Tag prefixes are buffered across engine chunks. Tool payloads are held
/// until structurally complete, then surfaced as OpenAI-neutral tool events.
pub struct ResponseParser {
    config: ParserConfig,
    tool: Option<ToolProfile>,
    pending: String,
    tool_payload: String,
    mode: Mode,
    next_tool_index: usize,
}

impl ResponseParser {
    pub fn new(config: ParserConfig) -> Self {
        let mode = if config.starts_in_reasoning {
            Mode::Reasoning
        } else {
            Mode::Content
        };
        let tool = config.tool_parser.as_deref().and_then(tool_profile);
        Self {
            config,
            tool,
            pending: String::new(),
            tool_payload: String::new(),
            mode,
            next_tool_index: 0,
        }
    }

    pub fn push(&mut self, chunk: &str) -> Vec<AssistantEvent> {
        self.pending.push_str(chunk);
        let mut output = Vec::new();
        loop {
            let delimiters = self.active_delimiters();
            if delimiters.is_empty() {
                self.emit_committed(self.pending.len(), &mut output);
                break;
            }
            if let Some((index, delimiter, target)) = earliest_delimiter(&self.pending, &delimiters)
            {
                self.emit_committed(index, &mut output);
                self.pending.drain(..delimiter.len());
                match (self.mode, target) {
                    (Mode::Content, Mode::Tool) => {
                        self.mode = Mode::Tool;
                        self.tool_payload.clear();
                    }
                    (Mode::Tool, Mode::Content) => {
                        self.finish_tool(&mut output);
                        self.mode = Mode::Content;
                    }
                    (_, target) => self.mode = target,
                }
                continue;
            }
            if self.mode == Mode::Tool
                && self.tool.as_ref().is_some_and(|tool| tool.close.is_none())
            {
                break;
            }
            let keep = delimiters
                .iter()
                .map(|(delimiter, _)| longest_prefix_suffix(&self.pending, delimiter))
                .max()
                .unwrap_or(0);
            self.emit_committed(self.pending.len() - keep, &mut output);
            break;
        }
        output
    }

    pub fn finish(&mut self) -> Vec<AssistantEvent> {
        let mut output = Vec::new();
        self.emit_committed(self.pending.len(), &mut output);
        if self.mode == Mode::Tool {
            self.finish_tool(&mut output);
            self.mode = Mode::Content;
        }
        output
    }

    fn active_delimiters(&self) -> Vec<(String, Mode)> {
        match self.mode {
            Mode::Reasoning => self
                .config
                .reasoning_close_tag
                .clone()
                .map(|tag| vec![(tag, Mode::Content)])
                .unwrap_or_default(),
            Mode::Tool => self
                .tool
                .as_ref()
                .and_then(|tool| tool.close.clone())
                .map(|tag| vec![(tag, Mode::Content)])
                .unwrap_or_default(),
            Mode::Content => {
                let mut delimiters = Vec::new();
                if let Some(tag) = self.config.reasoning_open_tag.clone() {
                    delimiters.push((tag, Mode::Reasoning));
                }
                if let Some(tool) = &self.tool {
                    delimiters.push((tool.open.clone(), Mode::Tool));
                }
                delimiters
            }
        }
    }

    fn emit_committed(&mut self, bytes: usize, output: &mut Vec<AssistantEvent>) {
        if bytes == 0 {
            return;
        }
        let text: String = self.pending.drain(..bytes).collect();
        match self.mode {
            Mode::Content => output.push(AssistantEvent::Content { text }),
            Mode::Reasoning => output.push(AssistantEvent::Reasoning { text }),
            Mode::Tool => self.tool_payload.push_str(&text),
        }
    }

    fn finish_tool(&mut self, output: &mut Vec<AssistantEvent>) {
        let Some(profile) = &self.tool else {
            return;
        };
        let calls = parse_tool_payload(profile.format, self.tool_payload.trim());
        for (name, arguments) in calls {
            let index = self.next_tool_index;
            self.next_tool_index += 1;
            output.push(AssistantEvent::ToolStart {
                index,
                id: format!("chatcmpl-tool-{}", Uuid::new_v4().simple()),
                name,
            });
            output.push(AssistantEvent::ToolArguments { index, arguments });
            output.push(AssistantEvent::ToolEnd { index });
        }
        self.tool_payload.clear();
    }
}

fn tool_profile(name: &str) -> Option<ToolProfile> {
    let profile = match name {
        "qwen" | "qwen2d5" | "qwen3" => ToolProfile {
            open: "<tool_call>".into(),
            close: Some("</tool_call>".into()),
            format: ToolFormat::Json,
        },
        "llama3" => ToolProfile {
            open: "<|python_tag|>".into(),
            close: None,
            format: ToolFormat::Json,
        },
        "internlm" | "intern-s1" => ToolProfile {
            open: "<|action_start|><|plugin|>".into(),
            close: Some("<|action_end|>".into()),
            format: ToolFormat::Json,
        },
        "qwen3coder" | "interns2-preview" => ToolProfile {
            open: "<tool_call>".into(),
            close: Some("</tool_call>".into()),
            format: ToolFormat::QwenXml,
        },
        "glm47" => ToolProfile {
            open: "<tool_call>".into(),
            close: Some("</tool_call>".into()),
            format: ToolFormat::GlmXml,
        },
        "deepseek-v32" | "deepseek-v3.2" => ToolProfile {
            open: "\n\n<｜DSML｜function_calls>".into(),
            close: Some("</｜DSML｜function_calls>".into()),
            format: ToolFormat::Dsml,
        },
        "deepseek-v4" => ToolProfile {
            open: "\n\n<｜DSML｜tool_calls>".into(),
            close: Some("</｜DSML｜tool_calls>".into()),
            format: ToolFormat::Dsml,
        },
        _ => return None,
    };
    Some(profile)
}

fn parse_tool_payload(format: ToolFormat, payload: &str) -> Vec<(String, String)> {
    match format {
        ToolFormat::Json => parse_json_tool(payload).into_iter().collect(),
        ToolFormat::QwenXml => parse_qwen_xml(payload).into_iter().collect(),
        ToolFormat::GlmXml => parse_glm_xml(payload).into_iter().collect(),
        ToolFormat::Dsml => parse_dsml(payload),
    }
}

fn parse_json_tool(payload: &str) -> Option<(String, String)> {
    let object = serde_json::from_str::<Value>(payload)
        .ok()?
        .as_object()?
        .clone();
    let name = object.get("name")?.as_str()?.to_owned();
    let arguments = object
        .get("arguments")
        .or_else(|| object.get("parameters"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let arguments = arguments
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| arguments.to_string());
    Some((name, arguments))
}

fn parse_qwen_xml(payload: &str) -> Option<(String, String)> {
    let name_start = payload.find("<function=")? + "<function=".len();
    let name_end = payload[name_start..].find('>')? + name_start;
    let name = payload[name_start..name_end].trim().to_owned();
    let mut arguments = Map::new();
    let mut offset = name_end + 1;
    while let Some(relative) = payload[offset..].find("<parameter=") {
        let key_start = offset + relative + "<parameter=".len();
        let key_end = payload[key_start..].find('>')? + key_start;
        let value_start = key_end + 1;
        let value_end = payload[value_start..].find("</parameter>")? + value_start;
        let raw = payload[value_start..value_end].trim();
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.into()));
        arguments.insert(payload[key_start..key_end].trim().into(), value);
        offset = value_end + "</parameter>".len();
    }
    Some((name, Value::Object(arguments).to_string()))
}

fn parse_glm_xml(payload: &str) -> Option<(String, String)> {
    let args_start = payload.find("<arg_key>").unwrap_or(payload.len());
    let name = payload[..args_start].trim().to_owned();
    if name.is_empty() {
        return None;
    }
    let mut arguments = Map::new();
    let mut offset = args_start;
    while let Some(relative) = payload[offset..].find("<arg_key>") {
        let key_start = offset + relative + "<arg_key>".len();
        let key_end = payload[key_start..].find("</arg_key>")? + key_start;
        let value_tag = payload[key_end + "</arg_key>".len()..].find("<arg_value>")?
            + key_end
            + "</arg_key>".len();
        let value_start = value_tag + "<arg_value>".len();
        let value_end = payload[value_start..].find("</arg_value>")? + value_start;
        let raw = &payload[value_start..value_end];
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.into()));
        arguments.insert(payload[key_start..key_end].trim().into(), value);
        offset = value_end + "</arg_value>".len();
    }
    Some((name, Value::Object(arguments).to_string()))
}

fn parse_dsml(payload: &str) -> Vec<(String, String)> {
    let mut calls = Vec::new();
    let mut offset = 0;
    while let Some(relative) = payload[offset..].find("<｜DSML｜invoke name=\"") {
        let name_start = offset + relative + "<｜DSML｜invoke name=\"".len();
        let Some(name_end) = payload[name_start..]
            .find("\">")
            .map(|value| value + name_start)
        else {
            break;
        };
        let Some(invoke_end) = payload[name_end + 2..]
            .find("</｜DSML｜invoke>")
            .map(|value| value + name_end + 2)
        else {
            break;
        };
        let body = &payload[name_end + 2..invoke_end];
        let mut arguments = Map::new();
        let mut body_offset = 0;
        while let Some(relative) = body[body_offset..].find("<｜DSML｜parameter name=\"") {
            let key_start = body_offset + relative + "<｜DSML｜parameter name=\"".len();
            let Some(key_end) = body[key_start..].find('"').map(|value| value + key_start) else {
                break;
            };
            let header_end = body[key_end..]
                .find('>')
                .map(|value| value + key_end)
                .unwrap_or(key_end);
            let string_value = body[key_end..header_end].contains("string=\"true\"");
            let value_start = header_end + 1;
            let Some(value_end) = body[value_start..]
                .find("</｜DSML｜parameter>")
                .map(|value| value + value_start)
            else {
                break;
            };
            let raw = &body[value_start..value_end];
            let value = if string_value {
                Value::String(raw.into())
            } else {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.into()))
            };
            arguments.insert(body[key_start..key_end].into(), value);
            body_offset = value_end + "</｜DSML｜parameter>".len();
        }
        calls.push((
            payload[name_start..name_end].into(),
            Value::Object(arguments).to_string(),
        ));
        offset = invoke_end + "</｜DSML｜invoke>".len();
    }
    calls
}

fn earliest_delimiter<'a>(
    text: &str,
    delimiters: &'a [(String, Mode)],
) -> Option<(usize, &'a str, Mode)> {
    delimiters
        .iter()
        .filter_map(|(delimiter, target)| {
            text.find(delimiter)
                .map(|index| (index, delimiter.as_str(), *target))
        })
        .min_by_key(|(index, _, _)| *index)
}

fn longest_prefix_suffix(text: &str, delimiter: &str) -> usize {
    let max = text.len().min(delimiter.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&length| {
            text.is_char_boundary(text.len() - length)
                && delimiter.is_char_boundary(length)
                && text[text.len() - length..] == delimiter[..length]
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_tags_are_safe_across_chunk_boundaries() {
        let mut parser = ResponseParser::new(ParserConfig {
            reasoning_open_tag: Some("<think>".into()),
            reasoning_close_tag: Some("</think>".into()),
            starts_in_reasoning: false,
            ..Default::default()
        });
        assert_eq!(
            parser.push("before<th"),
            vec![AssistantEvent::Content {
                text: "before".into()
            }]
        );
        assert_eq!(
            parser.push("ink>inside</thi"),
            vec![AssistantEvent::Reasoning {
                text: "inside".into()
            }]
        );
        assert_eq!(
            parser.push("nk>after"),
            vec![AssistantEvent::Content {
                text: "after".into()
            }]
        );
    }

    #[test]
    fn incomplete_delimiter_is_flushed_at_eof() {
        let mut parser = ResponseParser::new(ParserConfig {
            reasoning_open_tag: Some("<think>".into()),
            reasoning_close_tag: Some("</think>".into()),
            starts_in_reasoning: false,
            ..Default::default()
        });
        assert!(parser.push("answer<th").iter().any(|event| {
            matches!(event, AssistantEvent::Content { text } if text == "answer")
        }));
        assert_eq!(
            parser.finish(),
            vec![AssistantEvent::Content { text: "<th".into() }]
        );
    }

    #[test]
    fn qwen_json_tool_call_is_protocol_neutral() {
        let mut parser = ResponseParser::new(ParserConfig {
            tool_parser: Some("qwen3".into()),
            ..Default::default()
        });
        assert!(
            parser
                .push("text<tool_")
                .iter()
                .any(|event| matches!(event, AssistantEvent::Content { text } if text == "text"))
        );
        let events = parser
            .push("call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Shanghai\"}}</tool_call>");
        assert!(matches!(&events[0], AssistantEvent::ToolStart { name, .. } if name == "weather"));
        assert!(
            matches!(&events[1], AssistantEvent::ToolArguments { arguments, .. } if arguments.contains("Shanghai"))
        );
        assert!(matches!(&events[2], AssistantEvent::ToolEnd { .. }));
    }

    #[test]
    fn dsml_supports_multiple_calls() {
        let payload = concat!(
            "<｜DSML｜invoke name=\"a\"><｜DSML｜parameter name=\"x\" string=\"false\">1</｜DSML｜parameter></｜DSML｜invoke>",
            "<｜DSML｜invoke name=\"b\"></｜DSML｜invoke>"
        );
        let calls = parse_dsml(payload);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], ("a".into(), "{\"x\":1}".into()));
    }
}
