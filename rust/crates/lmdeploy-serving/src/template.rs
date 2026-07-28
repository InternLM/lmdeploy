// Copyright (c) OpenMMLab. All rights reserved.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use minijinja::Environment;
use serde::Deserialize;
use serde_json::{Map, Value};
use thiserror::Error;

use crate::protocol::{ChatMessage, Tool};

#[derive(Debug, Error)]
pub enum Error {
    #[error("tokenizer_config.json was not found at {0}")]
    MissingConfig(PathBuf),
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("tokenizer_config.json does not define a usable chat_template")]
    MissingTemplate,
    #[error("failed to compile or render the Hugging Face chat template: {0}")]
    Template(#[from] minijinja::Error),
}

#[derive(Debug, Deserialize)]
struct TemplateEntry {
    name: String,
    template: String,
}

struct CompiledTemplate {
    environment: Environment<'static>,
}

impl CompiledTemplate {
    fn new(source: String) -> Result<Self, Error> {
        let mut environment = Environment::new();
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_template_owned("chat", source)?;
        Ok(Self { environment })
    }

    fn render(&self, context: Value) -> Result<String, Error> {
        Ok(self.environment.get_template("chat")?.render(context)?)
    }
}

/// A Hugging Face Jinja chat template loaded entirely by Rust.
pub struct ChatTemplate {
    default: CompiledTemplate,
    tool_use: Option<CompiledTemplate>,
    special_tokens: Map<String, Value>,
}

impl ChatTemplate {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = model_dir.as_ref().join("tokenizer_config.json");
        if !path.is_file() {
            return Err(Error::MissingConfig(path));
        }
        let content = fs::read_to_string(&path).map_err(|source| Error::Read {
            path: path.clone(),
            source,
        })?;
        let root: Value = serde_json::from_str(&content).map_err(|source| Error::Parse {
            path: path.clone(),
            source,
        })?;
        let object = root.as_object().ok_or(Error::MissingTemplate)?;
        let template_value = object.get("chat_template").ok_or(Error::MissingTemplate)?;
        let (default_source, tool_source) = resolve_templates(template_value)?;

        let mut special_tokens = Map::new();
        for key in [
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "sep_token",
            "cls_token",
            "mask_token",
        ] {
            if let Some(token) = object.get(key).and_then(token_text) {
                special_tokens.insert(key.into(), Value::String(token));
            }
        }

        Ok(Self {
            default: CompiledTemplate::new(default_source)?,
            tool_use: tool_source.map(CompiledTemplate::new).transpose()?,
            special_tokens,
        })
    }

    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        template_kwargs: Option<&HashMap<String, Value>>,
    ) -> Result<String, Error> {
        let mut context = self.special_tokens.clone();
        context.insert(
            "messages".into(),
            serde_json::to_value(messages).expect("messages serialize"),
        );
        context.insert("add_generation_prompt".into(), Value::Bool(true));
        context.insert("continue_final_message".into(), Value::Bool(false));
        if let Some(tools) = tools {
            context.insert(
                "tools".into(),
                serde_json::to_value(tools).expect("tools serialize"),
            );
        }
        if let Some(kwargs) = template_kwargs {
            for (key, value) in kwargs {
                context.insert(key.clone(), value.clone());
            }
        }
        let template = if tools.is_some() {
            self.tool_use.as_ref().unwrap_or(&self.default)
        } else {
            &self.default
        };
        template.render(Value::Object(context))
    }
}

fn resolve_templates(value: &Value) -> Result<(String, Option<String>), Error> {
    if let Some(template) = value.as_str() {
        return Ok((template.to_owned(), None));
    }
    let entries: Vec<TemplateEntry> =
        serde_json::from_value(value.clone()).map_err(|_| Error::MissingTemplate)?;
    let mut default = None;
    let mut tool_use = None;
    for entry in entries {
        match entry.name.as_str() {
            "default" => default = Some(entry.template),
            "tool_use" => tool_use = Some(entry.template),
            _ => {}
        }
    }
    let default = default
        .or_else(|| tool_use.clone())
        .ok_or(Error::MissingTemplate)?;
    Ok((default, tool_use))
}

fn token_text(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(str::to_owned)
        .or_else(|| value.get("content")?.as_str().map(str::to_owned))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_objects_are_supported() {
        assert_eq!(
            token_text(&serde_json::json!({"content": "<s>"})).as_deref(),
            Some("<s>")
        );
    }

    #[test]
    fn named_template_prefers_default() {
        let (default, tool) = resolve_templates(&serde_json::json!([
            {"name": "tool_use", "template": "tool"},
            {"name": "default", "template": "plain"}
        ]))
        .unwrap();
        assert_eq!(default, "plain");
        assert_eq!(tool.as_deref(), Some("tool"));
    }
}
