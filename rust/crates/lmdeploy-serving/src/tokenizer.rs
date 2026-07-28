// Copyright (c) OpenMMLab. All rights reserved.

use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Hugging Face tokenizer.json was not found at {0}")]
    Missing(PathBuf),
    #[error("failed to load tokenizer.json at {path}: {source}")]
    Load {
        path: PathBuf,
        source: tokenizers::Error,
    },
    #[error("tokenizer operation failed: {0}")]
    Operation(tokenizers::Error),
}

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    eos_ids: Vec<i32>,
}

impl Tokenizer {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = model_dir.as_ref().join("tokenizer.json");
        if !path.is_file() {
            return Err(Error::Missing(path));
        }
        let inner = tokenizers::Tokenizer::from_file(&path).map_err(|source| Error::Load {
            path: path.clone(),
            source,
        })?;
        let eos_ids = load_eos_ids(model_dir.as_ref(), &inner);
        Ok(Self { inner, eos_ids })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, Error> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().iter().map(|&id| id as i32).collect())
            .map_err(Error::Operation)
    }

    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> Result<String, Error> {
        let ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner
            .decode(&ids, skip_special_tokens)
            .map_err(Error::Operation)
    }

    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        self.inner.token_to_id(token).map(|id| id as i32)
    }

    pub fn eos_ids(&self) -> &[i32] {
        &self.eos_ids
    }
}

fn load_eos_ids(model_dir: &Path, tokenizer: &tokenizers::Tokenizer) -> Vec<i32> {
    for filename in ["generation_config.json", "config.json"] {
        if let Some(ids) = read_numeric_eos_ids(&model_dir.join(filename)) {
            return ids;
        }
    }
    let path = model_dir.join("tokenizer_config.json");
    let Ok(content) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(root) = serde_json::from_str::<serde_json::Value>(&content) else {
        return Vec::new();
    };
    let Some(token) = root.get("eos_token") else {
        return Vec::new();
    };
    let text = token.as_str().or_else(|| token.get("content")?.as_str());
    text.and_then(|text| tokenizer.token_to_id(text))
        .map(|id| vec![id as i32])
        .unwrap_or_default()
}

fn read_numeric_eos_ids(path: &Path) -> Option<Vec<i32>> {
    let content = fs::read_to_string(path).ok()?;
    let root: serde_json::Value = serde_json::from_str(&content).ok()?;
    let value = root.get("eos_token_id")?;
    if let Some(id) = value.as_i64() {
        return i32::try_from(id).ok().map(|id| vec![id]);
    }
    let ids = value
        .as_array()?
        .iter()
        .map(|id| i32::try_from(id.as_i64()?).ok())
        .collect::<Option<Vec<_>>>()?;
    (!ids.is_empty()).then_some(ids)
}
