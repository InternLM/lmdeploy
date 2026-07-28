// Copyright (c) OpenMMLab. All rights reserved.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use thiserror::Error;
use tokio::sync::mpsc;
use turbomind_runtime::{GenerationConfig, NativeEngine, SubmitRequest};

pub const STATUS_FINISH: i32 = 7;
pub const STATUS_CANCEL: i32 = 8;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Native(#[from] turbomind_runtime::Error),
    #[error("TurboMind request failed with status {0}")]
    EngineStatus(i32),
    #[error("input must contain at least 2 tokens to compute ppl")]
    PplInputTooShort,
    #[error("generation worker stopped before returning a final result")]
    WorkerStopped,
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub input_ids: Vec<i32>,
    pub session_id: u64,
    pub generation: GenerationConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateChunk {
    pub token_ids: Vec<i32>,
    pub finished: bool,
    pub cancelled: bool,
}

pub type GenerateStream = Pin<Box<dyn Stream<Item = Result<GenerateChunk>> + Send>>;

#[derive(Debug, Clone, Copy, Default)]
pub struct ScheduleMetrics {
    pub total_sequences: i32,
    pub active_sequences: i32,
    pub waiting_sequences: i32,
    pub total_blocks: i32,
    pub active_blocks: i32,
    pub cached_blocks: i32,
    pub free_blocks: i32,
}

#[async_trait]
pub trait Engine: Send + Sync + 'static {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateStream>;
    async fn ppl(&self, input_ids: Vec<i32>, session_id: u64) -> Result<f32>;
    fn schedule_metrics(&self) -> Result<Option<ScheduleMetrics>>;
}

#[async_trait]
impl Engine for NativeEngine {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateStream> {
        let mut native = self.create_request()?;
        let prompt_len = request.input_ids.len();
        native.submit(&SubmitRequest {
            input_ids: request.input_ids,
            session_id: request.session_id,
            session_step: 0,
            stream_output: true,
            enable_metrics: true,
            generation: request.generation,
        })?;

        let (sender, receiver) = mpsc::channel(8);
        tokio::spawn(async move {
            let mut previous = prompt_len;
            loop {
                let state = match native.next_state().await {
                    Ok(state) => state,
                    Err(error) => {
                        let _ = sender.send(Err(error.into())).await;
                        return;
                    }
                };
                if state.sequence_length < previous {
                    let _ = sender
                        .send(Err(turbomind_runtime::Error::InvalidSequenceLength(
                            state.sequence_length as i32,
                        )
                        .into()))
                        .await;
                    return;
                }
                let ids = match native.output_ids(previous, state.sequence_length) {
                    Ok(ids) => ids,
                    Err(error) => {
                        let _ = sender.send(Err(error.into())).await;
                        return;
                    }
                };
                previous = state.sequence_length;
                let finished = state.status == STATUS_FINISH;
                let cancelled = state.status == STATUS_CANCEL;
                if state.status != 0 && !finished && !cancelled {
                    let _ = sender.send(Err(Error::EngineStatus(state.status))).await;
                    return;
                }
                if (!ids.is_empty() || finished || cancelled)
                    && sender
                        .send(Ok(GenerateChunk {
                            token_ids: ids,
                            finished,
                            cancelled,
                        }))
                        .await
                        .is_err()
                {
                    native.cancel();
                    return;
                }
                if finished || cancelled {
                    return;
                }
            }
        });

        Ok(Box::pin(futures::stream::unfold(
            receiver,
            |mut receiver| async move { receiver.recv().await.map(|item| (item, receiver)) },
        )))
    }

    async fn ppl(&self, input_ids: Vec<i32>, session_id: u64) -> Result<f32> {
        let scored_tokens = input_ids
            .len()
            .checked_sub(1)
            .filter(|&count| count > 0)
            .ok_or(Error::PplInputTooShort)?;
        let mut native = self.create_request()?;
        native.submit(&SubmitRequest {
            input_ids,
            session_id,
            session_step: 0,
            stream_output: false,
            enable_metrics: true,
            generation: GenerationConfig {
                max_new_tokens: 1,
                top_k: 1,
                return_ppl: true,
                ..Default::default()
            },
        })?;
        loop {
            let state = native.next_state().await?;
            match state.status {
                0 => continue,
                STATUS_FINISH => return Ok(native.ce_loss()? / scored_tokens as f32),
                status => return Err(Error::EngineStatus(status)),
            }
        }
    }

    fn schedule_metrics(&self) -> Result<Option<ScheduleMetrics>> {
        Ok(
            NativeEngine::schedule_metrics(self, 0)?.map(|metrics| ScheduleMetrics {
                total_sequences: metrics.total_sequences,
                active_sequences: metrics.active_sequences,
                waiting_sequences: metrics.waiting_sequences,
                total_blocks: metrics.total_blocks,
                active_blocks: metrics.active_blocks,
                cached_blocks: metrics.cached_blocks,
                free_blocks: metrics.free_blocks,
            }),
        )
    }
}
