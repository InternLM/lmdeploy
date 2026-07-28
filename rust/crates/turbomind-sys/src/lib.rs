// Copyright (c) OpenMMLab. All rights reserved.

//! Raw declarations for the TurboMind serving C ABI.

use std::ffi::{c_char, c_void};

pub const API_VERSION: u32 = 1;
pub const RESULT_OK: i32 = 0;

#[repr(C)]
pub struct EngineHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RequestHandle {
    _private: [u8; 0],
}

pub type NotifyFn = unsafe extern "C" fn(context: *mut c_void);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Error {
    pub code: i32,
    pub message: [c_char; 512],
}

impl Default for Error {
    fn default() -> Self {
        Self {
            code: RESULT_OK,
            message: [0; 512],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IntSlice {
    pub data: *const i32,
    pub len: usize,
}

impl IntSlice {
    pub fn new(values: &[i32]) -> Self {
        Self {
            data: values.as_ptr(),
            len: values.len(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GenerationConfig {
    pub max_new_tokens: i32,
    pub min_new_tokens: i32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub random_seed: u64,
    pub output_logprobs: i32,
    pub return_ppl: u8,
    pub reserved: [u8; 7],
    pub eos_ids: IntSlice,
    pub stop_ids: IntSlice,
    pub bad_ids: IntSlice,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SubmitParams {
    pub input_ids: IntSlice,
    pub session_id: u64,
    pub session_step: i32,
    pub stream_output: u8,
    pub enable_metrics: u8,
    pub reserved: [u8; 6],
    pub generation: GenerationConfig,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RequestState {
    pub available: u8,
    pub reserved: [u8; 3],
    pub status: i32,
    pub sequence_length: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogprobEntry {
    pub token_id: i32,
    pub logprob: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestMetrics {
    pub enqueue_time_us: i64,
    pub scheduled_time_us: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScheduleMetrics {
    pub total_sequences: i32,
    pub active_sequences: i32,
    pub waiting_sequences: i32,
    pub total_blocks: i32,
    pub active_blocks: i32,
    pub cached_blocks: i32,
    pub free_blocks: i32,
    pub scheduler_tick: i64,
}

unsafe extern "C" {
    pub fn tm_api_version() -> u32;
    pub fn tm_error_clear(error: *mut Error);

    pub fn tm_engine_retain(engine: *mut EngineHandle);
    pub fn tm_engine_release(engine: *mut EngineHandle);
    pub fn tm_engine_create_request(
        engine: *mut EngineHandle,
        request: *mut *mut RequestHandle,
        error: *mut Error,
    ) -> i32;
    pub fn tm_engine_get_schedule_metrics(
        engine: *mut EngineHandle,
        device_index: i32,
        metrics: *mut ScheduleMetrics,
        available: *mut u8,
        error: *mut Error,
    ) -> i32;

    pub fn tm_request_submit(
        request: *mut RequestHandle,
        params: *const SubmitParams,
        notify: Option<NotifyFn>,
        notify_context: *mut c_void,
        error: *mut Error,
    ) -> i32;
    pub fn tm_request_consume_state(
        request: *mut RequestHandle,
        state: *mut RequestState,
        error: *mut Error,
    ) -> i32;
    pub fn tm_request_copy_output_ids(
        request: *mut RequestHandle,
        begin: i32,
        end: i32,
        output: *mut i32,
        output_capacity: usize,
        error: *mut Error,
    ) -> i32;
    pub fn tm_request_get_logprob_count(
        request: *mut RequestHandle,
        generated_index: i32,
        count: *mut i32,
        error: *mut Error,
    ) -> i32;
    pub fn tm_request_copy_logprobs(
        request: *mut RequestHandle,
        generated_index: i32,
        output: *mut LogprobEntry,
        output_capacity: usize,
        written: *mut usize,
        error: *mut Error,
    ) -> i32;
    pub fn tm_request_get_ce_loss(
        request: *mut RequestHandle,
        ce_loss: *mut f32,
        error: *mut Error,
    ) -> i32;
    pub fn tm_request_get_metrics(
        request: *mut RequestHandle,
        metrics: *mut RequestMetrics,
        available: *mut u8,
        error: *mut Error,
    ) -> i32;
    pub fn tm_request_cancel(request: *mut RequestHandle);
    pub fn tm_request_release(request: *mut RequestHandle);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_layout_has_expected_field_sizes() {
        assert_eq!(std::mem::size_of::<LogprobEntry>(), 8);
        assert_eq!(std::mem::size_of::<RequestState>(), 12);
        assert_eq!(std::mem::size_of::<RequestMetrics>(), 16);
    }
}
