// Copyright (c) OpenMMLab. All rights reserved.

//! Safe asynchronous wrappers over the TurboMind serving C ABI.

use std::ffi::{CStr, c_void};
use std::ptr::NonNull;
use std::sync::Arc;

use thiserror::Error;
use tokio::sync::Notify;
use turbomind_sys as sys;

#[derive(Debug, Error)]
pub enum Error {
    #[error("TurboMind serving ABI version {actual} is incompatible with version {expected}")]
    AbiVersion { expected: u32, actual: u32 },
    #[error("TurboMind error {code}: {message}")]
    Native { code: i32, message: String },
    #[error("TurboMind returned a null {0} handle")]
    NullHandle(&'static str),
    #[error("invalid native sequence length {0}")]
    InvalidSequenceLength(i32),
}

pub type Result<T> = std::result::Result<T, Error>;

fn check(code: i32, error: &sys::Error) -> Result<()> {
    if code == sys::RESULT_OK {
        return Ok(());
    }
    // SAFETY: C API always NUL-terminates the fixed error buffer.
    let message = unsafe { CStr::from_ptr(error.message.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Err(Error::Native { code, message })
}

#[derive(Debug, Clone)]
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
    pub return_ppl: bool,
    pub eos_ids: Vec<i32>,
    pub stop_ids: Vec<i32>,
    pub bad_ids: Vec<i32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            min_new_tokens: 0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            temperature: 1.0,
            repetition_penalty: 1.0,
            random_seed: 0,
            output_logprobs: 0,
            return_ppl: false,
            eos_ids: Vec::new(),
            stop_ids: Vec::new(),
            bad_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubmitRequest {
    pub input_ids: Vec<i32>,
    pub session_id: u64,
    pub session_step: i32,
    pub stream_output: bool,
    pub enable_metrics: bool,
    pub generation: GenerationConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestState {
    pub status: i32,
    pub sequence_length: usize,
}

pub struct NativeEngine {
    handle: NonNull<sys::EngineHandle>,
}

// TurboMind owns its internal synchronization. The C handle is retained for
// every clone and may be used to create requests from multiple Tokio workers.
unsafe impl Send for NativeEngine {}
unsafe impl Sync for NativeEngine {}

impl NativeEngine {
    /// Take ownership of one retained native engine handle.
    ///
    /// # Safety
    /// `handle` must come from `turbomind::rust::RetainEngine` and represent one
    /// owned reference.
    pub unsafe fn from_raw(handle: *mut sys::EngineHandle) -> Result<Self> {
        let handle = NonNull::new(handle).ok_or(Error::NullHandle("engine"))?;
        // SAFETY: version query has no preconditions.
        let actual = unsafe { sys::tm_api_version() };
        if actual != sys::API_VERSION {
            // SAFETY: caller transferred one retained reference.
            unsafe { sys::tm_engine_release(handle.as_ptr()) };
            return Err(Error::AbiVersion {
                expected: sys::API_VERSION,
                actual,
            });
        }
        Ok(Self { handle })
    }

    pub fn create_request(&self) -> Result<NativeRequest> {
        let mut request = std::ptr::null_mut();
        let mut error = sys::Error::default();
        // SAFETY: pointers remain valid for the duration of the call.
        let code = unsafe {
            sys::tm_engine_create_request(self.handle.as_ptr(), &mut request, &mut error)
        };
        check(code, &error)?;
        let request = NonNull::new(request).ok_or(Error::NullHandle("request"))?;
        Ok(NativeRequest {
            handle: request,
            notify: Box::new(NotifyContext {
                notify: Arc::new(Notify::new()),
            }),
            submitted: false,
        })
    }

    pub fn schedule_metrics(&self, device_index: i32) -> Result<Option<sys::ScheduleMetrics>> {
        let mut metrics = sys::ScheduleMetrics::default();
        let mut available = 0;
        let mut error = sys::Error::default();
        // SAFETY: all output pointers are valid and uniquely borrowed.
        let code = unsafe {
            sys::tm_engine_get_schedule_metrics(
                self.handle.as_ptr(),
                device_index,
                &mut metrics,
                &mut available,
                &mut error,
            )
        };
        check(code, &error)?;
        Ok((available != 0).then_some(metrics))
    }
}

impl Clone for NativeEngine {
    fn clone(&self) -> Self {
        // SAFETY: handle is valid while self owns a retained reference.
        unsafe { sys::tm_engine_retain(self.handle.as_ptr()) };
        Self {
            handle: self.handle,
        }
    }
}

impl Drop for NativeEngine {
    fn drop(&mut self) {
        // SAFETY: this instance owns exactly one retained reference.
        unsafe { sys::tm_engine_release(self.handle.as_ptr()) };
    }
}

struct NotifyContext {
    notify: Arc<Notify>,
}

unsafe extern "C" fn notify_callback(context: *mut c_void) {
    if context.is_null() {
        return;
    }
    // SAFETY: context is boxed inside NativeRequest. The C release operation
    // disarms and synchronizes callbacks before that box is dropped.
    let context = unsafe { &*(context.cast::<NotifyContext>()) };
    context.notify.notify_one();
}

pub struct NativeRequest {
    handle: NonNull<sys::RequestHandle>,
    notify: Box<NotifyContext>,
    submitted: bool,
}

unsafe impl Send for NativeRequest {}

impl NativeRequest {
    pub fn submit(&mut self, request: &SubmitRequest) -> Result<()> {
        let generation = &request.generation;
        let params = sys::SubmitParams {
            input_ids: sys::IntSlice::new(&request.input_ids),
            session_id: request.session_id,
            session_step: request.session_step,
            stream_output: u8::from(request.stream_output),
            enable_metrics: u8::from(request.enable_metrics),
            generation: sys::GenerationConfig {
                max_new_tokens: generation.max_new_tokens,
                min_new_tokens: generation.min_new_tokens,
                top_k: generation.top_k,
                top_p: generation.top_p,
                min_p: generation.min_p,
                temperature: generation.temperature,
                repetition_penalty: generation.repetition_penalty,
                random_seed: generation.random_seed,
                output_logprobs: generation.output_logprobs,
                return_ppl: u8::from(generation.return_ppl),
                eos_ids: sys::IntSlice::new(&generation.eos_ids),
                stop_ids: sys::IntSlice::new(&generation.stop_ids),
                bad_ids: sys::IntSlice::new(&generation.bad_ids),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut error = sys::Error::default();
        // SAFETY: C++ copies all borrowed slices before returning. notify lives
        // until tm_request_release synchronously disarms the callback.
        let code = unsafe {
            sys::tm_request_submit(
                self.handle.as_ptr(),
                &params,
                Some(notify_callback),
                (&mut *self.notify as *mut NotifyContext).cast(),
                &mut error,
            )
        };
        check(code, &error)?;
        self.submitted = true;
        Ok(())
    }

    pub fn consume_state(&mut self) -> Result<Option<RequestState>> {
        let mut state = sys::RequestState::default();
        let mut error = sys::Error::default();
        // SAFETY: handle and output pointers are valid.
        let code =
            unsafe { sys::tm_request_consume_state(self.handle.as_ptr(), &mut state, &mut error) };
        check(code, &error)?;
        if state.available == 0 {
            return Ok(None);
        }
        let sequence_length = usize::try_from(state.sequence_length)
            .map_err(|_| Error::InvalidSequenceLength(state.sequence_length))?;
        Ok(Some(RequestState {
            status: state.status,
            sequence_length,
        }))
    }

    pub async fn next_state(&mut self) -> Result<RequestState> {
        loop {
            if let Some(state) = self.consume_state()? {
                return Ok(state);
            }
            self.notify.notify.notified().await;
        }
    }

    pub fn output_ids(&self, begin: usize, end: usize) -> Result<Vec<i32>> {
        let mut output = vec![0; end.saturating_sub(begin)];
        let mut error = sys::Error::default();
        // SAFETY: output owns enough writable elements for the requested range.
        let code = unsafe {
            sys::tm_request_copy_output_ids(
                self.handle.as_ptr(),
                i32::try_from(begin).map_err(|_| Error::InvalidSequenceLength(i32::MAX))?,
                i32::try_from(end).map_err(|_| Error::InvalidSequenceLength(i32::MAX))?,
                output.as_mut_ptr(),
                output.len(),
                &mut error,
            )
        };
        check(code, &error)?;
        Ok(output)
    }

    pub fn logprobs(&self, generated_index: usize) -> Result<Vec<sys::LogprobEntry>> {
        let generated_index =
            i32::try_from(generated_index).map_err(|_| Error::InvalidSequenceLength(i32::MAX))?;
        let mut count = 0;
        let mut error = sys::Error::default();
        // SAFETY: output pointers are valid.
        let code = unsafe {
            sys::tm_request_get_logprob_count(
                self.handle.as_ptr(),
                generated_index,
                &mut count,
                &mut error,
            )
        };
        check(code, &error)?;
        let mut output = vec![sys::LogprobEntry::default(); count.max(0) as usize];
        let mut written = 0;
        // SAFETY: output has the capacity reported by the native request.
        let code = unsafe {
            sys::tm_request_copy_logprobs(
                self.handle.as_ptr(),
                generated_index,
                output.as_mut_ptr(),
                output.len(),
                &mut written,
                &mut error,
            )
        };
        check(code, &error)?;
        output.truncate(written);
        Ok(output)
    }

    pub fn ce_loss(&self) -> Result<f32> {
        let mut output = 0.0;
        let mut error = sys::Error::default();
        // SAFETY: output pointers are valid.
        let code =
            unsafe { sys::tm_request_get_ce_loss(self.handle.as_ptr(), &mut output, &mut error) };
        check(code, &error)?;
        Ok(output)
    }

    pub fn metrics(&self) -> Result<Option<sys::RequestMetrics>> {
        let mut output = sys::RequestMetrics::default();
        let mut available = 0;
        let mut error = sys::Error::default();
        // SAFETY: output pointers are valid.
        let code = unsafe {
            sys::tm_request_get_metrics(
                self.handle.as_ptr(),
                &mut output,
                &mut available,
                &mut error,
            )
        };
        check(code, &error)?;
        Ok((available != 0).then_some(output))
    }

    pub fn cancel(&mut self) {
        // SAFETY: handle is valid while self is alive.
        unsafe { sys::tm_request_cancel(self.handle.as_ptr()) };
    }
}

impl Drop for NativeRequest {
    fn drop(&mut self) {
        // SAFETY: release disarms and synchronizes the C callback before the
        // NotifyContext box is dropped.
        unsafe { sys::tm_request_release(self.handle.as_ptr()) };
    }
}
