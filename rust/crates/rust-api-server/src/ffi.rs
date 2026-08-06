// Copyright (c) OpenMMLab. All rights reserved.

use std::ffi::c_char;
use std::slice;
use std::sync::Arc;

use turbomind_runtime::NativeEngine;
use turbomind_sys::EngineHandle;

use crate::{AppState, ServerConfig, tracing_level_directive};

/// Run the Rust API server and consume one retained TurboMind engine handle.
///
/// # Safety
/// `engine` must be a retained handle from the TurboMind C ABI. `config_json`
/// must reference `config_len` readable bytes. `error` must either be null or
/// reference `error_capacity` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmdeploy_rust_api_server_run(
    engine: *mut EngineHandle,
    config_json: *const u8,
    config_len: usize,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    let result = (|| -> anyhow::Result<()> {
        // Take ownership first so every later error path releases the handle.
        // SAFETY: caller transfers one retained engine reference.
        let engine = unsafe { NativeEngine::from_raw(engine) }?;
        if config_len != 0 && config_json.is_null() {
            anyhow::bail!("server config pointer is null");
        }
        // SAFETY: caller guarantees the config slice is readable.
        let config_json = if config_len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(config_json, config_len) }
        };
        let config: ServerConfig = serde_json::from_slice(config_json)?;

        let default_filter =
            tracing_subscriber::EnvFilter::try_new(tracing_level_directive(&config.log_level)?)?;
        let filter =
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or(default_filter);
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
        let state = Arc::new(AppState::new(config, Arc::new(engine))?);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("lmdeploy-rust-server")
            .build()?;
        runtime.block_on(crate::serve(state))
    })();

    match result {
        Ok(()) => 0,
        Err(cause) => {
            // SAFETY: caller guarantees the optional output buffer is writable.
            unsafe { write_error(error, error_capacity, &format!("{cause:#}")) };
            1
        }
    }
}

unsafe fn write_error(output: *mut c_char, capacity: usize, message: &str) {
    if output.is_null() || capacity == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let count = bytes.len().min(capacity - 1);
    // SAFETY: caller provided capacity writable bytes, and source has count bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.cast::<u8>(), count);
        *output.add(count) = 0;
    }
}
