// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/turbomind/rust/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Consumes one retained engine reference and blocks until the server exits.
int32_t lmdeploy_rust_api_server_run(tm_engine_handle* engine,
                                     const uint8_t*    config_json,
                                     size_t            config_len,
                                     char*             error,
                                     size_t            error_capacity);

#ifdef __cplusplus
}  // extern "C"
#endif
