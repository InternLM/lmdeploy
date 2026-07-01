// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <string>

#include "src/turbomind/core/logger.h"  // TM_LOG_FATAL

namespace turbomind {

// Per-side prefix-cache publication mode. cache_prompt uses {kAuto, kAll}
// (no kNone); cache_generation uses all three.
enum class CacheMode
{
    kNone,
    kAuto,
    kAll
};

// String -> CacheMode. cache_prompt never receives "none" (rejected by the
// Python TurbomindEngineConfig.__post_init__ assert); the shared parser still
// accepts it for cache_generation. TM_LOG_FATAL is [[noreturn]] (std::abort),
// so no trailing return is needed.
inline CacheMode ParseCacheMode(const std::string& s)
{
    if (s == "none") {
        return CacheMode::kNone;
    }
    if (s == "auto") {
        return CacheMode::kAuto;
    }
    if (s == "all") {
        return CacheMode::kAll;
    }
    TM_LOG_FATAL("invalid cache mode: {}", s);
}

// Pure prompt-boundary publish decision, given the mode, whether the geometric
// plan wants a partial fork_to node (else block-aligned checkpoint clamp), and
// whether that partial node's token range holds image tokens. 'all' publishes a
// partial node whenever the plan is partial and arms the clamp when
// block-aligned; 'auto' publishes only an image-bearing partial node and never
// arms the block-aligned clamp.
inline bool DecidePromptBoundaryPublish(CacheMode prompt_mode, bool plan_partial, bool has_image_in_node)
{
    if (plan_partial) {
        return prompt_mode == CacheMode::kAll
               || (prompt_mode == CacheMode::kAuto && has_image_in_node);
    }
    return prompt_mode == CacheMode::kAll;
}

}  // namespace turbomind
