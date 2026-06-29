#include "src/turbomind/engine/cache_boundary_policy.h"
#include "src/turbomind/core/check.h"  // TM_CHECK
#include "src/turbomind/engine/engine_config.h"
#include "src/turbomind/engine/request.h"

namespace turbomind {

AutoCacheBoundaryPolicy::AutoCacheBoundaryPolicy(int min_interval, bool prompt_boundary, bool generation_boundary):
    threshold_{min_interval > 0 ? min_interval / 2 : 0},
    prompt_boundary_{prompt_boundary},
    generation_boundary_{generation_boundary}
{
}

bool AutoCacheBoundaryPolicy::PublishPromptBoundary(const Sequence& s) const
{
    return prompt_boundary_ && threshold_ > 0 && (s.prompt_len - 1) - s.last_ckpt_pos >= threshold_;
}

bool AutoCacheBoundaryPolicy::PublishGenerationBoundary(const Sequence& s) const
{
    return generation_boundary_ && threshold_ > 0 && s.filled_len - s.last_ckpt_pos >= threshold_;
}

std::unique_ptr<CacheBoundaryPolicy> CreateCacheBoundaryPolicy(const EngineConfig& cfg)
{
    const std::string& name = cfg.cache_boundary_policy;

    if (name.empty() || name == "default") {
        return std::make_unique<DefaultCacheBoundaryPolicy>(cfg.cache_prompt_boundary, cfg.cache_generation_boundary);
    }

    if (name == "auto") {
        return std::make_unique<AutoCacheBoundaryPolicy>(
            cfg.linear_prefix_cache_min_interval, cfg.cache_prompt_boundary, cfg.cache_generation_boundary);
    }

    TM_CHECK(false) << "unknown cache_boundary_policy: " << name;
    return nullptr;
}

}  // namespace turbomind
