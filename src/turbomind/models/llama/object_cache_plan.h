// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "src/turbomind/comm/host_comm.h"

namespace turbomind {

class AttentionWeight;
class DeltaNetWeight;
class ModelWeight;
struct EngineParam;

struct AttentionCachePlan {
    size_t              object_bytes{};       // S: one KV block across all full-attention layers
    std::vector<size_t> layer_offsets_bytes;  // byte offsets in full-attention weight order
};

struct GdnCachePlan {
    int local_v_heads{};
    int layers_per_block{};
    int heads_per_block{};
    int num_layer_groups{};
    int num_head_groups{};
    int num_blocks{};

    size_t           conv_bytes{};
    size_t           conv_part_bytes{};
    std::vector<int> conv_state_offsets;  // Element offsets, in the same order as the GDN weights.

    size_t           recurrent_cell_elements{};  // one (layer, value-head) cell
    size_t           recurrent_total_bytes{};
    size_t           recurrent_block_bytes{};
    size_t           recurrent_part_bytes{};
    std::vector<int> recurrent_state_offsets;  // Element offsets, in the same order as the GDN weights.
};

struct ObjectCachePlan {
    size_t                            page_size{32 << 20UL};  // default 32MB
    std::optional<AttentionCachePlan> attention;
    std::optional<GdnCachePlan>       gdn;
};

// Equal byte budgets do not imply equal page counts: CUDA allocations can
// consume different alignment padding on each rank. Trim only the tail so the
// original allocation base remains available for CUDA IPC registration.
inline size_t CommonCacheRegionSize(comm::HostCommImpl* group, const void* base, size_t bytes, size_t page_size)
{
    if (!page_size)
        throw std::invalid_argument("cache page size must be positive");
    const auto remainder = reinterpret_cast<std::uintptr_t>(base) % page_size;
    const auto padding   = remainder ? page_size - remainder : 0;
    const auto local     = bytes > padding ? (bytes - padding) / page_size : 0;
    const auto pages     = comm::AllReduce(group, local, comm::RedOp::kMin);
    if (!pages)
        throw std::runtime_error("cache budget cannot fit one aligned page on every TP rank");
    return padding + pages * page_size;
}

AttentionCachePlan CreateDefaultAttentionCachePlan(const std::vector<AttentionWeight*>& weights,
                                                   const EngineParam&                   engine);

GdnCachePlan CreateDefaultGdnCachePlan(const std::vector<DeltaNetWeight*>& weights, const EngineParam& engine);

ObjectCachePlan CreateDefaultObjectCachePlan(const ModelWeight& model, const EngineParam& engine);

ObjectCachePlan TuneObjectCacheLayout(const ModelWeight& model, const EngineParam& engine);

std::optional<ObjectCachePlan> TuneObjectCacheLayout(ObjectCachePlan plan);

}  // namespace turbomind
