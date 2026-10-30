// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/object_cache_plan.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <tuple>
#include <utility>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/model_weight.h"

namespace turbomind {

// clang-format off
struct BlockConfig {
    int  head_dim_;
    int  head_num_;
    int  block_len_;
    int  t_bits_;
    int  q_bits_;
    bool share_kv_;

    int  t_bits() const { return t_bits_; }
    int  q_bits() const { return q_bits_; }
    int  head_dim() const { return head_dim_; }
    int  head_num() const { return head_num_; }
    int  block_len() const { return block_len_; }
    bool is_share_kv() const { return share_kv_; }
};
// clang-format on

std::vector<size_t> Divisors(size_t value)
{
    std::vector<size_t> lower;
    std::vector<size_t> upper;
    for (size_t divisor = 1; divisor <= value / divisor; ++divisor) {
        if (value % divisor == 0) {
            lower.push_back(divisor);
            if (divisor != value / divisor) {
                upper.push_back(value / divisor);
            }
        }
    }
    lower.insert(lower.end(), upper.rbegin(), upper.rend());
    return lower;
}

AttentionCachePlan CreateDefaultAttentionCachePlan(const std::vector<AttentionWeight*>& weights,
                                                   const EngineParam&                   engine)
{
    TM_CHECK(!weights.empty());

    const int dtype_bits = byte_size(engine.data_type, 8);
    const int quant_bits = engine.quant_policy ? engine.quant_policy : dtype_bits;

    auto get_block_config = [&](const AttentionWeight& w) {
        BlockConfig b{w.head_dim,
                      w.kv_head_num / w.tp_size,
                      engine.cache_block_seq_len,
                      dtype_bits == quant_bits ? 0 : dtype_bits,
                      quant_bits,
                      w.head_dim == 576};
        return b;
    };

    AttentionCachePlan result;
    result.layer_offsets_bytes.reserve(weights.size());
    result.object_bytes = 0;  // byte size (quantization aware)
    for (int i = 0; i < weights.size(); ++i) {
        block::Layout layout{get_block_config(*weights[i])};
        result.layer_offsets_bytes.push_back(result.object_bytes);
        result.object_bytes += layout.layer_size();
    }
    return result;
}

GdnCachePlan CreateDefaultGdnCachePlan(const std::vector<DeltaNetWeight*>& weights, const EngineParam& engine)
{
    TM_CHECK(!weights.empty());
    const auto& first   = *TM_CHECK_NOTNULL(weights.front());
    const int   tp_size = engine.attn_tp_size * engine.attn_cp_size;

    TM_CHECK_EQ(first.num_k_heads % tp_size, 0);
    TM_CHECK_EQ(first.num_v_heads % tp_size, 0);
    for (const auto* weight_ptr : weights) {
        const auto& weight = *TM_CHECK_NOTNULL(weight_ptr);
        TM_CHECK_EQ(weight.num_k_heads, first.num_k_heads);
        TM_CHECK_EQ(weight.num_v_heads, first.num_v_heads);
        TM_CHECK_EQ(weight.key_head_dim, first.key_head_dim);
        TM_CHECK_EQ(weight.value_head_dim, first.value_head_dim);
        TM_CHECK_EQ(weight.d_conv, first.d_conv);
        TM_CHECK_EQ(weight.data_type, first.data_type);
    }

    const int layer_num     = static_cast<int>(weights.size());
    const int local_k_heads = first.num_k_heads / tp_size;
    const int local_v_heads = first.num_v_heads / tp_size;

    int layers_per_block = 1;
    int heads_per_block  = local_v_heads;
    if (const char* value = std::getenv("TM_GDN_BLOCK_CONFIG")) {
        TM_CHECK_EQ(std::sscanf(value, "%d,%d", &layers_per_block, &heads_per_block), 2)
            << "expected TM_GDN_BLOCK_CONFIG=l,h (e.g. 4,16)";
    }
    TM_CHECK_GT(layers_per_block, 0);
    TM_CHECK_GT(heads_per_block, 0);

    GdnCachePlan result;
    result.local_v_heads    = local_v_heads;
    result.layers_per_block = layers_per_block;
    result.heads_per_block  = heads_per_block;
    result.num_layer_groups = ceil_div(layer_num, layers_per_block);
    result.num_head_groups  = ceil_div(local_v_heads, heads_per_block);
    result.num_blocks       = result.num_layer_groups * result.num_head_groups;

    const int    key_dim                 = local_k_heads * first.key_head_dim;
    const int    value_dim               = local_v_heads * first.value_head_dim;
    const size_t conv_dim                = 2 * key_dim + value_dim;
    const size_t conv_elements_per_layer = conv_dim * first.d_conv;
    const size_t conv_elements           = weights.size() * conv_elements_per_layer;
    result.conv_bytes                    = byte_size(first.data_type, conv_elements);
    result.conv_part_bytes               = result.conv_bytes;
    result.conv_state_offsets.reserve(weights.size());
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        result.conv_state_offsets.push_back(layer * conv_elements_per_layer);
    }

    result.recurrent_cell_elements    = first.key_head_dim * first.value_head_dim;
    const size_t recurrent_cell_bytes = byte_size(engine.state_dtype, result.recurrent_cell_elements);
    result.recurrent_total_bytes      = weights.size() * result.local_v_heads * recurrent_cell_bytes;
    result.recurrent_block_bytes      = layers_per_block * heads_per_block * recurrent_cell_bytes;
    result.recurrent_part_bytes       = result.recurrent_block_bytes;
    result.recurrent_state_offsets.reserve(weights.size());
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        result.recurrent_state_offsets.push_back((layer % layers_per_block) * heads_per_block
                                                 * result.recurrent_cell_elements);
    }
    return result;
}

ObjectCachePlan CreateDefaultObjectCachePlan(const ModelWeight& model, const EngineParam& engine)
{
    std::vector<AttentionWeight*> attention_weights;
    std::vector<DeltaNetWeight*>  gdn_weights;
    for (int layer = 0; layer < model.num_layer; ++layer) {
        const auto* layer_weights = TM_CHECK_NOTNULL(model.layer(layer));
        if (layer_weights->attention) {
            attention_weights.push_back(layer_weights->attention.get());
        }
        if (layer_weights->linear_attn) {
            gdn_weights.push_back(layer_weights->linear_attn.get());
        }
    }

    ObjectCachePlan result;
    if (!attention_weights.empty()) {
        result.attention = CreateDefaultAttentionCachePlan(attention_weights, engine);
    }
    if (!gdn_weights.empty()) {
        result.gdn = CreateDefaultGdnCachePlan(gdn_weights, engine);
    }
    return result;
}

std::optional<ObjectCachePlan> TuneObjectCacheLayout(ObjectCachePlan plan)
{
    constexpr size_t alignment     = 256;
    constexpr size_t max_page_size = 32 << 20UL;
    TM_CHECK(plan.attention || plan.gdn) << "plan must have at least one of attention or gdn";

    size_t attention_bytes{};
    size_t attention_units{1};
    if (plan.attention) {
        attention_bytes = plan.attention->object_bytes;
        if (attention_bytes == 0 || attention_bytes > max_page_size || attention_bytes % alignment != 0) {
            return std::nullopt;
        }
        attention_units = attention_bytes / alignment;
    }

    if (!plan.gdn) {
        plan.page_size = max_page_size / attention_bytes * attention_bytes;
        return plan;
    }

    auto&     gdn         = *plan.gdn;
    const int layer_count = gdn.conv_state_offsets.size();

    const size_t recurrent_cell_count = layer_count * static_cast<size_t>(gdn.local_v_heads);
    const size_t recurrent_cell_bytes = gdn.recurrent_total_bytes / recurrent_cell_count;

    const size_t conv_units     = ceil_div(gdn.conv_bytes, alignment);
    const size_t max_page_units = max_page_size / alignment;

    struct Geometry {
        size_t recurrent_bytes{};
        int    layers_per_block{};
        int    heads_per_block{};
        int    num_layer_groups{};
        int    num_head_groups{};
        int    num_blocks{};
    };

    std::vector<Geometry> geometries;
    for (int layers_per_block = 1; layers_per_block <= static_cast<int>(layer_count); ++layers_per_block) {
        for (int heads_per_block = 1; heads_per_block <= gdn.local_v_heads; ++heads_per_block) {
            const int num_layer_groups = ceil_div(layer_count, layers_per_block);
            const int num_head_groups  = ceil_div(gdn.local_v_heads, heads_per_block);
            const int num_blocks       = num_layer_groups * num_head_groups;
            if (num_blocks <= layer_count) {  // reduce num blocks
                geometries.push_back({static_cast<size_t>(layers_per_block) * heads_per_block * recurrent_cell_bytes,
                                      layers_per_block,
                                      heads_per_block,
                                      num_layer_groups,
                                      num_head_groups,
                                      num_blocks});
            }
        }
    }
    if (geometries.empty()) {
        return std::nullopt;
    }

    const size_t page_step_units = plan.attention ? attention_units : 1;
    const size_t min_page_units  = plan.attention ? attention_units : std::max<size_t>(conv_units, 1);

    struct Candidate {
        using Key = std::tuple<size_t, int, size_t, size_t, size_t, int, int>;

        Key      key;
        size_t   page_units{};
        size_t   conv_padded_units{};
        size_t   recurrent_padded_units{};
        Geometry geometry;
    };

    std::optional<Candidate> best;
    for (size_t page_units = min_page_units; page_units <= max_page_units; page_units += page_step_units) {
        const auto divisors = Divisors(page_units);
        const auto conv_it  = std::lower_bound(divisors.begin(), divisors.end(), conv_units);
        if (conv_it != divisors.end()) {
            const size_t conv_padded_units = *conv_it;
            const size_t conv_waste_bytes  = conv_padded_units * alignment - gdn.conv_bytes;

            for (const auto& geometry : geometries) {
                const size_t recurrent_units = ceil_div(geometry.recurrent_bytes, alignment);
                const auto   recurrent_it    = std::lower_bound(divisors.begin(), divisors.end(), recurrent_units);
                if (recurrent_it == divisors.end()) {
                    continue;
                }

                const size_t recurrent_padded_units = *recurrent_it;
                const size_t actual_page_units =
                    std::lcm(attention_units, std::lcm(conv_padded_units, recurrent_padded_units));
                const size_t recurrent_waste_bytes =
                    static_cast<size_t>(geometry.num_blocks) * recurrent_padded_units * alignment
                    - gdn.recurrent_total_bytes;
                const Candidate::Key key{conv_waste_bytes + recurrent_waste_bytes,
                                         geometry.num_blocks,
                                         actual_page_units,
                                         conv_waste_bytes,
                                         recurrent_waste_bytes,
                                         geometry.layers_per_block,
                                         geometry.heads_per_block};
                if (!best || key < best->key) {
                    best = Candidate{key, actual_page_units, conv_padded_units, recurrent_padded_units, geometry};
                }
            }
        }
    }

    if (!best) {
        return std::nullopt;
    }

    gdn.conv_part_bytes       = best->conv_padded_units * alignment;
    gdn.recurrent_block_bytes = best->geometry.recurrent_bytes;
    gdn.recurrent_part_bytes  = best->recurrent_padded_units * alignment;
    gdn.layers_per_block      = best->geometry.layers_per_block;
    gdn.heads_per_block       = best->geometry.heads_per_block;
    gdn.num_layer_groups      = best->geometry.num_layer_groups;
    gdn.num_head_groups       = best->geometry.num_head_groups;
    gdn.num_blocks            = best->geometry.num_blocks;
    gdn.recurrent_state_offsets.resize(layer_count);
    for (size_t layer = 0; layer < layer_count; ++layer) {
        gdn.recurrent_state_offsets[layer] =
            static_cast<int>(layer % gdn.layers_per_block * gdn.heads_per_block * gdn.recurrent_cell_elements);
    }

    plan.page_size = best->page_units * alignment;
    return plan;
}

ObjectCachePlan TuneObjectCacheLayout(const ModelWeight& model, const EngineParam& engine)
{
    auto result = TuneObjectCacheLayout(CreateDefaultObjectCachePlan(model, engine));
    TM_CHECK(result.has_value());
    return *result;
}

}  // namespace turbomind
