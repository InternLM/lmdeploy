// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/engine/request.h"

#define private public
#include "src/turbomind/models/llama/SequenceManager.h"
#undef private

namespace turbomind {
namespace {

struct DeviceTestContext {
    core::Stream       stream;
    core::Allocator    device_allocator;
    core::ContextGuard guard;

    DeviceTestContext():
        stream(core::Stream::create()), device_allocator(stream, false), guard(stream, device_allocator)
    {
    }
};

ModelParam MakeModelParam(bool linear_attention)
{
    ModelParam model{};
    model.head_num       = 1;
    model.head_dim       = 4;
    model.kv_head_num    = 1;
    model.hidden_units   = 16;
    model.layer_num      = linear_attention ? 2 : 1;
    model.vocab_size     = 32;
    model.embedding_size = 16;
    model.norm_eps       = 1e-5f;
    model.quant_policy   = 0;
    model.attn_bias      = false;
    model.attn_sink      = false;
    model.mlp_bias       = false;
    model.data_type      = kFloat;
    model.weight_type    = kFloat;
    model.expert_weight_type = kFloat;
    model.ffn_weight_type    = kFloat;
    model.group_size         = 0;
    model.qk_norm            = false;
    model.tune_layer_num     = 0;
    model.act_type           = ActivationType::kSilu;
    model.layer_types        = linear_attention ? std::vector<int>{0, 1} : std::vector<int>{0};
    model.window_size.assign(model.layer_num, 0);
    model.inter_size.assign(model.layer_num, 16);
    if (linear_attention) {
        model.linear_key_head_dim    = 4;
        model.linear_value_head_dim  = 4;
        model.linear_conv_kernel_dim = 2;
        model.linear_num_key_heads   = 1;
        model.linear_num_value_heads = 1;
        model.linear_state_dtype     = kFloat;
    }
    return model;
}

size_t KVBlockBytes(const ModelParam& model, DataType runtime_dtype, int cache_block_seq_len, int attn_tp_size)
{
    int cache_layer_num = model.layer_num;
    for (const auto& type : model.layer_types) {
        if (type == 1) {
            --cache_layer_num;
        }
    }
    const int dbits     = byte_size(runtime_dtype, 8);
    const int elem_bits = model.quant_policy ? model.quant_policy : dbits;
    SequenceManager::BlockConfig block_config{
        (int)model.head_dim,
        (int)model.kv_head_num / attn_tp_size,
        cache_block_seq_len,
        elem_bits == dbits ? 0 : dbits,
        elem_bits,
        model.head_dim == 576,
    };
    block::Layout layout{block_config};
    return layout.block_size(cache_layer_num);
}

size_t LinearSnapshotBytesPerBlock(const ModelParam& model, int attn_tp_size)
{
    int num_linear_layers = 0;
    for (const auto& type : model.layer_types) {
        if (type == 1) {
            ++num_linear_layers;
        }
    }
    if (num_linear_layers == 0) {
        return 0;
    }

    const int key_head_dim = model.linear_key_head_dim > 0 ? model.linear_key_head_dim : model.head_dim;
    const int value_head_dim = model.linear_value_head_dim > 0 ? model.linear_value_head_dim : model.head_dim;
    const int d_conv = model.linear_conv_kernel_dim > 0 ? model.linear_conv_kernel_dim : 4;
    const int num_k_heads = model.linear_num_key_heads / attn_tp_size;
    const int num_v_heads = model.linear_num_value_heads / attn_tp_size;
    const int conv_dim    = (num_k_heads * key_head_dim) * 2 + num_v_heads * value_head_dim;

    const size_t conv_bytes = static_cast<size_t>(num_linear_layers) * d_conv * conv_dim * byte_size(model.data_type);
    const size_t recurrent_bytes =
        static_cast<size_t>(num_linear_layers) * num_v_heads * key_head_dim * value_head_dim * byte_size(model.linear_state_dtype);
    return conv_bytes + recurrent_bytes;
}

size_t LinearActivePoolBytes(const ModelParam& model, int attn_tp_size, int max_batch_size)
{
    return LinearSnapshotBytesPerBlock(model, attn_tp_size) * max_batch_size;
}

void FillTensor(Ref<Tensor> tensor_, float value)
{
    auto& tensor = tensor_.get();
    REQUIRE(tensor.dtype() == kFloat);
    Tensor host{tensor.shape(), tensor.dtype(), kCPU};
    std::fill_n(host.data<float>(), host.size(), value);
    Copy(host, tensor);
    core::Context::stream().Sync();
}

void FillTensorByBlock(Ref<Tensor> tensor_, const std::vector<float>& block_values)
{
    auto& tensor = tensor_.get();
    REQUIRE(tensor.dtype() == kFloat);
    REQUIRE(!block_values.empty());

    Tensor host{tensor.shape(), tensor.dtype(), kCPU};
    auto*  data            = host.data<float>();
    auto   elements        = host.size();
    auto   elements_per_block = elements / static_cast<ssize_t>(block_values.size());
    REQUIRE(elements_per_block * static_cast<ssize_t>(block_values.size()) == elements);

    for (size_t i = 0; i < block_values.size(); ++i) {
        std::fill_n(data + i * elements_per_block, elements_per_block, block_values[i]);
    }
    Copy(host, tensor);
    core::Context::stream().Sync();
}

std::vector<float> ToHostVector(const Tensor& tensor)
{
    REQUIRE(tensor.dtype() == kFloat);
    Tensor host{tensor.shape(), tensor.dtype(), kCPU};
    Copy(tensor, host);
    core::Context::stream().Sync();
    return {host.data<float>(), host.data<float>() + host.size()};
}

bool AllClose(const std::vector<float>& values, float expected, float tol = 1e-6f)
{
    return std::all_of(values.begin(), values.end(), [&](float v) { return std::fabs(v - expected) <= tol; });
}

Sequence MakeActiveSequence(uint64_t id, const std::vector<int>& prompt, const BlockIds& block_ids, const UniqueIds& unique_ids)
{
    Sequence seq{id};
    seq.status           = Sequence::kActive;
    seq.prompt           = prompt;
    seq.blocks           = block_ids;
    seq.block_unique_ids = unique_ids;
    seq.cache_len        = static_cast<int>(block_ids.size()) * 2;
    return seq;
}

}  // namespace

TEST_CASE("BlockTrie caches and matches aligned linear snapshots", "[prefix_cache][linear][trie]")
{
    core::Allocator               allocator{kCPU};
    auto                          block_manager = std::make_shared<BlockManager>(64, 8, -1, allocator, [] { return size_t{1 << 20}; });
    std::vector<uint64_t>         snapshot_unique_ids(8, 0);
    uint64_t                      next_snapshot_unique_id = 1;
    BlockTrie::SnapshotPublisher  publisher = [&](const Sequence&, int block_idx, int slot_hint) {
        const int slot = slot_hint >= 0 ? slot_hint : block_idx;
        snapshot_unique_ids[slot] = next_snapshot_unique_id++;
        return std::pair<int, uint64_t>{slot, snapshot_unique_ids[slot]};
    };
    BlockTrie::SnapshotValidator  validator = [&](int slot, uint64_t unique_id) {
        return slot >= 0 && slot < (int)snapshot_unique_ids.size() && unique_id != 0 && snapshot_unique_ids[slot] == unique_id;
    };
    BlockTrie trie(2, block_manager, publisher, validator, {});

    const auto [block_ids, unique_ids] = block_manager->Allocate(2);
    Sequence   cached_seq = MakeActiveSequence(1, {1, 2, 3, 4, 5}, block_ids, unique_ids);

    const auto cache = trie.Cache(cached_seq, cached_seq.prompt);
    REQUIRE(cache.block_ids == block_ids);
    REQUIRE(cache.unique_ids == unique_ids);

    Sequence query{2};
    query.prompt = cached_seq.prompt;

    const auto match = trie.Match(query);
    REQUIRE(match.block_ids == block_ids);
    REQUIRE(match.unique_ids == unique_ids);
    REQUIRE(match.matched_block_count == 2);
    REQUIRE(match.snapshot_slot == 1);
    REQUIRE(match.snapshot_unique_id == snapshot_unique_ids[1]);
}

TEST_CASE("BlockTrie fast-forward refresh updates snapshot metadata", "[prefix_cache][linear][trie]")
{
    core::Allocator               allocator{kCPU};
    auto                          block_manager = std::make_shared<BlockManager>(64, 8, -1, allocator, [] { return size_t{1 << 20}; });
    std::vector<uint64_t>         snapshot_unique_ids(8, 0);
    uint64_t                      next_snapshot_unique_id = 1;
    BlockTrie::SnapshotPublisher  publisher = [&](const Sequence&, int block_idx, int slot_hint) {
        const int slot = slot_hint >= 0 ? slot_hint : block_idx;
        snapshot_unique_ids[slot] = next_snapshot_unique_id++;
        return std::pair<int, uint64_t>{slot, snapshot_unique_ids[slot]};
    };
    BlockTrie::SnapshotValidator validator = [&](int slot, uint64_t unique_id) {
        return slot >= 0 && slot < (int)snapshot_unique_ids.size() && unique_id != 0 && snapshot_unique_ids[slot] == unique_id;
    };
    BlockTrie trie(2, block_manager, publisher, validator, {});

    const std::vector<int> prompt{6, 7, 8, 9, 10};

    const auto [block_ids_1, unique_ids_1] = block_manager->Allocate(2);
    Sequence   seq1 = MakeActiveSequence(1, prompt, block_ids_1, unique_ids_1);
    trie.Cache(seq1, prompt);

    Sequence query{11};
    query.prompt = prompt;
    const auto first_match = trie.Match(query);

    const auto [block_ids_2, unique_ids_2] = block_manager->Allocate(2);
    Sequence   seq2 = MakeActiveSequence(2, prompt, block_ids_2, unique_ids_2);
    trie.Cache(seq2, prompt);

    const auto refreshed_match = trie.Match(query);
    REQUIRE(refreshed_match.block_ids == block_ids_2);
    REQUIRE(refreshed_match.unique_ids == unique_ids_2);
    REQUIRE(refreshed_match.snapshot_slot == first_match.snapshot_slot);
    REQUIRE(refreshed_match.snapshot_unique_id != first_match.snapshot_unique_id);
    REQUIRE(refreshed_match.snapshot_unique_id == snapshot_unique_ids[first_match.snapshot_slot]);
}

TEST_CASE("BlockTrie verify releases descendant snapshots when a parent snapshot goes stale",
          "[prefix_cache][linear][trie]")
{
    core::Allocator               allocator{kCPU};
    auto                          block_manager = std::make_shared<BlockManager>(64, 8, -1, allocator, [] { return size_t{1 << 20}; });
    std::vector<uint64_t>         snapshot_unique_ids(8, 0);
    std::vector<int>              released_slots;
    uint64_t                      next_snapshot_unique_id = 1;
    BlockTrie::SnapshotPublisher  publisher = [&](const Sequence&, int block_idx, int slot_hint) {
        const int slot = slot_hint >= 0 ? slot_hint : block_idx;
        snapshot_unique_ids[slot] = next_snapshot_unique_id++;
        return std::pair<int, uint64_t>{slot, snapshot_unique_ids[slot]};
    };
    BlockTrie::SnapshotValidator  validator = [&](int slot, uint64_t unique_id) {
        return slot >= 0 && slot < (int)snapshot_unique_ids.size() && unique_id != 0 && snapshot_unique_ids[slot] == unique_id;
    };
    BlockTrie::SnapshotReleaser releaser = [&](int slot, uint64_t unique_id) {
        if (validator(slot, unique_id)) {
            snapshot_unique_ids[slot] = 0;
            released_slots.push_back(slot);
        }
    };
    BlockTrie trie(2, block_manager, publisher, validator, releaser);

    const auto [block_ids, unique_ids] = block_manager->Allocate(2);
    Sequence   cached_seq = MakeActiveSequence(1, {10, 11, 12, 13, 14}, block_ids, unique_ids);
    trie.Cache(cached_seq, cached_seq.prompt);

    snapshot_unique_ids[0] = 0;  // Invalidate the parent snapshot only.
    trie.Verify();

    REQUIRE(released_slots == std::vector<int>{1});

    Sequence query{2};
    query.prompt = cached_seq.prompt;
    const auto match = trie.Match(query);
    REQUIRE(match.block_ids.empty());
    REQUIRE(match.unique_ids.empty());
    REQUIRE(match.snapshot_slot == -1);
}

TEST_CASE("SequenceManager fractional budgeting includes linear snapshot bytes",
          "[prefix_cache][linear][sequence_manager]")
{
    DeviceTestContext ctx;

    const auto  model                = MakeModelParam(true);
    constexpr int cache_block_seq_len = 2;
    constexpr int attn_tp_size        = 1;
    constexpr int max_batch_size      = 2;
    constexpr double ratio            = 0.5;

    const size_t kv_block_bytes = KVBlockBytes(model, kFloat, cache_block_seq_len, attn_tp_size);
    const size_t snapshot_bytes = LinearSnapshotBytesPerBlock(model, attn_tp_size);
    const size_t active_bytes   = LinearActivePoolBytes(model, attn_tp_size, max_batch_size);
    const size_t effective_block_bytes = kv_block_bytes + snapshot_bytes;
    const size_t expected_blocks       = 6;
    const size_t target_bytes          = active_bytes + expected_blocks * effective_block_bytes + effective_block_bytes / 2;
    const size_t free_before           = target_bytes * 2;

    SequenceManager manager(model,
                            kFloat,
                            cache_block_seq_len,
                            attn_tp_size,
                            max_batch_size,
                            ratio,
                            -1,
                            true,
                            1,
                            1,
                            ctx.device_allocator,
                            [free_before] { return free_before; });

    REQUIRE(manager.linear_active_pool_bytes_ == active_bytes);
    REQUIRE(manager.linear_snapshot_bytes_per_block_ == snapshot_bytes);
    REQUIRE(manager.max_block_count() == static_cast<int>(expected_blocks));
}

TEST_CASE("SequenceManager clamps integer reusable blocks with linear snapshot overhead",
          "[prefix_cache][linear][sequence_manager]")
{
    DeviceTestContext ctx;

    const auto  model                = MakeModelParam(true);
    constexpr int cache_block_seq_len = 2;
    constexpr int attn_tp_size        = 1;
    constexpr int max_batch_size      = 2;
    constexpr int requested_blocks    = 9;

    const size_t kv_block_bytes = KVBlockBytes(model, kFloat, cache_block_seq_len, attn_tp_size);
    const size_t snapshot_bytes = LinearSnapshotBytesPerBlock(model, attn_tp_size);
    const size_t active_bytes   = LinearActivePoolBytes(model, attn_tp_size, max_batch_size);
    const size_t effective_block_bytes = kv_block_bytes + snapshot_bytes;
    const size_t expected_blocks       = 4;
    const size_t free_before           = active_bytes + expected_blocks * effective_block_bytes + effective_block_bytes / 2;

    SequenceManager manager(model,
                            kFloat,
                            cache_block_seq_len,
                            attn_tp_size,
                            max_batch_size,
                            requested_blocks,
                            -1,
                            true,
                            1,
                            1,
                            ctx.device_allocator,
                            [free_before] { return free_before; });

    REQUIRE(manager.linear_active_pool_bytes_ == active_bytes);
    REQUIRE(manager.linear_snapshot_bytes_per_block_ == snapshot_bytes);
    REQUIRE(manager.max_block_count() == static_cast<int>(expected_blocks));
}

TEST_CASE("SequenceManager restores the last matched linear snapshot on a prefix hit",
          "[prefix_cache][linear][sequence_manager]")
{
    DeviceTestContext ctx;

    const auto  model                = MakeModelParam(true);
    constexpr int cache_block_seq_len = 2;
    constexpr int attn_tp_size        = 1;
    constexpr int max_batch_size      = 2;

    SequenceManager manager(model,
                            kFloat,
                            cache_block_seq_len,
                            attn_tp_size,
                            max_batch_size,
                            8,
                            -1,
                            true,
                            1,
                            1,
                            ctx.device_allocator,
                            [] { return size_t{1 << 20}; });

    auto seq1 = manager.Create(1);
    manager.AcquireLinearStateSlot(*seq1);

    auto& s1 = const_cast<Sequence&>(*seq1);
    std::tie(s1.blocks, s1.block_unique_ids) = manager.block_manager_->Allocate(2);
    s1.status                                = Sequence::kActive;
    s1.prompt                                = {21, 22, 23, 24, 25};
    s1.cache_len                             = 4;
    s1.staged_linear_block_begin             = 0;
    s1.staged_linear_block_count             = 2;
    s1.staged_conv_snapshots =
        {{2, manager.num_linear_layers_, manager.d_conv_, manager.conv_dim_}, manager.linear_conv_dtype_, kDEVICE};
    s1.staged_recurrent_snapshots = {{2, manager.num_linear_layers_, manager.num_v_heads_, manager.key_head_dim_, manager.value_head_dim_},
                                     manager.linear_state_dtype_,
                                     kDEVICE};
    FillTensorByBlock(s1.staged_conv_snapshots, {1.f, 2.f});
    FillTensorByBlock(s1.staged_recurrent_snapshots, {3.f, 4.f});
    s1.staged_linear_block_valid.assign(2, 1);

    manager.CachePrompt({seq1}, 1);

    auto seq2 = manager.Create(2);
    manager.AcquireLinearStateSlot(*seq2);

    auto& s2    = const_cast<Sequence&>(*seq2);
    s2.prompt   = s1.prompt;
    s2.cache_len = 0;

    Sequences sequences{seq2};
    manager.PrefixMatch(sequences, {0});

    REQUIRE(s2.blocks == s1.blocks);
    REQUIRE(s2.block_unique_ids == s1.block_unique_ids);
    REQUIRE(s2.cache_len == 4);
    REQUIRE(s2.linear_restore_snapshot_slot >= 0);
    REQUIRE(s2.linear_restore_snapshot_unique_id != 0);
    REQUIRE_FALSE(s2.linear_states_need_reset);
    REQUIRE(AllClose(ToHostVector(s2.conv_states), 2.f));
    REQUIRE(AllClose(ToHostVector(s2.recurrent_states), 4.f));
}

TEST_CASE("SequenceManager falls back to a prefix miss when linear snapshot metadata is missing",
          "[prefix_cache][linear][sequence_manager]")
{
    DeviceTestContext ctx;

    const auto  model                = MakeModelParam(true);
    constexpr int cache_block_seq_len = 2;
    constexpr int attn_tp_size        = 1;
    constexpr int max_batch_size      = 2;

    SequenceManager manager(model,
                            kFloat,
                            cache_block_seq_len,
                            attn_tp_size,
                            max_batch_size,
                            8,
                            -1,
                            true,
                            1,
                            1,
                            ctx.device_allocator,
                            [] { return size_t{1 << 20}; });

    auto seq1 = manager.Create(1);
    manager.AcquireLinearStateSlot(*seq1);

    auto& s1 = const_cast<Sequence&>(*seq1);
    std::tie(s1.blocks, s1.block_unique_ids) = manager.block_manager_->Allocate(2);
    s1.status                                = Sequence::kActive;
    s1.prompt                                = {31, 32, 33, 34, 35};
    s1.cache_len                             = 4;
    manager.CachePrompt({seq1}, 1);

    auto seq2 = manager.Create(2);
    manager.AcquireLinearStateSlot(*seq2);

    auto& s2    = const_cast<Sequence&>(*seq2);
    s2.prompt   = s1.prompt;
    s2.cache_len = 0;

    Sequences sequences{seq2};
    manager.PrefixMatch(sequences, {0});

    REQUIRE(s2.blocks.empty());
    REQUIRE(s2.block_unique_ids.empty());
    REQUIRE(s2.cache_len == 0);
    REQUIRE(s2.linear_restore_snapshot_slot == -1);
    REQUIRE(s2.linear_restore_snapshot_unique_id == 0);
    REQUIRE(s2.linear_states_need_reset);
}

TEST_CASE("PublishLinearSnapshot releases stale slot hint when staged snapshot is missing",
          "[prefix_cache][linear][sequence_manager]")
{
    DeviceTestContext ctx;

    const auto  model                = MakeModelParam(true);
    constexpr int cache_block_seq_len = 2;
    constexpr int attn_tp_size        = 1;
    constexpr int max_batch_size      = 2;

    SequenceManager manager(model,
                            kFloat,
                            cache_block_seq_len,
                            attn_tp_size,
                            max_batch_size,
                            8,
                            -1,
                            true,
                            1,
                            1,
                            ctx.device_allocator,
                            [] { return size_t{1 << 20}; });

    const auto stats_before = SequenceManager::LinearPrefixCacheStats();

    REQUIRE_FALSE(manager.free_linear_snapshot_slots_.empty());
    const int slot_hint = manager.free_linear_snapshot_slots_.back();
    manager.free_linear_snapshot_slots_.pop_back();
    manager.linear_snapshot_unique_ids_[slot_hint] = 99;

    Sequence seq{1};
    seq.staged_linear_block_valid.assign(1, 0);
    seq.staged_linear_block_begin = 0;
    seq.staged_linear_block_count = 1;
    seq.staged_conv_snapshots =
        {{1, manager.num_linear_layers_, manager.d_conv_, manager.conv_dim_}, manager.linear_conv_dtype_, kDEVICE};
    seq.staged_recurrent_snapshots = {
        {1, manager.num_linear_layers_, manager.num_v_heads_, manager.key_head_dim_, manager.value_head_dim_},
        manager.linear_state_dtype_,
        kDEVICE};

    const auto [out_slot, out_uid] = manager.PublishLinearSnapshot(seq, 0, slot_hint);
    REQUIRE(out_slot == -1);
    REQUIRE(out_uid == 0);
    REQUIRE(manager.linear_snapshot_unique_ids_[slot_hint] == 0);
    REQUIRE(std::find(manager.free_linear_snapshot_slots_.begin(), manager.free_linear_snapshot_slots_.end(), slot_hint)
            != manager.free_linear_snapshot_slots_.end());

    const auto stats_after = SequenceManager::LinearPrefixCacheStats();
    REQUIRE(std::get<1>(stats_after) == std::get<1>(stats_before) + 1);
}

TEST_CASE("PublishLinearSnapshot uses compact row relative to staged_linear_block_begin",
          "[prefix_cache][linear][sequence_manager]")
{
    DeviceTestContext ctx;

    const auto  model                = MakeModelParam(true);
    constexpr int cache_block_seq_len = 2;
    constexpr int attn_tp_size        = 1;
    constexpr int max_batch_size      = 2;

    SequenceManager manager(model,
                            kFloat,
                            cache_block_seq_len,
                            attn_tp_size,
                            max_batch_size,
                            8,
                            -1,
                            true,
                            1,
                            1,
                            ctx.device_allocator,
                            [] { return size_t{1 << 20}; });

    Sequence seq{1};
    seq.staged_linear_block_valid.assign(4, 0);
    seq.staged_linear_block_valid[2]          = 1;
    seq.staged_linear_block_begin             = 2;
    seq.staged_linear_block_count             = 1;
    seq.staged_conv_snapshots =
        {{1, manager.num_linear_layers_, manager.d_conv_, manager.conv_dim_}, manager.linear_conv_dtype_, kDEVICE};
    seq.staged_recurrent_snapshots = {
        {1, manager.num_linear_layers_, manager.num_v_heads_, manager.key_head_dim_, manager.value_head_dim_},
        manager.linear_state_dtype_,
        kDEVICE};
    FillTensor(seq.staged_conv_snapshots, 7.f);
    FillTensor(seq.staged_recurrent_snapshots, 8.f);

    const auto [slot, uid] = manager.PublishLinearSnapshot(seq, 2, -1);
    REQUIRE(slot >= 0);
    REQUIRE(uid != 0);
    Tensor host_conv{manager.pooled_prefix_conv_snapshots_.slice(slot, 1).squeeze(0).shape(),
                     manager.linear_conv_dtype_,
                     kCPU};
    Copy(manager.pooled_prefix_conv_snapshots_.slice(slot, 1).squeeze(0), host_conv);
    core::Context::stream().Sync();
    REQUIRE(AllClose(ToHostVector(host_conv), 7.f));
}

TEST_CASE("PublishLinearSnapshot returns miss when linear snapshot pool is exhausted",
          "[prefix_cache][linear][sequence_manager]")
{
    DeviceTestContext ctx;

    const auto  model                = MakeModelParam(true);
    constexpr int cache_block_seq_len = 2;
    constexpr int attn_tp_size        = 1;
    constexpr int max_batch_size      = 2;

    SequenceManager manager(model,
                            kFloat,
                            cache_block_seq_len,
                            attn_tp_size,
                            max_batch_size,
                            2,
                            -1,
                            true,
                            1,
                            1,
                            ctx.device_allocator,
                            [] { return size_t{1 << 20}; });

    while (!manager.free_linear_snapshot_slots_.empty()) {
        manager.free_linear_snapshot_slots_.pop_back();
    }
    REQUIRE(manager.free_linear_snapshot_slots_.empty());

    Sequence seq{1};
    seq.staged_linear_block_valid.assign(1, 1);
    seq.staged_linear_block_begin = 0;
    seq.staged_linear_block_count = 1;
    seq.staged_conv_snapshots =
        {{1, manager.num_linear_layers_, manager.d_conv_, manager.conv_dim_}, manager.linear_conv_dtype_, kDEVICE};
    seq.staged_recurrent_snapshots = {
        {1, manager.num_linear_layers_, manager.num_v_heads_, manager.key_head_dim_, manager.value_head_dim_},
        manager.linear_state_dtype_,
        kDEVICE};

    const auto stats_before = SequenceManager::LinearPrefixCacheStats();
    const auto [out_slot, out_uid] = manager.PublishLinearSnapshot(seq, 0, -1);
    const auto stats_after         = SequenceManager::LinearPrefixCacheStats();

    REQUIRE(out_slot == -1);
    REQUIRE(out_uid == 0);
    REQUIRE(std::get<2>(stats_after) == std::get<2>(stats_before) + 1);
}

TEST_CASE("PrefixMatch increments skipped-alpha counter when alpha is non-zero",
          "[prefix_cache][linear][sequence_manager]")
{
    DeviceTestContext ctx;

    const auto  model                = MakeModelParam(true);
    constexpr int cache_block_seq_len = 2;
    constexpr int attn_tp_size        = 1;
    constexpr int max_batch_size      = 2;

    SequenceManager manager(model,
                            kFloat,
                            cache_block_seq_len,
                            attn_tp_size,
                            max_batch_size,
                            8,
                            -1,
                            true,
                            1,
                            1,
                            ctx.device_allocator,
                            [] { return size_t{1 << 20}; });

    auto s = manager.Create(1);
    auto& seq = const_cast<Sequence&>(*s);
    seq.prompt    = {1, 2, 3, 4, 5};
    seq.cache_len = 0;

    const auto before = SequenceManager::LinearPrefixCacheStats();
    Sequences seqs{s};
    manager.PrefixMatch(seqs, {3});
    const auto after = SequenceManager::LinearPrefixCacheStats();
    REQUIRE(std::get<3>(after) == std::get<3>(before) + 1);
}

TEST_CASE("PrepareLinearCheckpointStaging compacts windows across passes and applies alpha",
          "[prefix_cache][linear][sequence_manager]")
{
    DeviceTestContext ctx;

    const auto    model               = MakeModelParam(true);
    constexpr int cache_block_seq_len = 4;
    constexpr int attn_tp_size        = 1;
    constexpr int max_batch_size      = 2;

    SequenceManager manager(model,
                            kFloat,
                            cache_block_seq_len,
                            attn_tp_size,
                            max_batch_size,
                            8,
                            -1,
                            true,
                            1,
                            1,
                            ctx.device_allocator,
                            [] { return size_t{1 << 20}; });

    const Sequence* s = manager.Create(1);
    auto&           seq = const_cast<Sequence&>(*s);
    manager.AcquireLinearStateSlot(seq);

    auto req = std::make_shared<Request>();
    RequestCache rc(req, *s);

    seq.prompt    = {};
    seq.cache_len = 0;

    rc.history_len = 0;
    rc.alpha       = 0;
    rc.input_len   = 16;
    manager.PrepareLinearCheckpointStaging(rc);
    REQUIRE(seq.staged_linear_block_begin == 0);
    REQUIRE(seq.staged_linear_block_count == 4);
    REQUIRE(seq.staged_conv_snapshots);
    REQUIRE(seq.staged_conv_snapshots.shape(0) == 4);

    rc.history_len = 16;
    rc.alpha       = 0;
    rc.input_len   = 8;
    manager.PrepareLinearCheckpointStaging(rc);
    REQUIRE(seq.staged_linear_block_begin == 4);
    REQUIRE(seq.staged_linear_block_count == 2);
    REQUIRE(seq.staged_conv_snapshots.shape(0) == 2);

    rc.history_len = 8;
    rc.alpha       = 8;
    rc.input_len   = 8;
    manager.PrepareLinearCheckpointStaging(rc);
    REQUIRE(seq.staged_linear_block_begin == 4);
    REQUIRE(seq.staged_linear_block_count == 2);
    REQUIRE(seq.staged_conv_snapshots.shape(0) == 2);
}

}  // namespace turbomind
