// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <math_constants.h>

#include "src/turbomind/kernels/attention/rotary_embedding.h"
#include "src/turbomind/kernels/attention/verification/attention.h"
#include "src/turbomind/kernels/attention/verification/paged_kv.cuh"
#include "src/turbomind/kernels/attention/verification/policy_sm90.cuh"
#include "src/turbomind/kernels/core/array_ops.h"

namespace turbomind::verification_attention {

// Generic mma.sync/ldmatrix fallback.  It is compiled into the SM90a
// verification library but uses the SM80 tensor-core instruction family.

CUTE_DEVICE float ReduceRowMax(float value)
{
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
    return value;
}

CUTE_DEVICE float ReduceRowSum(float value)
{
    value += __shfl_xor_sync(0xffffffffu, value, 1);
    value += __shfl_xor_sync(0xffffffffu, value, 2);
    return value;
}

template<class T, class Policy, bool StorePartial>
struct VerificationAttentionMainloop {
    Arguments                 arguments;
    SharedStorage<T, Policy>& storage;

    CUTE_DEVICE auto decode_m(int m, int m_begin) const
    {
        const int flat_m         = m_begin + m;
        int query_position;
        int head_in_group;
        arguments.query_group_size_divmod(query_position, head_in_group, flat_m);
        return cute::make_coord(query_position, head_in_group);
    }

    CUTE_DEVICE void run()
    {
        using Mma          = Sm90Mma<T, Policy>;
        using TiledMmaQK   = typename Mma::QK;
        using TiledMmaPV   = typename Mma::PV;
        using QueryCopy    = QTileCopy<T, Policy>;
        using KeyValueCopy = KvTileCopy<T, Policy>;

        static_assert(KeyValueCopy::AccessCount == (Policy::Threads == 256 ? 4 : 8));
        static_assert(Policy::Stages == 3);

        using QLayout  = SmemLayout2D<Policy::MTile, Policy::HeadDim>;
        using KVLayout = SmemLayout3D<Policy::KeyTile, Policy::HeadDim, Policy::Stages>;
        using PLayout  = SmemLayout2D<Policy::MTile, Policy::KeyTile>;

        auto shared_q = cute::make_tensor(cute::make_smem_ptr(storage.q), QLayout{});
        auto shared_k = cute::make_tensor(cute::make_smem_ptr(storage.body.k), KVLayout{});
        auto shared_v = cute::make_tensor(cute::make_smem_ptr(storage.body.v), KVLayout{});
        auto shared_probability =
            cute::make_tensor(cute::make_smem_ptr(storage.body.probability), PLayout{});
        auto global_out = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<T*>(arguments.out)),
            cute::make_layout(
                cute::make_shape(arguments.query_count,
                                 arguments.query_head_count,
                                 cute::Int<Policy::HeadDim>{}),
                cute::make_stride(arguments.query_head_count * Policy::HeadDim,
                                  cute::Int<Policy::HeadDim>{},
                                  cute::_1{})));
        auto partial_o = cute::make_tensor(
            cute::make_gmem_ptr(arguments.partial_o),
            cute::make_layout(
                cute::make_shape(arguments.query_count,
                                 arguments.split_count,
                                 arguments.query_head_count,
                                 cute::Int<Policy::HeadDim>{}),
                cute::make_stride(arguments.split_count * arguments.query_head_count * Policy::HeadDim,
                                  arguments.query_head_count * Policy::HeadDim,
                                  cute::Int<Policy::HeadDim>{},
                                  cute::_1{})));
        auto partial_ml = cute::make_tensor(
            cute::make_gmem_ptr(arguments.partial_ml),
            cute::make_layout(
                cute::make_shape(arguments.query_count,
                                 arguments.split_count,
                                 arguments.query_head_count,
                                 cute::_2{}),
                cute::make_stride(arguments.split_count * arguments.query_head_count * 2,
                                  arguments.query_head_count * 2,
                                  cute::_2{},
                                  cute::_1{})));

        const int request          = blockIdx.x;
        const int query_group_size = arguments.query_group_size;
        const int m_count          = arguments.max_query_length * query_group_size;
        const int m_slices         = (m_count + Policy::MTile - 1) / Policy::MTile;
        auto block_head = cute::idx2crd(
            static_cast<int>(blockIdx.y),
            cute::make_shape(m_slices, arguments.kv_head_count));
        const int m_begin           = cute::get<0>(block_head) * Policy::MTile;
        const int kv_head           = cute::get<1>(block_head);
        const int query_begin       = arguments.q_offsets[request];
        const int query_end         = arguments.q_offsets[request + 1];
        const int query_length      = query_end - query_begin;
        const int key_length        = arguments.k_offsets[request + 1] - arguments.k_offsets[request];
        const int history_length    = key_length - query_length;
        const int tile_count      = (key_length + Policy::KeyTile - 1) / Policy::KeyTile;
        const int tiles_per_split = (tile_count + arguments.split_count - 1) / arguments.split_count;
        const int first_tile      = blockIdx.z * tiles_per_split;
        const int last_tile       = min(tile_count, first_tile + tiles_per_split);
        const bool empty_split    = first_tile >= last_tile;
        const bool finished         = arguments.finished && arguments.finished[request];

        const bool inactive = finished || empty_split;

        auto thread_mma_pv = TiledMmaPV{}.get_slice(threadIdx.x);
        auto pv_identity = cute::make_identity_tensor(typename Policy::PvShape{});
        auto pv_output_coordinates = thread_mma_pv.partition_C(pv_identity);
        const int row0 = cute::get<0>(pv_output_coordinates(0));
        auto pv_output_prototype = thread_mma_pv.make_fragment_C(pv_output_coordinates);
        using PvOutputLayout = typename decltype(pv_output_prototype)::layout_type;
        static_assert(cute::rank_v<PvOutputLayout> == 3);
        static constexpr int PvOutputValues = cute::cosize_v<PvOutputLayout>;
        using PvTileMode = cute::Layout<
            cute::Shape<cute::Int<Policy::PvTileCount>>,
            cute::Stride<cute::Int<PvOutputValues>>>;
        using OutputFragmentsLayout = decltype(cute::append(PvOutputLayout{}, PvTileMode{}));

        cutlass::Array<float, PvOutputValues * Policy::PvTileCount> output_storage;
        auto output_fragments = cute::make_tensor(
            cute::make_rmem_ptr(output_storage.data()), OutputFragmentsLayout{});
        cute::clear(output_fragments);

        float running_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
        float running_sum[2] = {0.f, 0.f};

        if (!inactive) {

        auto global_q = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<const T*>(arguments.q)),
            cute::make_layout(
                cute::make_shape(arguments.query_count,
                                 arguments.query_head_count,
                                 cute::Int<Policy::HeadDim>{}),
                cute::make_stride(arguments.q_stride,
                                  cute::Int<Policy::HeadDim>{},
                                  cute::_1{})));
        auto global_q_bias = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<const T*>(arguments.q_bias)),
            cute::make_layout(
                cute::make_shape(arguments.query_head_count,
                                 cute::Int<Policy::HeadDim>{}),
                cute::make_stride(cute::Int<Policy::HeadDim>{}, cute::_1{})));

        using RegisterCopy = cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, T>;
        auto q_identity    = cute::make_identity_tensor(typename Policy::QShape{});
        auto tiled_q_copy  = typename QueryCopy::TiledCopy{};
        auto thread_q_copy = tiled_q_copy.get_thread_slice(threadIdx.x);
        auto q_coordinates = cute::group_modes<
            1, cute::rank_v<decltype(thread_q_copy.partition_S(q_identity))>>(
            thread_q_copy.partition_S(q_identity));
        auto q_destinations = cute::group_modes<
            1, cute::rank_v<decltype(thread_q_copy.partition_D(shared_q))>>(
            thread_q_copy.partition_D(shared_q));
        Array<T, QueryCopy::ValuesPerAccess> fragment;
        auto fragment_tensor = cute::make_tensor(
            cute::make_rmem_ptr(fragment.data()),
            cute::make_layout(cute::Int<QueryCopy::ValuesPerAccess>{}));
        Array<T, QueryCopy::ValuesPerAccess> bias;
        auto bias_fragment = cute::make_tensor(
            cute::make_rmem_ptr(bias.data()),
            cute::make_layout(cute::Int<QueryCopy::ValuesPerAccess>{}));

        CUTE_UNROLL
        for (int access = 0; access < QueryCopy::AccessCount; ++access) {
            const auto md            = q_coordinates(cute::_0{}, access);
            const auto qh            = decode_m(cute::get<0>(md), m_begin);
            const int query_position = cute::get<0>(qh);
            const int head_in_group  = cute::get<1>(qh);
            const int d_begin        = cute::get<1>(md);
            const bool valid = query_position < query_length && head_in_group < query_group_size;

            CUTE_UNROLL
            for (int value = 0; value < QueryCopy::ValuesPerAccess; ++value) {
                fragment[value] = T(0);
            }
            if (valid) {
                const int query_head = kv_head * query_group_size + head_in_group;
                auto source = cute::make_tensor(
                    cute::make_gmem_ptr(&global_q(query_begin + query_position, query_head, d_begin)),
                    cute::make_layout(cute::Int<QueryCopy::ValuesPerAccess>{}));
                cute::copy(RegisterCopy{}, source, fragment_tensor);

                if (arguments.q_bias) {
                    auto bias_source = cute::make_tensor(
                        cute::make_gmem_ptr(&global_q_bias(query_head, d_begin)),
                        cute::make_layout(cute::Int<QueryCopy::ValuesPerAccess>{}));
                    cute::copy(RegisterCopy{}, bias_source, bias_fragment);
                    CUTE_UNROLL
                    for (int value = 0; value < QueryCopy::ValuesPerAccess; ++value) {
                        fragment[value] = fragment[value] + bias[value];
                    }
                }

                FastRoPE<QueryCopy::ValuesPerAccess> rope(
                    arguments.rope,
                    request,
                    std::integral_constant<int, QueryCopy::ValuesPerAccess>{});
                rope.init(d_begin);
                rope.apply(fragment, history_length + query_position);

            }
            cute::copy(RegisterCopy{}, fragment_tensor, q_destinations(cute::_, access));
        }
        __syncthreads();

        auto thread_mma_qk = TiledMmaQK{}.get_slice(threadIdx.x);
        auto tiled_copy_q  = typename Mma::CopyQkA{};
        auto thread_copy_q = tiled_copy_q.get_slice(threadIdx.x);
        auto tiled_copy_k  = typename Mma::CopyQkB{};
        auto thread_copy_k = tiled_copy_k.get_slice(threadIdx.x);
        auto tiled_copy_v  = typename Mma::CopyPvB{};
        auto thread_copy_v = tiled_copy_v.get_slice(threadIdx.x);

        auto q_registers = thread_mma_qk.partition_fragment_A(shared_q);
        auto q_source    = thread_copy_q.partition_S(shared_q);
        auto q_target    = thread_copy_q.retile_D(q_registers);
        cute::copy(tiled_copy_q, q_source, q_target);
        // Q and K share storage; finish every Q read before staging K.
        __syncthreads();

        PagedKv<T, Policy::HeadDim> cache(arguments.block_ptrs,
                                          arguments.block_ptr_offsets,
                                          request,
                                          kv_head,
                                          arguments.kv_head_count,
                                          arguments.block_len,
                                          arguments.block_len_divmod,
                                          arguments.cache_block_offset);

        const int split_tile_count = last_tile - first_tile;
        int issued_tiles       = 0;
        int consumed_tiles     = 0;
        int outstanding_groups = 0;
        int read_stage         = 0;
        int write_stage        = 0;

        CUTE_UNROLL
        for (int prologue = 0; prologue < Policy::Stages - 1; ++prologue) {
            if (prologue < split_tile_count) {
                const int load_tile      = first_tile + issued_tiles;
                const int load_key_begin = load_tile * Policy::KeyTile;
                auto shared_k_write = shared_k(cute::_, cute::_, write_stage);
                auto shared_v_write = shared_v(cute::_, cute::_, write_stage);
                copy_paged_tile<false, T, Policy>(cache, load_key_begin, key_length, shared_k_write);
                copy_paged_tile<true, T, Policy>(cache, load_key_begin, key_length, shared_v_write);
                cute::cp_async_fence();
                ++issued_tiles;
                ++outstanding_groups;
                write_stage = (write_stage + 1) % Policy::Stages;
            }
        }

        for (; consumed_tiles < split_tile_count; ++consumed_tiles) {
            if (outstanding_groups == 1) {
                cute::cp_async_wait<0>();
            }
            else {
                cute::cp_async_wait<1>();
            }
            __syncthreads();
            --outstanding_groups;

            const int current_tile      = first_tile + consumed_tiles;
            const int current_key_begin = current_tile * Policy::KeyTile;
            auto shared_k_read = shared_k(cute::_, cute::_, read_stage);
            auto shared_v_read = shared_v(cute::_, cute::_, read_stage);

            if (issued_tiles < split_tile_count) {
                const int load_tile      = first_tile + issued_tiles;
                const int load_key_begin = load_tile * Policy::KeyTile;
                auto shared_k_write = shared_k(cute::_, cute::_, write_stage);
                auto shared_v_write = shared_v(cute::_, cute::_, write_stage);
                copy_paged_tile<false, T, Policy>(cache, load_key_begin, key_length, shared_k_write);
                copy_paged_tile<true, T, Policy>(cache, load_key_begin, key_length, shared_v_write);
                cute::cp_async_fence();
                ++issued_tiles;
                ++outstanding_groups;
                write_stage = (write_stage + 1) % Policy::Stages;
            }

            auto score_identity = cute::make_identity_tensor(typename Policy::ScoreShape{});
            auto score_coordinates = thread_mma_qk.partition_C(score_identity);
            auto score_fragment = thread_mma_qk.make_fragment_C(score_coordinates);
            cute::clear(score_fragment);

            CUTE_UNROLL
            for (int d_tile = 0; d_tile < Policy::HeadDim / 16; ++d_tile) {
                auto key_tile = cute::local_tile(
                    shared_k_read,
                    cute::Shape<cute::Int<Policy::KeyTile>, cute::_16>{},
                    cute::make_coord(cute::_0{}, d_tile));
                auto key_fragment = thread_mma_qk.partition_fragment_B(key_tile);
                cute::copy(tiled_copy_k,
                           thread_copy_k.partition_S(key_tile),
                           thread_copy_k.retile_D(key_fragment));
                auto q_slice = q_registers(cute::_, cute::_, d_tile);
                auto q_single_k = cute::make_tensor(q_slice.data(), cute::append(q_slice.layout()));
                cute::gemm(thread_mma_qk, q_single_k, key_fragment, score_fragment);
            }

            float local_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
            CUTE_UNROLL
            for (int i = 0; i < cute::size(score_fragment); ++i) {
                const auto mk            = score_coordinates(i);
                const auto qh            = decode_m(cute::get<0>(mk), m_begin);
                const int query_position = cute::get<0>(qh);
                const int head_in_group  = cute::get<1>(qh);
                const int key_in_tile    = cute::get<1>(mk);
                const int absolute_key   = current_key_begin + key_in_tile;
                const int last_valid     = history_length + query_position;
                const int first_valid    = max(0, last_valid - arguments.window_size + 1);
                const bool valid = query_position < query_length && head_in_group < query_group_size
                                   && absolute_key >= first_valid && absolute_key <= last_valid;
                const float score = valid ? score_fragment(i) * arguments.qk_scale_log2 :
                                            -CUDART_INF_F;
                score_fragment(i) = score;
                const int row_slot = cute::get<0>(mk) == row0 ? 0 : 1;
                local_max[row_slot] = fmaxf(local_max[row_slot], score);
            }

            float new_max[2];
            float old_scale[2];
            CUTE_UNROLL
            for (int row = 0; row < 2; ++row) {
                const float tile_max = ReduceRowMax(local_max[row]);
                new_max[row] = fmaxf(running_max[row], tile_max);
                old_scale[row] = running_max[row] == -CUDART_INF_F ?
                                     0.f : exp2f(running_max[row] - new_max[row]);
            }

            CUTE_UNROLL
            for (int pv_tile = 0; pv_tile < Policy::PvTileCount; ++pv_tile) {
                auto output_tile = output_fragments(cute::_, cute::_, cute::_, pv_tile);
                CUTE_UNROLL
                for (int i = 0; i < cute::size(output_tile); ++i) {
                    const int row = cute::get<0>(pv_output_coordinates(i));
                    output_tile(i) *= old_scale[row == row0 ? 0 : 1];
                }
            }

            using ProbabilityStoreLayout =
                typename decltype(cute::make_fragment_like<T>(score_fragment))::layout_type;
            auto probability_store_fragment = cute::make_tensor(
                cute::recast_ptr<T>(score_fragment.data()), ProbabilityStoreLayout{});
            float local_sum[2] = {0.f, 0.f};
            CUTE_UNROLL
            for (int i = 0; i < cute::size(score_fragment); ++i) {
                const int row = cute::get<0>(score_coordinates(i));
                const int row_slot = row == row0 ? 0 : 1;
                const float score = score_fragment(i);
                const float probability = score == -CUDART_INF_F ?
                                              0.f : exp2f(score - new_max[row_slot]);
                probability_store_fragment(i) = static_cast<T>(probability);
                local_sum[row_slot] += probability;
            }

            CUTE_UNROLL
            for (int row = 0; row < 2; ++row) {
                const float tile_sum = ReduceRowSum(local_sum[row]);
                running_sum[row] = running_sum[row] * old_scale[row] + tile_sum;
                running_max[row] = new_max[row];
            }

            auto store_p        = typename Mma::StoreProbability{};
            auto thread_store_p = store_p.get_thread_slice(threadIdx.x);
            cute::copy(store_p,
                       thread_store_p.retile_S(probability_store_fragment),
                       thread_store_p.partition_D(shared_probability));
            __syncthreads();

            auto probability_fragment = thread_mma_pv.partition_fragment_A(shared_probability);
            auto load_p        = typename Mma::CopyPvA{};
            auto thread_load_p = load_p.get_thread_slice(threadIdx.x);
            cute::copy(load_p,
                       thread_load_p.partition_S(shared_probability),
                       thread_load_p.retile_D(probability_fragment));

            auto value_for_pv = cute::composition(
                shared_v_read,
                cute::Layout<
                    cute::Shape<cute::Int<Policy::HeadDim>, cute::Int<Policy::KeyTile>>,
                    cute::Stride<cute::Int<Policy::KeyTile>, cute::_1>>{});
            CUTE_UNROLL
            for (int pv_tile = 0; pv_tile < Policy::PvTileCount; ++pv_tile) {
                auto output_tile = output_fragments(cute::_, cute::_, cute::_, pv_tile);
                auto value_tile = cute::local_tile(
                    value_for_pv,
                    cute::Shape<cute::Int<Policy::PvNtile>, cute::Int<Policy::KeyTile>>{},
                    cute::make_coord(pv_tile, cute::_0{}));
                auto value_fragment = thread_mma_pv.partition_fragment_B(value_tile);
                auto value_source   = thread_copy_v.partition_S(value_tile);
                auto value_target   = thread_copy_v.retile_D(value_fragment);
                CUTE_UNROLL
                for (int k_block = 0; k_block < cute::size<2>(value_fragment); ++k_block) {
                    const auto k_coord = cute::idx2crd(k_block, cute::shape<2>(value_fragment));
                    cute::copy(typename Mma::CopyPvBAtom{},
                               value_source(cute::_, cute::_, k_coord),
                               value_target(cute::_, cute::_, k_coord));
                }
                cute::gemm(thread_mma_pv, probability_fragment, value_fragment, output_tile);
            }

            __syncthreads();
            read_stage = (read_stage + 1) % Policy::Stages;
        }
        }

        CUTE_UNROLL
        for (int pv_tile = 0; pv_tile < Policy::PvTileCount; ++pv_tile) {
            auto output_tile = output_fragments(cute::_, cute::_, cute::_, pv_tile);
            CUTE_UNROLL
            for (int i = 0; i < cute::size(output_tile); ++i) {
                const auto mn            = pv_output_coordinates(i);
                const auto qh            = decode_m(cute::get<0>(mn), m_begin);
                const int query_position = cute::get<0>(qh);
                const int head_in_group  = cute::get<1>(qh);
                if (query_position < query_length && head_in_group < query_group_size) {
                    const int absolute_query = query_begin + query_position;
                    const int query_head = kv_head * query_group_size + head_in_group;
                    const int d = cute::crd2idx(
                        cute::make_coord(cute::get<1>(mn), pv_tile),
                        cute::Shape<cute::Int<Policy::PvNtile>, cute::Int<Policy::PvTileCount>>{});
                    const int row_slot = cute::get<0>(mn) == row0 ? 0 : 1;
                    if constexpr (!StorePartial) {
                        global_out(absolute_query, query_head, d) = running_sum[row_slot] == 0.f ?
                            T(0) : static_cast<T>(output_tile(i) / running_sum[row_slot]);
                    }
                    else {
                        const int local_query = absolute_query - arguments.query_offset;
                        partial_o(local_query,
                                  static_cast<int>(blockIdx.z),
                                  query_head,
                                  d) = output_tile(i);
                    }
                }
            }
        }

        if constexpr (StorePartial) {
            CUTE_UNROLL
            for (int i = 0; i < cute::size(pv_output_coordinates); ++i) {
                const auto mn = pv_output_coordinates(i);
                if (cute::get<1>(mn) == 0) {
                    const auto qh            = decode_m(cute::get<0>(mn), m_begin);
                    const int query_position = cute::get<0>(qh);
                    const int head_in_group  = cute::get<1>(qh);
                    if (query_position < query_length && head_in_group < query_group_size) {
                        const int absolute_query = query_begin + query_position;
                        const int local_query    = absolute_query - arguments.query_offset;
                        const int query_head = kv_head * query_group_size + head_in_group;
                        const int row_slot = cute::get<0>(mn) == row0 ? 0 : 1;
                        auto ml = partial_ml(local_query,
                                             static_cast<int>(blockIdx.z),
                                             query_head,
                                             cute::_);
                        ml(cute::_0{}) = running_max[row_slot];
                        ml(cute::_1{}) = running_sum[row_slot];
                    }
                }
            }
        }
    }
};

template<class T, class Policy, bool StorePartial>
__global__ __maxnreg__(Policy::HeadDim == 128 ? 209 : 255)
void VerificationAttentionKernel(Arguments arguments)
{
    extern __shared__ char dynamic_shared[];
    auto& storage = *reinterpret_cast<SharedStorage<T, Policy>*>(dynamic_shared);
    VerificationAttentionMainloop<T, Policy, StorePartial>{arguments, storage}.run();
}

void Reduce(const Arguments& arguments);

}  // namespace turbomind::verification_attention
