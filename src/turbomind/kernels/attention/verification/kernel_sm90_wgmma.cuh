// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <math_constants.h>

#include "src/turbomind/kernels/attention/rotary_embedding.h"
#include "src/turbomind/kernels/attention/verification/attention.h"
#include "src/turbomind/kernels/attention/verification/paged_kv.cuh"
#include "src/turbomind/kernels/core/array_ops.h"

namespace turbomind::verification_attention {

CUTE_DEVICE static void sync_named_barrier(int barrier_id, int thread_count)
{
    asm volatile("bar.sync %0, %1;"
                 :
                 : "r"(barrier_id), "r"(thread_count)
                 : "memory");
}

CUTE_DEVICE static void sync_warp_group_barrier(int warp_group)
{
    sync_named_barrier(8 + warp_group, 128);
}

CUTE_DEVICE static void sync_compute_groups()
{
    sync_named_barrier(10, 256);
}

struct Sm90WgmmaPolicy256 {
    static constexpr int HeadDim     = 256;
    static constexpr int Threads     = 128;
    static constexpr int MTile       = 32;
    static constexpr int KeyTile     = 64;
    static constexpr int PvNtile     = 64;
    static constexpr int PvTileCount = HeadDim / PvNtile;

    using QShape  = cute::Shape<cute::Int<MTile>, cute::Int<HeadDim>>;
    using KvShape = cute::Shape<cute::Int<KeyTile>, cute::Int<HeadDim>>;
};

template<class T, int MTile>
struct Sm90WgmmaMma;

template<>
struct Sm90WgmmaMma<cutlass::bfloat16_t, 32> {
    using QK = decltype(cute::make_tiled_mma(
        cute::SM90_64x32x16_F32BF16BF16_SS<
            cute::GMMA::Major::K, cute::GMMA::Major::K>{}));
    using PV = decltype(cute::make_tiled_mma(
        cute::SM90_64x32x16_F32BF16BF16_SS<
            cute::GMMA::Major::MN, cute::GMMA::Major::K>{}));
};

template<class T, class Policy>
struct Sm90WgmmaKvStorage {
    T k[Policy::KeyTile * Policy::HeadDim];
    T v[Policy::KeyTile * Policy::HeadDim];
};

template<class T, class Policy>
union alignas(128) Sm90WgmmaKvOrOutputStorage {
    Sm90WgmmaKvStorage<T, Policy> kv;
    float output[Policy::MTile * Policy::HeadDim];
};

template<class T, class Policy>
struct Sm90WgmmaSplitStorage {
    Sm90WgmmaKvOrOutputStorage<T, Policy> kv_or_output;
    alignas(128) T probability[Policy::MTile * Policy::KeyTile];
    alignas(16) float row_partials[4][Policy::MTile];
    alignas(16) float running_max[Policy::MTile];
    alignas(16) float running_sum[Policy::MTile];
    alignas(16) float old_scale[Policy::MTile];
};

template<class T, class Policy>
struct Sm90WgmmaSharedStorage {
    alignas(128) T q[Policy::MTile * Policy::HeadDim];
    Sm90WgmmaSplitStorage<T, Policy> split[2];
};

template<class T, bool StorePartial>
struct Sm90WgmmaMainloop {
    using Policy = Sm90WgmmaPolicy256;

    Arguments                       arguments;
    Sm90WgmmaSharedStorage<T, Policy>& storage;

    CUTE_DEVICE auto decode_row(int row) const
    {
        int query_position;
        int head_in_group;
        arguments.query_group_size_divmod(query_position, head_in_group, row);
        return cute::make_coord(query_position, head_in_group);
    }

    CUTE_DEVICE void run()
    {
        using Mma = Sm90WgmmaMma<T, Policy::MTile>;
        using QueryCopy = QTileCopy<T, Policy>;

        using QLayout = decltype(cute::tile_to_shape(
            cute::GMMA::Layout_K_SW128_Atom<T>{},
            cute::Shape<cute::Int<Policy::MTile>, cute::_256>{}));
        using KLayout = decltype(cute::tile_to_shape(
            cute::GMMA::Layout_K_SW128_Atom<T>{},
            cute::Shape<cute::_64, cute::_256>{}));
        using VLayout = decltype(cute::tile_to_shape(
            cute::GMMA::Layout_MN_SW128_Atom<T>{},
            cute::Shape<cute::_256, cute::_64>{}));
        using PLayout = decltype(cute::tile_to_shape(
            cute::GMMA::Layout_K_SW128_Atom<T>{},
            cute::Shape<cute::Int<Policy::MTile>, cute::_64>{}));

        const int warp_group = threadIdx.x / Policy::Threads;
        const int local_tid  = threadIdx.x % Policy::Threads;
        const int split      = StorePartial ? 2 * blockIdx.z + warp_group : warp_group;
        const int work_split_count = StorePartial ? arguments.split_count : 2;
        auto& split_storage = storage.split[warp_group];
        auto shared_q = cute::make_tensor(cute::make_smem_ptr(storage.q), QLayout{});
        auto shared_k = cute::make_tensor(
            cute::make_smem_ptr(split_storage.kv_or_output.kv.k), KLayout{});
        auto shared_v = cute::make_tensor(
            cute::make_smem_ptr(split_storage.kv_or_output.kv.v), VLayout{});
        auto shared_p = cute::make_tensor(cute::make_smem_ptr(split_storage.probability), PLayout{});
        auto v_copy_view = cute::composition(
            shared_v,
            cute::Layout<
                cute::Shape<cute::_64, cute::_256>,
                cute::Stride<cute::Int<256>, cute::_1>>{});

        const int request          = blockIdx.x;
        const int query_group_size = arguments.query_group_size;
        const int kv_head          = blockIdx.y;
        const int query_begin      = arguments.q_offsets[request];
        const int query_end        = arguments.q_offsets[request + 1];
        const int query_length     = query_end - query_begin;
        const int key_length       = arguments.k_offsets[request + 1] - arguments.k_offsets[request];
        const int history_length   = key_length - query_length;
        const int page_count       = (key_length + Policy::KeyTile - 1) / Policy::KeyTile;
        const int first_page = static_cast<int64_t>(page_count) * split / work_split_count;
        const int last_page = static_cast<int64_t>(page_count) * (split + 1) / work_split_count;
        const bool request_finished = arguments.finished && arguments.finished[request];
        const bool inactive = request_finished || first_page >= last_page;

        auto global_out = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<T*>(arguments.out)),
            cute::make_layout(
                cute::make_shape(arguments.query_count,
                                 arguments.query_head_count,
                                 cute::_256{}),
                cute::make_stride(arguments.query_head_count * Policy::HeadDim,
                                  cute::Int<Policy::HeadDim>{},
                                  cute::_1{})));
        auto partial_o = cute::make_tensor(
            cute::make_gmem_ptr(arguments.partial_o),
            cute::make_layout(
                cute::make_shape(arguments.query_count,
                                 arguments.split_count,
                                 arguments.query_head_count,
                                 cute::_256{}),
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

        auto pv_mma = typename Mma::PV{};
        auto thread_pv = pv_mma.get_slice(local_tid);
        auto pv_identity = cute::make_identity_tensor(
            cute::Shape<cute::_64, cute::Int<Policy::MTile>>{});
        auto pv_coordinates = thread_pv.partition_C(pv_identity);
        auto pv_prototype = thread_pv.make_fragment_C(pv_coordinates);
        using PvLayout = typename decltype(pv_prototype)::layout_type;
        static constexpr int PvValues = cute::cosize_v<PvLayout>;
        using PvTiles = cute::Layout<
            cute::Shape<cute::Int<Policy::PvTileCount>>,
            cute::Stride<cute::Int<PvValues>>>;
        using OutputLayout = decltype(cute::append(PvLayout{}, PvTiles{}));
        cutlass::Array<float, PvValues * Policy::PvTileCount> output_storage;
        auto output = cute::make_tensor(
            cute::make_rmem_ptr(output_storage.data()), OutputLayout{});
        cute::clear(output);

        if (local_tid < Policy::MTile) {
            split_storage.running_max[local_tid] = -CUDART_INF_F;
            split_storage.running_sum[local_tid] = 0.f;
            split_storage.old_scale[local_tid]   = 0.f;
        }
        sync_warp_group_barrier(warp_group);

        if (warp_group == 0 && !request_finished) {
            auto global_q = cute::make_tensor(
                cute::make_gmem_ptr(static_cast<const T*>(arguments.q)),
                cute::make_layout(
                    cute::make_shape(arguments.query_count,
                                     arguments.query_head_count,
                                     cute::_256{}),
                    cute::make_stride(arguments.q_stride,
                                      cute::Int<Policy::HeadDim>{},
                                      cute::_1{})));
            auto global_q_bias = cute::make_tensor(
                cute::make_gmem_ptr(static_cast<const T*>(arguments.q_bias)),
                cute::make_layout(
                    cute::make_shape(arguments.query_head_count, cute::_256{}),
                    cute::make_stride(cute::Int<Policy::HeadDim>{}, cute::_1{})));

            using RegisterCopy = cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, T>;
            auto q_identity    = cute::make_identity_tensor(typename Policy::QShape{});
            auto tiled_q_copy  = typename QueryCopy::TiledCopy{};
            auto thread_q_copy = tiled_q_copy.get_thread_slice(local_tid);
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
                const auto nd            = q_coordinates(cute::_0{}, access);
                const auto qh            = decode_row(cute::get<0>(nd));
                const int query_position = cute::get<0>(qh);
                const int head_in_group  = cute::get<1>(qh);
                const int d_begin        = cute::get<1>(nd);
                const bool valid = query_position < query_length
                                   && head_in_group < query_group_size;
                CUTE_UNROLL
                for (int value = 0; value < QueryCopy::ValuesPerAccess; ++value) {
                    fragment[value] = T(0);
                }
                if (valid) {
                    const int query_head = kv_head * query_group_size + head_in_group;
                    auto source = cute::make_tensor(
                        cute::make_gmem_ptr(&global_q(query_begin + query_position,
                                                      query_head,
                                                      d_begin)),
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
            cutlass::arch::fence_view_async_shared();
        }
        __syncthreads();

        if constexpr (StorePartial) {
            if (split >= arguments.split_count) {
                return;
            }
        }

        if (!inactive) {

            PagedKv<T, Policy::HeadDim> cache(arguments.block_ptrs,
                                              arguments.block_ptr_offsets,
                                              request,
                                              kv_head,
                                              arguments.kv_head_count,
                                              arguments.block_len,
                                              arguments.block_len_divmod,
                                              arguments.cache_block_offset);

            auto qk_mma = typename Mma::QK{};
            auto thread_qk = qk_mma.get_slice(local_tid);
            auto qk_identity = cute::make_identity_tensor(
                cute::Shape<cute::_64, cute::Int<Policy::MTile>>{});
            auto score_coordinates = thread_qk.partition_C(qk_identity);

            copy_paged_page<false, T, Policy>(
                cache, first_page, first_page * Policy::KeyTile, key_length, shared_k, local_tid);
            cute::cp_async_fence();
            copy_paged_page<true, T, Policy>(
                cache, first_page, first_page * Policy::KeyTile, key_length, v_copy_view, local_tid);
            cute::cp_async_fence();

            for (int page = first_page; page < last_page; ++page) {
                const int key_begin = page * Policy::KeyTile;
                cute::cp_async_wait<1>();
                sync_warp_group_barrier(warp_group);

                auto score = thread_qk.make_fragment_C(score_coordinates);
                cute::clear(score);
                auto k_source = thread_qk.partition_A(shared_k);
                auto q_source = thread_qk.partition_B(shared_q);
                auto k_fragment = thread_qk.make_fragment_A(k_source);
                auto q_fragment = thread_qk.make_fragment_B(q_source);
                cute::warpgroup_fence_operand(score);
                cute::warpgroup_arrive();
                cute::gemm(qk_mma, k_fragment, q_fragment, score);
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(score);

                const bool has_next_page = page + 1 < last_page;
                if (has_next_page) {
                    const int next_page = page + 1;
                    copy_paged_page<false, T, Policy>(
                        cache,
                        next_page,
                        next_page * Policy::KeyTile,
                        key_length,
                        shared_k,
                        local_tid);
                    cute::cp_async_fence();
                }

                static constexpr int RowSlots = Policy::MTile / 4;
                Array<float, RowSlots> row_max;
                CUTE_UNROLL
                for (int row_slot = 0; row_slot < RowSlots; ++row_slot) {
                    row_max[row_slot] = -CUDART_INF_F;
                }
                CUTE_UNROLL
                for (int i = 0; i < cute::size(score); ++i) {
                    const auto kn = score_coordinates(i);
                    const int key_in_page = cute::get<0>(kn);
                    const int row         = cute::get<1>(kn);
                    const auto qh         = decode_row(row);
                    const int query_position = cute::get<0>(qh);
                    const int head_in_group  = cute::get<1>(qh);
                    const int absolute_key   = key_begin + key_in_page;
                    const int last_valid     = history_length + query_position;
                    const int first_valid = max(0, last_valid - arguments.window_size + 1);
                    const bool valid = query_position < query_length
                                       && head_in_group < query_group_size
                                       && absolute_key >= first_valid
                                       && absolute_key <= last_valid;
                    score(i) = valid ? score(i) * arguments.qk_scale_log2 :
                                       -CUDART_INF_F;
                    const int row_slot = (i / 4) * 2 + i % 2;
                    row_max[row_slot] = fmaxf(row_max[row_slot], score(i));
                }

                CUTE_UNROLL
                for (int offset = 4; offset <= 16; offset *= 2) {
                    CUTE_UNROLL
                    for (int row_slot = 0; row_slot < RowSlots; ++row_slot) {
                        row_max[row_slot] = fmaxf(
                            row_max[row_slot],
                            __shfl_xor_sync(0xffffffffu,
                                            row_max[row_slot],
                                            offset));
                    }
                }
                const int lane = local_tid % 32;
                const int warp = local_tid / 32;
                if (lane < 4) {
                    CUTE_UNROLL
                    for (int row_slot = 0; row_slot < RowSlots; ++row_slot) {
                        const int fragment_index = (row_slot / 2) * 4 + row_slot % 2;
                        const int row = cute::get<1>(score_coordinates(fragment_index));
                        split_storage.row_partials[warp][row] = row_max[row_slot];
                    }
                }
                sync_warp_group_barrier(warp_group);

                if (local_tid < Policy::MTile) {
                    const int row = local_tid;
                    float page_max = -CUDART_INF_F;
                    CUTE_UNROLL
                    for (int source_warp = 0; source_warp < 4; ++source_warp) {
                        page_max = fmaxf(page_max,
                                         split_storage.row_partials[source_warp][row]);
                    }
                    const float old_max = split_storage.running_max[row];
                    const float new_max = fmaxf(old_max, page_max);
                    const float scale = old_max == -CUDART_INF_F ?
                                            0.f : exp2f(old_max - new_max);
                    split_storage.old_scale[row]   = scale;
                    split_storage.running_max[row] = new_max;
                }
                sync_warp_group_barrier(warp_group);

                Array<float, RowSlots> row_sum;
                CUTE_UNROLL
                for (int row_slot = 0; row_slot < RowSlots; ++row_slot) {
                    row_sum[row_slot] = 0.f;
                }
                CUTE_UNROLL
                for (int i = 0; i < cute::size(score); ++i) {
                    const int row = cute::get<1>(score_coordinates(i));
                    const float value = score(i);
                    const float probability = value == -CUDART_INF_F ?
                                                  0.f :
                                                  exp2f(value - split_storage.running_max[row]);
                    score(i) = probability;
                    const int row_slot = (i / 4) * 2 + i % 2;
                    row_sum[row_slot] += probability;
                }
                CUTE_UNROLL
                for (int offset = 4; offset <= 16; offset *= 2) {
                    CUTE_UNROLL
                    for (int row_slot = 0; row_slot < RowSlots; ++row_slot) {
                        row_sum[row_slot] += __shfl_xor_sync(
                            0xffffffffu, row_sum[row_slot], offset);
                    }
                }
                if (lane < 4) {
                    CUTE_UNROLL
                    for (int row_slot = 0; row_slot < RowSlots; ++row_slot) {
                        const int fragment_index = (row_slot / 2) * 4 + row_slot % 2;
                        const int row = cute::get<1>(score_coordinates(fragment_index));
                        split_storage.row_partials[warp][row] = row_sum[row_slot];
                    }
                }
                sync_warp_group_barrier(warp_group);

                if (local_tid < Policy::MTile) {
                    const int row = local_tid;
                    float page_sum = 0.f;
                    CUTE_UNROLL
                    for (int source_warp = 0; source_warp < 4; ++source_warp) {
                        page_sum += split_storage.row_partials[source_warp][row];
                    }
                    split_storage.running_sum[row] =
                        split_storage.running_sum[row] * split_storage.old_scale[row] + page_sum;
                }
                sync_warp_group_barrier(warp_group);

                CUTE_UNROLL
                for (int i = 0; i < cute::size(score); ++i) {
                    const auto kn = score_coordinates(i);
                    shared_p(cute::get<1>(kn), cute::get<0>(kn)) =
                        static_cast<T>(score(i));
                }
                cutlass::arch::fence_view_async_shared();
                if (has_next_page) {
                    cute::cp_async_wait<1>();
                }
                else {
                    cute::cp_async_wait<0>();
                }
                sync_warp_group_barrier(warp_group);

                CUTE_UNROLL
                for (int tile = 0; tile < Policy::PvTileCount; ++tile) {
                    auto output_tile = output(cute::_, cute::_, cute::_, tile);
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(output_tile); ++i) {
                        const int row = cute::get<1>(pv_coordinates(i));
                        output_tile(i) *= split_storage.old_scale[row];
                    }
                    cute::warpgroup_fence_operand(output_tile);
                }
                cute::warpgroup_arrive();
                CUTE_UNROLL
                for (int tile = 0; tile < Policy::PvTileCount; ++tile) {
                    auto output_tile = output(cute::_, cute::_, cute::_, tile);
                    auto value_tile = cute::local_tile(
                        shared_v,
                        cute::Shape<cute::_64, cute::_64>{},
                        cute::make_coord(tile, cute::_0{}));
                    auto value_source = thread_pv.partition_A(value_tile);
                    auto probability_source = thread_pv.partition_B(shared_p);
                    auto value_fragment = thread_pv.make_fragment_A(value_source);
                    auto probability_fragment = thread_pv.make_fragment_B(probability_source);
                    cute::gemm(pv_mma,
                               value_fragment,
                               probability_fragment,
                               output_tile);
                }
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
                CUTE_UNROLL
                for (int tile = 0; tile < Policy::PvTileCount; ++tile) {
                    auto output_tile = output(cute::_, cute::_, cute::_, tile);
                    cute::warpgroup_fence_operand(output_tile);
                }
                if (has_next_page) {
                    const int next_page = page + 1;
                    copy_paged_page<true, T, Policy>(
                        cache,
                        next_page,
                        next_page * Policy::KeyTile,
                        key_length,
                        v_copy_view,
                        local_tid);
                    cute::cp_async_fence();
                }
            }
        }

        CUTE_UNROLL
        for (int tile = 0; tile < Policy::PvTileCount; ++tile) {
            auto output_tile = output(cute::_, cute::_, cute::_, tile);
            CUTE_UNROLL
            for (int i = 0; i < cute::size(output_tile); ++i) {
                const auto dn = pv_coordinates(i);
                const int row = cute::get<1>(dn);
                const auto qh = decode_row(row);
                const int query_position = cute::get<0>(qh);
                const int head_in_group  = cute::get<1>(qh);
                if (query_position < query_length && head_in_group < query_group_size) {
                    const int absolute_query = query_begin + query_position;
                    const int query_head = kv_head * query_group_size + head_in_group;
                    const int d = tile * Policy::PvNtile + cute::get<0>(dn);
                    if constexpr (StorePartial) {
                        const int local_query = absolute_query - arguments.query_offset;
                        partial_o(local_query, split, query_head, d) = output_tile(i);
                    }
                    else {
                        auto merge_output = cute::make_tensor(
                            cute::make_smem_ptr(split_storage.kv_or_output.output),
                            cute::make_layout(cute::Shape<cute::Int<Policy::MTile>, cute::_256>{},
                                              cute::Stride<cute::_256, cute::_1>{}));
                        merge_output(row, d) = output_tile(i);
                    }
                }
            }
        }

        if constexpr (StorePartial) {
            if (local_tid < Policy::MTile) {
                const int row = local_tid;
                const auto qh = decode_row(row);
                const int query_position = cute::get<0>(qh);
                const int head_in_group  = cute::get<1>(qh);
                if (query_position < query_length && head_in_group < query_group_size) {
                    const int absolute_query = query_begin + query_position;
                    const int local_query = absolute_query - arguments.query_offset;
                    const int query_head = kv_head * query_group_size + head_in_group;
                    partial_ml(local_query, split, query_head, cute::_0{}) =
                        split_storage.running_max[row];
                    partial_ml(local_query, split, query_head, cute::_1{}) =
                        split_storage.running_sum[row];
                }
            }
        }
        else {
            __syncthreads();
            if (warp_group != 0) {
                return;
            }

            if (local_tid < Policy::MTile) {
                const int row = local_tid;
                const float max0 = storage.split[0].running_max[row];
                const float max1 = storage.split[1].running_max[row];
                const float merged_max = fmaxf(max0, max1);
                const float scale0 = max0 == -CUDART_INF_F ? 0.f : exp2f(max0 - merged_max);
                const float scale1 = max1 == -CUDART_INF_F ? 0.f : exp2f(max1 - merged_max);
                const float merged_sum = storage.split[0].running_sum[row] * scale0
                                         + storage.split[1].running_sum[row] * scale1;
                storage.split[0].old_scale[row] = scale0;
                storage.split[1].old_scale[row] = scale1;
                storage.split[0].running_sum[row] = merged_sum == 0.f ? 0.f : 1.f / merged_sum;
            }
            sync_warp_group_barrier(warp_group);

            auto output0 = cute::make_tensor(
                cute::make_smem_ptr(storage.split[0].kv_or_output.output),
                cute::make_layout(cute::Shape<cute::Int<Policy::MTile>, cute::_256>{},
                                  cute::Stride<cute::_256, cute::_1>{}));
            auto output1 = cute::make_tensor(
                cute::make_smem_ptr(storage.split[1].kv_or_output.output),
                cute::make_layout(cute::Shape<cute::Int<Policy::MTile>, cute::_256>{},
                                  cute::Stride<cute::_256, cute::_1>{}));
            CUTE_UNROLL
            for (int access = 0;
                 access < Policy::MTile * Policy::HeadDim / Policy::Threads;
                 ++access) {
                const int linear = local_tid + access * Policy::Threads;
                const int row = linear / Policy::HeadDim;
                const int d = linear % Policy::HeadDim;
                const auto qh = decode_row(row);
                const int query_position = cute::get<0>(qh);
                const int head_in_group  = cute::get<1>(qh);
                if (query_position < query_length && head_in_group < query_group_size) {
                    const float value = output0(row, d) * storage.split[0].old_scale[row]
                                        + output1(row, d) * storage.split[1].old_scale[row];
                    const int absolute_query = query_begin + query_position;
                    const int query_head = kv_head * query_group_size + head_in_group;
                    global_out(absolute_query, query_head, d) = static_cast<T>(
                        value * storage.split[0].running_sum[row]);
                }
            }
        }
    }
};

template<class T, bool StorePartial>
__global__ __launch_bounds__(256, 1)
void VerificationAttentionWgmmaKernel(Arguments arguments)
{
    using Policy = Sm90WgmmaPolicy256;
    extern __shared__ char dynamic_shared[];
    auto& storage = *reinterpret_cast<Sm90WgmmaSharedStorage<T, Policy>*>(dynamic_shared);
    Sm90WgmmaMainloop<T, StorePartial>{arguments, storage}.run();
}

template<class T>
void LaunchWgmma(const Arguments& arguments)
{
    using Policy = Sm90WgmmaPolicy256;
    const dim3 grid(arguments.request_count,
                    arguments.kv_head_count,
                    (arguments.split_count + 1) / 2);
    constexpr int smem_bytes = sizeof(Sm90WgmmaSharedStorage<T, Policy>);
    if (arguments.split_count == 1) {
        auto kernel = VerificationAttentionWgmmaKernel<T, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        kernel<<<grid, 2 * Policy::Threads, smem_bytes, arguments.stream>>>(arguments);
    }
    else {
        auto kernel = VerificationAttentionWgmmaKernel<T, true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        kernel<<<grid, 2 * Policy::Threads, smem_bytes, arguments.stream>>>(arguments);
    }
}

}  // namespace turbomind::verification_attention
