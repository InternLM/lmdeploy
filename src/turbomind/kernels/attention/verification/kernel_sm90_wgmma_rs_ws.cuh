// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <type_traits>

#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include "src/turbomind/kernels/attention/verification/kernel_sm90_wgmma_rs.cuh"

namespace turbomind::verification_attention {

struct Sm90WgmmaRsWsComputePolicy {
    static constexpr int HeadDim = 256;
    static constexpr int Threads = 128;
    static constexpr int MTile   = 64;
    static constexpr int KeyTile = 64;

    using QShape = cute::Shape<cute::_64, cute::_256>;
};

struct Sm90WgmmaRsWsCopyPolicy {
    static constexpr int HeadDim = 256;
    static constexpr int Threads = 128;
    static constexpr int KeyTile = 64;

    using KvShape = cute::Shape<cute::_64, cute::_256>;
};

template<class T>
struct Sm90WgmmaRsWsMma;

template<>
struct Sm90WgmmaRsWsMma<cutlass::bfloat16_t> {
    using QK = decltype(cute::make_tiled_mma(
        cute::SM90_64x64x16_F32BF16BF16_SS<
            cute::GMMA::Major::K, cute::GMMA::Major::K>{}));
    using PV = decltype(cute::make_tiled_mma(
        cute::SM90_64x256x16_F32BF16BF16_RS<
            cute::GMMA::Major::K, cute::GMMA::Major::MN>{}));
};

using Sm90WgmmaRsWsKPipeline = cutlass::PipelineAsync<2>;
using Sm90WgmmaRsWsVPipeline = cutlass::PipelineAsync<2>;

template<class T>
struct alignas(128) Sm90WgmmaRsWsSharedStorage {
    using ComputePolicy = Sm90WgmmaRsWsComputePolicy;
    using CopyPolicy    = Sm90WgmmaRsWsCopyPolicy;

    alignas(128) T q[2][ComputePolicy::MTile * ComputePolicy::HeadDim];
    alignas(128) T k[2][CopyPolicy::KeyTile * CopyPolicy::HeadDim];
    alignas(128) T v[2][CopyPolicy::KeyTile * CopyPolicy::HeadDim];
    alignas(16) typename Sm90WgmmaRsWsKPipeline::SharedStorage k_pipeline;
    alignas(16) typename Sm90WgmmaRsWsVPipeline::SharedStorage v_pipeline;
};

static_assert(sizeof(Sm90WgmmaRsWsSharedStorage<cutlass::bfloat16_t>) <= 232448);

template<class T, bool StorePartial>
struct Sm90WgmmaRsWsMainloop {
    using ComputePolicy = Sm90WgmmaRsWsComputePolicy;
    using CopyPolicy    = Sm90WgmmaRsWsCopyPolicy;
    using Mma           = Sm90WgmmaRsWsMma<T>;
    using KPipeline     = Sm90WgmmaRsWsKPipeline;
    using VPipeline     = Sm90WgmmaRsWsVPipeline;
    using Storage       = Sm90WgmmaRsWsSharedStorage<T>;

    Arguments arguments;
    Storage&  storage;

    CUTE_DEVICE auto decode_row(int compute_group, int row) const
    {
        const int flat_row = compute_group * ComputePolicy::MTile + row;
        int query_position;
        int head_in_group;
        arguments.query_group_size_divmod(query_position, head_in_group, flat_row);
        return cute::make_coord(query_position, head_in_group);
    }

    CUTE_DEVICE auto make_k_raw()
    {
        using Layout = decltype(cute::tile_to_shape(
            cute::GMMA::Layout_K_SW128_Atom<T>{},
            cute::Shape<cute::_64, cute::_256, cute::_2>{}));
        return cute::make_tensor(cute::make_smem_ptr(&storage.k[0][0]), Layout{});
    }

    CUTE_DEVICE auto make_v_raw()
    {
        using Layout = decltype(cute::tile_to_shape(
            cute::GMMA::Layout_MN_SW128_Atom<T>{},
            cute::Shape<cute::_256, cute::_64, cute::_2>{}));
        return cute::make_tensor(cute::make_smem_ptr(&storage.v[0][0]), Layout{});
    }

    CUTE_DEVICE void producer(KPipeline& k_pipeline,
                              VPipeline& v_pipeline,
                              int        request,
                              int        kv_head,
                              int        key_length,
                              int        first_page,
                              int        last_page,
                              bool       inactive)
    {
        cutlass::arch::warpgroup_reg_dealloc<40>();
        if (inactive) {
            return;
        }

        PagedKv<T, CopyPolicy::HeadDim> cache(arguments.block_ptrs,
                                               arguments.block_ptr_offsets,
                                               request,
                                               kv_head,
                                               arguments.kv_head_count,
                                               arguments.block_len,
                                               arguments.block_len_divmod,
                                               arguments.cache_block_offset);
        auto shared_k_raw = make_k_raw();
        auto shared_v_raw = make_v_raw();
        const int local_tid = threadIdx.x;
        auto k_write = cutlass::make_producer_start_state<KPipeline>();
        auto v_write = cutlass::make_producer_start_state<VPipeline>();
        const int full_page_count = key_length / CopyPolicy::KeyTile;
        auto load_k = [&](int page) {
            k_pipeline.producer_acquire(k_write);
            auto destination = shared_k_raw(cute::_, cute::_, k_write.index());
            if (page < full_page_count) {
                copy_paged_page_full<false, T, CopyPolicy>(
                    cache, page, destination, local_tid);
            }
            else {
                copy_paged_page<false, T, CopyPolicy>(
                    cache,
                    page,
                    page * CopyPolicy::KeyTile,
                    key_length,
                    destination,
                    local_tid);
            }
            k_pipeline.producer_commit(k_write, cutlass::arch::cpasync_barrier_arrive);
            ++k_write;
        };
        auto load_v = [&](int page) {
            v_pipeline.producer_acquire(v_write);
            auto destination = cute::composition(
                shared_v_raw(cute::_, cute::_, v_write.index()),
                cute::Layout<
                    cute::Shape<cute::_64, cute::_256>,
                    cute::Stride<cute::_256, cute::_1>>{});
            if (page < full_page_count) {
                copy_paged_page_full<true, T, CopyPolicy>(
                    cache, page, destination, local_tid);
            }
            else {
                copy_paged_page<true, T, CopyPolicy>(
                    cache,
                    page,
                    page * CopyPolicy::KeyTile,
                    key_length,
                    destination,
                    local_tid);
            }
            v_pipeline.producer_commit(v_write, cutlass::arch::cpasync_barrier_arrive);
            ++v_write;
        };
        for (int page = first_page; page < last_page; ++page) {
            load_k(page);
            load_v(page);
        }
        if (local_tid == 0) {
            k_pipeline.producer_tail(k_write);
            v_pipeline.producer_tail(v_write);
        }
    }

    CUTE_DEVICE void load_query(int compute_group,
                                int local_tid,
                                int request,
                                int kv_head,
                                int query_begin,
                                int query_length,
                                int history_length)
    {
        using QueryCopy = QTileCopy<T, ComputePolicy>;
        using QLayout = decltype(cute::tile_to_shape(
            cute::GMMA::Layout_K_SW128_Atom<T>{},
            cute::Shape<cute::_64, cute::_256>{}));
        auto shared_q = cute::make_tensor(
            cute::make_smem_ptr(storage.q[compute_group]), QLayout{});
        auto global_q = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<const T*>(arguments.q)),
            cute::make_layout(
                cute::make_shape(arguments.query_count,
                                 arguments.query_head_count,
                                 cute::_256{}),
                cute::make_stride(arguments.q_stride,
                                  cute::Int<ComputePolicy::HeadDim>{},
                                  cute::_1{})));
        auto global_q_bias = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<const T*>(arguments.q_bias)),
            cute::make_layout(
                cute::make_shape(arguments.query_head_count, cute::_256{}),
                cute::make_stride(cute::Int<ComputePolicy::HeadDim>{}, cute::_1{})));
        using RegisterCopy = cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, T>;
        auto q_identity    = cute::make_identity_tensor(typename ComputePolicy::QShape{});
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
        const int query_group_size = arguments.query_group_size;
        const int d_begin          = cute::get<1>(q_coordinates(cute::_0{}, cute::_0{}));
        FastRoPE<QueryCopy::ValuesPerAccess> rope(
            arguments.rope,
            request,
            std::integral_constant<int, QueryCopy::ValuesPerAccess>{});
        const bool rotary = d_begin < arguments.rope.dim;
        if (rotary) {
            rope.init(d_begin);
        }
        const auto first_qh = decode_row(
            compute_group, cute::get<0>(q_coordinates(cute::_0{}, cute::_0{})));
        int query_position = cute::get<0>(first_qh);
        int head_in_group  = cute::get<1>(first_qh);

        CUTE_UNROLL
        for (int access = 0; access < QueryCopy::AccessCount; ++access) {
            const bool valid         = query_position < query_length;
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
                if (rotary) {
                    rope.apply(fragment, history_length + query_position);
                }
            }
            cute::copy(RegisterCopy{}, fragment_tensor, q_destinations(cute::_, access));
            head_in_group += QueryCopy::RowsPerThreadTile;
            if (head_in_group >= query_group_size) {
                head_in_group -= query_group_size;
                ++query_position;
            }
        }
        cutlass::arch::fence_view_async_shared();
        sync_warp_group_barrier(compute_group);
    }

    CUTE_DEVICE void consumer(KPipeline& k_pipeline,
                              VPipeline& v_pipeline,
                              int        request,
                              int        kv_head,
                              int        query_begin,
                              int        query_length,
                              int        key_length,
                              int        history_length,
                              int        first_page,
                              int        last_page,
                              int        split,
                              bool       inactive)
    {
        cutlass::arch::warpgroup_reg_alloc<232>();
        const int compute_group = threadIdx.x / 128 - 1;
        const int local_tid     = threadIdx.x % 128;
        const int query_group_size = arguments.query_group_size;
        if (!inactive) {
            load_query(compute_group,
                       local_tid,
                       request,
                       kv_head,
                       query_begin,
                       query_length,
                       history_length);
        }

        using QLayout = decltype(cute::tile_to_shape(
            cute::GMMA::Layout_K_SW128_Atom<T>{},
            cute::Shape<cute::_64, cute::_256>{}));
        auto shared_q = cute::make_tensor(
            cute::make_smem_ptr(storage.q[compute_group]), QLayout{});
        auto shared_k = make_k_raw();
        auto shared_v = make_v_raw();
        auto qk_mma = typename Mma::QK{};
        auto pv_mma = typename Mma::PV{};
        auto thread_qk = qk_mma.get_slice(local_tid);
        auto thread_pv = pv_mma.get_slice(local_tid);
        auto qk_identity = cute::make_identity_tensor(cute::Shape<cute::_64, cute::_64>{});
        auto pv_identity = cute::make_identity_tensor(cute::Shape<cute::_64, cute::_256>{});
        auto score_coordinates = thread_qk.partition_C(qk_identity);
        auto pv_coordinates = thread_pv.partition_C(pv_identity);
        const int row0 = cute::get<0>(pv_coordinates(0));
        const int row1 = cute::get<0>(pv_coordinates(2));
        const auto qh0 = decode_row(compute_group, row0);
        const auto qh1 = decode_row(compute_group, row1);
        Array<int, 2> query_position;
        Array<int, 2> head_in_group;
        query_position[0] = cute::get<0>(qh0);
        query_position[1] = cute::get<0>(qh1);
        head_in_group[0]  = cute::get<1>(qh0);
        head_in_group[1]  = cute::get<1>(qh1);
        auto output = thread_pv.make_fragment_C(pv_coordinates);
        cute::clear(output);

        using ScoreTensor = decltype(thread_qk.make_fragment_C(score_coordinates));
        using ScoreLayout = typename ScoreTensor::layout_type;
        static_assert(cute::cosize_v<ScoreLayout> == 32);
        using ProbabilityLayout = cute::Layout<
            cute::Shape<cute::Shape<cute::_2, cute::_2, cute::_2>, cute::_1, cute::_4>,
            cute::Stride<cute::Stride<cute::_1, cute::_2, cute::_4>, cute::_0, cute::_8>>;
        cutlass::Array<float, 32> score_storage;
        auto score = cute::make_tensor(
            cute::make_rmem_ptr(score_storage.data()), ScoreLayout{});
        cutlass::Array<T, 32> probability_storage;
        auto probability = cute::make_tensor(
            cute::make_rmem_ptr(probability_storage.data()), ProbabilityLayout{});
        auto output_coordinates = pv_coordinates;

        Array<float, 2> running_max;
        Array<float, 2> running_sum;
        Array<float, 2> old_scale;
        CUTE_UNROLL
        for (int row_slot = 0; row_slot < 2; ++row_slot) {
            running_max[row_slot] = -CUDART_INF_F;
            running_sum[row_slot] = 0.f;
            old_scale[row_slot]   = 0.f;
        }

        typename KPipeline::PipelineState k_read;
        typename VPipeline::PipelineState v_read;
        if (!inactive) {
            const int first_flat_row = compute_group * ComputePolicy::MTile;
            const int valid_flat_row_end = min(first_flat_row + ComputePolicy::MTile,
                                               query_length * query_group_size);
            const int first_query_position =
                arguments.query_group_size_divmod.div(first_flat_row);
            const int last_query_position =
                arguments.query_group_size_divmod.div(valid_flat_row_end - 1);
            const bool full_query_tile = first_flat_row < valid_flat_row_end;
            const int full_page_first_key =
                max(0, history_length + last_query_position - arguments.window_size + 1);
            const int full_page_last_key = history_length + first_query_position;

            auto q_source   = thread_qk.partition_A(shared_q);
            auto q_fragment = thread_qk.make_fragment_A(q_source);

            auto issue_qk = [&] {
                cute::clear(score);
                auto k_source = thread_qk.partition_B(
                    shared_k(cute::_, cute::_, k_read.index()));
                auto k_fragment = thread_qk.make_fragment_B(k_source);
                cute::warpgroup_fence_operand(score);
                cute::warpgroup_arrive();
                cute::gemm(qk_mma, q_fragment, k_fragment, score);
                cute::warpgroup_commit_batch();
            };

            auto update_softmax = [&](int page, auto full_page_tag) {
                constexpr bool FullPage = decltype(full_page_tag)::value;
                const int key_begin = page * CopyPolicy::KeyTile;
                Array<float, 2> page_max;
                page_max[0] = -CUDART_INF_F;
                page_max[1] = -CUDART_INF_F;
                if constexpr (FullPage) {
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(score); ++i) {
                        const int row = cute::get<0>(score_coordinates(i));
                        const int row_slot = row == row0 ? 0 : 1;
                        page_max[row_slot] = fmaxf(page_max[row_slot], score(i));
                    }
                }
                else {
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(score); ++i) {
                        const auto rk = score_coordinates(i);
                        const int row = cute::get<0>(rk);
                        const int key_in_page = cute::get<1>(rk);
                        const int row_slot = row == row0 ? 0 : 1;
                        const int absolute_key   = key_begin + key_in_page;
                        const int last_valid     = history_length + query_position[row_slot];
                        const int first_valid = max(0, last_valid - arguments.window_size + 1);
                        const bool valid = query_position[row_slot] < query_length
                                           && absolute_key >= first_valid
                                           && absolute_key <= last_valid;
                        score(i) = valid ? score(i) : -CUDART_INF_F;
                        page_max[row_slot] = fmaxf(page_max[row_slot], score(i));
                    }
                }
                CUTE_UNROLL
                for (int offset = 1; offset <= 2; offset *= 2) {
                    page_max[0] = fmaxf(
                        page_max[0], __shfl_xor_sync(0xffffffffu, page_max[0], offset));
                    page_max[1] = fmaxf(
                        page_max[1], __shfl_xor_sync(0xffffffffu, page_max[1], offset));
                }
                CUTE_UNROLL
                for (int row_slot = 0; row_slot < 2; ++row_slot) {
                    page_max[row_slot] *= arguments.qk_scale_log2;
                    const float new_max = fmaxf(running_max[row_slot], page_max[row_slot]);
                    old_scale[row_slot] = running_max[row_slot] == -CUDART_INF_F ?
                                              0.f : exp2f(running_max[row_slot] - new_max);
                    running_max[row_slot] = new_max;
                }

                Array<float, 2> page_sum;
                page_sum[0] = 0.f;
                page_sum[1] = 0.f;
                if constexpr (FullPage) {
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(score); ++i) {
                        const int row = cute::get<0>(score_coordinates(i));
                        const int row_slot = row == row0 ? 0 : 1;
                        const float p = exp2f(
                            fmaf(score(i), arguments.qk_scale_log2, -running_max[row_slot]));
                        score(i) = p;
                        page_sum[row_slot] += p;
                    }
                }
                else {
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(score); ++i) {
                        const int row = cute::get<0>(score_coordinates(i));
                        const int row_slot = row == row0 ? 0 : 1;
                        const float value = score(i);
                        const float p = value == -CUDART_INF_F ?
                                            0.f : exp2f(fmaf(value,
                                                            arguments.qk_scale_log2,
                                                            -running_max[row_slot]));
                        score(i) = p;
                        page_sum[row_slot] += p;
                    }
                }
                CUTE_UNROLL
                for (int offset = 1; offset <= 2; offset *= 2) {
                    page_sum[0] += __shfl_xor_sync(0xffffffffu, page_sum[0], offset);
                    page_sum[1] += __shfl_xor_sync(0xffffffffu, page_sum[1], offset);
                }
                running_sum[0] = fmaf(running_sum[0], old_scale[0], page_sum[0]);
                running_sum[1] = fmaf(running_sum[1], old_scale[1], page_sum[1]);
                CUTE_UNROLL
                for (int i = 0; i < cute::size(probability); ++i) {
                    probability(i) = static_cast<T>(score(i));
                }
            };

            auto scale_output = [&] {
                const bool warp_needs_scale = __any_sync(
                    0xffffffffu, old_scale[0] != 1.f || old_scale[1] != 1.f);
                if (!warp_needs_scale) {
                    return;
                }
                CUTE_UNROLL
                for (int i = 0; i < cute::size(output); ++i) {
                    const int row = cute::get<0>(output_coordinates(i));
                    output(i) *= row == row0 ? old_scale[0] : old_scale[1];
                }
            };

            auto issue_pv = [&] {
                auto value_source = thread_pv.partition_B(
                    shared_v(cute::_, cute::_, v_read.index()));
                auto value_fragment = thread_pv.make_fragment_B(value_source);
                cute::warpgroup_fence_operand(probability);
                cute::warpgroup_fence_operand(output);
                cute::warpgroup_arrive();
                cute::gemm(pv_mma, probability, value_fragment, output);
                cute::warpgroup_commit_batch();
            };

            auto process_page = [&](int page, auto full_page_tag) {
                k_pipeline.consumer_wait(k_read);
                issue_qk();
                auto v_ready = v_pipeline.consumer_try_wait(v_read);
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(score);
                k_pipeline.consumer_release(k_read);
                ++k_read;
                update_softmax(page, full_page_tag);
                scale_output();
                v_pipeline.consumer_wait(v_read, v_ready);
                issue_pv();
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(probability);
                cute::warpgroup_fence_operand(output);
                v_pipeline.consumer_release(v_read);
                ++v_read;
            };

            const int full_page_begin = full_query_tile ?
                min(last_page,
                    max(first_page,
                        (full_page_first_key + CopyPolicy::KeyTile - 1) /
                            CopyPolicy::KeyTile)) : last_page;
            const int full_page_end = full_query_tile ?
                max(full_page_begin,
                    min(last_page,
                        (full_page_last_key + 1) / CopyPolicy::KeyTile)) : last_page;
            for (int page = first_page; page < full_page_begin; ++page) {
                process_page(page, std::false_type{});
            }
            for (int page = full_page_begin; page < full_page_end; ++page) {
                process_page(page, std::true_type{});
            }
            for (int page = full_page_end; page < last_page; ++page) {
                process_page(page, std::false_type{});
            }
        }

        auto global_out = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<T*>(arguments.out)),
            cute::make_layout(
                cute::make_shape(arguments.query_count,
                                 arguments.query_head_count,
                                 cute::_256{}),
                cute::make_stride(arguments.query_head_count * ComputePolicy::HeadDim,
                                  cute::Int<ComputePolicy::HeadDim>{},
                                  cute::_1{})));
        auto partial_o = cute::make_tensor(
            cute::make_gmem_ptr(arguments.partial_o),
            cute::make_layout(
                cute::make_shape(arguments.query_count,
                                 arguments.split_count,
                                 arguments.query_head_count,
                                 cute::_256{}),
                cute::make_stride(arguments.split_count * arguments.query_head_count * ComputePolicy::HeadDim,
                                  arguments.query_head_count * ComputePolicy::HeadDim,
                                  cute::Int<ComputePolicy::HeadDim>{},
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
        if constexpr (StorePartial) {
            CUTE_UNROLL
            for (int i = 0; i < cute::size(output); ++i) {
                const auto rd = output_coordinates(i);
                const int row = cute::get<0>(rd);
                const auto qh = decode_row(compute_group, row);
                const int position = cute::get<0>(qh);
                const int head = cute::get<1>(qh);
                if (position < query_length) {
                    const int absolute_query = query_begin + position;
                    const int local_query = absolute_query - arguments.query_offset;
                    const int query_head = kv_head * query_group_size + head;
                    const int d = cute::get<1>(rd);
                    partial_o(local_query, split, query_head, d) = output(i);
                }
            }
        }
        else {
            using OutputLayout = decltype(cute::tile_to_shape(
                cute::GMMA::Layout_K_SW128_Atom<T>{},
                cute::Shape<cute::_64, cute::_256>{}));
            using OutputStore = decltype(cute::make_tiled_copy_C(
                cute::Copy_Atom<cute::SM90_U32x4_STSM_N, T>{},
                typename Mma::PV{}));
            static constexpr int OutputElementsPerStore = 16 / sizeof(T);
            static constexpr int OutputThreadsPerRow = 64 / OutputElementsPerStore;
            using OutputGlobalCopy = decltype(cute::make_tiled_copy(
                cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, T>{},
                cute::Layout<
                    cute::Shape<cute::Int<ComputePolicy::Threads / OutputThreadsPerRow>,
                                cute::Int<OutputThreadsPerRow>>,
                    cute::Stride<cute::Int<OutputThreadsPerRow>, cute::_1>>{},
                cute::Layout<cute::Shape<cute::_1, cute::Int<OutputElementsPerStore>>>{}));

            cutlass::Array<float, 2> inverse_sum;
            CUTE_UNROLL
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const float sum = running_sum[row_slot];
                inverse_sum[row_slot] = sum == 0.f ? 0.f : 1.f / sum;
            }

            if (query_length * query_group_size < 2 * ComputePolicy::MTile) {
                CUTE_UNROLL
                for (int i = 0; i < cute::size(output); ++i) {
                    const auto rd = output_coordinates(i);
                    const int row = cute::get<0>(rd);
                    const auto qh = decode_row(compute_group, row);
                    const int position = cute::get<0>(qh);
                    if (position < query_length) {
                        const int row_slot = row == row0 ? 0 : 1;
                        const int head = cute::get<1>(qh);
                        const int absolute_query = query_begin + position;
                        const int query_head = kv_head * query_group_size + head;
                        const int d = cute::get<1>(rd);
                        global_out(absolute_query, query_head, d) = static_cast<T>(
                            output(i) * inverse_sum[row_slot]);
                    }
                }
                return;
            }

            auto normalized = cute::make_tensor_like<T>(output);
            CUTE_UNROLL
            for (int i = 0; i < cute::size(output); ++i) {
                const int row = cute::get<0>(output_coordinates(i));
                const int row_slot = row == row0 ? 0 : 1;
                normalized(i) = static_cast<T>(output(i) * inverse_sum[row_slot]);
            }

            // Both compute warp groups consumed the common K ring.  Reuse two
            // dead K stages as disjoint 64x256 epilogue tiles.
            sync_compute_groups();
            auto shared_output = cute::make_tensor(
                cute::make_smem_ptr(storage.k[compute_group]), OutputLayout{});
            auto output_store = OutputStore{};
            auto thread_output_store = output_store.get_thread_slice(local_tid);
            cute::copy(output_store,
                       thread_output_store.retile_S(normalized),
                       thread_output_store.partition_D(shared_output));
            cutlass::arch::fence_view_async_shared();
            sync_warp_group_barrier(compute_group);

            auto output_identity = cute::make_identity_tensor(
                cute::Shape<cute::_64, cute::_256>{});
            auto output_copy = OutputGlobalCopy{};
            auto thread_output_copy = output_copy.get_thread_slice(local_tid);
            auto shared_fragments = thread_output_copy.partition_S(shared_output);
            auto output_coordinates_copy = thread_output_copy.partition_D(output_identity);
            CUTE_UNROLL
            for (int m = 0; m < cute::size<1>(shared_fragments); ++m) {
                const int row = cute::get<0>(output_coordinates_copy(cute::_0{}, m, cute::_0{}));
                const auto qh = decode_row(compute_group, row);
                const int position = cute::get<0>(qh);
                const int head = cute::get<1>(qh);
                if (position < query_length) {
                    const int absolute_query = query_begin + position;
                    const int query_head = kv_head * query_group_size + head;
                    auto output_row = global_out(absolute_query, query_head, cute::_);
                    auto output_vectors = cute::tiled_divide(
                        output_row,
                        cute::Shape<cute::Int<OutputElementsPerStore>>{});
                    CUTE_UNROLL
                    for (int k = 0; k < cute::size<2>(shared_fragments); ++k) {
                        const int d = cute::get<1>(
                            output_coordinates_copy(cute::_0{}, cute::_0{}, k));
                        cute::copy(output_copy,
                                   shared_fragments(cute::_, m, k),
                                   output_vectors(cute::_, d / OutputElementsPerStore));
                    }
                }
            }
        }

        if constexpr (StorePartial) {
            const int lane = local_tid % 32;
            if (lane % 4 == 0) {
                CUTE_UNROLL
                for (int row_slot = 0; row_slot < 2; ++row_slot) {
                    if (query_position[row_slot] < query_length) {
                        const int absolute_query = query_begin + query_position[row_slot];
                        const int local_query = absolute_query - arguments.query_offset;
                        const int query_head =
                            kv_head * query_group_size + head_in_group[row_slot];
                        partial_ml(local_query, split, query_head, cute::_0{}) = running_max[row_slot];
                        partial_ml(local_query, split, query_head, cute::_1{}) = running_sum[row_slot];
                    }
                }
            }
        }
    }

    CUTE_DEVICE void run()
    {
        const int warp_group = threadIdx.x / 128;
        typename KPipeline::Params k_params;
        k_params.role = warp_group == 0 ? KPipeline::ThreadCategory::Producer :
                                          KPipeline::ThreadCategory::Consumer;
        k_params.producer_arv_count = 128;
        k_params.consumer_arv_count = 256;
        k_params.initializing_warp  = 0;
        typename VPipeline::Params v_params;
        v_params.role = warp_group == 0 ? VPipeline::ThreadCategory::Producer :
                                          VPipeline::ThreadCategory::Consumer;
        v_params.producer_arv_count = 128;
        v_params.consumer_arv_count = 256;
        v_params.initializing_warp  = 0;
        KPipeline k_pipeline(storage.k_pipeline, k_params);
        VPipeline v_pipeline(storage.v_pipeline, v_params);
        __syncthreads();

        const int request        = blockIdx.x;
        const int kv_head        = blockIdx.y;
        const int query_begin    = arguments.q_offsets[request];
        const int query_end      = arguments.q_offsets[request + 1];
        const int query_length   = query_end - query_begin;
        const int key_length     = arguments.k_offsets[request + 1] - arguments.k_offsets[request];
        const int history_length = key_length - query_length;
        const int page_count     = (key_length + CopyPolicy::KeyTile - 1) / CopyPolicy::KeyTile;
        const int split          = blockIdx.z;
        const int first_page = static_cast<int64_t>(page_count) * split / arguments.split_count;
        const int last_page = static_cast<int64_t>(page_count) * (split + 1) / arguments.split_count;
        const bool request_finished = arguments.finished && arguments.finished[request];
        const bool inactive = request_finished || first_page >= last_page;

        if (warp_group == 0) {
            producer(k_pipeline,
                     v_pipeline,
                     request,
                     kv_head,
                     key_length,
                     first_page,
                     last_page,
                     inactive);
        }
        else {
            consumer(k_pipeline,
                     v_pipeline,
                     request,
                     kv_head,
                     query_begin,
                     query_length,
                     key_length,
                     history_length,
                     first_page,
                     last_page,
                     split,
                     inactive);
        }
    }
};

template<class T, bool StorePartial>
__global__ __launch_bounds__(384, 1)
void VerificationAttentionWgmmaRsWsKernel(Arguments arguments)
{
    extern __shared__ char dynamic_shared[];
    auto& storage = *reinterpret_cast<Sm90WgmmaRsWsSharedStorage<T>*>(dynamic_shared);
    Sm90WgmmaRsWsMainloop<T, StorePartial>{arguments, storage}.run();
}

template<class T>
void LaunchWgmmaRsWs(const Arguments& arguments)
{
    const dim3 grid(arguments.request_count,
                    arguments.kv_head_count,
                    arguments.split_count);
    constexpr int smem_bytes = sizeof(Sm90WgmmaRsWsSharedStorage<T>);
    if (arguments.split_count == 1) {
        auto kernel = VerificationAttentionWgmmaRsWsKernel<T, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        kernel<<<grid, 384, smem_bytes, arguments.stream>>>(arguments);
    }
    else {
        auto kernel = VerificationAttentionWgmmaRsWsKernel<T, true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        kernel<<<grid, 384, smem_bytes, arguments.stream>>>(arguments);
    }
}

}  // namespace turbomind::verification_attention
