// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/verification/kernel_sm90_wgmma.cuh"

namespace turbomind::verification_attention {

struct Sm90WgmmaRsPolicy256 {
    static constexpr int HeadDim           = 256;
    static constexpr int Threads           = 128;
    static constexpr int MTile             = 64;
    static constexpr int KeyTile           = 64;
    static constexpr int QueryPartitionTile = 8;

    using QShape  = cute::Shape<cute::Int<MTile>, cute::Int<HeadDim>>;
    using KvShape = cute::Shape<cute::Int<KeyTile>, cute::Int<HeadDim>>;
};

template<class T>
struct Sm90WgmmaRsMma;

template<>
struct Sm90WgmmaRsMma<cutlass::bfloat16_t> {
    using QK = decltype(cute::make_tiled_mma(
        cute::SM90_64x64x16_F32BF16BF16_SS<
            cute::GMMA::Major::K, cute::GMMA::Major::K>{}));
    using PV = decltype(cute::make_tiled_mma(
        cute::SM90_64x256x16_F32BF16BF16_RS<
            cute::GMMA::Major::K, cute::GMMA::Major::MN>{}));
};

template<class T>
struct Sm90WgmmaRsKvStorage {
    using Policy = Sm90WgmmaRsPolicy256;
    T k[Policy::KeyTile * Policy::HeadDim];
    T v[Policy::KeyTile * Policy::HeadDim];
};

template<class T>
union alignas(128) Sm90WgmmaRsGroupBody {
    using Policy = Sm90WgmmaRsPolicy256;
    Sm90WgmmaRsKvStorage<T> kv;
    float output[Policy::MTile * Policy::HeadDim];
};

template<class T>
struct Sm90WgmmaRsGroupStorage {
    using Policy = Sm90WgmmaRsPolicy256;
    Sm90WgmmaRsGroupBody<T> body;
    alignas(16) float running_max[Policy::MTile];
    alignas(16) float running_sum[Policy::MTile];
};

template<class T>
struct Sm90WgmmaRsSharedStorage {
    using Policy = Sm90WgmmaRsPolicy256;
    alignas(128) T q[2][Policy::MTile * Policy::HeadDim];
    Sm90WgmmaRsGroupStorage<T> group[2];
};

template<class T, bool StorePartial, bool QueryPartition>
struct Sm90WgmmaRsMainloop {
    using Policy = Sm90WgmmaRsPolicy256;
    using Mma    = Sm90WgmmaRsMma<T>;

    Arguments                       arguments;
    Sm90WgmmaRsSharedStorage<T>&     storage;

    CUTE_DEVICE auto decode_row(int row) const
    {
        int query_position;
        int head_in_group;
        arguments.query_group_size_divmod(query_position, head_in_group, row);
        return cute::make_coord(query_position, head_in_group);
    }

    CUTE_DEVICE void run()
    {
        // The registered M64 path uses two warp groups over disjoint context
        // ranges and merges their online-softmax states.
        using QueryCopy = QTileCopy<T, Policy>;
        using QLayout = decltype(cute::tile_to_shape(
            cute::GMMA::Layout_K_SW128_Atom<T>{},
            cute::Shape<cute::_64, cute::_256>{}));
        using KLayout = QLayout;
        using VLayout = decltype(cute::tile_to_shape(
            cute::GMMA::Layout_MN_SW128_Atom<T>{},
            cute::Shape<cute::_256, cute::_64>{}));

        const int warp_group = threadIdx.x / Policy::Threads;
        const int local_tid  = threadIdx.x % Policy::Threads;
        const int query_base = QueryPartition ? warp_group * Policy::QueryPartitionTile : 0;
        const int split = QueryPartition ? blockIdx.z :
                              (StorePartial ? 2 * blockIdx.z + warp_group : warp_group);
        const int work_split_count = QueryPartition ? arguments.split_count :
                                         (StorePartial ? arguments.split_count : 2);
        auto& group_storage = storage.group[warp_group];
        auto shared_q = cute::make_tensor(
            cute::make_smem_ptr(storage.q[QueryPartition ? warp_group : 0]), QLayout{});
        auto& kv_storage = QueryPartition ? storage.group[0] : group_storage;
        auto shared_k = cute::make_tensor(cute::make_smem_ptr(kv_storage.body.kv.k), KLayout{});
        auto shared_v = cute::make_tensor(cute::make_smem_ptr(kv_storage.body.kv.v), VLayout{});
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
        auto pv_identity = cute::make_identity_tensor(cute::Shape<cute::_64, cute::_256>{});
        auto pv_coordinates = thread_pv.partition_C(pv_identity);
        const int row0 = cute::get<0>(pv_coordinates(0));
        const int row1 = cute::get<0>(pv_coordinates(2));
        auto output = thread_pv.make_fragment_C(pv_coordinates);
        cute::clear(output);
        using ProbabilityLayout = cute::Layout<
            cute::Shape<cute::Shape<cute::_2, cute::_2, cute::_2>, cute::_1, cute::_4>,
            cute::Stride<cute::Stride<cute::_1, cute::_2, cute::_4>, cute::_0, cute::_8>>;
        cutlass::Array<T, 32> probability_storage;
        auto probability = cute::make_tensor(
            cute::make_rmem_ptr(probability_storage.data()), ProbabilityLayout{});

        Array<float, 2> running_max;
        Array<float, 2> running_sum;
        Array<float, 2> old_scale;
        CUTE_UNROLL
        for (int row_slot = 0; row_slot < 2; ++row_slot) {
            running_max[row_slot] = -CUDART_INF_F;
            running_sum[row_slot] = 0.f;
            old_scale[row_slot]   = 0.f;
        }

        if ((QueryPartition || warp_group == 0) && !request_finished) {
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
                const int query_position = query_base + cute::get<0>(qh);
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

        if constexpr (QueryPartition) {
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
                auto qk_identity = cute::make_identity_tensor(cute::Shape<cute::_64, cute::_64>{});
                auto score_coordinates = thread_qk.partition_C(qk_identity);

                const int initial_page = first_page + warp_group;
                if (initial_page < last_page) {
                    auto initial_k = cute::make_tensor(
                        cute::make_smem_ptr(storage.group[warp_group].body.kv.k), KLayout{});
                    auto initial_v = cute::make_tensor(
                        cute::make_smem_ptr(storage.group[warp_group].body.kv.v), VLayout{});
                    auto initial_v_copy = cute::composition(
                        initial_v,
                        cute::Layout<
                            cute::Shape<cute::_64, cute::_256>,
                            cute::Stride<cute::Int<256>, cute::_1>>{});
                    copy_paged_page<false, T, Policy>(
                        cache,
                        initial_page,
                        initial_page * Policy::KeyTile,
                        key_length,
                        initial_k,
                        local_tid);
                    cute::cp_async_fence();
                    copy_paged_page<true, T, Policy>(
                        cache,
                        initial_page,
                        initial_page * Policy::KeyTile,
                        key_length,
                        initial_v_copy,
                        local_tid);
                    cute::cp_async_fence();
                }

                for (int pair_page = first_page; pair_page < last_page; pair_page += 2) {
                    // K-ready, V-ready, K-reuse, and V-reuse are the four CTA
                    // rendezvous for a nonfinal pair. The final two omit reuse.
                    const bool has_second_page = pair_page + 1 < last_page;
                    const bool has_next_pair = pair_page + 2 < last_page;
                    const int next_page = pair_page + 2 + warp_group;

                    cute::cp_async_wait<1>();
                    __syncthreads();

                    CUTE_UNROLL
                    for (int slot = 0; slot < 2; ++slot) {
                        if (slot == 0 || has_second_page) {
                            const int page = pair_page + slot;
                            const int key_begin = page * Policy::KeyTile;
                            auto shared_k_slot = cute::make_tensor(
                                cute::make_smem_ptr(storage.group[slot].body.kv.k), KLayout{});
                            auto score = thread_qk.make_fragment_C(score_coordinates);
                            cute::clear(score);
                            auto q_source = thread_qk.partition_A(shared_q);
                            auto k_source = thread_qk.partition_B(shared_k_slot);
                            auto q_fragment = thread_qk.make_fragment_A(q_source);
                            auto k_fragment = thread_qk.make_fragment_B(k_source);
                            cute::warpgroup_fence_operand(score);
                            cute::warpgroup_arrive();
                            cute::gemm(qk_mma, q_fragment, k_fragment, score);
                            cute::warpgroup_commit_batch();
                            cute::warpgroup_wait<0>();
                            cute::warpgroup_fence_operand(score);

                            Array<float, 2> page_max;
                            page_max[0] = -CUDART_INF_F;
                            page_max[1] = -CUDART_INF_F;
                            CUTE_UNROLL
                            for (int i = 0; i < cute::size(score); ++i) {
                                const auto rk = score_coordinates(i);
                                const int row = cute::get<0>(rk);
                                const int key_in_page = cute::get<1>(rk);
                                const auto qh = decode_row(row);
                                const int query_position = query_base + cute::get<0>(qh);
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
                                const int row_slot = row == row0 ? 0 : 1;
                                page_max[row_slot] = fmaxf(page_max[row_slot], score(i));
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
                                const float new_max = fmaxf(running_max[row_slot], page_max[row_slot]);
                                old_scale[row_slot] = running_max[row_slot] == -CUDART_INF_F ?
                                                          0.f : exp2f(running_max[row_slot] - new_max);
                                running_max[row_slot] = new_max;
                            }

                            Array<float, 2> page_sum;
                            page_sum[0] = 0.f;
                            page_sum[1] = 0.f;
                            CUTE_UNROLL
                            for (int i = 0; i < cute::size(score); ++i) {
                                const int row = cute::get<0>(score_coordinates(i));
                                const int row_slot = row == row0 ? 0 : 1;
                                const float value = score(i);
                                const float probability_value = value == -CUDART_INF_F ?
                                                                    0.f : exp2f(value - running_max[row_slot]);
                                score(i) = probability_value;
                                page_sum[row_slot] += probability_value;
                            }
                            CUTE_UNROLL
                            for (int offset = 1; offset <= 2; offset *= 2) {
                                page_sum[0] += __shfl_xor_sync(0xffffffffu, page_sum[0], offset);
                                page_sum[1] += __shfl_xor_sync(0xffffffffu, page_sum[1], offset);
                            }
                            running_sum[0] = running_sum[0] * old_scale[0] + page_sum[0];
                            running_sum[1] = running_sum[1] * old_scale[1] + page_sum[1];

                            CUTE_UNROLL
                            for (int i = 0; i < cute::size(output); ++i) {
                                const int row = cute::get<0>(pv_coordinates(i));
                                output(i) *= row == row0 ? old_scale[0] : old_scale[1];
                            }
                            CUTE_UNROLL
                            for (int i = 0; i < cute::size(score); ++i) {
                                probability(i) = static_cast<T>(score(i));
                            }

                            if (slot == 0) {
                                cute::cp_async_wait<0>();
                                __syncthreads();
                            }
                            else if (has_next_pair) {
                                __syncthreads();
                                if (next_page < last_page) {
                                    auto next_k = cute::make_tensor(
                                        cute::make_smem_ptr(storage.group[warp_group].body.kv.k), KLayout{});
                                    copy_paged_page<false, T, Policy>(
                                        cache,
                                        next_page,
                                        next_page * Policy::KeyTile,
                                        key_length,
                                        next_k,
                                        local_tid);
                                    cute::cp_async_fence();
                                }
                            }

                            auto shared_v_slot = cute::make_tensor(
                                cute::make_smem_ptr(storage.group[slot].body.kv.v), VLayout{});
                            auto value_source = thread_pv.partition_B(shared_v_slot);
                            auto value_fragment = thread_pv.make_fragment_B(value_source);
                            cute::warpgroup_fence_operand(probability);
                            cute::warpgroup_fence_operand(output);
                            cute::warpgroup_arrive();
                            cute::gemm(pv_mma, probability, value_fragment, output);
                            cute::warpgroup_commit_batch();
                            cute::warpgroup_wait<0>();
                            cute::warpgroup_fence_operand(probability);
                            cute::warpgroup_fence_operand(output);
                        }
                    }

                    if (has_next_pair) {
                        __syncthreads();
                        if (next_page < last_page) {
                            auto next_v = cute::make_tensor(
                                cute::make_smem_ptr(storage.group[warp_group].body.kv.v), VLayout{});
                            auto next_v_copy = cute::composition(
                                next_v,
                                cute::Layout<
                                    cute::Shape<cute::_64, cute::_256>,
                                    cute::Stride<cute::Int<256>, cute::_1>>{});
                            copy_paged_page<true, T, Policy>(
                                cache,
                                next_page,
                                next_page * Policy::KeyTile,
                                key_length,
                                next_v_copy,
                                local_tid);
                            cute::cp_async_fence();
                        }
                    }
                }
            }
        }
        else if (!inactive) {
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
            auto qk_identity = cute::make_identity_tensor(cute::Shape<cute::_64, cute::_64>{});
            auto score_coordinates = thread_qk.partition_C(qk_identity);
            if (!QueryPartition || warp_group == 0) {
                copy_paged_page<false, T, Policy>(
                    cache, first_page, first_page * Policy::KeyTile, key_length, shared_k, local_tid);
                cute::cp_async_fence();
                copy_paged_page<true, T, Policy>(
                    cache, first_page, first_page * Policy::KeyTile, key_length, v_copy_view, local_tid);
                cute::cp_async_fence();
            }

            for (int page = first_page; page < last_page; ++page) {
                const int key_begin = page * Policy::KeyTile;
                if constexpr (QueryPartition) {
                    if (warp_group == 0) {
                        cute::cp_async_wait<1>();
                    }
                    __syncthreads();
                }
                else {
                    cute::cp_async_wait<1>();
                    sync_warp_group_barrier(warp_group);
                }

                auto score = thread_qk.make_fragment_C(score_coordinates);
                cute::clear(score);
                auto q_source = thread_qk.partition_A(shared_q);
                auto k_source = thread_qk.partition_B(shared_k);
                auto q_fragment = thread_qk.make_fragment_A(q_source);
                auto k_fragment = thread_qk.make_fragment_B(k_source);
                cute::warpgroup_fence_operand(score);
                cute::warpgroup_arrive();
                cute::gemm(qk_mma, q_fragment, k_fragment, score);
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(score);
                if constexpr (QueryPartition) {
                    __syncthreads();
                }

                const bool has_next_page = page + 1 < last_page;
                if (has_next_page && (!QueryPartition || warp_group == 0)) {
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

                Array<float, 2> page_max;
                page_max[0] = -CUDART_INF_F;
                page_max[1] = -CUDART_INF_F;
                CUTE_UNROLL
                for (int i = 0; i < cute::size(score); ++i) {
                    const auto rk = score_coordinates(i);
                    const int row = cute::get<0>(rk);
                    const int key_in_page = cute::get<1>(rk);
                    const auto qh = decode_row(row);
                    const int query_position = query_base + cute::get<0>(qh);
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
                    const int row_slot = row == row0 ? 0 : 1;
                    page_max[row_slot] = fmaxf(page_max[row_slot], score(i));
                }
                CUTE_UNROLL
                for (int offset = 1; offset <= 2; offset *= 2) {
                    page_max[0] = fmaxf(page_max[0],
                                        __shfl_xor_sync(0xffffffffu, page_max[0], offset));
                    page_max[1] = fmaxf(page_max[1],
                                        __shfl_xor_sync(0xffffffffu, page_max[1], offset));
                }
                CUTE_UNROLL
                for (int row_slot = 0; row_slot < 2; ++row_slot) {
                    const float new_max = fmaxf(running_max[row_slot], page_max[row_slot]);
                    old_scale[row_slot] = running_max[row_slot] == -CUDART_INF_F ?
                                              0.f : exp2f(running_max[row_slot] - new_max);
                    running_max[row_slot] = new_max;
                }

                Array<float, 2> page_sum;
                page_sum[0] = 0.f;
                page_sum[1] = 0.f;
                CUTE_UNROLL
                for (int i = 0; i < cute::size(score); ++i) {
                    const int row = cute::get<0>(score_coordinates(i));
                    const int row_slot = row == row0 ? 0 : 1;
                    const float value = score(i);
                    const float probability = value == -CUDART_INF_F ?
                                                  0.f : exp2f(value - running_max[row_slot]);
                    score(i) = probability;
                    page_sum[row_slot] += probability;
                }
                CUTE_UNROLL
                for (int offset = 1; offset <= 2; offset *= 2) {
                    page_sum[0] += __shfl_xor_sync(0xffffffffu, page_sum[0], offset);
                    page_sum[1] += __shfl_xor_sync(0xffffffffu, page_sum[1], offset);
                }
                running_sum[0] = running_sum[0] * old_scale[0] + page_sum[0];
                running_sum[1] = running_sum[1] * old_scale[1] + page_sum[1];

                CUTE_UNROLL
                for (int i = 0; i < cute::size(output); ++i) {
                    const int row = cute::get<0>(pv_coordinates(i));
                    output(i) *= row == row0 ? old_scale[0] : old_scale[1];
                }
                CUTE_UNROLL
                for (int i = 0; i < cute::size(score); ++i) {
                    probability(i) = static_cast<T>(score(i));
                }
                if constexpr (QueryPartition) {
                    if (warp_group == 0) {
                        if (has_next_page) {
                            cute::cp_async_wait<1>();
                        }
                        else {
                            cute::cp_async_wait<0>();
                        }
                    }
                    __syncthreads();
                }
                else {
                    if (has_next_page) {
                        cute::cp_async_wait<1>();
                    }
                    else {
                        cute::cp_async_wait<0>();
                    }
                    sync_warp_group_barrier(warp_group);
                }

                auto value_source = thread_pv.partition_B(shared_v);
                auto value_fragment = thread_pv.make_fragment_B(value_source);
                cute::warpgroup_fence_operand(probability);
                cute::warpgroup_fence_operand(output);
                cute::warpgroup_arrive();
                cute::gemm(pv_mma, probability, value_fragment, output);
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(probability);
                cute::warpgroup_fence_operand(output);
                if constexpr (QueryPartition) {
                    __syncthreads();
                }

                if (has_next_page && (!QueryPartition || warp_group == 0)) {
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
        for (int i = 0; i < cute::size(output); ++i) {
            const auto rd = pv_coordinates(i);
            const int row = cute::get<0>(rd);
            const auto qh = decode_row(row);
            const int query_position = query_base + cute::get<0>(qh);
            const int head_in_group  = cute::get<1>(qh);
            if (query_position < query_length && head_in_group < query_group_size) {
                const int absolute_query = query_begin + query_position;
                const int query_head = kv_head * query_group_size + head_in_group;
                const int d = cute::get<1>(rd);
                if constexpr (StorePartial) {
                    const int local_query = absolute_query - arguments.query_offset;
                    partial_o(local_query, split, query_head, d) = output(i);
                }
                else if constexpr (QueryPartition) {
                    const int row_slot = row == row0 ? 0 : 1;
                    const float sum = running_sum[row_slot];
                    global_out(absolute_query, query_head, d) =
                        sum == 0.f ? T(0) : static_cast<T>(output(i) / sum);
                }
                else {
                    auto merge_output = cute::make_tensor(
                        cute::make_smem_ptr(group_storage.body.output),
                        cute::make_layout(cute::Shape<cute::_64, cute::_256>{},
                                          cute::Stride<cute::_256, cute::_1>{}));
                    merge_output(row, d) = output(i);
                }
            }
        }

        const int lane = local_tid % 32;
        if ((!QueryPartition || StorePartial) && lane % 4 == 0) {
            group_storage.running_max[row0] = running_max[0];
            group_storage.running_max[row1] = running_max[1];
            group_storage.running_sum[row0] = running_sum[0];
            group_storage.running_sum[row1] = running_sum[1];
        }

        if constexpr (StorePartial) {
            if (lane % 4 == 0) {
                CUTE_UNROLL
                for (int row_slot = 0; row_slot < 2; ++row_slot) {
                    const int row = row_slot == 0 ? row0 : row1;
                    const auto qh = decode_row(row);
                    const int query_position = query_base + cute::get<0>(qh);
                    const int head_in_group  = cute::get<1>(qh);
                    if (query_position < query_length && head_in_group < query_group_size) {
                        const int absolute_query = query_begin + query_position;
                        const int local_query = absolute_query - arguments.query_offset;
                        const int query_head = kv_head * query_group_size + head_in_group;
                        partial_ml(local_query, split, query_head, cute::_0{}) = running_max[row_slot];
                        partial_ml(local_query, split, query_head, cute::_1{}) = running_sum[row_slot];
                    }
                }
            }
        }
        else if constexpr (!QueryPartition) {
            __syncthreads();
            if (warp_group != 0) {
                return;
            }
            auto output0 = cute::make_tensor(
                cute::make_smem_ptr(storage.group[0].body.output),
                cute::make_layout(cute::Shape<cute::_64, cute::_256>{},
                                  cute::Stride<cute::_256, cute::_1>{}));
            auto output1 = cute::make_tensor(
                cute::make_smem_ptr(storage.group[1].body.output),
                cute::make_layout(cute::Shape<cute::_64, cute::_256>{},
                                  cute::Stride<cute::_256, cute::_1>{}));
            CUTE_UNROLL
            for (int access = 0; access < Policy::MTile * Policy::HeadDim / Policy::Threads; ++access) {
                const int linear = local_tid + access * Policy::Threads;
                const int row = linear / Policy::HeadDim;
                const int d = linear % Policy::HeadDim;
                const auto qh = decode_row(row);
                const int query_position = cute::get<0>(qh);
                const int head_in_group  = cute::get<1>(qh);
                if (query_position < query_length && head_in_group < query_group_size) {
                    const float max0 = storage.group[0].running_max[row];
                    const float max1 = storage.group[1].running_max[row];
                    const float merged_max = fmaxf(max0, max1);
                    const float scale0 = max0 == -CUDART_INF_F ? 0.f : exp2f(max0 - merged_max);
                    const float scale1 = max1 == -CUDART_INF_F ? 0.f : exp2f(max1 - merged_max);
                    const float sum0 = storage.group[0].running_sum[row];
                    const float sum1 = storage.group[1].running_sum[row];
                    const float merged_sum = sum0 * scale0 + sum1 * scale1;
                    const float inverse_sum = merged_sum == 0.f ? 0.f : 1.f / merged_sum;
                    const float value = output0(row, d) * scale0 + output1(row, d) * scale1;
                    const int absolute_query = query_begin + query_position;
                    const int query_head = kv_head * query_group_size + head_in_group;
                    global_out(absolute_query, query_head, d) = static_cast<T>(value * inverse_sum);
                }
            }
        }
    }
};

template<class T, bool StorePartial, bool QueryPartition>
__global__ __launch_bounds__(256, 1)
void VerificationAttentionWgmmaRsKernel(Arguments arguments)
{
    extern __shared__ char dynamic_shared[];
    auto& storage = *reinterpret_cast<Sm90WgmmaRsSharedStorage<T>*>(dynamic_shared);
    Sm90WgmmaRsMainloop<T, StorePartial, QueryPartition>{arguments, storage}.run();
}

template<class T>
void LaunchWgmmaRs(const Arguments& arguments)
{
    using Policy = Sm90WgmmaRsPolicy256;
    const dim3 grid(arguments.request_count,
                    arguments.kv_head_count,
                    (arguments.split_count + 1) / 2);
    constexpr int smem_bytes = sizeof(Sm90WgmmaRsSharedStorage<T>);
    if (arguments.split_count == 1) {
        auto kernel = VerificationAttentionWgmmaRsKernel<T, false, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        kernel<<<grid, 2 * Policy::Threads, smem_bytes, arguments.stream>>>(arguments);
    }
    else {
        auto kernel = VerificationAttentionWgmmaRsKernel<T, true, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        kernel<<<grid, 2 * Policy::Threads, smem_bytes, arguments.stream>>>(arguments);
    }
}

}  // namespace turbomind::verification_attention
