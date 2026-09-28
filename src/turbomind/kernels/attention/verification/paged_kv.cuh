// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cute/tensor.hpp>

#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/attention/verification/policy_sm90.cuh"

namespace turbomind::verification_attention {

template<class T, int HeadDim>
class PagedKv {
public:
    using Config = block::Config<T, T, HeadDim, false>;
    using Layout = block::Layout<Config>;

    CUTE_DEVICE PagedKv(char* const* block_ptrs,
                        const int*   block_ptr_offsets,
                        int          request,
                        int          kv_head,
                        int          kv_head_count,
                        int          block_len,
                        cutlass::FastDivmod block_len_divmod,
                        int          cache_block_offset):
        pages_{block_ptrs + block_ptr_offsets[request]},
        layout_{Config{kv_head_count, block_len}},
        kv_head_{kv_head},
        block_len_divmod_{block_len_divmod},
        cache_block_offset_{cache_block_offset}
    {
    }

    template<bool Value>
    CUTE_DEVICE auto row(int token) const
    {
        int page_index;
        int page_token;
        block_len_divmod_(page_index, page_token, token);
        char*     page       = pages_[page_index];
        const auto byte_offset = Value ? layout_.v_data(kv_head_, page_token) :
                                         layout_.k_data(kv_head_, page_token);
        const T* row_ptr = reinterpret_cast<const T*>(page + cache_block_offset_ + byte_offset);
        return cute::make_tensor(cute::make_gmem_ptr(row_ptr), cute::make_layout(cute::Int<HeadDim>{}));
    }

    template<bool Value, int RowCount>
    CUTE_DEVICE auto page(int page_index) const
    {
        char* page = pages_[page_index];
        const auto byte_offset = Value ? layout_.v_data(kv_head_, 0) :
                                         layout_.k_data(kv_head_, 0);
        const T* page_ptr = reinterpret_cast<const T*>(page + cache_block_offset_ + byte_offset);
        return cute::make_tensor(
            cute::make_gmem_ptr(page_ptr),
            cute::make_layout(
                cute::make_shape(cute::Int<RowCount>{}, cute::Int<HeadDim>{}),
                cute::make_stride(cute::Int<HeadDim>{}, cute::_1{})));
    }

private:
    char* const* pages_;
    Layout       layout_;
    int          kv_head_;
    cutlass::FastDivmod block_len_divmod_;
    int          cache_block_offset_;
};

template<bool Value, class T, class Policy, class SmemTensor>
CUTE_DEVICE void copy_paged_tile(const PagedKv<T, Policy::HeadDim>& cache,
                                 int                         key_begin,
                                 int                         key_limit,
                                 SmemTensor                  destination)
{
    using Copy = KvTileCopy<T, Policy>;
    auto logical_tile = cute::make_identity_tensor(typename Policy::KvShape{});
    auto tiled_copy   = typename Copy::TiledCopy{};
    auto thread_copy  = tiled_copy.get_thread_slice(threadIdx.x);
    auto coordinates  = cute::group_modes<
        1, cute::rank_v<decltype(thread_copy.partition_S(logical_tile))>>(
        thread_copy.partition_S(logical_tile));
    auto destinations = cute::group_modes<
        1, cute::rank_v<decltype(thread_copy.partition_D(destination))>>(
        thread_copy.partition_D(destination));

    const int d_begin = cute::get<1>(coordinates(cute::_0{}, cute::_0{}));
    auto copy_atom    = typename Copy::Atom{};
    CUTE_UNROLL
    for (int access = 0; access < Copy::AccessCount; ++access) {
        const int key = key_begin + cute::get<0>(coordinates(cute::_0{}, access));
        const bool valid = key < key_limit;
        const int source_key = valid ? key : key_begin;
        auto source_row = cache.template row<Value>(source_key);
        auto source = cute::make_tensor(
            cute::make_gmem_ptr(&source_row(d_begin)),
            cute::make_layout(cute::Int<Copy::ValuesPerAccess>{}));
        cute::copy(copy_atom.with(valid), source, destinations(cute::_, access));
    }
}

template<bool Value, bool GuardTail, class T, class Policy, class SmemTensor>
CUTE_DEVICE void copy_paged_page_impl(const PagedKv<T, Policy::HeadDim>& cache,
                                      int                                page_index,
                                      int                                key_begin,
                                      int                                key_limit,
                                      SmemTensor                         destination,
                                      int                                thread_idx)
{
    using Copy = KvTileCopy<T, Policy>;
    auto logical_tile = cute::make_identity_tensor(typename Policy::KvShape{});
    auto source_page  = cache.template page<Value, Policy::KeyTile>(page_index);
    auto tiled_copy   = typename Copy::TiledCopy{};
    auto thread_copy  = tiled_copy.get_thread_slice(thread_idx);
    auto coordinates  = cute::group_modes<
        1, cute::rank_v<decltype(thread_copy.partition_S(logical_tile))>>(
        thread_copy.partition_S(logical_tile));
    auto sources = cute::group_modes<
        1, cute::rank_v<decltype(thread_copy.partition_S(source_page))>>(
        thread_copy.partition_S(source_page));
    auto destinations = cute::group_modes<
        1, cute::rank_v<decltype(thread_copy.partition_D(destination))>>(
        thread_copy.partition_D(destination));

    auto copy_atom = typename Copy::Atom{};
    CUTE_UNROLL
    for (int access = 0; access < Copy::AccessCount; ++access) {
        if constexpr (GuardTail) {
            const int key = key_begin + cute::get<0>(coordinates(cute::_0{}, access));
            cute::copy(copy_atom.with(key < key_limit),
                       sources(cute::_, access),
                       destinations(cute::_, access));
        }
        else {
            cute::copy(copy_atom, sources(cute::_, access), destinations(cute::_, access));
        }
    }
}

template<bool Value, class T, class Policy, class SmemTensor>
CUTE_DEVICE void copy_paged_page(const PagedKv<T, Policy::HeadDim>& cache,
                                 int                                page_index,
                                 int                                key_begin,
                                 int                                key_limit,
                                 SmemTensor                         destination,
                                 int                                thread_idx = threadIdx.x)
{
    copy_paged_page_impl<Value, true, T, Policy>(
        cache, page_index, key_begin, key_limit, destination, thread_idx);
}

template<bool Value, class T, class Policy, class SmemTensor>
CUTE_DEVICE void copy_paged_page_full(const PagedKv<T, Policy::HeadDim>& cache,
                                      int                                page_index,
                                      SmemTensor                         destination,
                                      int                                thread_idx = threadIdx.x)
{
    copy_paged_page_impl<Value, false, T, Policy>(
        cache, page_index, 0, 0, destination, thread_idx);
}

}  // namespace turbomind::verification_attention
