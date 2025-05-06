// Copyright (c) OpenMMLab. All rights reserved.

#include <type_traits>
#include <utility>

#include "decoding.h"
#include "decoding_config.h"
#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class Kernel>
bool invokeDecoding(const typename Kernel::ParamType& params);

template<int... idxs>
using seq = std::integer_sequence<int, idxs...>;

template<class T, int is_kv_int8>
constexpr auto get_kv_type(std::integral_constant<int, is_kv_int8>)
{
    if constexpr (is_kv_int8) {
        return int8_t{};
    }
    else {
        return T{};
    }
}

template<class T>
void dispatchDecoding(const AttentionParams<T>& params)
{
    const bool is_kv_int8     = params.quant_policy & QuantPolicy::kCacheKVInt8;
    const bool is_kv_int4     = params.quant_policy & QuantPolicy::kCacheKVInt4;
    const int  query_group_sz = params.num_heads / params.num_kv_heads;

    using namespace attention;

    /// TODO: we need better Qh dispatching, when #waves < 1, smaller Qh may outperform larger Qh due to better
    // concurrency
    auto dispatch_h = [&](auto arch, auto kv, const auto dim) -> bool {
        using Arch             = decltype(arch);
        using Tkv              = decltype(kv);
        constexpr int kHeadDim = dim;
        if (0) {}
        else if (query_group_sz > 8) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 9, kHeadDim>>(params);
        }
        else if (query_group_sz == 8) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 8, kHeadDim>>(params);
        }
        else if (query_group_sz == 7) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 7, kHeadDim>>(params);
        }
        else if (query_group_sz == 6) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 6, kHeadDim>>(params);
        }
        else if (query_group_sz == 5) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 5, kHeadDim>>(params);
        }
        else if (query_group_sz == 4) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 4, kHeadDim>>(params);
        }
        else if (query_group_sz == 3) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 3, kHeadDim>>(params);
        }
        else if (query_group_sz == 2) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 2, kHeadDim>>(params);
        }
        else {
            return invokeDecoding<Decoding<Arch, T, Tkv, 1, kHeadDim>>(params);
        }
        return false;
    };

    auto dispatch_kv = [&](auto arch, const auto dim) -> bool {
        FT_CHECK(!(is_kv_int4 && is_kv_int8));
        if (is_kv_int4) {
            return dispatch_h(arch, uint4_t{}, dim);
        }
        else if (is_kv_int8) {
            return dispatch_h(arch, uint8_t{}, dim);
        }
        else {
            return dispatch_h(arch, T{}, dim);
        }
        return false;
    };

    auto dispatch_head_dim = [&](auto arch) {
        if (params.size_per_head == 128) {
            return dispatch_kv(arch, std::integral_constant<int, 128>{});
        }
        else if (params.size_per_head == 64) {
            return dispatch_kv(arch, std::integral_constant<int, 64>{});
        }
        return false;
    };

    auto dispatch = [&]() {
        if (params.arch >= 80) {
            return dispatch_head_dim(arch::Sm80{});
        }

        if constexpr (!std::is_same_v<T, nv_bfloat16>) {
            if (params.arch == 75) {
                return dispatch_head_dim(arch::Sm75{});
            }
            else if (params.arch >= 70) {
                return dispatch_head_dim(arch::Sm70{});
            }
        }

        return false;
    };

    if (params.size_per_head == 192) {

        if (is_kv_int8) {
            invokeDecoding<Decoding<arch::Sm80, T, uint8_t, 1, 192>>(params);
        }
        else if (is_kv_int4) {
            FT_CHECK_WITH_INFO(!is_kv_int4, "not implemented");
            // invokeDecoding<Decoding<arch::Sm80, T, uint4_t, 1, 192>>(params);
        }
        else {
            invokeDecoding<Decoding<arch::Sm80, T, T, 1, 192>>(params);
        }
        return;
    }

    auto success = dispatch();

    FT_CHECK(success);
}

template void dispatchDecoding(const AttentionParams<half>& params);
#if ENABLE_BF16
template void dispatchDecoding(const AttentionParams<nv_bfloat16>& params);
#endif

}  // namespace turbomind
