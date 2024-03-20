// Copyright (c) OpenMMLab. All rights reserved.

#include "decoding.h"
#include "decoding_config.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/dispatch.h"
#include <type_traits>
#include <utility>

namespace turbomind {

template<class Kernel>
void invokeDecoding(const typename Kernel::ParamType& params);

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
    constexpr int kHeadDim = 128;

    const bool is_kv_int8     = params.quant_policy & QuantPolicy::kCacheKVInt8;
    const bool is_kv_int4     = params.quant_policy & QuantPolicy::kCacheKVInt4;
    const int  query_group_sz = params.num_heads / params.num_kv_heads;

    using namespace attention;

    // TODO: we need better Qh dispatching, when #waves < 1, smaller Qh may outperform larger Qh due to better
    // concurrency

    if (is_kv_int8) {
        if (params.arch >= 80) {
            if (0) {}
            // else if (query_group_sz % 2 == 0) {
            //     return invokeDecoding<typename DecodingConfig<arch::Sm80, T, int8_t, 2, kHeadDim>::Kernel>(params);
            // }
            // else {
                return invokeDecoding<typename DecodingConfig<arch::Sm80, T, uint8_t, 1, kHeadDim>::Kernel>(params);
            // }
        }
        else {
            // if (0) {}
            // else if (query_group_sz % 4 == 0) {
            //     return invokeDecoding<typename DecodingConfig<arch::Sm70, T, int8_t, 4, kHeadDim>::Kernel>(params);
            // }
            // else if (query_group_sz % 2 == 0) {
            //     return invokeDecoding<typename DecodingConfig<arch::Sm70, T, int8_t, 2, kHeadDim>::Kernel>(params);
            // }
            // else {
            //     return invokeDecoding<typename DecodingConfig<arch::Sm70, T, int8_t, 1, kHeadDim>::Kernel>(params);
            // }
        }
    }
    else if (is_kv_int4) {
        return invokeDecoding<typename DecodingConfig<arch::Sm80, T, uint4_t, 8, kHeadDim>::Kernel>(params);
    }
    else {
        if (params.arch >= 80) {  // tensor core & async copy
            if (0) {}
            // else if (query_group_sz % 8 == 0) {
            //     return invokeDecoding<typename DecodingConfig<arch::Sm80, T, T, 8, kHeadDim>::Kernel>(params);
            // }
            // else if (query_group_sz % 4 == 0) {
            //     return invokeDecoding<typename DecodingConfig<arch::Sm80, T, T, 4, kHeadDim>::Kernel>(params);
            // }
            // else if (query_group_sz % 2 == 0) {
            //     return invokeDecoding<typename DecodingConfig<arch::Sm80, T, T, 2, kHeadDim>::Kernel>(params);
            // }
            else {
                return invokeDecoding<typename DecodingConfig<arch::Sm80, T, T, 1, kHeadDim>::Kernel>(params);
            }
        }
        else {  // SIMT & sync copy
            // if (0) {}
            // else if (query_group_sz % 4 == 0) {
            //     return invokeDecoding<typename DecodingConfig<arch::Sm70, T, T, 4, kHeadDim>::Kernel>(params);
            // }
            // else if (query_group_sz % 2 == 0) {
            //     return invokeDecoding<typename DecodingConfig<arch::Sm70, T, T, 2, kHeadDim>::Kernel>(params);
            // }
            // else {
            //     return invokeDecoding<typename DecodingConfig<arch::Sm70, T, T, 1, kHeadDim>::Kernel>(params);
            // }
        }
    }

    FT_CHECK(0);
}

template<>
void dispatchDecoding(const AttentionParams<nv_bfloat16>& params)
{
    constexpr int kHeadDim = 128;

    const bool is_kv_int8     = params.quant_policy & QuantPolicy::kCacheKVInt8;
    const int  query_group_sz = params.num_heads / params.num_kv_heads;

    using namespace attention;

    // TODO: we need better Qh dispatching, when #waves < 1, smaller Qh may outperform larger Qh due to better
    // concurrency

    // if (is_kv_int8) {
    //     if (params.arch >= 80) {
    //         if (0) {}
    //         else if (query_group_sz % 2 == 0) {
    //             return invokeDecoding<typename DecodingConfig<arch::Sm80, nv_bfloat16, int8_t, 2, kHeadDim>::Kernel>(
    //                 params);
    //         }
    //         else {
    //             return invokeDecoding<typename DecodingConfig<arch::Sm80, nv_bfloat16, int8_t, 1, kHeadDim>::Kernel>(
    //                 params);
    //         }
    //     }
    // }
    // else {
    //     if (params.arch >= 80) {
    //         if (0) {}
    //         else if (query_group_sz % 8 == 0) {
    //             return invokeDecoding<
    //                 typename DecodingConfig<arch::Sm80, nv_bfloat16, nv_bfloat16, 8, kHeadDim>::Kernel>(params);
    //         }
    //         else if (query_group_sz % 4 == 0) {
    //             return invokeDecoding<
    //                 typename DecodingConfig<arch::Sm80, nv_bfloat16, nv_bfloat16, 4, kHeadDim>::Kernel>(params);
    //         }
    //         else if (query_group_sz % 2 == 0) {
    //             return invokeDecoding<
    //                 typename DecodingConfig<arch::Sm80, nv_bfloat16, nv_bfloat16, 2, kHeadDim>::Kernel>(params);
    //         }
    //         else {
    //             return invokeDecoding<
    //                 typename DecodingConfig<arch::Sm80, nv_bfloat16, nv_bfloat16, 1, kHeadDim>::Kernel>(params);
    //         }
    //     }
    // }

    FT_CHECK(0);
}

template void dispatchDecoding(const AttentionParams<half>& params);

}  // namespace turbomind
