// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/reduce_kernel.h"

#include <type_traits>

namespace turbomind::attention {

template<int HeadDim, class T>
void invokeReduce(T*           out,
                  float*       partial_M,
                  float*       partial_L,
                  float*       partial_O,
                  const int*   split_cnt,
                  int          partial_len,
                  int          max_split_cnt,
                  int          query_num,
                  int          head_num,
                  float        exp_scale,
                  cudaStream_t stream)
{
    constexpr int CTA_K = 32;  // warp size

    using Reduce = attention::Reduce<T, 1, CTA_K, HeadDim, 4>;

    static constexpr size_t kSmemSize = sizeof(typename Reduce::SharedStorage);
    static_assert(kSmemSize < (48 << 10));

    auto invoke = [&](auto is_final, int stride_k) {
        const dim3 block = Reduce::kWarpCnt * 32;
        const dim3 grid  = ReduceCtaMap::get_grid_shape(query_num, head_num, max_split_cnt, CTA_K);
        reduce_kernel<Reduce, is_final><<<grid, block, kSmemSize, stream>>>(out,  //
                                                                            partial_M,
                                                                            partial_L,
                                                                            partial_O,
                                                                            nullptr,
                                                                            split_cnt,
                                                                            partial_len,
                                                                            head_num,
                                                                            exp_scale,
                                                                            stride_k);
    };

    int stride_k = 1;

    while (max_split_cnt > CTA_K) {
        invoke(std::false_type{}, stride_k);
        max_split_cnt = (max_split_cnt + CTA_K - 1) / CTA_K;
        stride_k *= CTA_K;
    }

    invoke(std::true_type{}, stride_k);
}

template void invokeReduce<128>(half*        out,
                                float*       partial_M,
                                float*       partial_L,
                                float*       partial_O,
                                const int*   split_cnt,
                                int          partial_len,
                                int          max_split_cnt,
                                int          query_num,
                                int          head_num,
                                float        exp_scale,
                                cudaStream_t stream);

#if ENABLE_BF16
template void invokeReduce<128>(nv_bfloat16* out,
                                float*       partial_M,
                                float*       partial_L,
                                float*       partial_O,
                                const int*   split_cnt,
                                int          partial_len,
                                int          max_split_cnt,
                                int          query_num,
                                int          head_num,
                                float        exp_scale,
                                cudaStream_t stream);
#endif

}  // namespace turbomind::attention
