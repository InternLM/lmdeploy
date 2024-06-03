// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include <iterator>

namespace turbomind::gemm {

struct Transform {
    template<class FragA, class FragB, class DataA, class DataB, class DataU, class DataV>
    __device__ static void transform(FragA& frag_A,  //
                                     FragB& frag_B,
                                     int    k,
                                     DataA& data_A,
                                     DataB& data_B,
                                     DataU&,
                                     DataV&)
    {
        static_assert(sizeof(frag_A) == sizeof(data_A));
        PRAGMA_UNROLL
        for (int i = 0; i < std::size(frag_A[k]); ++i) {
            frag_A[k][i] = data_A[k][i];
        }

        static_assert(sizeof(frag_B) == sizeof(data_B));
        auto& frag_B_k = (decltype(data_B[k])&)frag_B[k];
        PRAGMA_UNROLL
        for (int i = 0; i < std::size(frag_B_k); ++i) {
            frag_B_k[i] = data_B[k][i];
        }
    }
};

}  // namespace turbomind::gemm