// Copyright (c) OpenMMLab. All rights reserved.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "src/turbomind/kernels/draft_carry_kernels.h"

namespace turbomind {
namespace {

__global__ void SelectDraftCarryKernel(const unsigned char* local_residual,
                                       const int*           selected_local_rows,
                                       const bool*          candidate_active,
                                       unsigned char*       carry,
                                       int                  local_token_num,
                                       int                  row_bytes,
                                       int                  first,
                                       int                  last)
{
    const int candidate = blockIdx.x;

    auto*     destination  = reinterpret_cast<uint4*>(carry + static_cast<size_t>(candidate) * row_bytes);
    const int vector_count = row_bytes / sizeof(uint4);

    bool owned = candidate_active[candidate];
    int  row   = 0;
    if (owned) {
        row   = selected_local_rows[candidate];
        owned = 0 <= row && row < local_token_num && first <= row && row < last;
    }

    if (owned) {
        const auto* source = reinterpret_cast<const uint4*>(local_residual + static_cast<size_t>(row) * row_bytes);
        for (int i = threadIdx.x; i < vector_count; i += blockDim.x) {
            destination[i] = source[i];
        }
    }
    else {
        const uint4 zero{};
        for (int i = threadIdx.x; i < vector_count; i += blockDim.x) {
            destination[i] = zero;
        }
    }
}

}  // namespace

void invokeSelectDraftCarry(const void*  local_residual,
                            const int*   selected_local_rows,
                            const bool*  candidate_active,
                            void*        carry,
                            int          local_token_num,
                            int          candidate_count,
                            int          hidden_size,
                            int          element_bits,
                            int          first,
                            int          last,
                            cudaStream_t stream)
{
    if (candidate_count == 0) {
        return;
    }

    const int64_t row_bits  = static_cast<int64_t>(hidden_size) * element_bits;
    const int     row_bytes = row_bits / 8;

    constexpr int block_size = 256;
    SelectDraftCarryKernel<<<candidate_count, block_size, 0, stream>>>(
        static_cast<const unsigned char*>(local_residual),
        selected_local_rows,
        candidate_active,
        static_cast<unsigned char*>(carry),
        local_token_num,
        row_bytes,
        first,
        last);
}

}  // namespace turbomind
