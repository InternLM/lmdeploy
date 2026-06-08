#include "src/turbomind/models/qwen3_5vit/fused_embed_merge.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/core/array_ops.h"

namespace turbomind {

namespace {

template<typename T, int N>
__device__ Array<float, N> roundToStorageDtype(Array<float, N> x)
{
    return cast<float>(cast<T>(x));
}

template<int vec_size, typename T>
__global__ void fusedPosEmbedMergeKernel(T*         hidden_states,
                                         const T*   pos_embeds,
                                         const T*   pos_embed_weights,
                                         const int* mapped_idx,
                                         const T*   bias,
                                         int        hidden,
                                         int        vdim)
{
    const int index  = blockIdx.x;
    const int mapped = mapped_idx[index];  // same address for all threads in block -> L1 broadcast

    Array<T, 4> w4;
    Ldg(w4, pos_embed_weights + mapped * 4);

    const int row_off = index * hidden;
    const int pe_row0 = mapped * 4 * hidden;

    using namespace ops;
    for (int d = threadIdx.x; d < vdim; d += blockDim.x) {
        Array<float, vec_size> pos{};
        Array<T, vec_size>     tmp;
        Load(tmp, hidden_states + row_off + d * vec_size);
        auto hidden_acc = cast<float>(tmp);

        if (bias) {
            Ldg(tmp, bias + d * vec_size);
            hidden_acc = roundToStorageDtype<T>(hidden_acc + cast<float>(tmp));
        }
        PRAGMA_UNROLL
        for (int k = 0; k < 4; ++k) {
            Ldg(tmp, pos_embeds + pe_row0 + k * hidden + d * vec_size);
            pos = pos + cast<float>(tmp * w4[k]);
        }
        const auto out = hidden_acc + roundToStorageDtype<T>(pos);
        Store(hidden_states + row_off + d * vec_size, cast<T>(out));
    }
}

}  // namespace

void invokeFusedPosEmbedMerge(void*        hidden_states,
                              const void*  pos_embeds,
                              const void*  pos_embed_weights,
                              const int*   mapped_idx,
                              const void*  bias,
                              int          batch,
                              int          hidden,
                              DataType     dtype,
                              cudaStream_t stream)
{
    if (batch <= 0) {
        return;
    }

    const dim3 grid(batch);
    const dim3 block(128);

    auto invoke = [&](auto t) {
        using T                = decltype(t);
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        TM_CHECK(hidden % vec_size == 0);
        fusedPosEmbedMergeKernel<vec_size, T><<<grid, block, 0, stream>>>((T*)hidden_states,
                                                                          (const T*)pos_embeds,
                                                                          (const T*)pos_embed_weights,
                                                                          mapped_idx,
                                                                          (const T*)bias,
                                                                          hidden,
                                                                          hidden / vec_size);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
}

}  // namespace turbomind
