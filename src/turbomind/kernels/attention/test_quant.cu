#include "quantization.h"
#include "src/turbomind/kernels/attention/array_ops.h"
#include "src/turbomind/kernels/attention/test_utils.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

#include <thrust/universal_vector.h>

using namespace turbomind;

__global__ void test(half* dst, const half* src, int n, float scale, float zero)
{
    ConvertKvCache<half, uint4_t> f16_to_u4{scale, zero};
    ConvertKvCache<uint4_t, half> u4_to_f16{scale, zero};

    constexpr int kVecSize = 8;

    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * kVecSize; i < n; i += (blockDim.x * gridDim.x) * kVecSize) {
        Array<half, kVecSize> v;
        Load(v, &src[i]);
        auto tmp = f16_to_u4(v);
        auto u   = u4_to_f16(tmp);
        Store(&dst[i], u);
    }
}

__global__ void cvt_u4_f16(uint4_t* dst, const half* src, int n, float scale, float zero)
{
    ConvertKvCache<half, uint4_t> f16_to_u4{scale, zero};

    constexpr int kVecSize = 8;

    using VecF16 = Array<half, kVecSize>;
    using VecU4  = Array<uint4_t, kVecSize>;

    auto v_src = (const VecF16*)src;
    auto v_dst = (VecU4*)dst;

    n /= 8;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        VecF16 vi;
        Load(vi, v_src[i].data());

        VecU4 vo = f16_to_u4(vi);

        Store((uint4_t*)&v_dst[i], vo);
    }
}

template<int M>
__global__ void __launch_bounds__(128) cvt_f16_u4(half* dst, const uint4_t* src, int n, float scale, float zero)
{
    ConvertKvCache<uint4_t, half> u4_to_f16{scale, zero};

    constexpr int kVecSize = 8;

    constexpr int ITER = M / kVecSize;

    using VecF16 = Array<half, kVecSize>;
    using VecU4  = Array<uint4_t, kVecSize>;

    auto v_src = (const VecU4*)src;
    auto v_dst = (VecF16*)dst;

    n /= M;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        PRAGMA_UNROLL
        for (int m = 0; m < ITER; ++m) {
            VecU4 vi;
            Load(vi, (uint4_t*)&v_src[i * ITER + m]);
            VecF16 vo = u4_to_f16(vi);
            Store(v_dst[i * ITER + m].data(), vo);
        }
    }
}

int main(int argc, char* argv[])
{

    using namespace thrust;

    universal_vector<half>              src(1 << 30);
    universal_vector<Array<uint4_t, 8>> tmp(src.size() / 8);
    universal_vector<half>              dst(src.size());

    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = half(rand() % 16);
    }

    // test<<<1024, 128>>>(dst.data().get(), src.data().get(), src.size(), 1, 0);

    cvt_u4_f16<<<1024, 128>>>((uint4_t*)tmp.data().get(), src.data().get(), src.size(), 1, 0);
    cvt_f16_u4<32><<<1024, 128>>>(dst.data().get(), (uint4_t*)tmp.data().get(), src.size(), 1, 0);

    cudaDeviceSynchronize();

    // Compare(dst.data().get(), src.data().get(), src.size(), src.size(), 1);

    return 0;
}