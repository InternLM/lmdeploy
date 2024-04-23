
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/gemm/cache_utils.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/test_utils.h"
#include "src/turbomind/kernels/gemm/transcript.h"
#include <cublas_v2.h>
#include <limits>
#include <thrust/universal_vector.h>
#include <type_traits>

using namespace turbomind;
using thrust::universal_vector;

cublasHandle_t cublas_handle{};

void ComputeRefCpu(half* C, const half* A, const half* B, int m, int n, int k)
{
    for (int mm = 0; mm < m; ++mm) {
        for (int nn = 0; nn < n; ++nn) {
            float c = 0;
            for (int kk = 0; kk < k; ++kk) {
                c += (float)A[mm * k + kk] * (float)B[nn * k + kk];
            }
            C[mm * n + nn] = c;
        }
    }
}

void computeRefCublas(half* C, const half* A, const half* B, int m, int n, int k, cudaStream_t stream)
{
    // cublas
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
    float alpha = 1.f;
    float beta  = 0.f;
    // TNT (A and B are swapped for transposing C)
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_T,
                 CUBLAS_OP_N,
                 n,
                 m,
                 k,
                 &alpha,
                 B,
                 CUDA_R_16F,
                 k,
                 A,
                 CUDA_R_16F,
                 k,
                 &beta,
                 C,
                 CUDA_R_16F,
                 n,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "*** Check failed: (%s) @ %s:%d\n", #cond, __FILE__, __LINE__);                            \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

RNG& gRNG()
{
    static RNG inst;
    return inst;
}

// quantize using `scale` and `zeros`,
template<class T>
__global__ void find_stats(Array<T, 2>* minmax, const T* src, int N, int K, int G)
{
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.y;

    if (n_idx >= N || k_idx * G >= K) {
        return;
    }

    float min = std::numeric_limits<float>::infinity();
    float max = -min;

    for (int k = 0; k < G; k += 8) {
        Array<T, 8> vec;
        Load(vec, &src[n_idx * K + k_idx * G + k]);
        PRAGMA_UNROLL
        for (int i = 0; i < vec.size(); ++i) {
            min = __hmin(min, vec[i]);
            max = __hmax(max, vec[i]);
        }
    }

    // store in n-major
    Store(minmax[k_idx * N + n_idx].data(), Array<T, 2>{min, max});
}

template<class Q, bool asym, class T>
__global__ void find_params(T* param, const Array<T, 2>* minmax, int count)
{
    // int global_idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= count) {
        return;
    }
    const auto  stats     = minmax[global_idx];
    const float inv_q_max = fdividef(1.f, (1 << bitsof<Q>)-1);

    static_assert(asym);

    float scale = (T)(((float)stats[1] - (float)stats[0]) * inv_q_max);
    Store(param + global_idx * 2, Array<T, 2>{scale, stats[0]});
}

template<class Q, class T>
__global__ void quantize(uint16_t* dst, T* fake, const T* src, const T* stats, int N, int K, int G)
{
    static_assert(bitsof<Q> <= 16);
    static_assert(bitsof<T> == 16);  // fp16 & bf16

    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.y;

    if (n_idx >= N || k_idx * G >= K) {
        return;
    }

    Array<T, 2> param;
    Load(param, stats + (k_idx * N + n_idx) * 2);

    float inv_scale = fdividef(1.f, param[0]);

    for (int k = 0; k < G; k += 8) {
        Array<T, 8>        vi;
        Array<uint16_t, 8> vo;
        Load(vi, &src[n_idx * K + k_idx * G + k]);

        PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) {
            float u = (static_cast<float>(vi[i] - param[1])) * inv_scale;
            vo[i]   = quant<uint16_t>(u, bitsof<Q>);
        }
        Store(&dst[n_idx * K + k_idx * G + k], vo);

        if (fake) {
            Array<T, 8> vf;
            PRAGMA_UNROLL
            for (int i = 0; i < 8; ++i) {
                vf[i] = __hfma(static_cast<T>(vo[i]), param[0], param[1]);
            }
            Store(&fake[n_idx * K + k_idx * G + k], vf);
        }
    }
}

template<class T, class Tb>
void prepare_test_data(universal_vector<T>&        a,    // [m,k]
                       universal_vector<T>&        b,    // [n,k]
                       universal_vector<T>&        c,    // [m,n]
                       universal_vector<uint16_t>& b_q,  // [n*k]
                       universal_vector<T>&        q,    // [k/g,n,2]
                       int                         m,
                       int                         n,
                       int                         k,
                       int                         g)
{
    a.resize(m * k);
    b.resize(n * k);
    c.resize(m * n);

    gRNG().GenerateUniform(a.data().get(), a.size(), 1, -0.5);
    gRNG().GenerateUniform(b.data().get(), b.size(), 1, -0.5);

    universal_vector<T> c_ref(m * n);
    computeRefCublas(c_ref.data().get(), a.data().get(), b.data().get(), m, n, k, 0);

    if constexpr (!std::is_same_v<T, Tb>) {
        b_q.resize(n * k);

        CHECK(k % g == 0);

        q.resize(k / g * n * 2);
        universal_vector<Array<T, 2>> minmax(k / g * n);

        const int  threads = std::min(256, n);
        const dim3 blocks((n + threads - 1) / threads, k / g);

        find_stats<<<blocks, threads>>>(minmax.data().get(),  //
                                        b.data().get(),
                                        n,
                                        k,
                                        g);

        find_params<Tb, true><<<(minmax.size() + 255) / 256, 256>>>(q.data().get(),  //
                                                                    minmax.data().get(),
                                                                    minmax.size());

        // cudaDeviceSynchronize();
        // for (int i = 0; i < q.size(); i += 2) {
        //     std::cout << i << " " << (float)q[i] << " " << (float)q[i + 1] << "\n";
        // }

        universal_vector<T> b_f(b.size());
        quantize<Tb><<<blocks, threads>>>(b_q.data().get(),  //
                                          b_f.data().get(),
                                          b.data().get(),
                                          q.data().get(),
                                          n,
                                          k,
                                          g);

        cudaDeviceSynchronize();
        Compare(b_f.data().get(), b.data().get(), k, k, n);

        b.swap(b_f);
    }

    for (int i = 0; i < 5; ++i) {
        gemm::CacheFlushing::flush();
        computeRefCublas(c.data().get(), a.data().get(), b.data().get(), m, n, k, 0);
    }

    cudaDeviceSynchronize();
    Compare(c.data().get(), c_ref.data().get(), n, n, m);
}

template<class T, class Tb>
void Run(int m, int n, int k, int g = 128)
{
    constexpr int kMaxSplits = 32;

    universal_vector<T> a;
    universal_vector<T> b;
    universal_vector<T> c_ref;
    universal_vector<T> q;  //((k + g) / g * n * 2);

    universal_vector<uint16_t> b0;

    prepare_test_data<T, Tb>(a, b, c_ref, b0, q, m, n, k, g);

    universal_vector<float> workspace(c_ref.size() * (kMaxSplits + 1));
    thrust::fill(workspace.begin(), workspace.end(), 0);

    cudaDeviceSynchronize();

    // std::vector<T> c_cpu(m * n);
    // ComputeRefCpu(c_cpu.data(), a.data().get(), b.data().get(), m, n, k);

    if (1) {
        universal_vector<T>            c(m * n);
        universal_vector<Array<Tb, 8>> b1(n * k / 8);
        universal_vector<T>            q_pack(q.size());
        auto                           B1 = (Tb*)b1.data().get();

        // for (int i = 0; i < q.size(); i += 2) {
        //     std::cout << "q_AAA: " << (float)q[i] << " " << (float)q[i + 1] << std::endl;
        // }

        for (int i = 0; i < 1; ++i) {
            if constexpr (std::is_same_v<T, Tb>) {
                gemm::transcript<T>(B1, nullptr, b.data().get(), nullptr, n, k, g, 0);
            }
            else {
                gemm::transcript<T>(B1, q_pack.data().get(), b0.data().get(), q.data().get(), n, k, g, 0);
            }
        }

        cudaDeviceSynchronize();

        // for (int i = 0; i < q.size(); i += 2) {
        //     std::cout << "q_BBB: " << (float)q[i] << " " << (float)q[i + 1] << std::endl;
        // }

        for (int i = 0; i < 10; ++i) {
            gemm::CacheFlushing::flush();
            gemm::invoke(
                c.data().get(), a.data().get(), B1, q_pack.data().get(), m, n, k, 1, workspace.data().get(), 0);
        }

        // for (int i = 0; i < 5; ++i) {
        //     computeRefCublas(c_ref.data().get(), a.data().get(), b.data().get(), m, n, k, 0);
        // }

        cudaDeviceSynchronize();

        // Compare(c_ref.data().get(), c_cpu.data(), n, n, m, 1);
        Compare(c.data().get(), c_ref.data().get(), n, n, m, 0);
    }

    cudaDeviceSynchronize();
}

template<class T, class Tb>
void Test(int bsz, int tp)
{
    Run<T, Tb>(8192, 8192, 8192);
    // Run<T, Tb>(4096, 4096, 4096);
    // Run<half, uint4_t>(64, 11008, 4096);
    // Run<half, uint4_t>(128, 128, 32);
    // Run<half, uint4_t>(128, 128, 1024);

    // llama2-7b
    // Run<T, Tb>(bsz, 11008 * 2 / tp, 4096); // mlp.up/gate
    // Run<T, Tb>(bsz, 4096, 11008 / tp);  // mlp.down

    // llama2-7b
    // Run<T, Tb>(bsz, 10240 / tp, 8192);  // attn.qkv

    // Run<T, Tb>(8, 128, 512);
}

int main(int argc, char* argv[])
{
    Test<half, uint4_t>(8, 1);
    return 0;
}