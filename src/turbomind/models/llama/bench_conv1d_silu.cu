
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"

using namespace turbomind;
using namespace turbomind::core;

struct Args {
    int      batch_size  = 32;
    int      seq_len     = 1;
    int      num_v_heads = 64;
    int      num_k_heads = 16;
    int      d_conv      = 4;
    int      warmup      = 10;
    int      iters       = 100;
    DataType dtype       = kFloat16;

    static DataType ParseDtype(const char* s)
    {
        if (strcmp(s, "half") == 0 || strcmp(s, "fp16") == 0)
            return kFloat16;
        if (strcmp(s, "bf16") == 0)
            return kBfloat16;
        fprintf(stderr, "Unknown dtype: %s (expected half/fp16/bf16)\n", s);
        exit(1);
    }

    static Args Parse(int argc, char** argv)
    {
        Args a;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", argv[i]);
                exit(1);
            }
            if (strcmp(argv[i], "--batch_size") == 0)
                a.batch_size = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--seq_len") == 0)
                a.seq_len = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--num_v_heads") == 0)
                a.num_v_heads = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--num_k_heads") == 0)
                a.num_k_heads = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--d_conv") == 0)
                a.d_conv = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--warmup") == 0)
                a.warmup = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--iters") == 0)
                a.iters = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--dtype") == 0)
                a.dtype = ParseDtype(argv[i + 1]);
            else {
                fprintf(stderr, "Unknown arg: %s\n", argv[i]);
                exit(1);
            }
        }
        return a;
    }

    void Print() const
    {
        printf("batch_size=%d  seq_len=%d  num_v_heads=%d  num_k_heads=%d  d_conv=%d  "
               "warmup=%d  iters=%d  dtype=%s\n",
               batch_size,
               seq_len,
               num_v_heads,
               num_k_heads,
               d_conv,
               warmup,
               iters,
               to_string(dtype));
    }
};

static float benchmark_kernel(const char*           name,
                              std::function<void()> launch,
                              cudaStream_t          stream,
                              int                   warmup,
                              int                   iters)
{
    for (int i = 0; i < warmup; ++i)
        launch();
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; ++i)
        launch();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;

    printf("  %-45s  %8.3f ms (avg over %d iters)\n", name, avg_ms, iters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return avg_ms;
}

// CPU reference: conv1d + SiLU with transposed weight/state layout [d_conv, conv_dim]
template<typename T>
static void cpu_conv1d_silu(T*        h_out,
                            const T*  h_in,
                            const T*  h_weight,
                            T*        h_state,
                            const int* h_q_offsets,
                            int       batch_size,
                            int       conv_dim,
                            int       d_conv,
                            int       in_stride,
                            int       total_tokens)
{
    const int state_stride = d_conv * conv_dim;

    for (int t = 0; t < total_tokens; ++t) {
        int b = 0;
        for (int bb = 0; bb < batch_size; ++bb) {
            if (t < h_q_offsets[bb + 1]) {
                b = bb;
                break;
            }
        }
        const int t_local = t - h_q_offsets[b];
        const int seq_len = h_q_offsets[b + 1] - h_q_offsets[b];

        for (int c = 0; c < conv_dim; ++c) {
            float acc = 0.0f;
            for (int d = 0; d < d_conv; ++d) {
                int   src = t_local - (d_conv - 1 - d);
                float val;
                if (src >= 0) {
                    val = static_cast<float>(h_in[(h_q_offsets[b] + src) * in_stride + c]);
                }
                else {
                    val = static_cast<float>(h_state[b * state_stride + (d_conv + src) * conv_dim + c]);
                }
                float w = static_cast<float>(h_weight[d * conv_dim + c]);
                acc += val * w;
            }
            h_out[t * conv_dim + c] = static_cast<T>(acc / (1.0f + std::exp(-acc)));

            if (t_local == seq_len - 1) {
                T* sb = h_state + b * state_stride;
                for (int d = 0; d < d_conv; ++d) {
                    int src_t = seq_len - d_conv + d;
                    if (src_t >= 0) {
                        sb[d * conv_dim + c] = h_in[(h_q_offsets[b] + src_t) * in_stride + c];
                    }
                    else {
                        sb[d * conv_dim + c] = sb[(d + seq_len) * conv_dim + c];
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    auto args = Args::Parse(argc, argv);
    args.Print();

    constexpr int kHeadDim = 128;

    const int num_v_heads = args.num_v_heads;
    const int num_k_heads = args.num_k_heads;
    const int batch_size  = args.batch_size;
    const int seq_len     = args.seq_len;
    const int d_conv      = args.d_conv;

    const int k_dim     = num_k_heads * kHeadDim;
    const int v_dim     = num_v_heads * kHeadDim;
    const int conv_dim  = 2 * k_dim + v_dim;
    const int in_stride = conv_dim + v_dim + 2 * num_v_heads;
    const int total_tok = batch_size * seq_len;

    const int      conv_state_size = conv_dim * d_conv;
    const DataType dtype           = args.dtype;
    const auto     elem_bytes      = byte_size(dtype);

    auto         stream    = Stream::create();
    ContextGuard ctx{stream, Allocator{kCPU}, Allocator{kCPUpinned}, Allocator{stream, false}};
    cudaStream_t cu_stream = stream.handle();

    int sm_count = 1;
    {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    Buffer_<int> work_counter{1, kDEVICE};

    printf("\nconv_dim=%d  d_conv=%d  in_stride=%d  total_tokens=%d\n", conv_dim, d_conv, in_stride, total_tok);

    Tensor all_proj{Layout{{total_tok, in_stride}}, dtype, kDEVICE};
    Tensor weight{Layout{{d_conv, conv_dim}}, dtype, kDEVICE};

    Tensor out_ref{Layout{{total_tok, conv_dim}}, dtype, kDEVICE};
    Tensor out_v2{Layout{{total_tok, conv_dim}}, dtype, kDEVICE};

    Tensor state_ref{Layout{{batch_size, conv_state_size}}, dtype, kDEVICE};
    Tensor state_v2{Layout{{batch_size, conv_state_size}}, dtype, kDEVICE};

    Buffer_<void*> state_ptrs_v2_host{batch_size, kCPUpinned};
    Buffer_<void*> state_ptrs_v2_dev{batch_size, kDEVICE};

    Buffer_<int> q_offsets_host{batch_size + 1, kCPUpinned};
    Buffer_<int> q_offsets_dev{batch_size + 1, kDEVICE};

    RNG rng;
    rng.UniformFloat(all_proj, 0.1f);
    rng.UniformFloat(weight, 0.1f);

    for (int i = 0; i <= batch_size; ++i)
        q_offsets_host.data()[i] = i * seq_len;
    Copy(q_offsets_host, batch_size + 1, q_offsets_dev);

    for (int i = 0; i < batch_size; ++i) {
        state_ptrs_v2_host.data()[i] = (char*)state_v2.raw_data() + i * conv_state_size * elem_bytes;
    }
    Copy(state_ptrs_v2_host, batch_size, state_ptrs_v2_dev);
    stream.Sync();

    auto launch_v2 = [&] {
        invokeFusedConv1dSiLU(out_v2,
                              all_proj,
                              weight,
                              Tensor{},
                              state_ptrs_v2_dev,
                              q_offsets_dev,
                              batch_size,
                              0,
                              sm_count,
                              work_counter.data(),
                              cu_stream);
    };

    // === Benchmark ===
    printf("\n=== Benchmark ===\n");
    float v2_ms = benchmark_kernel("v2   (templated + vectorized)", launch_v2, cu_stream, args.warmup, args.iters);

    // === Bandwidth ===
    {
        double in_bytes    = (double)total_tok * conv_dim * elem_bytes;
        double out_bytes   = (double)total_tok * conv_dim * elem_bytes;
        double wt_bytes    = (double)conv_dim * d_conv * elem_bytes;
        double state_bytes = (double)batch_size * conv_state_size * elem_bytes * 2.0;
        double total_bytes = in_bytes + out_bytes + wt_bytes + state_bytes;

        printf("\n=== Bandwidth ===\n");
        printf("  in:     %.1f MB\n", in_bytes / 1e6);
        printf("  out:    %.1f MB\n", out_bytes / 1e6);
        printf("  weight: %.3f MB\n", wt_bytes / 1e6);
        printf("  state:  %.1f MB  (R+W)\n", state_bytes / 1e6);
        printf("  total:  %.1f MB\n", total_bytes / 1e6);
        printf("  v2  BW: %.1f GB/s\n", total_bytes / (v2_ms * 1e6));
    }

    // === Cross-comparison (correctness): CPU ref vs GPU v2 ===
    printf("\n=== Cross-comparison (CPU ref vs GPU v2) ===\n");

    Clear(state_ref);
    Clear(state_v2);
    Clear(out_ref);
    Clear(out_v2);
    stream.Sync();

    // Run GPU kernel
    launch_v2();
    stream.Sync();

    // Run CPU reference
    {
        const size_t in_bytes     = (size_t)total_tok * in_stride * elem_bytes;
        const size_t wt_bytes     = (size_t)d_conv * conv_dim * elem_bytes;
        const size_t state_bytes  = (size_t)batch_size * conv_state_size * elem_bytes;
        const size_t out_bytes    = (size_t)total_tok * conv_dim * elem_bytes;

        std::vector<char> h_in(in_bytes), h_wt(wt_bytes), h_state(state_bytes), h_out(out_bytes);

        cudaMemcpy(h_in.data(), all_proj.raw_data(), in_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_wt.data(), weight.raw_data(), wt_bytes, cudaMemcpyDeviceToHost);
        std::memset(h_state.data(), 0, state_bytes);
        std::memset(h_out.data(), 0, out_bytes);

        auto run_cpu = [&](auto t) {
            using T = decltype(t);
            cpu_conv1d_silu((T*)h_out.data(),
                            (const T*)h_in.data(),
                            (const T*)h_wt.data(),
                            (T*)h_state.data(),
                            q_offsets_host.data(),
                            batch_size,
                            conv_dim,
                            d_conv,
                            in_stride,
                            total_tok);
        };

        if (dtype == kFloat16)
            run_cpu(half{});
        else
            run_cpu(nv_bfloat16{});

        cudaMemcpy(out_ref.raw_data(), h_out.data(), out_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(state_ref.raw_data(), h_state.data(), state_bytes, cudaMemcpyHostToDevice);
    }

    printf("  output comparison:\n");
    FC_Header();
    FC_Print(FastCompare(out_ref, out_v2, cu_stream));

    printf("  state comparison:\n");
    FC_Header();
    FC_Print(FastCompare(state_ref, state_v2, cu_stream));

    printf("\nDone.\n");
    return 0;
}
