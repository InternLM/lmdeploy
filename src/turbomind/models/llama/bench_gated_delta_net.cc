
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"

using namespace turbomind;
using namespace turbomind::core;

struct Args {
    int      batch_size  = 32;
    int      seq_len     = 64;
    int      num_v_heads = 16;
    int      num_k_heads = 4;
    int      warmup      = 10;
    int      iters       = 100;
    DataType dtype       = kFloat16;
    DataType state_dtype = kFloat32;

    static DataType ParseDtype(const char* s)
    {
        if (strcmp(s, "half") == 0 || strcmp(s, "fp16") == 0)
            return kFloat16;
        if (strcmp(s, "bf16") == 0)
            return kBfloat16;
        if (strcmp(s, "fp32") == 0 || strcmp(s, "float") == 0)
            return kFloat32;
        fprintf(stderr, "Unknown dtype: %s (expected half/fp16/bf16/fp32/float)\n", s);
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
            else if (strcmp(argv[i], "--warmup") == 0)
                a.warmup = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--iters") == 0)
                a.iters = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--dtype") == 0)
                a.dtype = ParseDtype(argv[i + 1]);
            else if (strcmp(argv[i], "--state_dtype") == 0)
                a.state_dtype = ParseDtype(argv[i + 1]);
            else {
                fprintf(stderr, "Unknown arg: %s\n", argv[i]);
                exit(1);
            }
        }
        return a;
    }

    void Print() const
    {
        printf(
            "batch_size=%d  seq_len=%d  num_v_heads=%d  num_k_heads=%d  warmup=%d  iters=%d  dtype=%s  state_dtype=%s\n",
            batch_size,
            seq_len,
            num_v_heads,
            num_k_heads,
            warmup,
            iters,
            to_string(dtype),
            to_string(state_dtype));
    }
};

static float
benchmark_kernel(const char* name, std::function<void()> launch, cudaStream_t stream, int warmup, int iters)
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

int main(int argc, char** argv)
{
    auto args = Args::Parse(argc, argv);
    args.Print();

    constexpr int kHeadDim = 128;

    const int num_v_heads = args.num_v_heads;
    const int num_k_heads = args.num_k_heads;
    const int batch_size  = args.batch_size;
    const int seq_len     = args.seq_len;

    const int k_dim     = num_k_heads * kHeadDim;
    const int v_dim     = num_v_heads * kHeadDim;
    const int conv_dim  = 2 * k_dim + v_dim;
    const int total_tok = batch_size * seq_len;

    const int state_size = num_v_heads * kHeadDim * kHeadDim;  // per request

    const DataType dtype       = args.dtype;
    const DataType state_dtype = args.state_dtype;

    // --- Context setup ---
    auto         stream = Stream::create();
    ContextGuard ctx{stream, Allocator{kCPU}, Allocator{kCPUpinned}, Allocator{stream, false}};
    cudaStream_t cu_stream = stream.handle();

    const bool is_decode = (seq_len == 1);

    // --- Allocate tensors ---
    Tensor qkv_in{Layout{{total_tok, conv_dim}}, dtype, kDEVICE};
    Tensor v_out_v2{Layout{{total_tok, v_dim}}, dtype, kDEVICE};
    Tensor v_out_chunked{Layout{{total_tok, v_dim}}, dtype, kDEVICE};
    Tensor v_out_v3{Layout{{total_tok, v_dim}}, dtype, kDEVICE};
    Tensor beta{Layout{{total_tok, num_v_heads}}, dtype, kDEVICE};
    Tensor g{Layout{{total_tok, num_v_heads}}, dtype, kDEVICE};

    // State buffers — v2/chunked use state_dtype, v3 always uses 16-bit (dtype)
    Tensor state_v2{Layout{{batch_size, state_size}}, state_dtype, kDEVICE};
    Tensor state_chunked{Layout{{batch_size, state_size}}, state_dtype, kDEVICE};
    Tensor state_v3{Layout{{batch_size, state_size}}, dtype, kDEVICE};  // S = T

    // State pointer arrays: host pinned + device
    Buffer_<void*> state_ptrs_v2_host{batch_size, kCPUpinned};
    Buffer_<void*> state_ptrs_v2_dev{batch_size, kDEVICE};
    Buffer_<void*> state_ptrs_chunked_host{batch_size, kCPUpinned};
    Buffer_<void*> state_ptrs_chunked_dev{batch_size, kDEVICE};
    Buffer_<void*> state_ptrs_v3_host{batch_size, kCPUpinned};
    Buffer_<void*> state_ptrs_v3_dev{batch_size, kDEVICE};

    // q_offsets: host + device
    Buffer_<int> q_offsets_host{batch_size + 1, kCPUpinned};
    Buffer_<int> q_offsets_dev{batch_size + 1, kDEVICE};

    // --- Fill random data ---
    RNG rng;
    rng.UniformFloat(qkv_in, 0.1f);
    rng.UniformFloat(beta, 1.0f);        // will be passed through sigmoid inside kernel
    rng.UniformFloat(g, 0.02f, -0.01f);  // small values around 0
    Clear(state_v2);
    Clear(state_chunked);
    Clear(state_v3);

    // --- Build q_offsets ---
    for (int i = 0; i <= batch_size; ++i)
        q_offsets_host.data()[i] = i * seq_len;
    Copy(q_offsets_host, batch_size + 1, q_offsets_dev);

    // --- Build state_ptrs ---
    const auto state_elem_bytes    = byte_size(state_dtype);
    const auto state_elem_bytes_v3 = byte_size(dtype);
    for (int i = 0; i < batch_size; ++i) {
        state_ptrs_v2_host.data()[i]      = (char*)state_v2.raw_data() + i * state_size * state_elem_bytes;
        state_ptrs_chunked_host.data()[i] = (char*)state_chunked.raw_data() + i * state_size * state_elem_bytes;
        state_ptrs_v3_host.data()[i]      = (char*)state_v3.raw_data() + i * state_size * state_elem_bytes_v3;
    }
    Copy(state_ptrs_v2_host, batch_size, state_ptrs_v2_dev);
    Copy(state_ptrs_chunked_host, batch_size, state_ptrs_chunked_dev);
    Copy(state_ptrs_v3_host, batch_size, state_ptrs_v3_dev);
    stream.Sync();

    // Shared resources for all three kernel launchers
    int sm_count = 1;
    {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    Buffer_<int> work_counter_buf{1, kDEVICE};
    int*         work_counter = work_counter_buf.data();

    // --- Benchmark recurrent (v2) kernel ---
    printf("\n=== Benchmarks ===\n");
    auto launch_v2 = [&] {
        invokeGatedDeltaRuleBatched_v2(v_out_v2,
                                       qkv_in,
                                       beta,
                                       g,
                                       state_ptrs_v2_dev,
                                       q_offsets_dev,
                                       batch_size,
                                       num_k_heads,
                                       0,
                                       state_dtype,
                                       sm_count,
                                       work_counter,
                                       cu_stream);
    };
    float v2_ms = benchmark_kernel("invokeGatedDeltaRuleBatched_v2", launch_v2, cu_stream, args.warmup, args.iters);

    // --- Benchmark chunked kernel ---
    auto launch_chunked = [&] {
        invokeChunkedGatedDeltaRuleBatched(v_out_chunked,
                                           qkv_in,
                                           beta,
                                           g,
                                           state_ptrs_chunked_dev,
                                           q_offsets_dev,
                                           batch_size,
                                           num_k_heads,
                                           0,
                                           state_dtype,
                                           sm_count,
                                           work_counter,
                                           cu_stream);
    };
    float chunked_ms =
        benchmark_kernel("invokeChunkedGatedDeltaRuleBatched", launch_chunked, cu_stream, args.warmup, args.iters);

    // --- Benchmark v3 persistent decode kernel (seq_len == 1 only) ---
    float v3_ms     = -1.f;
    auto  launch_v3 = [&] {
        invokeGatedDeltaRuleBatched_v3(v_out_v3,
                                       qkv_in,
                                       beta,
                                       g,
                                       state_ptrs_v3_dev,
                                       q_offsets_dev,
                                       batch_size,
                                       num_k_heads,
                                       0,
                                       dtype,
                                       sm_count,
                                       work_counter,
                                       cu_stream);
    };
    if (is_decode) {
        v3_ms = benchmark_kernel(
            "invokeGatedDeltaRuleBatched_v3 (persistent)", launch_v3, cu_stream, args.warmup, args.iters);
    }
    else {
        printf("  %-45s  (skipped — seq_len > 1)\n", "invokeGatedDeltaRuleBatched_v3 (persistent)");
    }

    printf("\n  Speedup v2 / chunked:  %.2fx\n", v2_ms / chunked_ms);
    if (is_decode)
        printf("  Speedup v2 / v3:       %.2fx\n", v2_ms / v3_ms);

    // --- Bandwidth stats ---
    {
        double state_bytes    = (double)batch_size * state_size * state_elem_bytes * 2.0;
        double state_bytes_v3 = (double)batch_size * state_size * state_elem_bytes_v3 * 2.0;
        printf("\n=== Bandwidth ===\n");
        printf("  v2:      state BW = %.1f GB/s\n", state_bytes / (v2_ms * 1e6));
        printf("  chunked: state BW = %.1f GB/s\n", state_bytes / (chunked_ms * 1e6));
        if (is_decode)
            printf("  v3:      state BW = %.1f GB/s\n", state_bytes_v3 / (v3_ms * 1e6));
        printf("  total_tokens = %d\n", total_tok);
    }

    // === Cross-comparison: run both kernels on identical input, compare outputs ===
    printf("\n=== Cross-comparison (v2 vs chunked) ===\n");

    // Reset states to identical initial values (zero)
    Clear(state_v2);
    Clear(state_chunked);
    Clear(v_out_v2);
    Clear(v_out_chunked);
    stream.Sync();

    // Single invocation of each kernel
    launch_v2();
    launch_chunked();
    stream.Sync();

    // Compare v_out
    printf("  v_out comparison:\n");
    FC_Header();
    auto v_out_stats = FastCompare(v_out_v2, v_out_chunked, cu_stream);
    FC_Print(v_out_stats);

    // Compare final states
    printf("  state comparison:\n");
    FC_Header();
    auto state_stats = FastCompare(state_v2, state_chunked, cu_stream);
    FC_Print(state_stats);

    // === Cross-comparison: v2 vs v3 (decode only) ===
    if (is_decode) {
        printf("\n=== Cross-comparison (v2 vs v3, state_dtype=%s) ===\n", to_string(state_dtype));

        Clear(state_v2);
        Clear(state_v3);
        Clear(v_out_v2);
        Clear(v_out_v3);
        stream.Sync();

        launch_v2();
        launch_v3();
        stream.Sync();

        printf("  v_out comparison:\n");
        FC_Header();
        FC_Print(FastCompare(v_out_v2, v_out_v3, cu_stream));

        printf("  state comparison:\n");
        FC_Header();
        FC_Print(FastCompare(state_v2, state_v3, cu_stream));
    }

    printf("\nDone.\n");
    return 0;
}
