// Copyright (c) OpenMMLab. All rights reserved.

#include "nvbench/main.cuh"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/test/testbed.h"
#include <map>
#include <nvbench/nvbench.cuh>
#include <string>

std::vector<std::pair<int64_t, int64_t>> config{
    {11008 * 2, 4096}, {4096, 11008}, {12288, 4096}, {4096, 4096},  // llama2-7b
    {14336 * 2, 4096}, {4096, 14336}, {6144, 4096},  {4096, 4096},  // llama3-8b / internlm2.5-7b
    {16384 * 2, 6144}, {6144, 16384}, {8192, 6144},  {6144, 6144},  // internlm2-20b
    {13696 * 2, 4096}, {4096, 13696}, {4608, 4096},  {4096, 4096},  // glm4-9b
    {18944 * 2, 3584}, {3584, 18944}, {4608, 3584},  {3584, 3584},  // qwen2-7b
    {20480 * 2, 7168}, {7168, 20480}, {9216, 7168},  {7168, 7168},  // yi-34b
    {28672 * 2, 8192}, {8192, 28672}, {10240, 8192}, {8192, 8192},  // llama2-70b / llama3-70b
    {29696 * 2, 8192}, {8192, 29696}, {10240, 8192}, {8192, 8192}   // qwen2-72b-instruct-awq
};

// {29568 * 2, 8192}, {8192, 29568}, {10240, 8192}, {8192, 8192},  // qwen2-72b

void gemm_bench(nvbench::state& state)
{
    const auto idx = state.get_int64("idx");

    const auto bs = state.get_int64("bs");
    const auto tp = state.get_int64("tp");

    auto [output_dims, input_dims] = config[idx];

    constexpr int group_size = 128;

    if (idx % 4 == 0 || idx % 4 == 2) {
        if (output_dims % tp)
            return;
        output_dims /= tp;
    }
    else {
        if (input_dims % tp)
            return;
        input_dims /= tp;
    }

    if (input_dims % group_size)
        return;

    using turbomind::gemm::get_test;

    {
        int m = bs;
        int n = output_dims;
        int k = input_dims;
        if (get_test().kBatchDim == 1) {
            std::swap(m, n);
        }
        std::cerr << "m" << m << "n" << n << "k" << k << "\n";
        get_test().Initialize(m, n, k, group_size, state.get_cuda_stream());
    }

    state.add_element_count((size_t)bs * output_dims * input_dims * 2);  // mul + add

    // state.collect_dram_throughput();
    // state.collect_l2_hit_rates();

    if constexpr (1) {
        state.add_global_memory_reads(get_test().global_memory_reads());
        get_test().Run();
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {  //
            get_test().Run();
        });
    }
    else {
        state.add_global_memory_reads(sizeof(half) * (bs * input_dims + output_dims * input_dims));
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {  //
            get_test().RunCublas();
        });
    }
}

NVBENCH_BENCH(gemm_bench)
    .add_int64_axis("idx", nvbench::range(0, (int)config.size() - 1))
    .add_int64_power_of_two_axis("bs", nvbench::range(0, 10))
    .add_int64_axis("tp", {1, 2, 4});

int main(int argc, char* argv[])
{
    NVBENCH_MAIN_BODY(argc, argv);
}
