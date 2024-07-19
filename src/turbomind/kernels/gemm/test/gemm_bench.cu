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
    {28672 * 2, 8192}, {8192, 28672}, {10240, 8192}, {8192, 8192},  // llama2-70b / llama3-70b
    {29696 * 2, 8192}, {8192, 29696}, {10240, 8192}, {8192, 8192}   // qwen2-72b-instruct-awq
};

// {29568 * 2, 8192}, {8192, 29568}, {10240, 8192}, {8192, 8192},  // qwen2-72b

void gemm_bench(nvbench::state& state)
{
    const auto idx = state.get_int64("idx");

    const auto bs = state.get_int64("bs");
    const auto tp = state.get_int64("tp");

    auto [n, k] = config[idx];

    // const auto n      = state.get_int64("batch size");
    // const auto [m, k] = config[index];

    if (idx % 4 == 0 || idx % 4 == 2) {
        n /= tp;
    }
    else {
        k /= tp;
    }

    using turbomind::gemm::get_test;

    get_test().Initialize(bs, n, k, 128, state.get_cuda_stream());

    state.add_element_count((size_t)bs * n * k * 2);  // mul + add

    // state.collect_dram_throughput();
    // state.collect_l2_hit_rates();

    if constexpr (1) {
        // state.add_global_memory_reads(m * k / 2 + sizeof(half) * n * k);
        state.add_global_memory_reads(bs * k * 2 + n * k / 2);
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {  //
            get_test().Run();
        });
    }
    else {
        state.add_global_memory_reads(sizeof(half) * (bs * k + n * k));
        state.exec([&](nvbench::launch&) {  //
            // g_testbed->RunCublas();
        });
    }
}

NVBENCH_BENCH(gemm_bench)
    .add_int64_axis("idx", nvbench::range(0, 27))
    .add_int64_power_of_two_axis("bs", nvbench::range(0, 10))
    .add_int64_axis("tp", {1, 2, 4});

int main(int argc, char* argv[])
{
    NVBENCH_MAIN_BODY(argc, argv);
}
