// Copyright (c) OpenMMLab. All rights reserved.

#include "nvbench/main.cuh"
#include "src/turbomind/kernels/gemm/testbed.h"
#include <map>
#include <nvbench/nvbench.cuh>
#include <string>

std::map<std::string, std::vector<std::pair<int, int>>> config{
    {"llama2-7b", {{2 * 11008, 4096}, {4096, 11008}, {12288, 4096}, {4096, 4096}}}};

std::unique_ptr<turbomind::gemm::Testbed<half, turbomind::uint4_t>> g_testbed;

void gemm_bench(nvbench::state& state)
{
    const auto& weights = config["llama2-7b"];

    const auto index  = state.get_int64("index");
    const auto m      = state.get_int64("batch size");
    const auto [n, k] = weights[index];

    g_testbed->Initialize(m, n, k, 128, true, state.get_cuda_stream());
    // g_testbed->Run();

    state.add_element_count((size_t)m * n * k * 2);  // mul + add
    state.collect_dram_throughput();
    state.collect_l2_hit_rates();

    if constexpr (0) {
        state.add_global_memory_reads(sizeof(half) * m * k + n * k / 2);
        state.exec([&](nvbench::launch&) {  //
            g_testbed->Run();
        });
    }
    else {
        // state.add_global_memory_reads(sizeof(half) * (m * k + n * k));
        // state.exec([&](nvbench::launch&) {  //
        //     g_testbed->RunCublas();
        // });
    }
}

NVBENCH_BENCH(gemm_bench)
    .add_int64_power_of_two_axis("batch size", nvbench::range(0, 13))
    .add_int64_axis("index", nvbench::range(0, 3));

int main(int argc, char* argv[])
{
    g_testbed = std::make_unique<turbomind::gemm::Testbed<half, turbomind::uint4_t>>(
        turbomind::gemm::DispatchPolicy::kUseCached, "cache");

    NVBENCH_MAIN_BODY(argc, argv);

    g_testbed.reset();
}