// Copyright (c) OpenMMLab. All rights reserved.

#include "nvbench/main.cuh"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/test/models.h"
#include "src/turbomind/kernels/gemm/test/testbed.h"
#include <map>
#include <nvbench/nvbench.cuh>
#include <string>

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
