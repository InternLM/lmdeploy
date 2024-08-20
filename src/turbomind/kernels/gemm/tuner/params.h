// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <array>
#include <string>
#include <vector>

namespace turbomind::gemm {

struct TuningParams {
    // Split-k params
    int max_splits = 8;
    int max_waves  = 10;

    // Swizzling params
    std::vector<int> swizzle{3};

    // Sampling params
    float top_k    = 0;
    int   clusters = 5;
    int   min_iter = 1;
    int   max_iter = 10;
    float max_time = 1.f;

    std::vector<int> seq;
};

// example
//   max_splits=8,top_splits=5,max_waves=16,top_k=10,swizzle=[2,3,4],clusters=5,max_iter=10,min_iter=1,max_time=10.0
void ParseTuningParams(TuningParams& params, const std::string& str);

// example
//   16-16-128,256-128-1024,8192
std::vector<int> ParseTuningSequence(const std::string& str);

std::vector<int> GenerateTuningSequence(const std::vector<std::array<int, 3>>& generators);

std::vector<std::array<int, 3>> GetDefaultTuningGenerators();

}  // namespace turbomind::gemm
