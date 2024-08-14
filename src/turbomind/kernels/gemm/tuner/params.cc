// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/tuner/params.h"
#include "src/turbomind/utils/parser.h"
#include <algorithm>
#include <iostream>
#include <regex>

namespace turbomind::gemm {

void ParseTuningParams(TuningParams& params, const std::string& str)
{
    const auto list = ParseArgsList(str);

    auto try_parse = [&](auto& value, auto name) {
        auto it = std::find_if(list.begin(), list.end(), [&](auto a) { return a.first == name; });
        if (it != list.end()) {
            std::cout << name << " " << it->second << "\n";
            Parse(value, it->second);
        }
    };

    try_parse(params.max_splits, "max_splits");
    try_parse(params.max_waves, "max_waves");
    try_parse(params.swizzle, "swizzle");
    try_parse(params.top_k, "top_k");
    try_parse(params.clusters, "clusters");
    try_parse(params.min_iter, "min_iter");
    try_parse(params.max_iter, "max_iter");
    try_parse(params.max_time, "max_time");

    if (auto it = std::find_if(list.begin(), list.end(), [&](auto a) { return a.first == "seq"; }); it != list.end()) {
        params.seq = ParseTuningSequence(it->second);
    }
}

std::vector<int> ParseTuningSequence(const std::string& str)
{
    const std::regex triplet(R"((\d+)-(\d+)-(\d+))");

    std::vector<std::array<int, 3>> generators;

    const auto tokens = ParseListOrTuple(str);

    for (const auto& token : tokens) {
        std::smatch match;
        if (std::regex_match(token, match, triplet)) {
            generators.push_back({std::stoi(match[1].str()),  //
                                  std::stoi(match[2].str()),
                                  std::stoi(match[3].str())});
        }
        else {  // must be an integer string
            generators.push_back({std::stoi(token), 0, 0});
        }
    }

    if (generators.size() == 1) {  // Replace sentinel of the default generators
        auto fallback   = GetDefaultTuningGenerators();
        fallback.back() = {generators.front().front(), 0, 0};
        generators      = std::move(fallback);
    }

    return GenerateTuningSequence(generators);
}

std::vector<int> GenerateTuningSequence(const std::vector<std::array<int, 3>>& generators)
{
    std::vector<int> ret;
    if (generators.empty()) {
        return ret;
    }
    const int last = generators.back().front();
    // The last generator is a sentinel `(max_bs, 0, 0)`
    for (int i = 0; i < (int)generators.size() - 1; ++i) {
        auto [curr, next, step] = generators[i];
        if (curr >= last) {
            break;
        }
        if (next == 0 && step == 0) {  // single value
            ret.push_back(curr);
        }
        else {  // generator
            const int end = std::min(generators[i + 1][0], last);
            while (curr < end) {
                ret.push_back(curr);
                if (curr == next) {
                    step *= 2;
                    next *= 2;
                }
                curr += step;
            }
        }
    }
    ret.push_back(last);
    return ret;
}

std::vector<std::array<int, 3>> GetDefaultTuningGenerators()
{
    /// TODO: set generators based on device
    return {{8, 16, 8}, {16, 64, 16}, {8192}};
}

}  // namespace turbomind::gemm
