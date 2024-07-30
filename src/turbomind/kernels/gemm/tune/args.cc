// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/tune/args.h"
#include "src/turbomind/utils/parser.h"
#include <algorithm>
#include <iostream>
#include <regex>

namespace turbomind::gemm {

void ParseTuningArgs(TuningArgs& args, const std::string& str)
{
    const auto list = ParseArgsList(str);

    auto try_parse = [&](auto& value, auto name) {
        auto it = std::find_if(list.begin(), list.end(), [&](auto a) { return a.first == name; });
        if (it != list.end()) {
            std::cout << name << " " << it->second << "\n";
            Parse(value, it->second);
        }
    };

    try_parse(args.max_splits, "max_splits");
    try_parse(args.top_splits, "top_splits");
    try_parse(args.max_waves, "max_waves");
    try_parse(args.swizzle, "swizzle");
    try_parse(args.clusters, "clusters");
    try_parse(args.max_iter, "max_iter");
    try_parse(args.max_time, "max_time");
}

std::vector<int> ParseTuningSequence(const std::string& str)
{
    const std::regex triplet(R"((\d+)-(\d+)-(\d+))");
    const std::regex delim(",");

    std::vector<int> ret;

    std::vector<std::array<int, 3>> generators;

    std::sregex_token_iterator begin(str.begin(), str.end(), delim, -1);
    std::for_each(begin, std::sregex_token_iterator{}, [&](const std::string& token) {
        std::sregex_iterator beg(token.begin(), token.end(), triplet);
        std::sregex_iterator end{};
        bool                 is_triplet{};
        std::for_each(beg, end, [&](std::smatch match) {
            if (match.size() == 4) {
                is_triplet = true;
                generators.push_back({std::stoi(match[1].str()),  //
                                      std::stoi(match[2].str()),
                                      std::stoi(match[3].str())});
            }
        });
        if (!is_triplet) {
            generators.push_back({std::stoi(token) + 1, 0, 0});
        }
    });

    if (generators.size() == 1) {
        auto fallback   = GetDefaultTuningGenerators();
        fallback.back() = generators.front();
        generators      = std::move(fallback);
    }

    return GenerateTuningSequence(generators);
}

std::vector<int> GenerateTuningSequence(const std::vector<std::array<int, 3>>& generators)
{
    std::vector<int> ret;
    if (generators.size() < 2) {
        return ret;
    }
    // The last generator is a sentinel `(max_bs, 0, 0)`
    for (int i = 0; i < (int)generators.size() - 1; ++i) {
        const int end           = generators[i + 1][0];
        auto [curr, step, next] = generators[i];
        while (curr < end) {
            ret.push_back(curr);
            if (curr == next) {
                step *= 2;
                next *= 2;
            }
            curr += step;
        }
    }
    return ret;
}

std::vector<std::array<int, 3>> GetDefaultTuningGenerators()
{
    /// TODO: set generators based device
    return {{16, 16, 64}, {8192 + 1}};
}

}  // namespace turbomind::gemm