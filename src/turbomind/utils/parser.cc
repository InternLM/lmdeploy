// Copyright (c) OpenMMLab. All rights reserved.

#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace turbomind {

std::vector<std::pair<std::string, std::string>> ParseArgsList(const std::string& str)
{
    const std::regex regex(R"((\w+)=([^,\[\(]+|\[.*\]|\(.*\)))");

    std::sregex_iterator beg(str.begin(), str.end(), regex);
    std::sregex_iterator end{};

    std::vector<std::pair<std::string, std::string>> ret;
    for (auto it = beg; it != end; ++it) {
        std::smatch match = *it;
        ret.emplace_back(match[1], match[2]);
    }

    return ret;
}

std::vector<std::string> ParseListOrTuple(const std::string& str)
{
    const std::regex regex(R"([,\[\]\(\)]+)");

    std::vector<std::string> ret;
    std::copy_if(std::sregex_token_iterator(str.begin(), str.end(), regex, -1),
                 std::sregex_token_iterator{},
                 std::back_inserter(ret),
                 [](const std::string& s) { return !s.empty(); });

    return ret;
}

}  // namespace turbomind
