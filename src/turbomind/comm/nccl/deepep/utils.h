#pragma once

#include "src/turbomind/core/check.h"

#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <regex>
#include <string>

namespace turbomind {

inline std::string RunCommand(const std::string& command)
{
    std::string output;
    if (auto* fp = popen(command.c_str(), "r"); fp) {
        constexpr int buffer_size = 4096;
        char          buffer[buffer_size];
        while (std::fgets(buffer, buffer_size, fp) != nullptr) {
            output += buffer;
        }
        pclose(fp);
    }
    return output;
}

inline bool IsFastRdmaAtomicSupport(const std::string& nic_name)
{
    std::string      output = RunCommand("ibstat 2>/dev/null");
    std::smatch      match;
    const std::regex pattern("CA '" + nic_name + R"('([\s\S]*?)CA type:\s*(\S+))");
    return std::regex_search(output, match, pattern) && match[2] == "MT4131";
}

inline std::string FindCudaHome()
{
    if (const char* path = std::getenv("CUDA_HOME"); path && *path) {
        return std::string(path);
    }
    if (const char* path = std::getenv("CUDA_PATH"); path && *path) {
        return std::string(path);
    }
    if (auto nvcc = RunCommand("command -v nvcc 2>/dev/null"); !nvcc.empty()) {
        nvcc.erase(nvcc.find_last_not_of("\r\n") + 1);
        return std::filesystem::path(nvcc).parent_path().parent_path().string();
    }

    constexpr const char* default_path = "/usr/local/cuda";
    std::string           cuda_home    = std::filesystem::exists(default_path) ? default_path : "";
    TM_CHECK(!cuda_home.empty()) << "CUDA_HOME not found. Please set the environment variable CUDA_HOME or CUDA_PATH";
    return cuda_home;
}

inline std::string FindNcclRoot()
{
    std::string nccl_root{};
    Dl_info     info{};
    if (dladdr(reinterpret_cast<void*>(&ncclGetVersion), &info) != 0 && info.dli_fname) {
        nccl_root = std::filesystem::absolute(info.dli_fname).parent_path().parent_path();
    }
    TM_CHECK(!nccl_root.empty()) << "nccl not found. Please install nccl with pip";
    return nccl_root;
}

inline std::string FindLibRoot()
{
    std::vector<std::filesystem::path> candidates{};
#ifdef TM_DEEPEP_BUILD_ROOT
    candidates.emplace_back(TM_DEEPEP_BUILD_ROOT);
#endif
    std::string libroot{};
    Dl_info     info{};
    if (dladdr(reinterpret_cast<void*>(&FindLibRoot), &info) != 0 && info.dli_fname) {
        candidates.push_back(std::filesystem::absolute(info.dli_fname).parent_path());
    }
    for (const auto& root : candidates) {
        if (std::filesystem::exists(root / "include/deep_ep/common/compiled.cuh")) {
            libroot = std::filesystem::weakly_canonical(root);
            break;
        }
    }
    TM_CHECK(!libroot.empty()) << "cannot find include files";
    return libroot;
}

inline float GetRdmaGbs(std::string nic_name)
{
    auto output = RunCommand("ibstat 2>/dev/null");

    std::smatch      match;
    const std::regex pattern("CA '" + nic_name + R"('([\s\S]*?)Port \d+:([\s\S]*?)Rate:\s*(\d+))");

    if (std::regex_search(output, match, pattern)) {
        return std::stof(match[3]) / 8.0f;
    }

    return 0.0f;
}

inline float GetNvlinkGbs(float factor = 0.9)
{
    const auto output = RunCommand("nvidia-smi nvlink -s 2>/dev/null");
    const auto begin  = output.find("GPU ");
    const auto end    = output.find("\nGPU ", begin + 1);

    if (begin == std::string::npos)
        return 0.0f;

    const auto       gpu_block = output.substr(begin, end - begin);
    const std::regex pattern(R"(Link \d+:\s*([\d.]+) GB/s)");

    float total = 0.0f;
    bool  found = false;
    for (std::sregex_iterator it(gpu_block.begin(), gpu_block.end(), pattern), last; it != last; ++it) {
        total += std::stof((*it)[1]);
        found = true;
    }

    return found ? total * factor : 0.0f;
}

}  // namespace turbomind
