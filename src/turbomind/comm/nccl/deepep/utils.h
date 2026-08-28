#pragma once

#include "src/turbomind/core/check.h"

#include <nccl.h>

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace turbomind {

inline bool IsFastRdmaAtomicSupport(const std::string& nic_name)
{
    std::ifstream file(std::filesystem::path{"/sys/class/infiniband"} / nic_name / "hca_type");
    std::string   hca_type;
    return static_cast<bool>(file >> hca_type) && hca_type == "MT4131";
}

inline std::optional<std::filesystem::path> FindExecutable(std::string_view name)
{
    if (const char* path = std::getenv("PATH"); path && *path) {
        std::string_view paths{path};
        while (true) {
            const auto pos        = paths.find(':');
            const auto dir        = paths.substr(0, pos);
            const auto executable = (dir.empty() ? std::filesystem::path{"."} : std::filesystem::path{dir}) / name;

            std::error_code ec;
            if (std::filesystem::is_regular_file(executable, ec) && access(executable.c_str(), X_OK) == 0) {
                auto resolved = std::filesystem::canonical(executable, ec);
                if (!ec) {
                    return resolved;
                }
            }

            if (pos == std::string_view::npos) {
                break;
            }
            paths.remove_prefix(pos + 1);
        }
    }
    return std::nullopt;
}

inline std::string FindCudaHome()
{
    if (const char* path = std::getenv("CUDA_HOME"); path && *path) {
        return std::string(path);
    }
    if (const char* path = std::getenv("CUDA_PATH"); path && *path) {
        return std::string(path);
    }
    if (auto nvcc = FindExecutable("nvcc")) {
        return nvcc->parent_path().parent_path().string();
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

}  // namespace turbomind
