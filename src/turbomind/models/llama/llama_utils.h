// Copyright (c) OpenMMLab. All rights reserved.

#pragma once
#include "src/turbomind/models/llama/Barrier.h"
#include "src/turbomind/utils/Tensor.h"
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <vector>

namespace turbomind {

enum QuantPolicy
{
    kNone = 0x00,
    // reserve 0x01 and 0x02 for backward compatibility
    kReserve1 = 0x01,
    kReserve2 = 0x02,
    // quantize cache kv
    kCacheKVInt8 = 0x04,
};

enum CmpMode
{
    kCmpNone,
    kCmpRead,
    kCmpWrite,
};

extern CmpMode compare_mode;

template<typename T>
void Compare(T* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream, std::string msg = {});

template<typename T>
void CheckNan(const T* ptr, size_t size, std::string key, cudaStream_t stream);

namespace detail {

template<typename T>
std::string to_string(T x)
{
    return std::to_string(x);
}

inline std::string to_string(std::string x)
{
    return x;
}

}  // namespace detail

template<typename... Args>
std::string Concat(std::string key, Args&&... args)
{
    std::vector<std::string> args_str{detail::to_string((Args &&) args)...};
    for (const auto& s : args_str) {
        key.append("_");
        key.append(s);
    }
    return key;
}

std::string format(const std::pair<std::string, Tensor>& p);

size_t curandStateGetSize();

bool isDebug();

template<typename T>
void CheckValues(const T* data, int count, const std::string& msg, cudaStream_t stream);

Barrier*& model_instance_barrier();

template<typename T>
inline void CheckBatchConsistency(T* ptr, size_t size, int batch_size, std::string key, int rank, cudaStream_t stream)
{
    if (compare_mode == kCmpNone) {
        return;
    }
    model_instance_barrier()->wait();
    if (compare_mode == kCmpWrite) {
        if (rank == 0) {
            Compare(ptr, size, key, compare_mode, stream);
        }
    }
    else {
        if (rank == 0) {
            for (int i = 0; i < batch_size; ++i) {
                Compare(ptr + i * size, size, key, compare_mode, stream, Concat("", rank, i));
            }
        }
    }
    model_instance_barrier()->wait();
}

}  // namespace turbomind
