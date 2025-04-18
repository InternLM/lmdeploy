
// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>

#include "src/turbomind/core/core.h"

namespace turbomind {

class AnomalyHandler {
public:
    static constexpr size_t max_entries = 65536;

    using size_type = unsigned long long;

    ~AnomalyHandler();

    static AnomalyHandler& instance();

    static int level() noexcept;

    void Init(int rank, int vocab_size, int fallback, int max_batch_size, cudaStream_t stream) noexcept;

    template<class T>
    void CountAndFix(T* data, int64_t size, std::string key, int level);

    template<class T>
    void FixLogits(T* logits, int batch_size, int level);

    void Summarize(std::function<void(const int*, int)> handler);

    void Reset();

private:
    AnomalyHandler();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

template<class T>
void count_and_fix(T* data, size_t size, std::string key, int level)
{
    AnomalyHandler::instance().CountAndFix(data, size, key, level);
}

void DebugTensor(Tensor& tensor, const std::string& key, int level);

inline void DebugTensor(Tensor&& tensor, const std::string& key, int level)
{
    DebugTensor(tensor, key, level);
}

#define TM_DEBUG_RAW(ptr, size, key, __level)                                                                          \
    if (::turbomind::AnomalyHandler::level() >= __level) {                                                             \
        ::turbomind::count_and_fix(ptr, size, key, __level);                                                           \
    }

#define TM_DEBUG_TENSOR(tensor, key, __level)                                                                          \
    if (::turbomind::AnomalyHandler::level() >= __level) {                                                             \
        ::turbomind::DebugTensor(tensor, key, __level);                                                                \
    }

}  // namespace turbomind
