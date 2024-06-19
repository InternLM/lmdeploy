
// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>

namespace turbomind {

class AnomalyHandler {
public:
    static constexpr size_t max_entries = 65536;

    using size_type = unsigned long long;

    ~AnomalyHandler();

    static AnomalyHandler& instance();

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

}  // namespace turbomind
