// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#ifndef _MSC_VER
#include <pthread.h>
#endif

namespace turbomind {

#ifdef _MSC_VER

class Barrier {
public:
    Barrier(unsigned count)
    {
        TM_LOG_INFO("Barrier(%d)", (int)count);
        FT_CHECK(count == 1);
    }

    Barrier(const Barrier&) = delete;
    Barrier& operator=(const Barrier&) = delete;
    Barrier(Barrier&&) noexcept        = delete;
    Barrier& operator=(Barrier&&) noexcept = delete;

    void wait() {}

    ~Barrier() {}
};

#else

class Barrier {
public:
    Barrier(unsigned count): count_(count)
    {
        if (count_ > 1) {
            pthread_barrier_init(&barrier_, nullptr, count);
        }
    }

    Barrier(const Barrier&) = delete;
    Barrier& operator=(const Barrier&) = delete;
    Barrier(Barrier&&) noexcept        = delete;
    Barrier& operator=(Barrier&&) noexcept = delete;

    void wait()
    {
        if (count_ > 1) {
            pthread_barrier_wait(&barrier_);
        }
    }

    ~Barrier()
    {
        if (count_ > 1) {
            pthread_barrier_destroy(&barrier_);
        }
    }

private:
    const int         count_;
    pthread_barrier_t barrier_{};
};

#endif

}  // namespace turbomind
