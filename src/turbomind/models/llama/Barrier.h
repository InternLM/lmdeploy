// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/utils/logger.h"
#include <pthread.h>

namespace turbomind {

class Barrier {
public:
    Barrier(unsigned count)
    {
        TM_LOG_INFO("Barrier(%d)", (int)count);
        pthread_barrier_init(&barrier_, nullptr, count);
    }

    Barrier(const Barrier&) = delete;
    Barrier& operator=(const Barrier&) = delete;
    Barrier(Barrier&&) noexcept        = delete;
    Barrier& operator=(Barrier&&) noexcept = delete;

    void wait()
    {
        pthread_barrier_wait(&barrier_);
    }

    ~Barrier()
    {
        pthread_barrier_destroy(&barrier_);
    }

private:
    pthread_barrier_t barrier_{};
};

}  // namespace turbomind
