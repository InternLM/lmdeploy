// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#if defined(_MSC_VER) && !defined(__clang__)

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace turbomind::comm {

class Barrier {
public:
    explicit Barrier(int count): threshold_{count}, count_{count} {}

    void arrive_and_wait()
    {
        std::unique_lock lock{mutex_};
        auto             phase = phase_;
        if (--count_ == 0) {
            ++phase_;
            count_ = threshold_;
            cv_.notify_all();
        }
        else {
            cv_.wait(lock, [this, phase] { return phase_ != phase; });
        }
    }

private:
    std::mutex              mutex_;
    std::condition_variable cv_;

    int threshold_;
    int count_;

    uint32_t phase_{};
};

}  // namespace turbomind::comm

#else

#include <pthread.h>

namespace turbomind::comm {

class Barrier {
public:
    explicit Barrier(int count): barrier_{}
    {
        pthread_barrier_init(&barrier_, {}, count);
    }

    ~Barrier()
    {
        pthread_barrier_destroy(&barrier_);
    }

    void arrive_and_wait()
    {
        pthread_barrier_wait(&barrier_);
    }

private:
    pthread_barrier_t barrier_;
};

}  // namespace turbomind::comm

#endif
