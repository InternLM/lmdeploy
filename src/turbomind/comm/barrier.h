// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

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
