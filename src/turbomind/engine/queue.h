// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace turbomind {

template<class T>
class Queue {
public:
    template<class X>
    void push(X&& x)
    {
        {
            std::lock_guard lock{mutex_};
            queue_.push(std::forward<X>(x));
        }
        cv_.notify_one();
    }

    bool pop(T& x)
    {
        std::unique_lock lock{mutex_};
        cv_.wait(lock, [&] { return !queue_.empty() || is_closed_; });
        if (is_closed_) {
            return false;
        }
        x = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void close()
    {
        {
            std::lock_guard lock{mutex_};
            is_closed_ = true;
        }
        cv_.notify_all();
    }

private:
    std::queue<T>           queue_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    is_closed_{false};
};

}  // namespace turbomind
