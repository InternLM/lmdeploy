// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>

namespace turbomind {

using Signal = std::function<void()>;

class SignalBuffer {
public:
    void push(std::vector<Signal> signals)
    {
        if (signals.empty()) {
            return;
        }
        {
            std::lock_guard lock{mutex_};
            signals_.insert(signals_.end(), std::move_iterator{signals.begin()}, std::move_iterator{signals.end()});
        }
        cv_.notify_one();
    }

    void close()
    {
        {
            std::lock_guard lock{mutex_};
            aborted_ = true;
        }
        cv_.notify_all();
    }

    std::vector<Signal> take_all(bool& abort)
    {
        std::vector<Signal> signals;
        {
            std::unique_lock lock{mutex_};
            cv_.wait(lock, [&] { return !signals_.empty() || aborted_; });
            if (aborted_) {
                abort = true;
            }
            else {
                signals.swap(signals_);
            }
        }
        return signals;
    }

private:
    std::vector<Signal> signals_;

    std::mutex              mutex_;
    std::condition_variable cv_;

    bool aborted_{false};
};

}  // namespace turbomind
