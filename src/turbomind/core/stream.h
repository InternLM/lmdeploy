#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/common.h"

namespace turbomind::core {

class StreamImpl {
public:
    StreamImpl(int priority): stream_{}
    {
        check_cuda_error(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority));
    }

    ~StreamImpl()
    {
        if (auto ec = cudaStreamDestroy(stream_); ec != cudaSuccess) {
            TM_LOG_ERROR(cudaGetErrorString(ec));
        }
        stream_ = {};
    }

    void Sync()
    {
        check_cuda_error(cudaStreamSynchronize(stream_));
    }

    void Wait(const Event& event);

    cudaStream_t handle() const
    {
        return stream_;
    }

public:
    cudaStream_t stream_;
};

class Stream {
public:
    Stream() = default;

    static Stream create(int priority = 0);

    void Sync()
    {
        impl_->Sync();
    }

    void Wait(const Event& event)
    {
        impl_->Wait(event);
    }

    cudaStream_t handle() const
    {
        return TM_CHECK_NOTNULL(impl_)->handle();
    }

    explicit operator cudaStream_t() const
    {
        return handle();
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    friend bool operator==(const Stream& a, const Stream& b)
    {
        return a.impl_ == b.impl_;
    }

    friend bool operator!=(const Stream& a, const Stream& b)
    {
        return !(a == b);
    }

    friend std::ostream& operator<<(std::ostream& os, const Stream& s)
    {
        os << s.impl_;
        return os;
    }

private:
    shared_ptr<StreamImpl> impl_;
};

class EventImpl {
public:
    explicit EventImpl(unsigned flags)
    {
        check_cuda_error(cudaEventCreateWithFlags(&event_, flags));
    }

    ~EventImpl()
    {
        if (auto ec = cudaEventDestroy(event_); ec != cudaSuccess) {
            TM_LOG_ERROR(cudaGetErrorString(ec));
        }
    }

    void Record(const Stream& stream)
    {
        check_cuda_error(cudaEventRecord(event_, stream.handle()));
    }

    void Sync() const
    {
        check_cuda_error(cudaEventSynchronize(event_));
    }

    cudaEvent_t handle() const
    {
        return event_;
    }

private:
    cudaEvent_t event_;
};

class Event {
public:
    Event() = default;

    static Event create(bool timing = false)
    {
        Event e{};
        e.impl_ = std::make_shared<EventImpl>(timing ? 0 : cudaEventDisableTiming);
        return e;
    }

    void Record(const Stream& stream)
    {
        TM_CHECK_NOTNULL(impl_)->Record(stream);
    }

    void Sync() const
    {
        TM_CHECK_NOTNULL(impl_)->Sync();
    }

    operator cudaEvent_t() const
    {
        return TM_CHECK_NOTNULL(impl_)->handle();
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

private:
    shared_ptr<EventImpl> impl_;
};

}  // namespace turbomind::core
