// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/check.h"

namespace turbomind::core {

class BatchCopy {
public:
    ~BatchCopy();

    BatchCopy();

    BatchCopy(const BatchCopy&) = delete;
    BatchCopy& operator=(const BatchCopy&) = delete;
    BatchCopy(BatchCopy&&) noexcept        = delete;
    BatchCopy& operator=(BatchCopy&&) noexcept = delete;

    // clang-format off
    class Group {
    public:
        ~Group() { parent_.group_end(); }
        Group(BatchCopy& parent): parent_{parent} { parent_.group_begin(); }
        explicit constexpr operator bool() const noexcept { return true; }
    private:
        BatchCopy& parent_;
    };
    // clang-format on

    friend Group;

    Group group()
    {
        return {*this};
    }

    template<class T>
    T* operator()(const T* src, ssize_t size, T* dst)
    {
        // return core::Copy(src, size, dst);

        /// TODO: verify this is actually a fast path in a loop (without extra jump)
        if (TM_LIKELY(group_ && src == (const T*)src_ptr_ && dst == (T*)dst_ptr_)) {
            src_ptr_ += sizeof(T) * size;
            dst_ptr_ += sizeof(T) * size;
            gsize_ += sizeof(T) * size;
            count_ += 1;
            return dst + size;
        }
        else if (group_) {
            group_commit();
            gsize_   = sizeof(T) * size;
            src_ptr_ = reinterpret_cast<const char*>(src + size);
            dst_ptr_ = reinterpret_cast<char*>(dst + size);
            count_ += 1;
            return dst + size;
        }
        else {
            gsize_   = sizeof(T) * size;
            src_ptr_ = reinterpret_cast<const char*>(src + size);
            dst_ptr_ = reinterpret_cast<char*>(dst + size);
            count_   = 1;
            group_commit();
            return dst + size;
        }
    }

    void operator()(const Buffer& src, ssize_t size, Ref<Buffer> dst_)
    {
        auto& dst = dst_.get();
        TM_CHECK_EQ(src.dtype(), dst.dtype());
        TM_CHECK_LE(size, src.size());
        TM_CHECK_LE(size, dst.size());
        (*this)((const char*)src.raw_data(), byte_size(src.dtype(), size), (char*)dst.raw_data());
    }

    void Run();

    Buffer_<BatchCopy*> buf()
    {
        return {&self_, 1, kCPU};
    }

    friend std::ostream& operator<<(std::ostream& os, const BatchCopy& a)
    {
        os << "(" << a.count_ << ", " << a.src_.size() << ")";
        return os;
    }

private:
    void Reset()
    {
        src_.clear();
        dst_.clear();
        size_.clear();
        count_ = 0;
    }

    void group_begin()
    {
        TM_CHECK(!group_) << "Nested group is not supported";
        group_ = true;
    }

    void group_end()
    {
        TM_CHECK(group_) << "Mismatched group end";
        group_commit();
        group_ = false;
    }

    void group_commit()
    {
        if (gsize_) {
            src_.push_back(src_ptr_ - gsize_);
            dst_.push_back(dst_ptr_ - gsize_);
            size_.push_back(gsize_);
        }
        src_ptr_ = dst_ptr_ = {};
        gsize_              = {};
    }

private:
    std::vector<const char*> src_;
    std::vector<char*>       dst_;
    std::vector<size_t>      size_;

    int         group_   = 0;
    size_t      gsize_   = 0;
    const char* src_ptr_ = {};
    char*       dst_ptr_ = {};

    size_t count_;

    BatchCopy* self_;
};

}  // namespace turbomind::core
