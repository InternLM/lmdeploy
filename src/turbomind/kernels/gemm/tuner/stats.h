// Copyright (c) OpenMMLab. All rights reserved.

#include <limits>

namespace turbomind::gemm {

class Stats {
public:
    Stats(): count_{}, mean_{}, m2_{} {}

    float mean() const noexcept
    {
        return mean_;
    }

    float sum() const noexcept
    {
        return mean_ * count_;
    }

    int count() const noexcept
    {
        return count_;
    }

    float get_variance() const noexcept
    {
        return count_ < 2 ? std::numeric_limits<float>::quiet_NaN() : m2_ / count_;
    }

    void add_sample(float x) noexcept
    {
        ++count_;
        float delta = x - mean_;
        mean_ += delta / count_;
        float delta2 = x - mean_;
        m2_ += delta * delta2;
    }

private:
    int   count_;
    float mean_;
    float m2_;
};

}  // namespace turbomind::gemm
