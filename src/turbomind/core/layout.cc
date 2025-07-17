
#include <numeric>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/layout.h"

namespace turbomind::core {

Layout::Layout(std::vector<ssize_t> shape): shape_{std::move(shape)}
{
    TM_CHECK(shape_.size());
    stride_.resize(shape_.size());
    size_ = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        stride_[i] = size_;
        size_ *= shape_[i];
    }
}

Layout::Layout(vector<ssize_t> shape, vector<ssize_t> stride): shape_{std::move(shape)}, stride_{std::move(stride)}
{
    TM_CHECK(shape_.size());
    TM_CHECK_EQ(shape_.size(), stride_.size());

    size_ = std::accumulate(shape_.begin(), shape_.end(), ssize_t{1}, std::multiplies<>{});

    TM_CHECK_GE(size_, 0);
}

ssize_t Layout::cosize() const noexcept
{
    if (rank() == 0) {
        return 0;
    }
    ssize_t value{1};
    for (size_t i = 0; i < shape_.size(); ++i) {
        value += (shape_[i] - 1) * stride_[i];
    }
    return value;
}

Layout Layout::coalesce() const noexcept
{
    vector<ssize_t> shape{shape_.front()};
    vector<ssize_t> stride{stride_.front()};

    for (size_t i = 1; i < shape_.size(); ++i) {
        if (shape_[i] == 1) {
            continue;
        }
        else if (shape.back() == 1) {
            shape.back()  = shape_[i];
            stride.back() = stride_[i];
        }
        else if (stride.back() == shape_[i] * stride_[i]) {
            stride.back() = stride_[i];
            shape.back() *= shape_[i];
        }
        else {
            shape.push_back(shape_[i]);
            stride.push_back(stride_[i]);
        }
    }

    return Layout{shape, stride};
}

Layout Layout::view(vector<ssize_t> shape) const
{
    if (shape == shape_) {
        return *this;
    }

    TM_CHECK(!shape.empty());

    // size check & wildcard resolution
    auto wildcard = std::find(shape.begin(), shape.end(), -1);
    if (wildcard != shape.end()) {
        TM_CHECK(std::find(wildcard + 1, shape.end(), -1) == shape.end());
        *wildcard = 1;
    }
    auto new_size = std::accumulate(shape.begin(), shape.end(), ssize_t{1}, std::multiplies<>{});
    if (wildcard != shape.end()) {
        TM_CHECK(size_ % new_size == 0) << size_ << " % " << new_size;
        *wildcard = size_ / new_size;
    }
    else {
        TM_CHECK_EQ(size_, new_size);
    }

    if (is_contiguous()) {
        return Layout{shape};
    }

    const Layout c = coalesce();  // merge contiguous dimensions

    ssize_t p = c.rank();
    ssize_t s = 1;
    ssize_t d = 0;

    vector<ssize_t> stride(shape.size());

    for (int i = shape.size() - 1; i >= 0; --i) {
        if (shape[i] == 1) {
            stride[i] = 0;
        }
        else {
            if (s == 1) {
                --p;
                s = c.shape().at(p);
                d = c.stride().at(p);
            }
            TM_CHECK_EQ(s % shape[i], 0);  // crossing non-contiguous dimensions
            stride[i] = d;
            d *= shape[i];
            s /= shape[i];
        }
    }
    return Layout{std::move(shape), std::move(stride)};
}

std::pair<Layout, ssize_t> Layout::slice(const vector<ssize_t>& base, vector<ssize_t> shape) const
{
    TM_CHECK_EQ(base.size(), shape.size());
    TM_CHECK_EQ(shape_.size(), shape.size());
    ssize_t offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        const auto space = shape_[i] - base[i];
        TM_CHECK_GE(space, 0);
        if (shape[i] == -1) {
            shape[i] = space;
        }
        TM_CHECK_LE(shape[i], space);
        offset += base[i] * stride_[i];
    }
    return {Layout{std::move(shape), stride_}, offset};
}

std::ostream& operator<<(std::ostream& os, const Layout& x)
{
    os << "(";
    for (int i = 0; i < x.rank(); ++i) {
        os << (i ? "," : "") << x.shape_[i];
    }
    os << "):(";
    for (int i = 0; i < x.rank(); ++i) {
        os << (i ? "," : "") << x.stride_[i];
    }
    os << ")";
    return os;
}

}  // namespace turbomind::core
