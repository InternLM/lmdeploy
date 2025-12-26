
#pragma once

#include <initializer_list>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/common.h"

namespace turbomind::core {

class Layout {
public:
    Layout(): size_{0} {}

    /* implicit */ Layout(vector<ssize_t> shape);

    /* implicit */ Layout(std::initializer_list<ssize_t> shape): Layout(vector(shape)) {}

    Layout(vector<ssize_t> shape, vector<ssize_t> stride);

    ssize_t size() const noexcept
    {
        return size_;
    }

    ssize_t cosize() const noexcept;

    ssize_t rank() const noexcept
    {
        return shape_.size();
    }

    auto& shape() const noexcept
    {
        return shape_;
    }

    auto shape(int i) const
    {
        return shape_.at(wrap(i));
    }

    template<class... Is>
    auto shapes(Is... is) const
    {
        return std::make_tuple(shape(is)...);
    }

    auto& stride() const noexcept
    {
        return stride_;
    }

    auto stride(int i) const
    {
        return stride_.at(wrap(i));
    }

    template<class... Is>
    auto strides(Is... is) const
    {
        return std::make_tuple(stride(is)...);
    }

    bool is_contiguous() const noexcept
    {
        if (stride_.back() != 1) {
            return false;
        }
        if (size() != cosize()) {
            return false;
        }
        for (int i = 0; i < rank() - 1; ++i) {
            // TODO: skip when shape == 1
            if (stride_[i] < stride_[i + 1]) {
                return false;
            }
        }
        return true;
    }

    Layout permute(const vector<int>& dims) const
    {
        TM_CHECK((int)dims.size() == rank());
        auto a = *this;
        for (int i = 0; i < rank(); ++i) {
            a.shape_[i]  = shape_[dims[i]];
            a.stride_[i] = stride_[dims[i]];
        }
        return a;
    }

    Layout transpose(int a, int b) const
    {
        TM_CHECK_LT(a, rank());
        TM_CHECK_LT(b, rank());
        auto x = *this;
        std::swap(x.shape_[a], x.shape_[b]);
        std::swap(x.stride_[a], x.stride_[b]);
        return x;
    }

    ssize_t offset(const vector<ssize_t>& idxs) const
    {
        TM_CHECK((int)idxs.size() < rank());
        ssize_t val = 0;
        for (size_t i = 0; i < idxs.size(); ++i) {
            TM_CHECK_LT(idxs[i], shape_[i]);
            val += idxs[i] * stride_[i];
        }
        return val;
    }

    ssize_t offset(ssize_t idx0) const
    {
        TM_CHECK(rank());
        TM_CHECK_LT(idx0, shape_[0]);
        return stride_[0] * idx0;
    }

    Layout coalesce() const noexcept;

    Layout view(vector<ssize_t> shape) const;

    std::pair<Layout, ssize_t> slice(const vector<ssize_t>& base, vector<ssize_t> shape) const;

    Layout squeeze(int dim) const
    {
        if (rank() == 1 || shape(dim) != 1) {
            return *this;
        }
        Layout a;
        a.shape_.reserve(rank() - 1);
        a.stride_.reserve(rank() - 1);
        for (int i = 0; i < rank(); ++i) {
            if (i != dim) {
                a.shape_.push_back(shape_[i]);
                a.stride_.push_back(stride_[i]);
            }
        }
        a.size_ = size_;
        return a;
    }

    friend std::ostream& operator<<(std::ostream& os, const Layout& x);

    friend bool operator==(const Layout& a, const Layout& b)
    {
        return a.shape_ == b.shape_ && a.stride_ == b.stride_;
    }

    friend bool operator!=(const Layout& a, const Layout& b)
    {
        return !(a == b);
    }

private:
    int wrap(int dim) const noexcept
    {
        return dim < 0 ? dim + shape_.size() : dim;
    }

private:
    vector<ssize_t> shape_;
    vector<ssize_t> stride_;
    ssize_t         size_;
};

inline std::string to_string(const Layout& x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

// clang-format off
template<class Archive>
void save(Archive& ar, const Layout& layout)
{
    ar & layout.shape();
    ar & layout.stride();
}

template<class Archive>
void load(Archive& ar, Layout& layout)
{
    vector<ssize_t> shape;
    vector<ssize_t> stride;
    ar & shape;
    ar & stride;
    layout = Layout(std::move(shape), std::move(stride));
}
// clang-format on

}  // namespace turbomind::core
