
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
    auto strides(Is... is)
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

    Layout permute(const vector<int>& dims)
    {
        TM_CHECK((int)dims.size() == rank());
        auto a = *this;
        for (int i = 0; i < rank(); ++i) {
            a.shape_[i]  = shape_[dims[i]];
            a.stride_[i] = stride_[dims[i]];
        }
        return a;
    }

    Layout coalesce() const noexcept;

    Layout view(vector<ssize_t> shape) const;

    std::pair<Layout, ssize_t> slice(const vector<ssize_t>& base, vector<ssize_t> shape) const;

    friend std::ostream& operator<<(std::ostream& os, const Layout& x);

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

}  // namespace turbomind::core