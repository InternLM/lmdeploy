// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <climits>
#include <ostream>

namespace turbomind {

class Interval {
public:
    struct Size {
        int      x;
        explicit operator int() const noexcept
        {
            return x;
        }
        friend bool operator<(const Size& a, const Size& b)
        {
            return a.x < b.x;
        }
    };

    Interval(): first_{0}, last_{0} {}

    explicit Interval(int first): first_{first}, last_{INT_MAX} {};

    Interval(int first, int last): first_{first}, last_{last} {}

    Interval(int first, Size size): first_{first}, last_{first + (int)size} {}

    bool empty() const noexcept
    {
        return first_ >= last_;
    }

    explicit operator bool() const noexcept
    {
        return !empty();
    }

    Size size() const noexcept
    {
        return Size{std::max(0, last_ - first_)};
    }

    int begin() const noexcept
    {
        return first_;
    }

    int end() const noexcept
    {
        return last_;
    }

    friend Interval operator&(const Interval& a, const Interval& b)
    {
        return {std::max(a.first_, b.first_), std::min(a.last_, b.last_)};
    }

    friend Interval operator|(const Interval& a, const Interval& b)
    {
        return {std::min(a.first_, b.first_), std::max(a.last_, b.last_)};
    }

    // dilate / erode left
    friend Interval operator|(int x, const Interval& a)
    {
        return {a.begin() - x, a.end()};
    }

    // dilate / erode right
    friend Interval operator|(const Interval& a, int x)
    {
        return {a.begin(), a.end() + x};
    }

    friend std::ostream& operator<<(std::ostream& os, const Interval& a)
    {
        return os << "[" << a.first_ << ", " << a.last_ << ")";
    }

    friend std::ostream& operator<<(std::ostream& os, const Interval* a)
    {
        return os << *a;
    }

private:
    int first_;
    int last_;
};

}  // namespace turbomind
