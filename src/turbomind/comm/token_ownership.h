#pragma once

#include <algorithm>
#include <numeric>

namespace turbomind::comm {

class OwnedTokenRows {
public:
    constexpr OwnedTokenRows() = default;

    constexpr OwnedTokenRows(int global_offset, int local_begin, int local_end):
        global_offset_(global_offset), local_begin_(local_begin), local_end_(local_end)
    {
    }

    constexpr int global_offset() const noexcept
    {
        return global_offset_;
    }

    constexpr int local_begin() const noexcept
    {
        return local_begin_;
    }

    constexpr int local_end() const noexcept
    {
        return local_end_;
    }

    constexpr int row_count() const noexcept
    {
        return local_end_ - local_begin_;
    }

    constexpr int global_begin() const noexcept
    {
        return global_offset_ + local_begin_;
    }

    constexpr int global_end() const noexcept
    {
        return global_offset_ + local_end_;
    }

private:
    int global_offset_{};
    int local_begin_{};
    int local_end_{};
};

inline OwnedTokenRows ComputeTokenOwnership(int global_rank, int tp0, int tp1, const int* local_token_nums)
{
    const int inner_tp = std::min(tp0, tp1);

    const int dp_index = global_rank / inner_tp;
    const int tp_index = global_rank % inner_tp;
    const int num      = local_token_nums[dp_index];

    const int slice  = (num + inner_tp - 1) / inner_tp;
    const int first  = std::min(num, tp_index * slice);
    const int last   = std::min(num, first + slice);
    const int offset = std::accumulate(local_token_nums, local_token_nums + dp_index, 0);

    return {offset, first, last};
}

}  // namespace turbomind::comm
