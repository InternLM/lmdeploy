#pragma once

#include <algorithm>

#include "src/turbomind/engine/request.h"

namespace turbomind {

class ContextTokenResource final: public Resource {
public:
    explicit ContextTokenResource(int max_context_tokens) noexcept: max_context_tokens_{max_context_tokens} {}

    int Test(const Sequence& s, const SubmittedRow& row) const noexcept override
    {
        const int q = row.query_count;
        if (q <= 0) {
            return 0;
        }
        return Charge(s, row) <= max_context_tokens_ ? q : 0;
    }

    void Commit(const Sequence& s, const SubmittedRow& row) noexcept override
    {
        max_context_tokens_ -= Charge(s, row);
    }

    int remaining_tokens() const noexcept
    {
        return max_context_tokens_;
    }

private:
    static int ContextLen(const Sequence& s) noexcept
    {
        return s.seq_len + s.inflight_new_tokens;
    }

    static int Charge(const Sequence& s, const SubmittedRow& row) noexcept
    {
        if (row.is_verification_row()) {
            return row.key_capacity_end;
        }

        const int context_len     = ContextLen(s);
        const int remaining_input = context_len - s.inflight_input_len - row.history_len;
        return (remaining_input > 1 || !s.is_active) ? context_len : 0;
    }

    int max_context_tokens_{};
};

}  // namespace turbomind
