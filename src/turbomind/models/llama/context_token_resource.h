#pragma once

#include <algorithm>

#include "src/turbomind/engine/request.h"

namespace turbomind {

class ContextTokenResource final: public Resource {
public:
    explicit ContextTokenResource(int max_context_tokens) noexcept: max_context_tokens_{max_context_tokens} {}

    int Test(const Sequence& s) const noexcept override
    {
        const int input_len = InputLen(s, s.resume_len);
        if (input_len <= 0) {
            return 0;
        }
        if (TempLen(s, input_len) > max_context_tokens_) {
            return 0;
        }
        return input_len;
    }

    void Commit(const Sequence& s) noexcept override
    {
        const int input_len = InputLen(s, s.history_len);
        max_context_tokens_ -= TempLen(s, input_len);
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

    static int InputLen(const Sequence& s, int history_len) noexcept
    {
        return ContextLen(s) - s.inflight_input_len - history_len;
    }

    static int TempLen(const Sequence& s, int input_len) noexcept
    {
        return (input_len > 1 || !s.is_active) ? ContextLen(s) : 0;
    }

    int max_context_tokens_{};
};

}  // namespace turbomind
