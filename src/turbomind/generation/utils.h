
#include <functional>
#include <vector>

#include "src/turbomind/core/core.h"

namespace turbomind {

constexpr int kMaxStopBadWordsLen = 32;
constexpr int kMaxEndIdsSize      = 32;

namespace {

template<class G, class Rs, class T, class Copy>
void init_stop_bad_words(G getter, const char* key, const Rs& rs, T* h_buf, T* d_buf, Tensor_<T>& out, Copy& copy)
{
    const int bsz        = rs.size();
    int       max_length = 0;

    std::vector<std::pair<const int*, int>> copy_tokens(bsz);
    std::vector<std::pair<const int*, int>> copy_offsets(bsz);
    for (int i = 0; i < bsz; ++i) {
        const auto& [token_ids, offsets] = std::invoke(getter, rs[i]->gen_cfg);
        if (offsets.size() == 0 || token_ids.size() == 0) {
            continue;
        }
        FT_CHECK(offsets.back() == token_ids.size());
        if (offsets.back() <= kMaxStopBadWordsLen) {
            copy_tokens[i]  = std::make_pair(token_ids.data(), (int)token_ids.size());
            copy_offsets[i] = std::make_pair(offsets.data(), (int)offsets.size());
            max_length      = std::max(max_length, (int)token_ids.size());
        }
        else {
            auto trunc_offset_size =
                std::upper_bound(offsets.begin(),
                                 offsets.begin() + std::min(kMaxStopBadWordsLen, (int)offsets.size()),
                                 kMaxStopBadWordsLen)
                - offsets.begin();
            TM_LOG_WARNING("[InitializeSampling] [%ld] %s length (%d) exceeds %d, truncated to %d",
                           rs[i]->req->id,
                           key,
                           offsets.back(),
                           kMaxStopBadWordsLen,
                           trunc_offset_size);
            if (trunc_offset_size > 0) {
                int trunc_token_size = offsets[trunc_offset_size - 1];
                copy_tokens[i]       = std::make_pair(token_ids.data(), trunc_token_size);
                copy_offsets[i]      = std::make_pair(offsets.data(), trunc_offset_size);
                max_length           = std::max(max_length, trunc_token_size);
            }
        }
    }
    if (!max_length) {
        return;
    }
    std::fill_n(h_buf, bsz * 2 * max_length, -1);
    for (int i = 0; i < bsz; ++i) {
        if (copy_tokens[i].first != nullptr) {
            std::copy_n(copy_tokens[i].first, copy_tokens[i].second, h_buf + i * 2 * max_length);
        }
        if (copy_offsets[i].first != nullptr) {
            std::copy_n(copy_offsets[i].first, copy_offsets[i].second, h_buf + i * 2 * max_length + max_length);
        }
    }
    copy(h_buf, bsz * 2 * max_length, d_buf);
    // Construct a tensor from the device buffer
    out = {d_buf, {bsz, 2, max_length}, kDEVICE};
};

}  // namespace

}  // namespace turbomind
