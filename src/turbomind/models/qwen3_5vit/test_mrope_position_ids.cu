// Copyright (c) OpenMMLab. All rights reserved.
//
// Standalone test for invokeMropePositionIds. Builds segment descriptors with the same
// logic the production Qwen3_5Vit::Impl::Setup() uses, runs the device kernel, and
// compares against a CPU reference that replicates the pre-refactor scalar loop.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/models/qwen3_5vit/mrope_position_ids.h"

using namespace turbomind;

namespace {

struct ImageSpec {
    int seq_start;
    int t;
    int h;  // pre-merge (kernel expects h/S)
    int w;
};

struct RequestSpec {
    int                    seq_len;
    int                    active_start;  // history_len + alpha
    int                    active_end;    // active_start + input_len
    bool                   autoregres;
    std::vector<ImageSpec> images;  // assumed in seq-order, non-overlapping
};

// Pre-refactor scalar reference: builds the full (seq_len, 3) row for each request,
// then we read out the [active_start, active_end) slice to compare against kernel output.
// Mirrors qwen3_5vit.cc:440-516 (pre-refactor) exactly.
std::vector<int> cpu_reference_full_row(const RequestSpec& r, int S, int& out_delta)
{
    std::vector<int> row(r.seq_len * 3, 0);
    int              write_ptr = 0;
    int              next_pos  = 0;
    int              mm_offset = 0;

    auto emit_text = [&](int upto) {
        while (next_pos < upto && write_ptr < r.seq_len) {
            row[write_ptr * 3 + 0] = next_pos;
            row[write_ptr * 3 + 1] = next_pos;
            row[write_ptr * 3 + 2] = next_pos;
            ++next_pos;
            ++write_ptr;
        }
    };

    for (const auto& img : r.images) {
        const int h2    = img.h / S;
        const int w2    = img.w / S;
        const int n_tok = img.t * h2 * w2;

        const int mm_start = img.seq_start + mm_offset;
        emit_text(mm_start);

        for (int k = 0; k < n_tok && write_ptr < r.seq_len; ++k) {
            const int t_idx        = k / (h2 * w2);
            const int h_idx        = (k / w2) % h2;
            const int w_idx        = k % w2;
            row[write_ptr * 3 + 0] = t_idx + mm_start;
            row[write_ptr * 3 + 1] = h_idx + mm_start;
            row[write_ptr * 3 + 2] = w_idx + mm_start;
            ++write_ptr;
        }

        const int new_pos = std::max({img.t, h2, w2});
        next_pos          = mm_start + new_pos;
        mm_offset += new_pos - n_tok;
    }
    while (write_ptr < r.seq_len) {
        row[write_ptr * 3 + 0] = next_pos;
        row[write_ptr * 3 + 1] = next_pos;
        row[write_ptr * 3 + 2] = next_pos;
        ++next_pos;
        ++write_ptr;
    }
    out_delta = next_pos - r.seq_len;
    return row;
}

// Mirrors the host walk in qwen3_5vit.cc Setup() — same logic, returns the segment list.
void emit_segments(const RequestSpec& r, int request_idx, int S, std::vector<MropeSegment>& out)
{
    if (r.autoregres || r.images.empty()) {
        return;
    }
    int  row = 0, pos = 0, mm_off = 0;
    auto emit = [&](int run_start, int run_n, int run_base, int h2, int w2) {
        const int a = std::max(run_start, r.active_start);
        const int b = std::min(run_start + run_n, r.active_end);
        if (a >= b) {
            return;
        }
        const int    local_off = a - run_start;
        MropeSegment seg{};
        seg.dst_row    = request_idx;
        seg.dst_offset = a;
        seg.n_tok      = b - a;
        seg.base_pos   = (h2 == 0) ? run_base + local_off : run_base;
        seg.h2         = h2;
        seg.w2         = w2;
        seg.k_offset   = (h2 == 0) ? 0 : local_off;
        out.push_back(seg);
    };
    for (const auto& img : r.images) {
        const int h2    = img.h / S;
        const int w2    = img.w / S;
        const int n_tok = img.t * h2 * w2;
        if (img.seq_start > row) {
            emit(row, img.seq_start - row, pos, 0, 0);
        }
        const int img_base = img.seq_start + mm_off;
        emit(img.seq_start, n_tok, img_base, h2, w2);
        row               = img.seq_start + n_tok;
        const int new_pos = std::max({img.t, h2, w2});
        pos               = img_base + new_pos;
        mm_off += new_pos - n_tok;
    }
    if (row < r.seq_len) {
        emit(row, r.seq_len - row, pos, 0, 0);
    }
}

int run_case(const std::string& name, const std::vector<RequestSpec>& batch, int S)
{
    const int bsz            = (int)batch.size();
    int       max_active_end = 0;
    int       max_seg_len    = 0;
    bool      any_table      = false;

    std::vector<MropeSegment> segs;
    for (int i = 0; i < bsz; ++i) {
        const size_t before = segs.size();
        emit_segments(batch[i], i, S, segs);
        for (size_t j = before; j < segs.size(); ++j) {
            max_seg_len = std::max(max_seg_len, segs[j].n_tok);
        }
        if (!batch[i].autoregres && !batch[i].images.empty()) {
            max_active_end = std::max(max_active_end, batch[i].active_end);
            any_table      = true;
        }
    }

    // Run kernel
    std::vector<int> kernel_out;
    if (any_table) {
        const ssize_t pos_ids_count = (ssize_t)bsz * max_active_end * 3;
        int*          d_pos_ids     = nullptr;
        cudaMalloc(&d_pos_ids, pos_ids_count * sizeof(int));
        cudaMemset(d_pos_ids, 0xCC, pos_ids_count * sizeof(int));  // poison

        MropeSegment* d_segs = nullptr;
        cudaMalloc(&d_segs, segs.size() * sizeof(MropeSegment));
        cudaMemcpy(d_segs, segs.data(), segs.size() * sizeof(MropeSegment), cudaMemcpyHostToDevice);

        invokeMropePositionIds(d_pos_ids,
                               max_active_end * 3,
                               d_segs,
                               (int)segs.size(),
                               max_seg_len,
                               /*stream=*/0);
        cudaDeviceSynchronize();

        kernel_out.resize(pos_ids_count);
        cudaMemcpy(kernel_out.data(), d_pos_ids, pos_ids_count * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_pos_ids);
        cudaFree(d_segs);
    }

    // Compare against CPU reference within each request's active range
    int errors = 0;
    for (int i = 0; i < bsz; ++i) {
        const auto&      r         = batch[i];
        int              delta_ref = 0;
        std::vector<int> full_row  = cpu_reference_full_row(r, S, delta_ref);
        if (r.autoregres || r.images.empty()) {
            continue;  // no table writes for this slot
        }
        for (int k = r.active_start; k < r.active_end; ++k) {
            for (int c = 0; c < 3; ++c) {
                const int got = kernel_out[(size_t)i * max_active_end * 3 + (size_t)k * 3 + c];
                const int ref = full_row[k * 3 + c];
                if (got != ref) {
                    if (errors < 16) {
                        std::printf(
                            "  [%s] mismatch req=%d tok=%d c=%d  got=%d ref=%d\n", name.c_str(), i, k, c, got, ref);
                    }
                    ++errors;
                }
            }
        }
    }

    if (errors == 0) {
        std::printf("[PASS] %s — bsz=%d segs=%zu max_active_end=%d\n", name.c_str(), bsz, segs.size(), max_active_end);
    }
    else {
        std::printf("[FAIL] %s — %d mismatches\n", name.c_str(), errors);
    }
    return errors;
}

}  // namespace

int main()
{
    const int S      = 2;  // spatial_merge_size
    int       errors = 0;

    // (a) Decode-only batch — no table writes expected.
    errors += run_case("decode_only",
                       {RequestSpec{/*seq_len=*/64,
                                    /*active_start=*/64,
                                    /*active_end=*/65,
                                    /*autoregres=*/true,
                                    /*images=*/{}}},
                       S);

    // (b) Pure-text prefill — empty images, identity positions.
    errors += run_case("pure_text", {RequestSpec{32, 0, 32, false, {}}}, S);

    // (c) Single-image prefill (image in the middle).
    errors += run_case("single_image",
                       {RequestSpec{40, 0, 40, false, {{/*seq_start=*/8, /*t=*/1, /*h=*/8, /*w=*/8}}}},  // 16 tokens
                       S);

    // (d) Interleaved text-image-text-image-text.
    errors += run_case("interleaved", {RequestSpec{80, 0, 80, false, {{4, 1, 4, 6}, {30, 1, 6, 4}}}}, S);

    // (e) Video (t > 1).
    errors += run_case("video",
                       {RequestSpec{50, 0, 50, false, {{2, 2, 4, 4}}}},  // 2*2*2 = 8 tokens
                       S);

    // (f) Mixed-batch: one decode + two prefills (one text-only, one with image).
    errors += run_case("mixed_batch",
                       {RequestSpec{16, 16, 17, true, {}},               // decode
                        RequestSpec{20, 0, 20, false, {}},               // text prefill
                        RequestSpec{30, 0, 30, false, {{6, 1, 6, 4}}}},  // image prefill
                       S);

    // (g) Chunked prefill — second chunk, history_len > 0, active range mid-prompt.
    //     Image overlaps both chunks.
    errors += run_case("chunked_prefill",
                       {RequestSpec{60, 16, 32, false, {{8, 1, 8, 8}}}},  // image spans 8..24
                       S);

    // (h) Multi-image with clipping.
    errors += run_case("multi_image_clip", {RequestSpec{80, 10, 50, false, {{6, 1, 4, 4}, {30, 2, 4, 4}}}}, S);

    if (errors == 0) {
        std::printf("All cases passed.\n");
        return 0;
    }
    std::printf("FAILED — %d total mismatches.\n", errors);
    return 1;
}
