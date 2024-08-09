#pragma once

namespace turbomind {
namespace marlin_qqq {

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages = 4;  // 4 pipeline stages fit into shared memory

static constexpr int min_thread_n = 64;
// TODO(HandH1998): 64 or 128?
static constexpr int min_thread_k = 64;

static constexpr int tile_size        = 16;
static constexpr int max_par          = 16;
static constexpr int pack_factor_4bit = 8;  // We have 8 4-bit vals inside a 32 bit

void invokeMarlinQQQGemm(const int4*  A,
                         const int4*  B,
                         int4*        C,
                         int4*        D,
                         const float* s1,
                         const int4*  s2,
                         const int4*  s3,
                         int          prob_m,
                         int          prob_n,
                         int          prob_k,
                         int*         locks,
                         int          thread_n_blocks,
                         int          thread_k_blocks,
                         int          group_blocks,
                         int          num_threads,
                         cudaStream_t stream);

}  // namespace marlin_qqq
}  // namespace turbomind
