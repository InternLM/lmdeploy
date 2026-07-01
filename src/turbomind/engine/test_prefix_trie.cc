// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/interval.h"
#include "src/turbomind/engine/cache_mode.h"
#include "src/turbomind/engine/fingerprint.h"
#include "src/turbomind/engine/prefix_key.h"
#include "src/turbomind/engine/prefix_trie.h"
#include "src/turbomind/engine/prompt_boundary.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/engine/scheduler.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

using namespace turbomind;

namespace {
Fingerprint FP(uint64_t a)
{
    return Fingerprint{{a, a + 1, a + 2, a + 3}};
}
}  // namespace

TEST_CASE("Fingerprint: empty never equals; distinct differ; identical match", "[fingerprint]")
{
    Fingerprint empty{};
    REQUIRE(empty.empty());
    REQUIRE_FALSE(empty == empty);  // empty never equals anything -- including itself
    REQUIRE(empty != empty);

    const Fingerprint a = FP(100), b = FP(200), a2 = FP(100);
    REQUIRE(a == a2);
    REQUIRE_FALSE(a == b);
    REQUIRE_FALSE(a == empty);
    REQUIRE_FALSE(empty == a);
}

TEST_CASE("PrefixTrie::Find honors image_fps", "[prefix_trie]")
{
    const int  bs = 4;
    PrefixTrie trie{bs};

    std::vector<int>  toks = {1, 2, 3, 4};
    const Fingerprint fpA = FP(1), fpB = FP(2);

    LogicalBlock blkA{};
    blkA.parent    = nullptr;
    blkA.size      = bs;
    blkA.tokens    = toks;
    blkA.image_fps = {fpA};
    blkA.key       = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks), {fpA});
    REQUIRE(trie.Insert(blkA));

    // Same tokens + same fingerprint -> hit.
    {
        const auto key = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks), {fpA});
        REQUIRE(trie.Find(nullptr, key, MakeTokenSpan(toks), {fpA}) == &blkA);
    }
    // Same tokens, DIFFERENT fingerprint -> miss (no false hit).
    {
        const auto key = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks), {fpB});
        REQUIRE(trie.Find(nullptr, key, MakeTokenSpan(toks), {fpB}) == nullptr);
    }
    // Same tokens, EMPTY fingerprint -> miss (empty never equals).
    {
        const std::vector<Fingerprint> empty_fps = {Fingerprint{}};
        const auto                     key       = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks), empty_fps);
        REQUIRE(trie.Find(nullptr, key, MakeTokenSpan(toks), empty_fps) == nullptr);
    }
}

TEST_CASE("PrefixTrie::Find: plain text block matches with empty fps", "[prefix_trie]")
{
    const int  bs = 4;
    PrefixTrie trie{bs};

    std::vector<int> toks = {5, 6, 7, 8};
    LogicalBlock     blk{};
    blk.parent = nullptr;
    blk.size   = bs;
    blk.tokens = toks;  // no image_fps -> empty
    blk.key    = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks));
    REQUIRE(trie.Insert(blk));

    const auto key = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks));
    REQUIRE(trie.Find(nullptr, key, MakeTokenSpan(toks)) == &blk);  // default fps = {}
    REQUIRE(trie.Find(nullptr, key, MakeTokenSpan(toks), {}) == &blk);
}

TEST_CASE("PrefixTrie::Search finds a partial block and sub-selects fingerprints", "[prefix_trie]")
{
    const int  bs = 4;
    PrefixTrie trie{bs};

    // Insert a PARTIAL block of length 2 whose first token carries an image start.
    std::vector<int>  part = {1, 2};
    const Fingerprint fpA  = FP(1);
    LogicalBlock      blkA{};
    blkA.parent    = nullptr;
    blkA.size      = (int)part.size();
    blkA.tokens    = part;
    blkA.image_fps = {fpA};
    blkA.key       = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(part), {fpA});
    REQUIRE(trie.Insert(blkA));

    // Search a full 4-token span that shares the {1,2} prefix; the image starts at
    // relative position 0. Search must enforce a partial match and land on blkA.
    std::vector<int> full = {1, 2, 3, 4};
    PrefixKey        key{};
    LogicalBlock*    hit = trie.Search(nullptr, key, MakeTokenSpan(full), {fpA}, /*fp_pos=*/{0});
    REQUIRE(hit == &blkA);
    REQUIRE(key == blkA.key);  // on a hit, key is replaced with the matched node's key
}

TEST_CASE("PrefixTrie::Search excludes an image that starts beyond the matched prefix", "[prefix_trie]")
{
    const int  bs = 4;
    PrefixTrie trie{bs};

    // Insert a PARTIAL length-2 block with NO image in its first two tokens.
    std::vector<int> part = {1, 2};
    LogicalBlock     blk{};
    blk.parent = nullptr;
    blk.size   = (int)part.size();
    blk.tokens = part;  // empty image_fps
    blk.key    = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(part));
    REQUIRE(trie.Insert(blk));

    // Image starts at relative position 2 (token index 2). For the length-2 prefix the
    // sub-selection (fp_pos < 2) is empty, so it must match the empty-fps block.
    std::vector<int>  full = {1, 2, 3, 4};
    const Fingerprint fpA  = FP(7);
    PrefixKey         key{};
    LogicalBlock*     hit = trie.Search(nullptr, key, MakeTokenSpan(full), {fpA}, /*fp_pos=*/{2});
    REQUIRE(hit == &blk);
}

TEST_CASE("PrefixTrie::Search is bounded when fp_pos is shorter than fps (no OOB)", "[prefix_trie]")
{
    const int  bs = 4;
    PrefixTrie trie{bs};

    // Partial length-2 block whose first token carries one image start (fpA).
    std::vector<int>  part = {1, 2};
    const Fingerprint fpA  = FP(1);
    LogicalBlock      blkA{};
    blkA.parent    = nullptr;
    blkA.size      = (int)part.size();
    blkA.tokens    = part;
    blkA.image_fps = {fpA};
    blkA.key       = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(part), {fpA});
    REQUIRE(trie.Insert(blkA));

    // Regression for the `j < fp_pos.size()` bound: fps has 2 entries but fp_pos only 1.
    // The loop must stop at fp_pos.size() and never read fp_pos[1].
    std::vector<int>  full = {1, 2, 3, 4};
    const Fingerprint fpB  = FP(2);
    PrefixKey         key{};
    LogicalBlock*     hit = trie.Search(nullptr, key, MakeTokenSpan(full), {fpA, fpB}, /*fp_pos=*/{0});
    REQUIRE(hit == &blkA);
}

TEST_CASE("PlanPromptBoundary: geometry and guards", "[prompt_boundary]")
{
    const int bs = 8;

    // K=1, last block has >1 token (prompt%bs==3): partial node at prompt_len-1, j==last.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/19, bs, /*skip=*/1, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE(p.partial);
        REQUIRE(p.pos == 18);       // 19 - 1
        REQUIRE(p.block == 2);      // (18-1)/8
        REQUIRE(p.node_size == 2);  // 18 - 16
    }
    // K=1, last block has exactly 1 token (prompt%bs==1): block-aligned, no partial node.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/17, bs, /*skip=*/1, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE_FALSE(p.partial);
        REQUIRE(p.pos == 16);   // 17 - 1, block-aligned
        REQUIRE(p.block == 1);  // (16-1)/8
    }
    // K=2, last block has >2 tokens (prompt%bs==3): partial node at prompt_len-2.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/19, bs, /*skip=*/2, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE(p.partial);
        REQUIRE(p.pos == 17);       // 19 - 2
        REQUIRE(p.block == 2);      // (17-1)/8
        REQUIRE(p.node_size == 1);  // 17 - 16
    }
    // K pushes B into the prior block (prompt%bs==2, K=3): B=18 in block 2, partial.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/21, bs, /*skip=*/3, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE(p.partial);
        REQUIRE(p.pos == 18);       // 21 - 3
        REQUIRE(p.block == 2);      // (18-1)/8
        REQUIRE(p.node_size == 2);  // 18 - 16
    }
    // Partial node needs st.miss < j: miss at j blocks the node.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/19, bs, /*skip=*/1, /*miss=*/2);  // j==2
        REQUIRE_FALSE(p.valid);
    }
    // Block-aligned allows st.miss <= j: miss at j still publishes the clamp target.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/17, bs, /*skip=*/1, /*miss=*/1);  // j==1
        REQUIRE(p.valid);
        REQUIRE_FALSE(p.partial);
        REQUIRE(p.pos == 16);
    }
    // Block-aligned PROMPT at K=1 (prompt%bs==0): NOT suppressed -- a matchable
    // boundary is published (option B; old code skipped this).
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/16, bs, /*skip=*/1, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE(p.partial);
        REQUIRE(p.pos == 15);       // 16 - 1
        REQUIRE(p.block == 1);      // (15-1)/8
        REQUIRE(p.node_size == 7);  // 15 - 8
    }
    // Think + full-block case: block-aligned prompt, K=2 -> partial node before
    // the volatile suffix that lives in the last full block.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/16, bs, /*skip=*/2, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE(p.partial);
        REQUIRE(p.pos == 14);       // 16 - 2
        REQUIRE(p.block == 1);      // (14-1)/8
        REQUIRE(p.node_size == 6);  // 14 - 8
    }
    // Partial geometry with j==0 (B < block_size): must be invalid -- no parent
    // block exists, and miss < 0 is impossible. Locks Task 3's block_ids[j-1] safety.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/4, bs, /*skip=*/1, /*miss=*/0);  // B=3, j=0
        REQUIRE_FALSE(p.valid);
    }
    // Block-aligned but miss past j: not matchable -> invalid.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/17, bs, /*skip=*/1, /*miss=*/2);  // B=16, j=1
        REQUIRE_FALSE(p.valid);
    }
    // B < 1 -> no boundary.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/1, bs, /*skip=*/1, /*miss=*/0);
        REQUIRE_FALSE(p.valid);
    }
}

TEST_CASE("ParseCacheMode maps strings to CacheMode", "[cache_mode]")
{
    using turbomind::CacheMode;
    using turbomind::ParseCacheMode;
    CHECK(ParseCacheMode("none") == CacheMode::kNone);
    CHECK(ParseCacheMode("auto") == CacheMode::kAuto);
    CHECK(ParseCacheMode("all") == CacheMode::kAll);
}

TEST_CASE("DecidePromptBoundaryPublish gates by mode/partial/image", "[cache_mode]")
{
    using turbomind::CacheMode;
    using turbomind::DecidePromptBoundaryPublish;

    // Partial node (B mid-block): 'all' always publishes; 'auto' only with image.
    CHECK(DecidePromptBoundaryPublish(CacheMode::kAll, /*partial=*/true, /*has_image=*/false));
    CHECK(DecidePromptBoundaryPublish(CacheMode::kAll, true, true));
    CHECK_FALSE(DecidePromptBoundaryPublish(CacheMode::kAuto, true, false));
    CHECK(DecidePromptBoundaryPublish(CacheMode::kAuto, true, true));

    // Block-aligned B (no partial node): only 'all' arms the checkpoint clamp.
    CHECK(DecidePromptBoundaryPublish(CacheMode::kAll, /*partial=*/false, /*has_image=*/false));
    CHECK_FALSE(DecidePromptBoundaryPublish(CacheMode::kAuto, false, false));
    CHECK_FALSE(DecidePromptBoundaryPublish(CacheMode::kAuto, false, true));
}

TEST_CASE("HasMultimodalOverlap: overlaps [lo, hi) with ascending spans", "[multimodal_overlap]")
{
    using turbomind::Interval;
    using turbomind::MultiModalSpan;
    using turbomind::Scheduler;
    using turbomind::Sequence;

    auto make_seq = [](std::vector<std::pair<int, int>> spans) {
        auto s = std::make_shared<Sequence>(std::make_shared<turbomind::Request>());
        for (const auto& [begin, end] : spans) {
            s->multimodal_spans.push_back(MultiModalSpan{Interval{begin, end}, {}});
        }
        return s;
    };

    // (a) A span fully inside [lo, hi) -> true.
    {
        auto s = make_seq({{12, 14}});
        CHECK(Scheduler::HasMultimodalOverlap(*s, 8, 16));
    }
    // (b) No span, span entirely before lo, or span entirely at-or-after hi -> false.
    {
        auto empty = make_seq({});
        CHECK_FALSE(Scheduler::HasMultimodalOverlap(*empty, 8, 16));

        auto before = make_seq({{2, 8}});  // end == lo, so [2,8) is entirely before lo=8
        CHECK_FALSE(Scheduler::HasMultimodalOverlap(*before, 8, 16));

        auto after = make_seq({{16, 20}});  // begin == hi -> at-or-after hi
        CHECK_FALSE(Scheduler::HasMultimodalOverlap(*after, 8, 16));
    }
    // (c) A span that begins before lo but ends after lo (begin < hi) -> true.
    {
        auto s = make_seq({{4, 10}});  // begins before lo=8, extends into [8,16)
        CHECK(Scheduler::HasMultimodalOverlap(*s, 8, 16));
    }
    // (d) Ascending early-break: a later span with begin >= hi (would return false)
    //     never masks an earlier overlapping span, which returns true first.
    {
        auto s = make_seq({{10, 12}, {20, 24}});  // first overlaps, second is >= hi
        CHECK(Scheduler::HasMultimodalOverlap(*s, 8, 16));
    }
}
