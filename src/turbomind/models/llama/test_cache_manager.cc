#include "BlockManager.h"
#include "SequenceManager.h"

#include "src/turbomind/utils/allocator.h"

#include "src/turbomind/utils/dbg.h"
#include <catch2/catch_test_macros.hpp>
#include <iterator>

using namespace turbomind;

std::ostream& operator<<(std::ostream& os, const Block* b)
{
    os << "(" << b->id << "," << b->timestamp << ")";
    return os;
}

TEST_CASE("BlockManager")
{
    Allocator<AllocatorType::CUDA> allocator(0);

    BlockManager m(1024, 32, 8, &allocator);
    REQUIRE(m.max_block_count() == 32);
    REQUIRE(m.free_count() == 32);

    auto blocks1 = m.Allocate(10);

    dbg(blocks1);

    REQUIRE(blocks1.size() == 10);
    REQUIRE(m.active_count() == blocks1.size());
    REQUIRE(m.free_count() == 22);

    auto blocks2 = m.Allocate(6);
    REQUIRE(blocks2.size() == 6);
    REQUIRE(m.active_count() == blocks1.size() + blocks2.size());
    REQUIRE(m.free_count() == 16);

    auto blocks3 = m.Allocate(16);
    REQUIRE(blocks3.size() == 16);
    REQUIRE(m.active_count() == 32);
    REQUIRE(m.free_count() == 0);

    std::copy(blocks3.begin(), blocks3.end(), std::back_inserter(blocks1));
    std::copy(blocks2.begin(), blocks2.end(), std::back_inserter(blocks1));

    REQUIRE(m.Release(blocks1) == 32);
    REQUIRE(m.active_count() == 0);
    REQUIRE(m.free_count() == 0);
    REQUIRE(m.cached_count() == 32);

    m.Evict(16);
    REQUIRE(m.active_count() == 0);
    REQUIRE(m.free_count() == 16);
    REQUIRE(m.cached_count() == 16);

    auto blocks4 = m.Allocate(14);
    REQUIRE(m.active_count() == 14);
    REQUIRE(m.free_count() == 2);
    REQUIRE(m.cached_count() == 16);
}

TEST_CASE("SequenceManager basic test")
{
    Allocator<AllocatorType::CUDA> allocator(0);

    SequenceManager manager(32, 32, 128, 128, 20, 4, 16, 0, &allocator);

    REQUIRE(manager.max_block_count() == 20);
    REQUIRE(manager.Contains(1) == false);

    auto s1 = manager.Create(1);
    dbg(*s1);
    REQUIRE(manager.Contains(1) == true);

    manager.Erase(1);
    REQUIRE(manager.Contains(1) == false);

    s1 = manager.Create(1);
    REQUIRE(manager.Contains(1) == true);

    auto outcome = manager.Materialize({s1}, {128}, {100}, 1);
    dbg(s1->blocks);
    REQUIRE(s1->blocks.size() == 2);

    auto s2 = manager.Create(2);
    REQUIRE(manager.Contains(2));

    outcome = manager.Materialize({s1, s2}, {128, 2559}, {2, 1}, 1);
    dbg(outcome);
    REQUIRE(outcome.allocation == 20);
    REQUIRE(outcome.swap_in == 1);
    REQUIRE(outcome.swap_out == 1);

    auto s3 = manager.Create(3);
    outcome = manager.Materialize({s1, s2, s3}, {127, 2559, 255}, {1, 100, 2}, 1);
    dbg(outcome);
}

TEST_CASE("SequenceManager functional test")
{
    Allocator<AllocatorType::CUDA> allocator(0);
    SequenceManager                manager(32, 32, 128, 128, 20, 4, 16, 0, &allocator);

    auto seq = manager.Create(1);
    for (int i = 0; i < 1024; ++i) {
        auto outcome = manager.Materialize({seq}, {i}, {0}, 1);
        if (outcome.allocation) {
            dbg(i, outcome);
        }
    }
}