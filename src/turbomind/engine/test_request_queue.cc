// Copyright (c) OpenMMLab. All rights reserved.

#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "catch2/catch_test_macros.hpp"

#include "src/turbomind/engine/request_queue.h"

namespace turbomind {
namespace {

using Policy = SchedulePolicy;

std::shared_ptr<Request> make_request(uint64_t id, uint8_t priority)
{
    auto req              = std::make_shared<Request>();
    req->id               = id;
    req->session          = SessionParam{id, 0, true, true, false};
    req->gen_cfg.priority = priority;
    req->cancel_flag.store(0, std::memory_order_relaxed);
    return req;
}

void pop(RequestQueue&                          queue,
         std::vector<std::shared_ptr<Request>>& infer_reqs,
         std::vector<std::shared_ptr<Request>>& kill_reqs,
         unsigned                               max_infer,
         bool                                   blocking = false)
{
    bool abort{};
    queue.pop(infer_reqs, kill_reqs, max_infer, blocking, abort);
    REQUIRE_FALSE(abort);
}

std::vector<uint64_t> ids(const std::vector<std::shared_ptr<Request>>& reqs)
{
    std::vector<uint64_t> ret;
    for (const auto& req : reqs) {
        ret.push_back(req->id);
    }
    return ret;
}

void expect_canceled_queued_request_is_skipped(RequestQueue& queue)
{
    auto canceled = make_request(1, 0);
    auto valid    = make_request(2, 5);
    canceled->cancel_flag.store(-1, std::memory_order_relaxed);
    queue.push(canceled);
    queue.push(valid);

    std::vector<std::shared_ptr<Request>> infer_reqs;
    std::vector<std::shared_ptr<Request>> kill_reqs;
    pop(queue, infer_reqs, kill_reqs, 2);

    REQUIRE(ids(infer_reqs) == std::vector<uint64_t>{2});
    REQUIRE(kill_reqs.empty());
}

}  // namespace

TEST_CASE("fifo policy ignores priority", "[request_queue]")
{
    FifoRequestQueue queue;
    queue.push(make_request(1, 0));
    queue.push(make_request(2, 10));
    queue.push(make_request(3, 5));

    std::vector<std::shared_ptr<Request>> infer_reqs;
    std::vector<std::shared_ptr<Request>> kill_reqs;
    pop(queue, infer_reqs, kill_reqs, 3);

    REQUIRE(ids(infer_reqs) == std::vector<uint64_t>{1, 2, 3});
    REQUIRE(kill_reqs.empty());
}

TEST_CASE("priority policy returns smallest priority value first", "[request_queue]")
{
    PriorityRequestQueue queue;
    queue.push(make_request(1, 0));
    queue.push(make_request(2, 10));
    queue.push(make_request(3, 5));

    std::vector<std::shared_ptr<Request>> infer_reqs;
    std::vector<std::shared_ptr<Request>> kill_reqs;
    pop(queue, infer_reqs, kill_reqs, 3);

    REQUIRE(ids(infer_reqs) == std::vector<uint64_t>{1, 3, 2});
    REQUIRE(kill_reqs.empty());
}

TEST_CASE("priority policy keeps fifo order for equal priority", "[request_queue]")
{
    PriorityRequestQueue queue;
    queue.push(make_request(1, 3));
    queue.push(make_request(2, 3));
    queue.push(make_request(3, 3));

    std::vector<std::shared_ptr<Request>> infer_reqs;
    std::vector<std::shared_ptr<Request>> kill_reqs;
    pop(queue, infer_reqs, kill_reqs, 3);

    REQUIRE(ids(infer_reqs) == std::vector<uint64_t>{1, 2, 3});
}

TEST_CASE("priority policy max infer limits batch", "[request_queue]")
{
    PriorityRequestQueue queue;
    queue.push(make_request(1, 1));
    queue.push(make_request(2, 9));
    queue.push(make_request(3, 8));

    std::vector<std::shared_ptr<Request>> infer_reqs;
    std::vector<std::shared_ptr<Request>> kill_reqs;
    pop(queue, infer_reqs, kill_reqs, 2);
    REQUIRE(ids(infer_reqs) == std::vector<uint64_t>{1, 3});

    infer_reqs.clear();
    kill_reqs.clear();
    pop(queue, infer_reqs, kill_reqs, 2);
    REQUIRE(ids(infer_reqs) == std::vector<uint64_t>{2});
}

TEST_CASE("canceled queued request is skipped in all policies", "[request_queue]")
{
    SECTION("fifo")
    {
        FifoRequestQueue queue;
        expect_canceled_queued_request_is_skipped(queue);
    }

    SECTION("priority")
    {
        PriorityRequestQueue queue;
        expect_canceled_queued_request_is_skipped(queue);
    }
}

TEST_CASE("kill requests are drained independently", "[request_queue]")
{
    PriorityRequestQueue queue;
    auto                 infer = make_request(1, 10);
    auto                 kill  = make_request(2, 0);
    queue.push(infer);
    queue.kill(kill);

    std::vector<std::shared_ptr<Request>> infer_reqs;
    std::vector<std::shared_ptr<Request>> kill_reqs;
    pop(queue, infer_reqs, kill_reqs, 1);

    REQUIRE(infer_reqs == std::vector<std::shared_ptr<Request>>{infer});
    REQUIRE(kill_reqs == std::vector<std::shared_ptr<Request>>{kill});
}

TEST_CASE("blocking pop wakes on push", "[request_queue]")
{
    PriorityRequestQueue queue;

    std::promise<std::vector<uint64_t>> promise;
    auto                                future = promise.get_future();

    std::thread worker{[&] {
        std::vector<std::shared_ptr<Request>> infer_reqs;
        std::vector<std::shared_ptr<Request>> kill_reqs;
        pop(queue, infer_reqs, kill_reqs, 1, true);
        promise.set_value(ids(infer_reqs));
    }};

    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    queue.push(make_request(42, 1));

    REQUIRE(future.get() == std::vector<uint64_t>{42});
    worker.join();
}

TEST_CASE("invalid schedule policy string is rejected", "[request_queue]")
{
    REQUIRE(parse_schedule_policy("fifo") == Policy::kFifo);
    REQUIRE(parse_schedule_policy("priority") == Policy::kPriority);
    REQUIRE_THROWS_AS(parse_schedule_policy("bad"), std::invalid_argument);
}

}  // namespace turbomind
