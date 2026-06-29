
#pragma once

#include <cstddef>
#include <future>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/request.h"

namespace turbomind {

enum class BatchOp
{
    kAdd,      //  Request -> Seq    H        Sched
    kSetup,    //  Seq -> (B  -> D)  H2D      Sched
    kPrepare,  // (D  ->  St)        D        Exec
    kForward,  //  St ->  St         D        Exec
    kUnprep,   // (St ->  D)         D        Exec
    kFetch,    // (D  ->  B)         D2H      Sched
    kUpdate,   //  B  ->  Seq        H        Sched
    kDel,      //  Seq ->  Request   H        Sched
};

// Request -> Seq -> (B -> D) -> St -> (D -> B) -> Seq -> Request

/*
Request -> Sequence        (add)
    Sequence -> B
        (B -> D)           (setup: Sequence, d, copy)
            (D -> St)
                St -> St   (forward)
            (St -> D)
        (D -> B)
    B -> Sequence          (sync)
Sequence -> Request        (del)
*/

// A scheduler-planned cache copy resolved to device pointers on the engine
// thread. Executed by the model executor as a whole-object device copy.
struct ResolvedCopy {
    void*  src;
    void*  dst;
    size_t bytes;
};

/// TODO: The strcuture itself should not be passed as a part of ENV
struct BatchData {

    explicit BatchData(int phase): self{this}, phase{phase}
    {
        ready = Event::create();
        done  = Event::create();
        next  = Event::create();
    }

    BatchData(const BatchData&)     = delete;
    BatchData(BatchData&&) noexcept = delete;
    BatchData& operator=(const BatchData&) = delete;
    BatchData& operator=(BatchData&&) noexcept = delete;

    BatchData* self;

    const int phase;

    int bs0 = 0;  // prev batch size
    int bsz = 0;  // curr batch size

    Buffer_<int> perm;

    std::vector<ResolvedCopy> restore_copies;  // run before BatchOp::kPrepare
    std::vector<ResolvedCopy> publish_copies;  // run after BatchOp::kUnprep

    std::vector<int> local_token_num;
    int              global_token_num = 0;

    Event ready;
    Event done;
    Event next;

    std::promise<Event> promise;

    Buffer buf()
    {
        return Buffer{&self, 1, kCPU};
    }

    void Notify()
    {
        next.Record(core::Context::stream());
        promise.set_value(next);
    }
};

}  // namespace turbomind
