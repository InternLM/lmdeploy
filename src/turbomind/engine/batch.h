
#pragma once

#include <future>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/request.h"

namespace turbomind {

enum class BatchOp
{
    kAdd,      //  Se ->  Rc         H
    kSetup,    //  Rc -> (B  -> D)   H2D
    kPrepare,  // (D  ->  St)        D
    kForward,  //  St ->  St         D
    kUnprep,   // (St ->  D)         D
    kFetch,    // (D  ->  B)         D2H
    kUpdate,   //  B  ->  Rc         H
    kDel,      //  Rc ->  Se         H
};

// Se -> Rc -> (B -> D) -> St -> (D -> B) -> Rc -> Se

/*
Se -> Rc                   (add: rc)
    Rc -> B
        (B -> D)           (setup: rc, d, copy)
            (D -> St)
                St -> St   (forward)
            (St -> D)
        (D -> B)
    B -> Rc                (sync)
Rc -> Se                   (del: rc)
*/

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

    int bs0 = 0;
    int bsz = 0;

    Buffer_<int> perm;

    std::vector<std::shared_ptr<RequestCache>> rc;

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
