
#pragma once

#include <future>

#include "src/turbomind/core/core.h"

namespace turbomind {

struct RequestCache;

struct BatchData {

    explicit BatchData(int phase): phase{phase}
    {
        ready = Event::create();
        done  = Event::create();
        next  = Event::create();
    }

    const int phase;

    int bs0 = 0;
    int bsz = 0;

    std::vector<int> perm;

    std::vector<int> local_token_num;
    int              global_token_num = 0;

    Event ready;
    Event done;
    Event next;

    std::promise<Event> promise;

    void Notify()
    {
        next.Record(core::Context::stream());
        promise.set_value(next);
    }
};

}  // namespace turbomind