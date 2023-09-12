// Copyright (c) OpenMMLab. All rights reserved.

#include "pycb_utils.h"
#include <memory>

namespace turbomind {

thread_local std::shared_ptr<int> _current;
thread_local std::shared_ptr<int> _total;

void set_batch_info(int current, int total)
{
    if (!_current) {
        _current = std::make_shared<int>();
        _total   = std::make_shared<int>();
    }
    *_current = current;
    *_total   = total;
}

int is_first_in_batch()
{
    return *_current == 0;
}

int is_last_in_batch()
{
    return *_current == (*_total - 1);
}

}  // namespace turbomind
