// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/Barrier.h"
#include "src/turbomind/utils/instance_comm.h"

namespace fastertransformer {

class LlamaInstanceComm: public AbstractInstanceComm {
public:
    LlamaInstanceComm(int count): barrier_(count) {}

    void barrier() override
    {
        barrier_.wait();
    }

    void setSharedObject(void* p) override
    {
        ptr = p;
    }

    void* getSharedObject() override
    {
        return ptr;
    }

private:
    Barrier barrier_;
    void*   ptr{};
};

}  // namespace fastertransformer
