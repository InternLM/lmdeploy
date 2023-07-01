#pragma once

namespace turbomind {

class AbstractInstanceComm {
public:
    virtual ~AbstractInstanceComm() = default;

    virtual void barrier() = 0;

    virtual void setSharedObject(void*) = 0;

    virtual void* getSharedObject() = 0;
};

}  // namespace turbomind
