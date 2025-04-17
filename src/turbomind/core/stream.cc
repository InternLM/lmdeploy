
#include "src/turbomind/core/stream.h"
#include <memory>

namespace turbomind::core {

Stream Stream::create(int priority)
{
    Stream stream;
    stream.impl_ = std::make_shared<StreamImpl>(priority);
    return stream;
}

void StreamImpl::Wait(const Event& event)
{
    check_cuda_error(cudaStreamWaitEvent(stream_, event));
}

}  // namespace turbomind::core
