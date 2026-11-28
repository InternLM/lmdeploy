#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/core.h"

namespace turbomind {

Tensor PaddedRowAllGather(Tensor                projected_local,
                          Tensor                gathered_padded,
                          int                   logical_row_count,
                          int                   local_row_count,
                          int                   model_tp_rank,
                          int                   model_tp_size,
                          comm::DeviceCommImpl& communicator,
                          int                   model_tp_group,
                          cudaStream_t          stream);

}  // namespace turbomind
