#include "src/turbomind/comm/padded_row_allgather.h"

#include <cstddef>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

Tensor PaddedRowAllGather(Tensor                projected_local,
                          Tensor                gathered_padded,
                          int                   logical_row_count,
                          int                   local_row_count,
                          int                   model_tp_rank,
                          int                   model_tp_size,
                          comm::DeviceCommImpl& communicator,
                          int                   model_tp_group,
                          cudaStream_t          stream)
{
    const int n = logical_row_count;
    const int T = model_tp_size;
    const int H = projected_local.shape(1);

    if (n == 0) {
        return projected_local.slice({0, 0}, {0, H});
    }
    if (T == 1) {
        return projected_local.slice({0, 0}, {n, H});
    }

    const int slice = cdiv(n, T);
    if (local_row_count < slice) {
        cudaMemset2DAsync(static_cast<std::byte*>(projected_local.raw_data())
                              + local_row_count * projected_local.stride(0) * byte_size(projected_local.dtype()),
                          projected_local.stride(0) * byte_size(projected_local.dtype()),
                          0,
                          projected_local.shape(1) * byte_size(projected_local.dtype()),
                          slice - local_row_count,
                          stream);
    }

    communicator.AllGather(projected_local.raw_data(),
                           gathered_padded.raw_data(),
                           slice * H,
                           projected_local.dtype(),
                           model_tp_group,
                           stream);

    return gathered_padded.slice({0, 0}, {n, H});
}

}  // namespace turbomind
