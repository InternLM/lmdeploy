
#include <array>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

struct LayoutConverter {

    Order order;
    Pack  pack;

    virtual int Convert(const void*         S,  //
                        const MatrixLayout& Sdesc,
                        void*               D,
                        MatrixLayout&       Ddesc,
                        cudaStream_t        stream) const = 0;
};

// Pointers to singletons
std::array<const LayoutConverter*, 2> GetConverters(DataType data_type,
                                                    DataType weight_type,  //
                                                    DataType input_type,
                                                    bool     grouped,
                                                    int      sm);

// Free with `cudaFree`
void* MakeStridedPtrs(const std::vector<std::pair<void*, int>>& ptrs, cudaStream_t stream);
void* MakeBlockedPtrs(const std::vector<std::pair<void*, int>>& ptrs, cudaStream_t stream);

}  // namespace turbomind::gemm
