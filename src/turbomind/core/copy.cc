
#include "src/turbomind/core/copy.h"

#include <cstdint>
#include <type_traits>
#include <variant>

#include <cuda_runtime.h>

#include "src/turbomind/core/check.h"

namespace turbomind::core {

// picked from "cudaTypedefs.h" / "cuda.h"

typedef enum CUmemcpyFlags_enum
{
    CU_MEMCPY_FLAG_DEFAULT                     = 0x0,
    CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE = 0x1
} CUmemcpyFlags;

typedef enum CUmemcpySrcAccessOrder_enum
{
    CU_MEMCPY_SRC_ACCESS_ORDER_INVALID         = 0x0,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM          = 0x1,
    CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL = 0x2,
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY             = 0x3,
    CU_MEMCPY_SRC_ACCESS_ORDER_MAX             = 0x7FFFFFFF
} CUmemcpySrcAccessOrder;

typedef struct CUmemcpyAttributes_st {
    CUmemcpySrcAccessOrder srcAccessOrder;
    CUmemLocation          srcLocHint;
    CUmemLocation          dstLocHint;
    unsigned int           flags;
} CUmemcpyAttributes_v1;

typedef CUresult(CUDAAPI* PFN_cuMemcpyBatchAsync_v12080)(CUdeviceptr_v2*        dsts,
                                                         CUdeviceptr_v2*        srcs,
                                                         size_t*                sizes,
                                                         size_t                 count,
                                                         CUmemcpyAttributes_v1* attrs,
                                                         size_t*                attrIdxs,
                                                         size_t                 numAttrs,
                                                         size_t*                failIdx,
                                                         CUstream               hStream);

/// TODO: add `PFN_cuMemcpyBatchAsync_v13000`

namespace {

const auto& GetCopyAPI()
{
    static auto inst = []() -> std::variant<std::monostate, PFN_cuMemcpyBatchAsync_v12080> {
        const auto                      symbol = "cuMemcpyBatchAsync";
        cudaDriverEntryPointQueryResult status{};
        void*                           fpn{};
        TM_CHECK_EQ(cudaGetDriverEntryPoint(symbol, &fpn, cudaEnableDefault, &status), 0);
        if (fpn && status == cudaDriverEntryPointSuccess) {
            return (PFN_cuMemcpyBatchAsync_v12080)fpn;
        }
        else {
            return {};
        }
    }();
    return inst;
}

}  // namespace

BatchCopy::~BatchCopy() = default;

BatchCopy::BatchCopy(): self_{this}
{
    Reset();
}

void BatchCopy::Run()
{
    if (src_.empty()) {
        return;
    }

    std::visit(
        [&](auto&& copy) {
            using T = std::decay_t<decltype(copy)>;
            if constexpr (std::is_same_v<T, PFN_cuMemcpyBatchAsync_v12080>) {
                CUmemcpyAttributes_v1 attr{};
                attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
                attr.flags          = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
                std::vector<size_t> ais(src_.size(), 0);
                size_t              fail_idx{SIZE_MAX};

                auto status = copy((CUdeviceptr_v2*)dst_.data(),
                                   (CUdeviceptr_v2*)src_.data(),
                                   size_.data(),
                                   src_.size(),
                                   &attr,
                                   ais.data(),
                                   1,
                                   &fail_idx,
                                   core::Context::stream().handle());

                if (auto i = fail_idx; i != SIZE_MAX) {
                    TM_CHECK(0) << (void*)src_[i] << " " << size_[i] << " " << (void*)dst_[i] << " code " << status;
                }
            }
            else {
                for (unsigned i = 0; i < src_.size(); ++i) {
                    core::Copy(src_[i], size_[i], dst_[i]);
                }
            }
        },
        GetCopyAPI());

    Reset();
}

}  // namespace turbomind::core
