
#include "src/turbomind/core/copy.h"

#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

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

typedef CUresult(CUDAAPI* PFN_cuMemcpyBatchAsync_v13000)(CUdeviceptr_v2*        dsts,
                                                         CUdeviceptr_v2*        srcs,
                                                         size_t*                sizes,
                                                         size_t                 count,
                                                         CUmemcpyAttributes_v1* attrs,
                                                         size_t*                attrIdxs,
                                                         size_t                 numAttrs,
                                                         CUstream               hStream);

namespace {

using MemcpyBatchAsync = std::variant<std::monostate, PFN_cuMemcpyBatchAsync_v12080, PFN_cuMemcpyBatchAsync_v13000>;

template<class F>
F QueryDriverEntryPoint(const char* symbol, unsigned cuda_version)
{
    cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
    void*                           fpn{};
#if CUDA_VERSION >= 13000
    if (cudaGetDriverEntryPointByVersion(symbol, &fpn, cuda_version, cudaEnableDefault, &status) != cudaSuccess) {
        return {};
    }
#else
    (void)cuda_version;
    if (cudaGetDriverEntryPoint(symbol, &fpn, cudaEnableDefault, &status) != cudaSuccess) {
        return {};
    }
#endif
    if (status != cudaDriverEntryPointSuccess || !fpn) {
        return {};
    }
    return reinterpret_cast<F>(fpn);
}

const MemcpyBatchAsync& GetMemcpyBatchAsync()
{
    static thread_local const MemcpyBatchAsync inst = []() -> MemcpyBatchAsync {
        // cuMemcpyBatchAsync crashes on sm_100 (Blackwell); use serialized Copy().
        int device = 0;
        TM_CUDA_CHECK(cudaGetDevice(&device));
        int compute_capability_major = 0;
        TM_CUDA_CHECK(cudaDeviceGetAttribute(&compute_capability_major, cudaDevAttrComputeCapabilityMajor, device));
        if (compute_capability_major >= 10) {
            return std::monostate{};
        }

#if CUDA_VERSION >= 13000
        // CUDA 13.0 removed failIdx from batched memcpy APIs.
        if (auto copy = QueryDriverEntryPoint<PFN_cuMemcpyBatchAsync_v13000>("cuMemcpyBatchAsync", 13000)) {
            return copy;
        }
#endif

        if (auto copy = QueryDriverEntryPoint<PFN_cuMemcpyBatchAsync_v12080>("cuMemcpyBatchAsync", 12080)) {
            return copy;
        }
        return std::monostate{};
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
        [&](auto copy) {
            using T = decltype(copy);
            if constexpr (std::is_same_v<T, std::monostate>) {
                for (size_t i = 0; i < src_.size(); ++i) {
                    core::Copy(src_[i], size_[i], dst_[i]);
                }
            }
            else {
                CUmemcpyAttributes_v1 attr{};
                attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
                attr.flags          = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
                std::vector<size_t> attr_idxs(src_.size(), 0);
                size_t              fail_idx{SIZE_MAX};

                CUresult status;
                if constexpr (std::is_same_v<T, PFN_cuMemcpyBatchAsync_v13000>) {
                    status = copy(reinterpret_cast<CUdeviceptr_v2*>(dst_.data()),
                                  reinterpret_cast<CUdeviceptr_v2*>(src_.data()),
                                  size_.data(),
                                  src_.size(),
                                  &attr,
                                  attr_idxs.data(),
                                  1,
                                  core::Context::stream().handle());
                }
                else {
                    status = copy(reinterpret_cast<CUdeviceptr_v2*>(dst_.data()),
                                  reinterpret_cast<CUdeviceptr_v2*>(src_.data()),
                                  size_.data(),
                                  src_.size(),
                                  &attr,
                                  attr_idxs.data(),
                                  1,
                                  &fail_idx,
                                  core::Context::stream().handle());
                }

                if (status != CUDA_SUCCESS || fail_idx != SIZE_MAX) {
                    const size_t i = fail_idx != SIZE_MAX ? fail_idx : 0;
                    TM_LOG_FATAL("copy failed: src={} size={} dst={} code={}",
                                 static_cast<const void*>(src_[i]),
                                 size_[i],
                                 static_cast<void*>(dst_[i]),
                                 static_cast<int>(status));
                }
            }
        },
        GetMemcpyBatchAsync());

    Reset();
}

}  // namespace turbomind::core
