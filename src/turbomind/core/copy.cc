
#include "src/turbomind/core/copy.h"

#include <cstdint>
#include <type_traits>
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

struct MemcpyBatchAsync {
    enum class Api { kNone, kV12080, kV13000 };

    Api                            api = Api::kNone;
    PFN_cuMemcpyBatchAsync_v12080  v12080{};
    PFN_cuMemcpyBatchAsync_v13000  v13000{};

    explicit operator bool() const noexcept
    {
        return api != Api::kNone;
    }
};

bool QueryDriverEntryPoint(const char* symbol, unsigned cuda_version, void** fpn)
{
    cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
#if CUDA_VERSION >= 13000
    if (cudaGetDriverEntryPointByVersion(symbol, fpn, cuda_version, cudaEnableDefault, &status) != cudaSuccess) {
        return false;
    }
#else
    (void)cuda_version;
    if (cudaGetDriverEntryPoint(symbol, fpn, cudaEnableDefault, &status) != cudaSuccess) {
        return false;
    }
#endif
    return status == cudaDriverEntryPointSuccess && *fpn != nullptr;
}

const MemcpyBatchAsync& GetMemcpyBatchAsync()
{
    static const MemcpyBatchAsync inst = [] {
        MemcpyBatchAsync api{};

        int runtime_version = 0;
        if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
            return api;
        }

        // cuMemcpyBatchAsync crashes on sm_100 (Blackwell); use serialized Copy().
        int device = 0;
        (void)cudaGetDevice(&device);
        int major = 0;
        (void)cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        if (major >= 10) {
            return api;
        }

        void* fpn = nullptr;

        // CUDA 13.0 removed failIdx from batched memcpy APIs; dispatch by runtime version.
        if (runtime_version >= 13000 && QueryDriverEntryPoint("cuMemcpyBatchAsync", 13000, &fpn)) {
            api.api     = MemcpyBatchAsync::Api::kV13000;
            api.v13000  = reinterpret_cast<PFN_cuMemcpyBatchAsync_v13000>(fpn);
            return api;
        }

        fpn = nullptr;
        if (QueryDriverEntryPoint("cuMemcpyBatchAsync", 12080, &fpn)) {
            api.api     = MemcpyBatchAsync::Api::kV12080;
            api.v12080  = reinterpret_cast<PFN_cuMemcpyBatchAsync_v12080>(fpn);
        }
        return api;
    }();
    return inst;
}

CUresult RunMemcpyBatchAsync(const MemcpyBatchAsync&               api,
                             CUdeviceptr_v2*                       dsts,
                             CUdeviceptr_v2*                       srcs,
                             size_t*                               sizes,
                             size_t                                count,
                             CUmemcpyAttributes_v1*                attrs,
                             size_t*                               attr_idxs,
                             size_t                                num_attrs,
                             CUstream                              stream,
                             size_t*                               fail_idx)
{
    if (api.api == MemcpyBatchAsync::Api::kV13000) {
        *fail_idx = SIZE_MAX;
        return api.v13000(dsts, srcs, sizes, count, attrs, attr_idxs, num_attrs, stream);
    }

    return api.v12080(dsts, srcs, sizes, count, attrs, attr_idxs, num_attrs, fail_idx, stream);
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

    const auto& batch = GetMemcpyBatchAsync();
    if (batch) {
        CUmemcpyAttributes_v1 attr{};
        attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
        attr.flags          = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
        std::vector<size_t> attr_idxs(src_.size(), 0);
        size_t              fail_idx{SIZE_MAX};

        const auto status = RunMemcpyBatchAsync(batch,
                                               (CUdeviceptr_v2*)dst_.data(),
                                               (CUdeviceptr_v2*)src_.data(),
                                               size_.data(),
                                               src_.size(),
                                               &attr,
                                               attr_idxs.data(),
                                               1,
                                               core::Context::stream().handle(),
                                               &fail_idx);

        if (status != CUDA_SUCCESS) {
            const size_t i = fail_idx != SIZE_MAX ? fail_idx : 0;
            TM_LOG_FATAL("copy failed: src={} size={} dst={} code={}",
                         (void*)src_[i],
                         size_[i],
                         (void*)dst_[i],
                         (int)status);
        }
        else if (fail_idx != SIZE_MAX) {
            TM_LOG_FATAL("copy failed: src={} size={} dst={} code={}",
                         (void*)src_[fail_idx],
                         size_[fail_idx],
                         (void*)dst_[fail_idx],
                         (int)status);
        }
    }
    else {
        for (unsigned i = 0; i < src_.size(); ++i) {
            core::Copy(src_[i], size_[i], dst_[i]);
        }
    }

    Reset();
}

}  // namespace turbomind::core
