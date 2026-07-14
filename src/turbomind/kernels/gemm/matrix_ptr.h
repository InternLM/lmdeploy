#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

struct __align__(16) StridedPtr
{
    void* ptr;
    int   stride;
};

struct MatrixParam {
    void* ptr;
    int   stride;
    int*  offsets;
    int*  idxs;
};

struct MatrixData {
    StridedPtr ptr;
    const int* idxs;
};

inline MatrixParam to_param(void* ptr, MatrixLayout layout)
{
    return {ptr, layout.ld, layout.offsets, layout.idxs};
}

#if 0
template<Striding mode>
__inline__ __device__ MatrixData resolve(const MatrixParam& param, int gemm_id)
{
    if constexpr (mode == Striding::kFlat) {
        return {{param.ptr, param.stride}, nullptr};
    }
    else if constexpr (mode == Striding::kBlocked) {
        StridedPtr ptr{param.ptr, param.stride};
        if (param.stride == 0) {
            (uint4&)ptr = __ldg((const uint4*)param.ptr + gemm_id);
        }
        return {ptr, nullptr};
    }
    else if constexpr (mode == Striding::kIndexed) {
        const uintptr_t idx = param.idxs ? __ldg((uintptr_t*)param.idxs + gemm_id) : 0;
        StridedPtr      ptr{param.ptr, param.stride};
        if (param.stride == 0) {
            (uint4&)ptr = __ldg((const uint4*)param.ptr + gemm_id);
        }
        return {ptr, reinterpret_cast<const int*>(idx)};
    }
    else {
        static_assert(mode != mode, "Not implemented.");
        return {};
    }
}
#endif

template<class T, Striding mode>
__inline__ __device__ MatrixData resolve(const MatrixParam& param, int g)
{
    StridedPtr ptr{param.ptr, param.stride};
    const int* idxs{};
    if constexpr (mode == Striding::kFlat) {
        // pass
    }
    else if constexpr (mode == Striding::kBlocked) {
        if (ptr.stride == 0) {
            (uint4&)ptr = __ldg((const uint4*)param.ptr + g);
        }  // Post-condition: ptr.stride != 0
        if (param.offsets) {
            ptr.ptr = (char*)ptr.ptr + __ldg(param.offsets + g) * (size_t)ptr.stride * bitsof<T> / bitsof<char>;
        }
    }
    else if constexpr (mode == Striding::kIndexed) {
        idxs = param.idxs;
        if (ptr.stride == 0) {
            (uint4&)ptr = __ldg((const uint4*)param.ptr + g);
            idxs        = idxs ? ((int**)idxs)[g] : nullptr;
        }  // Post-condition: ptr.stride != 0
        if (param.offsets) {
            const int offset = __ldg(param.offsets + g);
            if (idxs) {
                idxs += offset;
            }
            else {
                ptr.ptr = (char*)ptr.ptr + offset * (size_t)ptr.stride * bitsof<T> / bitsof<char>;
            }
        }
    }
    else {
        static_assert(mode != mode, "Not implemented.");
    }
    return {ptr, idxs};
}

// p <- dat_ptrs[g]
// i <- idx_ptrs[g]

// pitch offset idxs
//    1     0     0   -> {ptr, pitch}       , 0
//    1     0     1   -> {ptr, pitch}       , idxs
//    1     1     0   -> {ptr, pitch} + o[g], 0
//    1     1     1   -> {ptr, pitch}       , idxs + o[g]
//    0     0     0   ->       p            , 0
//    0     0     1   ->       p            , i
//    0     1     0   ->       p      + o[g], 0
//    0     1     1   ->       p            , i    + o[g]

}  // namespace turbomind::gemm
