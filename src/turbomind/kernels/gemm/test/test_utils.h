// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/macro.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "src/turbomind/core/core.h"

namespace turbomind {

template<typename T>
void Compare(const T* src,
             const T* ref,
             size_t   stride,
             int      dims,
             int      bsz,
             bool     show = false,
             float    rtol = 1e-2,
             float    atol = 1e-4);

void Compare(const void* x,
             const void* r,
             DataType    dtype,
             size_t      stride,
             int         dim,
             int         bsz,
             bool        show,
             float       rtol = 1e-2,
             float       atol = 1e-4);

template<class T>
std::vector<float> FastCompare(const T*     src,  //
                               const T*     ref,
                               int          dims,
                               int          bsz,
                               cudaStream_t stream,
                               float        rtol = 1e-2,
                               float        atol = 1e-4);

std::vector<float> FastCompare(const Tensor& x,  //
                               const Tensor& r,
                               cudaStream_t  stream,
                               float         rtol = 1e-2,
                               float         atol = 1e-4);

void FC_Header();

void FC_Print(const std::vector<float>& d);

void LoadBinary(const std::string& path, size_t size, void* dst);

class RNG {
public:
    RNG();
    ~RNG();
    void GenerateUInt(uint* out, size_t count);

    template<typename T>
    void GenerateUniform(T* out, size_t count, float scale = 1.f, float shift = 0.f);

    template<typename T>
    void GenerateNormal(T* out, size_t count, float scale = 1.f, float shift = 0.f);

    void RandomBytes(Ref<Tensor> out_);

    void UniformFloat(Ref<Tensor> out_, float scale = 1.f, float shift = 0.f);

    void NormalFloat(Ref<Tensor> out_, float scale = 1.f, float shift = 0.f);

    cudaStream_t stream() const;

    void set_stream(cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
