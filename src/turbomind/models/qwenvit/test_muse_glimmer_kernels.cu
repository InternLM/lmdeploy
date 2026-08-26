// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/models/qwenvit/qwenvit_kernels.h"

using namespace turbomind;

namespace {

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err = (call);                                                                                \
        if (err != cudaSuccess) {                                                                                      \
            std::printf("[CUDA] %s failed: %s\n", #call, cudaGetErrorString(err));                                     \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

template<class T>
T* copy_to_device(const std::vector<T>& host)
{
    T* device{};
    cudaMalloc(&device, host.size() * sizeof(T));
    cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return device;
}

int test_position_embedding(const int* grid_thws, const int* grid_offsets)
{
    constexpr int grid_side = 4;
    constexpr int total_hw  = 20;
    int*          indices{};
    float*        weights{};
    CHECK_CUDA(cudaMalloc(&indices, total_hw * 4 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&weights, total_hw * 4 * sizeof(float)));

    invokeFastPosEmbedIdxWeight(
        indices, weights, DataType::kFloat32, grid_thws, grid_offsets, 2, total_hw, grid_side, true, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int>   got_indices(total_hw * 4);
    std::vector<float> got_weights(total_hw * 4);
    CHECK_CUDA(cudaMemcpy(got_indices.data(), indices, got_indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(got_weights.data(), weights, got_weights.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const int grids[][2] = {{2, 6}, {4, 2}};
    int       errors     = 0;
    int       offset     = 0;
    for (const auto& grid : grids) {
        const int h = grid[0];
        const int w = grid[1];
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                const float y              = (i + .5f) * grid_side / h - .5f;
                const float x              = (j + .5f) * grid_side / w - .5f;
                const int   y0             = (int)std::floor(y);
                const int   x0             = (int)std::floor(x);
                const int   y1             = y0 + 1;
                const int   x1             = x0 + 1;
                const float dy             = y - y0;
                const float dx             = x - x0;
                const int   ref_indices[4] = {
                    std::clamp(y0, 0, grid_side - 1) * grid_side + std::clamp(x0, 0, grid_side - 1),
                    std::clamp(y0, 0, grid_side - 1) * grid_side + std::clamp(x1, 0, grid_side - 1),
                    std::clamp(y1, 0, grid_side - 1) * grid_side + std::clamp(x0, 0, grid_side - 1),
                    std::clamp(y1, 0, grid_side - 1) * grid_side + std::clamp(x1, 0, grid_side - 1),
                };
                const float ref_weights[4] = {
                    (1 - dy) * (1 - dx) * (y0 >= 0 && y0 < grid_side) * (x0 >= 0 && x0 < grid_side),
                    (1 - dy) * dx * (y0 >= 0 && y0 < grid_side) * (x1 >= 0 && x1 < grid_side),
                    dy * (1 - dx) * (y1 >= 0 && y1 < grid_side) * (x0 >= 0 && x0 < grid_side),
                    dy * dx * (y1 >= 0 && y1 < grid_side) * (x1 >= 0 && x1 < grid_side),
                };
                const int pos = offset + i * w + j;
                for (int k = 0; k < 4; ++k) {
                    errors += got_indices[pos * 4 + k] != ref_indices[k];
                    errors += std::abs(got_weights[pos * 4 + k] - ref_weights[k]) > 1e-6f;
                }
            }
        }
        offset += h * w;
    }

    CHECK_CUDA(cudaFree(indices));
    CHECK_CUDA(cudaFree(weights));
    std::printf("[%s] zero-padded position interpolation\n", errors ? "FAIL" : "PASS");
    return errors;
}

int test_rotary_embedding(const int* grid_thws, const int* grid_offsets)
{
    constexpr int   total_hw = 20;
    constexpr int   head_dim = 96;
    constexpr float theta    = 10000.f;
    __nv_bfloat16*  output{};
    CHECK_CUDA(cudaMalloc(&output, total_hw * head_dim * sizeof(__nv_bfloat16)));
    invokeQwenVitRotaryPosEmb(
        output, DataType::kBfloat16, grid_thws, grid_offsets, 2, total_hw, head_dim, theta, true, 1, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> got(total_hw * head_dim);
    CHECK_CUDA(cudaMemcpy(got.data(), output, got.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    const int grids[][2] = {{2, 6}, {4, 2}};
    int       errors     = 0;
    int       offset     = 0;
    for (const auto& grid : grids) {
        const int h = grid[0];
        const int w = grid[1];
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                const int pos = offset + i * w + j;
                for (int pair = 0; pair < head_dim / 2; ++pair) {
                    const int   frequency     = pair % (head_dim / 4);
                    const int   coordinate    = (pair < head_dim / 4 ? j : i) + 1;
                    const float inv_frequency = std::pow(theta, -(float)frequency / (head_dim / 4));
                    const float angle         = coordinate * inv_frequency;
                    errors += std::abs(__bfloat162float(got[pos * head_dim + pair * 2]) - std::cos(angle)) > 5e-3f;
                    errors += std::abs(__bfloat162float(got[pos * head_dim + pair * 2 + 1]) - std::sin(angle)) > 5e-3f;
                }
            }
        }
        offset += h * w;
    }

    CHECK_CUDA(cudaFree(output));
    std::printf("[%s] W-first offset vision RoPE\n", errors ? "FAIL" : "PASS");
    return errors;
}

int test_pixel_shuffle(const int* grid_thws, const int* grid_offsets)
{
    constexpr int              token_num = 28;
    constexpr int              hidden    = 3;
    constexpr int              merge     = 2;
    std::vector<__nv_bfloat16> input(token_num * hidden);
    for (int i = 0; i < token_num * hidden; ++i) {
        input[i] = __float2bfloat16((float)i);
    }

    __nv_bfloat16* device_input = copy_to_device(input);
    __nv_bfloat16* device_output{};
    CHECK_CUDA(cudaMalloc(&device_output, input.size() * sizeof(__nv_bfloat16)));
    Tensor input_tensor{device_input, {token_num, hidden}, kDEVICE};
    Tensor output_tensor{device_output, {token_num / (merge * merge), hidden * merge * merge}, kDEVICE};
    invokeQwenVitPixelShuffle(output_tensor, input_tensor, grid_thws, grid_offsets, 2, merge, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> got(input.size());
    CHECK_CUDA(cudaMemcpy(got.data(), device_output, got.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    std::vector<__nv_bfloat16> expected(input.size(), __float2bfloat16(-1.f));
    const int                  grids[][3]    = {{1, 2, 6}, {2, 4, 2}};
    int                        input_offset  = 0;
    int                        output_offset = 0;
    for (const auto& grid : grids) {
        const int t = grid[0];
        const int h = grid[1];
        const int w = grid[2];
        for (int frame = 0; frame < t; ++frame) {
            for (int row = 0; row < h; ++row) {
                for (int col = 0; col < w; ++col) {
                    const int input_token = input_offset + (frame * h + row) * w + col;
                    const int output_token =
                        output_offset + (frame * (h / merge) + row / merge) * (w / merge) + col / merge;
                    const int inner = row % merge * merge + col % merge;
                    for (int dim = 0; dim < hidden; ++dim) {
                        expected[output_token * hidden * merge * merge + dim * merge * merge + inner] =
                            input[input_token * hidden + dim];
                    }
                }
            }
        }
        input_offset += t * h * w;
        output_offset += t * h * w / (merge * merge);
    }

    int errors = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        errors += __bfloat162float(got[i]) != __bfloat162float(expected[i]);
    }
    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));
    std::printf("[%s] pixel shuffle\n", errors ? "FAIL" : "PASS");
    return errors;
}

int test_logit_transform()
{
    const std::vector<float>   input{-100.f, -20.f, -1.f, 0.f, 1.f, 20.f, 100.f};
    std::vector<__nv_bfloat16> input_bf16(input.size());
    std::transform(input.begin(), input.end(), input_bf16.begin(), [](float value) { return __float2bfloat16(value); });
    __nv_bfloat16* device = copy_to_device(input_bf16);
    Tensor         logits{device, {(int)input.size()}, kDEVICE};
    invokeLogitTransform(logits, 0.19611613513818404f, 20.f, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> got(input.size());
    CHECK_CUDA(cudaMemcpy(got.data(), device, got.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    int errors = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        const float expected = 20.f * std::tanh(input[i] * 0.19611613513818404f / 20.f);
        errors += std::abs(__bfloat162float(got[i]) - expected) > 1e-1f;
    }
    CHECK_CUDA(cudaFree(device));
    std::printf("[%s] logit transform\n", errors ? "FAIL" : "PASS");
    return errors;
}

}  // namespace

int main()
{
    const std::vector<int> grid_thws{1, 2, 6, 2, 4, 2};
    const std::vector<int> grid_offsets{0, 0, 12, 12};
    int*                   device_grids   = copy_to_device(grid_thws);
    int*                   device_offsets = copy_to_device(grid_offsets);

    int errors = 0;
    errors += test_position_embedding(device_grids, device_offsets);
    errors += test_rotary_embedding(device_grids, device_offsets);
    errors += test_pixel_shuffle(device_grids, device_offsets);
    errors += test_logit_transform();

    CHECK_CUDA(cudaFree(device_grids));
    CHECK_CUDA(cudaFree(device_offsets));
    std::printf(errors ? "FAILED - %d mismatches.\n" : "All Muse-Glimmer kernel cases passed.\n", errors);
    return errors ? 1 : 0;
}
