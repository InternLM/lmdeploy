#include <algorithm>  // std::fill_n
#include <iostream>   // snprintf
#include <math.h>     // expf, log
#include <stdlib.h>   // rand
#include <string>     // std::string
#include <vector>     // std::vector

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"
#include "src/turbomind/layers/DynamicDecodeLayer.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/constant.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

#include "gtest_utils.h"

using namespace turbomind;

namespace {

__global__ void get_curand_uniform(curandState_t* curandstate, float* output, int n)
{
    int   batch_id   = blockIdx.x;
    float rand_num   = (float)curand_uniform(curandstate + batch_id);
    output[batch_id] = rand_num;
}

template<typename T>
bool checkSorted(int  batch_size,
                 T*   expected_logits,
                 T*   output_logits,
                 int* expected_indices,
                 int* output_indices,
                 int* expected_kept,
                 int* output_kept,
                 int  vocab_size)
{
    for (int i = 0; i < batch_size; i++) {
        if (expected_kept[i] != output_kept[i]) {
            printf("batch=%d, expected_kept[i]=%d, output_kept[i]=%d\n", i, expected_kept[i], output_kept[i]);
            return false;
        }

        for (int j = 0; j < expected_kept[i]; j++) {
            int index = i * vocab_size + j;
            // soft check
            if (std::abs(expected_logits[index] - output_logits[index]) > 1e-6
                && expected_indices[index] != output_indices[index]) {
                printf("batch=%d, ith=%d, expected=(%d, %.5f), output=(%d, %.5f)\n",
                       i,
                       j,
                       expected_indices[index],
                       expected_logits[index],
                       output_indices[index],
                       output_logits[index]);
                return false;
            }
        }
    }
    return true;
}

bool checkSample(int*   expected_output_ids,
                 int*   output_ids,
                 int    batch_size,
                 float* expected_sampled_logprobs,
                 int*   expected_sampled_indices,
                 int*   expected_sampled_nums,
                 float* output_sampled_logprobs,
                 int*   output_sampled_indices,
                 int*   output_sampled_nums)
{
    for (int i = 0; i < batch_size; i++) {
        if (expected_output_ids[i] != output_ids[i]) {
            return false;
        }

        if (expected_sampled_nums[i] != output_sampled_nums[i]) {
            printf("sampled_nums, cpu=%d, gpu=%d\n", expected_sampled_nums[i], output_sampled_nums[i]);
            return false;
        }
        for (int j = 0; j < expected_sampled_nums[i]; j++) {
            int   offset  = i * kMaxLogProb + j;
            float gpu_val = output_sampled_logprobs[offset];
            float cpu_val = expected_sampled_logprobs[offset];
            int   gpu_idx = output_sampled_indices[offset];
            int   cpu_idx = expected_sampled_indices[offset];
            if (std::abs(gpu_val - cpu_val) > 1e-5) {
                if (gpu_idx != cpu_idx) {
                    printf("%d %d\n", expected_output_ids[i], output_ids[i]);
                    printf("batch=%d, ith=%d, idx cpu=%d, gpu=%d, val cpu=%.5f, gpu=%.5f\n",
                           i,
                           j,
                           cpu_idx,
                           gpu_idx,
                           cpu_val,
                           gpu_val);
                    return false;
                }
            }
        }
    }
    return true;
}

template<typename T>
void sampleCpu(int    batch_size,
               int    vocab_size,
               T*     logits,
               int*   indices,
               int*   kept,
               float* uniforms,
               int*   output_ids,
               float* sampled_logprobs,
               int*   sampled_indices,
               int*   sampled_nums)
{

    for (int i = 0; i < batch_size; i++) {
        int   selected = -1;
        float sum_val  = 0.f;
        for (int j = 0; j < kept[i]; j++) {
            sum_val += logits[i * vocab_size + j];
            if (sum_val > uniforms[i]) {
                selected      = j;
                output_ids[i] = indices[i * vocab_size + j];
                break;
            }
        }

        if (sampled_logprobs && sampled_indices && sampled_nums) {
            for (int j = 0; j < min(kept[i], kMaxLogProb); ++j) {
                sampled_logprobs[i * kMaxLogProb + j] = std::log(logits[i * vocab_size + j]);
                sampled_indices[i * kMaxLogProb + j]  = indices[i * vocab_size + j];
            }
            if (kept[i] > kMaxLogProb && selected >= kMaxLogProb) {
                sampled_logprobs[i * kMaxLogProb + kMaxLogProb - 1] = std::log(logits[i * vocab_size + selected]);
                sampled_indices[i * kMaxLogProb + kMaxLogProb - 1]  = indices[i * vocab_size + selected];
            }
            sampled_nums[i] = min(kept[i], kMaxLogProb);
        }
    }
}

template<typename T>
void softmax(T* input, int batch_size, int vocab_size, int* kept, T* output)
{
    for (int i = 0; i < batch_size; i++) {
        int offset  = i * vocab_size;
        T   max_val = input[offset];
        for (int j = 0; j < kept[i]; j++) {
            max_val = std::max(input[offset + j], max_val);
        }
        T sum_val{};
        for (int j = 0; j < kept[i]; j++) {
            output[offset + j] = std::exp((float)input[offset + j] - max_val);
            sum_val += output[offset + j];
        }
        for (int j = 0; j < kept[i]; j++) {
            output[offset + j] /= sum_val;
        }
    }
}

template<typename T>
void filterCpu(int    batch_size,
               int*   top_ks,
               float* top_ps,
               float* min_ps,
               T*     logits,
               T*     sorted_logits,
               int*   sorted_indices,
               int*   kept,
               int    vocab_size,
               bool   filter_topp = false,
               bool   filter_minp = false)
{
    for (int i = 0; i < batch_size; i++) {
        // fill
        std::vector<std::pair<T, int>> work(vocab_size);
        for (int j = 0; j < vocab_size; j++) {
            work[j] = {logits[i * vocab_size + j], j};
        }

        // sort
        if (top_ks && top_ks[i] != 0) {
            std::partial_sort(work.begin(), work.begin() + top_ks[i], work.end(), std::greater{});
            kept[i] = top_ks[i];
        }
        else {
            std::sort(work.begin(), work.end(), std::greater{});
            kept[i] = vocab_size;
        }
        for (int j = 0; j < kept[i]; j++) {
            sorted_logits[i * vocab_size + j]  = work[j].first;
            sorted_indices[i * vocab_size + j] = work[j].second;
        }
        // softmax
        softmax(sorted_logits + i * vocab_size, 1, vocab_size, kept + i, sorted_logits + i * vocab_size);
        if (top_ks && top_ks[i] == 0) {
            if (top_ps && sorted_logits[i * vocab_size] > top_ps[i]) {
                sorted_logits[i * vocab_size] = 1.f;
                kept[i]                       = 1;
            }
        }

        // topp filter
        if (filter_topp && top_ps[i] != 1.f) {
            float topp    = top_ps[i];
            float sum_val = 0;
            int   n       = kept[i];
            for (int j = 0; j < kept[i]; j++) {
                sum_val += sorted_logits[i * vocab_size + j];
                if (sum_val > topp) {
                    n = j + 1;
                    break;
                }
            }
            if (n != kept[i]) {
                kept[i] = n;
                for (int j = 0; j < n; j++) {
                    sorted_logits[i * vocab_size + j] /= (sum_val + 1e-6f);
                }
            }
        }

        // minp filter
        if (filter_minp && min_ps[i] != 0.f) {
            float minp      = min_ps[i];
            float threshold = sorted_logits[i * vocab_size] * minp;
            float sum_val   = 0;
            int   n         = kept[i];
            for (int j = 0; j < kept[i]; j++) {
                if (sorted_logits[i * vocab_size + j] < threshold) {
                    n = j;
                    break;
                }
                sum_val += sorted_logits[i * vocab_size + j];
            }
            if (n != kept[i]) {
                kept[i] = n;
                for (int j = 0; j < n; j++) {
                    sorted_logits[i * vocab_size + j] /= (sum_val + 1e-6f);
                }
            }
        }
    }
}

template<typename T>
class SamplingKernelTest: public testing::Test {
public:
    void SetUp() override
    {
        check_cuda_error(cudaStreamCreate(&stream));
        allocator = new Allocator<AllocatorType::CUDA>(getDevice());
        allocator->setStream(stream);
    }
    void TearDown() override
    {
        delete allocator;
        check_cuda_error(cudaStreamDestroy(stream));
    }

protected:
    cudaStream_t                    stream;
    Allocator<AllocatorType::CUDA>* allocator;
    curandState_t*                  curand_states;
};

template<typename T>
class TopKTopPSortTest: public SamplingKernelTest<T> {
protected:
    using SamplingKernelTest<T>::stream;
    using SamplingKernelTest<T>::allocator;

public:
    void runTest(int batch_size, int* top_ks, float* top_ps, int vocab_size)
    {

        TopKSortFilterParams params1{};
        params1.batch_size = batch_size;
        int max_top_k      = *std::max_element(top_ks, top_ks + batch_size);
        params1.max_top_k  = std::min(1024, std::max(0, max_top_k));
        invokeTopKSortFilter<T>(params1, stream);

        TopPSortParams params2{};
        params2.batch_size        = batch_size;
        params2.vocab_size        = vocab_size;
        params2.vocab_size_padded = vocab_size;
        invokeTopPSort<T>(params2, stream);

        // host buffer
        std::vector<T>   logits(batch_size * vocab_size);
        std::vector<T>   expected_logits(batch_size * vocab_size);
        std::vector<int> expected_indices(batch_size * vocab_size);
        std::vector<int> expected_kept(batch_size);

        std::vector<T>   output_logits(batch_size * vocab_size);
        std::vector<int> output_indices(batch_size * vocab_size);
        std::vector<int> output_kept(batch_size);

        // device buffer
        void*  d_ws_topk        = allocator->malloc(params1.workspace_size);
        void*  d_ws_topp        = allocator->malloc(params2.workspace_size);
        T*     d_logits         = (T*)allocator->malloc(sizeof(T) * batch_size * vocab_size);
        T*     d_sorted_logits  = (T*)allocator->malloc(sizeof(T) * batch_size * vocab_size);
        int*   d_sorted_indices = (int*)allocator->malloc(sizeof(int) * batch_size * vocab_size);
        int*   d_kept           = (int*)allocator->malloc(sizeof(int) * batch_size);
        int*   d_top_ks         = (int*)allocator->malloc(sizeof(int) * batch_size);
        float* d_top_ps         = (float*)allocator->malloc(sizeof(float) * batch_size);

        initRandom(logits.data(), batch_size * vocab_size, -200.0f, 200.0f);

        std::fill_n(expected_kept.data(), batch_size, vocab_size);

        cudaAutoCpy(d_logits, logits.data(), batch_size * vocab_size, stream);
        cudaAutoCpy(d_top_ps, top_ps, batch_size, stream);
        cudaAutoCpy(d_top_ks, top_ks, batch_size, stream);
        cudaAutoCpy(d_kept, expected_kept.data(), batch_size, stream);

        // gpu
        params1.workspace         = d_ws_topk;
        params1.logits            = d_logits;
        params1.sorted_logits     = d_sorted_logits;
        params1.sorted_indices    = d_sorted_indices;
        params1.kept              = d_kept;
        params1.top_ks            = d_top_ks;
        params1.vocab_size        = vocab_size;
        params1.vocab_size_padded = vocab_size;
        invokeTopKSortFilter<T>(params1, stream);

        invokeSoftmax<T>(d_logits, vocab_size, vocab_size, batch_size, d_kept, stream);
        params2.workspace      = d_ws_topp;
        params2.logits         = d_logits;
        params2.sorted_logits  = d_sorted_logits;
        params2.sorted_indices = d_sorted_indices;
        params2.kept           = d_kept;
        params2.top_ks         = d_top_ks;
        params2.top_ps         = d_top_ps;
        invokeTopPSort<T>(params2, stream);

        // outputs
        cudaAutoCpy(output_logits.data(), d_sorted_logits, batch_size * vocab_size);
        cudaAutoCpy(output_indices.data(), d_sorted_indices, batch_size * vocab_size);
        cudaAutoCpy(output_kept.data(), d_kept, batch_size, stream);
        cudaStreamSynchronize(stream);

        // cpu
        filterCpu(batch_size,
                  top_ks,
                  top_ps,
                  nullptr,
                  logits.data(),
                  expected_logits.data(),
                  expected_indices.data(),
                  expected_kept.data(),
                  vocab_size);

        EXPECT_TRUE(checkSorted(batch_size,
                                expected_logits.data(),
                                output_logits.data(),
                                expected_indices.data(),
                                output_indices.data(),
                                expected_kept.data(),
                                output_kept.data(),
                                vocab_size));
    }
};

TYPED_TEST_SUITE(TopKTopPSortTest, FloatType);

TYPED_TEST(TopKTopPSortTest, OnlyTopKBatch)
{
    int   top_ks[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float top_ps[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    this->runTest(sizeof(top_ks) / sizeof(int), top_ks, top_ps, 20);
};

TYPED_TEST(TopKTopPSortTest, OnlyTopKLargeVocab)
{
    int   top_ks[] = {1, 2, 4, 8, 16, 32, 64, 1024};
    float top_ps[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    this->runTest(sizeof(top_ks) / sizeof(int), top_ks, top_ps, 32000);
};

TYPED_TEST(TopKTopPSortTest, OnlyTopPBatch)
{
    int   top_ks[] = {0, 0, 0, 0, 0, 0, 0, 0};
    float top_ps[] = {0.0f, 0.1f, 0.3f, 0.4f, 0.5f, 0.7f, 0.9f, 1.0f};
    this->runTest(sizeof(top_ks) / sizeof(int), top_ks, top_ps, 20);
};

TYPED_TEST(TopKTopPSortTest, OnlyTopPLargeVocab)
{
    int   top_ks[] = {0, 0, 0, 0, 0, 0, 0, 0};
    float top_ps[] = {0.0f, 0.1f, 0.3f, 0.4f, 0.5f, 0.7f, 0.9f, 1.0f};
    this->runTest(sizeof(top_ks) / sizeof(int), top_ks, top_ps, 32000);
};

TYPED_TEST(TopKTopPSortTest, MixedTopKTopP)
{
    int   top_ks[] = {1, 0, 16, 0, 32, 0, 64, 1024};
    float top_ps[] = {0.0f, 0.1f, 0.0f, 0.4f, 0.5f, 0.7f, 0.9f, 1.0f};
    this->runTest(sizeof(top_ks) / sizeof(int), top_ks, top_ps, 32000);
};

template<typename T>
class TopPMinPFilterTest: public SamplingKernelTest<T> {
protected:
    using SamplingKernelTest<T>::stream;
    using SamplingKernelTest<T>::allocator;

public:
    void runTest(int batch_size, float* top_ps, float* min_ps, int vocab_size)
    {

        // host buffer
        std::vector<T>   logits(batch_size * vocab_size);
        std::vector<T>   expected_logits(batch_size * vocab_size);
        std::vector<int> expected_indices(batch_size * vocab_size);
        std::vector<int> expected_kept(batch_size);

        std::vector<T>   output_logits(batch_size * vocab_size);
        std::vector<int> output_indices(batch_size * vocab_size);
        std::vector<int> output_kept(batch_size);

        // device buffer
        T*     d_sorted_logits  = (T*)allocator->malloc(sizeof(T) * batch_size * vocab_size);
        int*   d_sorted_indices = (int*)allocator->malloc(sizeof(int) * batch_size * vocab_size);
        int*   d_kept           = (int*)allocator->malloc(sizeof(int) * batch_size);
        float* d_top_ps         = (float*)allocator->malloc(sizeof(float) * batch_size);
        float* d_min_ps         = (float*)allocator->malloc(sizeof(float) * batch_size);

        initRandom(logits.data(), batch_size * vocab_size, -200.0f, 200.0f);
        std::fill_n(expected_kept.data(), batch_size, vocab_size);

        filterCpu(batch_size,
                  nullptr,
                  top_ps,
                  min_ps,
                  logits.data(),
                  expected_logits.data(),
                  expected_indices.data(),
                  expected_kept.data(),
                  vocab_size);

        cudaAutoCpy(d_sorted_logits, expected_logits.data(), batch_size * vocab_size);
        cudaAutoCpy(d_sorted_indices, expected_indices.data(), batch_size * vocab_size);
        cudaAutoCpy(d_kept, expected_kept.data(), batch_size, stream);
        cudaAutoCpy(d_top_ps, top_ps, batch_size, stream);
        cudaAutoCpy(d_min_ps, min_ps, batch_size, stream);

        TopPMinPFilterParams params{};
        params.sorted_logits     = d_sorted_logits;
        params.sorted_indices    = d_sorted_indices;
        params.kept              = d_kept;
        params.top_ps            = d_top_ps;
        params.min_ps            = d_min_ps;
        params.batch_size        = batch_size;
        params.vocab_size        = vocab_size;
        params.vocab_size_padded = vocab_size;
        invokeTopPMinPFilter<T>(params, stream);
        cudaStreamSynchronize(stream);

        // outputs
        cudaAutoCpy(output_logits.data(), d_sorted_logits, batch_size * vocab_size);
        cudaAutoCpy(output_indices.data(), d_sorted_indices, batch_size * vocab_size);
        cudaAutoCpy(output_kept.data(), d_kept, batch_size, stream);
        cudaStreamSynchronize(stream);

        // cpu
        filterCpu(batch_size,
                  nullptr,
                  top_ps,
                  min_ps,
                  logits.data(),
                  expected_logits.data(),
                  expected_indices.data(),
                  expected_kept.data(),
                  vocab_size,
                  true,
                  true);

        EXPECT_TRUE(checkSorted(batch_size,
                                expected_logits.data(),
                                output_logits.data(),
                                expected_indices.data(),
                                output_indices.data(),
                                expected_kept.data(),
                                output_kept.data(),
                                vocab_size));
    }
};

TYPED_TEST_SUITE(TopPMinPFilterTest, FloatType);

TYPED_TEST(TopPMinPFilterTest, OnlyTopP)
{
    float top_ps[] = {0.8f, 0.82f, 0.84f, 0.86f, 0.88f, 0.90f, 0.92f, 0.94f, 0.96f, 0.98f, 1.0f};
    float min_ps[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    this->runTest(sizeof(top_ps) / sizeof(float), top_ps, min_ps, 200);
};

TYPED_TEST(TopPMinPFilterTest, OnlyMinP)
{
    float min_ps[] = {0.0f, 0.002f, 0.004f, 0.006f, 0.008f, 0.01f, 0.012f, 0.014f, 0.016f, 0.018f, 0.02f};
    float top_ps[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    this->runTest(sizeof(top_ps) / sizeof(float), top_ps, min_ps, 200);
};

TYPED_TEST(TopPMinPFilterTest, MixedTopPMinP)
{
    float min_ps[] = {0.0f, 0.002f, 0.004f, 0.006f, 0.008f, 0.01f, 0.012f, 0.014f, 0.016f, 0.018f, 0.02f};
    float top_ps[] = {0.8f, 0.82f, 0.84f, 0.86f, 0.88f, 0.90f, 0.92f, 0.94f, 0.96f, 0.98f, 1.0f};
    this->runTest(sizeof(top_ps) / sizeof(float), top_ps, min_ps, 200);
};

template<typename T>
class SamplingTest: public SamplingKernelTest<T> {
protected:
    using SamplingKernelTest<T>::stream;
    using SamplingKernelTest<T>::allocator;

public:
    void runTest(int batch_size, int vocab_size, int top_logprobs)
    {

        // host buffer
        std::vector<T>     logits(batch_size * vocab_size);
        std::vector<T>     expected_logits(batch_size * vocab_size);
        std::vector<int>   expected_indices(batch_size * vocab_size);
        std::vector<int>   expected_kept(batch_size);
        std::vector<int>   expected_output_ids(batch_size);
        std::vector<float> uniforms(batch_size);

        std::vector<float> sampled_logprobs(batch_size * kMaxLogProb);
        std::vector<int>   sampled_indexes(batch_size * kMaxLogProb);
        std::vector<int>   sampled_nums(batch_size);

        // std::vector<T>     output_logits(batch_size * vocab_size);
        // std::vector<int>   output_indices(batch_size * vocab_size);
        // std::vector<int>   output_kept(batch_size);
        std::vector<int>   output_ids(batch_size);
        std::vector<float> output_sampled_logprobs(batch_size * kMaxLogProb);
        std::vector<int>   output_sampled_indexes(batch_size * kMaxLogProb);
        std::vector<int>   output_sampled_nums(batch_size);

        // device buffer
        T*             d_sorted_logits    = (T*)allocator->malloc(sizeof(T) * batch_size * vocab_size);
        int*           d_sorted_indices   = (int*)allocator->malloc(sizeof(int) * batch_size * vocab_size);
        int*           d_kept             = (int*)allocator->malloc(sizeof(int) * batch_size);
        float*         d_top_ps           = (float*)allocator->malloc(sizeof(float) * batch_size);
        float*         d_min_ps           = (float*)allocator->malloc(sizeof(float) * batch_size);
        float*         d_uniforms         = (float*)(allocator->malloc(sizeof(float) * batch_size));
        int*           d_output_ids       = (int*)(allocator->malloc(sizeof(int) * batch_size));
        float*         d_sampled_logprobs = (float*)(allocator->malloc(sizeof(float) * batch_size * kMaxLogProb));
        int*           d_sampled_indexes  = (int*)(allocator->malloc(sizeof(int) * batch_size * kMaxLogProb));
        int*           d_sampled_nums     = (int*)(allocator->malloc(sizeof(int) * batch_size));
        curandState_t* curand_states =
            reinterpret_cast<curandState_t*>(allocator->malloc(sizeof(curandState_t) * batch_size, false));

        initRandom(logits.data(), batch_size * vocab_size, -3.0f, 3.0f);
        std::fill_n(expected_kept.data(), batch_size, vocab_size);

        // sort & softmax
        filterCpu(batch_size,
                  nullptr,
                  nullptr,
                  nullptr,
                  logits.data(),
                  expected_logits.data(),
                  expected_indices.data(),
                  expected_kept.data(),
                  vocab_size);

        cudaAutoCpy(d_sorted_logits, expected_logits.data(), batch_size * vocab_size);
        cudaAutoCpy(d_sorted_indices, expected_indices.data(), batch_size * vocab_size);
        cudaAutoCpy(d_kept, expected_kept.data(), batch_size, stream);

        // uniforms
        for (int i = 0; i < batch_size; i++) {
            invokeCurandInitialize(curand_states + i, 1, i, stream);
        }
        get_curand_uniform<<<batch_size, 1, 0, stream>>>(curand_states, d_uniforms, batch_size);
        cudaAutoCpy(uniforms.data(), d_uniforms, batch_size, stream);
        for (int i = 0; i < batch_size; i++) {
            invokeCurandInitialize(curand_states + i, 1, i, stream);
        }

        // sample
        SamplingParams params{};
        params.logits           = d_sorted_logits;
        params.stride           = vocab_size;
        params.indices          = d_sorted_indices;
        params.kept             = d_kept;
        params.curandstate      = curand_states;
        params.batch_size       = batch_size;
        params.output_ids       = d_output_ids;
        params.sequence_length  = nullptr;
        params.sampled_logprobs = d_sampled_logprobs;
        params.sampled_indexes  = (uint32_t*)d_sampled_indexes;
        params.sampled_nums     = (uint32_t*)d_sampled_nums;
        invokeSampling<T>(params, stream);

        // outputs
        cudaAutoCpy(output_ids.data(), d_output_ids, batch_size, stream);
        cudaAutoCpy(output_sampled_logprobs.data(), d_sampled_logprobs, batch_size * kMaxLogProb, stream);
        cudaAutoCpy(output_sampled_indexes.data(), d_sampled_indexes, batch_size * kMaxLogProb, stream);
        cudaAutoCpy(output_sampled_nums.data(), d_sampled_nums, batch_size, stream);
        cudaStreamSynchronize(stream);

        sampleCpu(batch_size,
                  vocab_size,
                  expected_logits.data(),
                  expected_indices.data(),
                  expected_kept.data(),
                  uniforms.data(),
                  expected_output_ids.data(),
                  sampled_logprobs.data(),
                  sampled_indexes.data(),
                  sampled_nums.data());

        EXPECT_TRUE(checkSample(expected_output_ids.data(),
                                output_ids.data(),
                                batch_size,
                                sampled_logprobs.data(),
                                sampled_indexes.data(),
                                sampled_nums.data(),
                                output_sampled_logprobs.data(),
                                output_sampled_indexes.data(),
                                output_sampled_nums.data()));
    }
};

TYPED_TEST_SUITE(SamplingTest, FloatType);

TYPED_TEST(SamplingTest, Single)
{
    this->runTest(1, 20, 5);
};

TYPED_TEST(SamplingTest, Batch)
{
    this->runTest(32, 9700, 1024);
};

}  // end of namespace
