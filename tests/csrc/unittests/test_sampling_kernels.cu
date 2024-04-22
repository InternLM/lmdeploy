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

#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"
#include "src/turbomind/layers/DynamicDecodeLayer.h"
#include "src/turbomind/layers/sampling_layers/TopKSamplingLayer.h"
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

struct SamplingKernelTestParam {
    size_t batch_size;
    size_t vocab_size;
    size_t beam_width;
    uint   top_k;
    float  top_p;
    size_t output_len;

    std::string toString()
    {
        return fmtstr("SamplingKernelTestParam[batch=%ld, vocab=%ld, beam=%ld, k=%u, p=%3.1f, output_len=%ld]",
                      batch_size,
                      vocab_size,
                      beam_width,
                      top_k,
                      top_p,
                      output_len);
    }
};

/////////////////////////////////// Tests //////////////////////////////////////////

template<typename T>
void computeProb(T* probs, T* logits, int batch_size, int vocab_size)
{
    // Compute the log probability from logits.
    //   logits = batch_size x vocab_size.
    //   probs =  softmax(logits) (softmax along with vocab dimension)
    // float is used for either T=float or half, since operations of half are
    // not fully supported in a host function.
    for (int bidx = 0; bidx < batch_size; ++bidx) {
        float maxval = -FLT_MAX;
        for (int i = 0; i < vocab_size; ++i) {
            float logit = static_cast<float>(logits[bidx * vocab_size + i]);
            if (logit > maxval) {
                maxval = logit;
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            sum += expf(static_cast<float>(logits[bidx * vocab_size + i]) - maxval);
        }
        for (int i = 0; i < vocab_size; ++i) {
            int   idx   = bidx * vocab_size + i;
            float logit = static_cast<float>(logits[idx]) - maxval;
            probs[idx]  = static_cast<T>(expf(logit) / (sum + EPSILON));
        }
    }
}

template<typename T>
void computeLogProb(T* logprobs, T* logits, int batch_size, int vocab_size)
{
    // Compute the log probability from logits.
    //   logits = batch_size x vocab_size.
    //   logprobs = log(softmax(logits)) (softmax along with vocab dimension)
    // float is used for either T=float or half, since operations of half are
    // not fully supported in a host function.
    for (int bidx = 0; bidx < batch_size; ++bidx) {
        float maxval = -FLT_MAX;
        for (int i = 0; i < vocab_size; ++i) {
            float logit = static_cast<float>(logits[bidx * vocab_size + i]);
            if (logit > maxval) {
                maxval = logit;
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            sum += expf(static_cast<float>(logits[bidx * vocab_size + i]) - maxval);
        }
        for (int i = 0; i < vocab_size; ++i) {
            int   idx     = bidx * vocab_size + i;
            float logit   = static_cast<float>(logits[idx]) - maxval;
            logprobs[idx] = static_cast<T>(logit - logf(sum + EPSILON));
        }
    }
}

template<typename T>
void computeSampledLogProb(T*     logits,
                           int*   top_ks,
                           float* top_ps,
                           float* uniform,
                           float* sampled_logprobs,
                           int*   sampled_indexes,
                           int*   sampled_nums,
                           int    batch_size,
                           int    vocab_size)
{
    for (int i = 0; i < batch_size; i++) {
        // select top k
        typedef std::pair<float, int> item;
        std::vector<item>             items;
        for (int j = 0; j < vocab_size; j++) {
            items.emplace_back(static_cast<float>(logits[j]), j);
        }

        int sort_k = (top_ks[i] > 0) ? top_ks[i] : vocab_size;
        std::partial_sort(items.begin(), items.begin() + sort_k, items.end(), [](item a, item b) { return a > b; });

        // calculate cum expf
        float sum_v = 0.f;
        float max_v = items[0].first;
        for (int j = 0; j < sort_k; j++) {
            float v        = expf(items[j].first - max_v);
            items[j].first = v;
            sum_v += v;
        }

        // filter top_p
        int   sampled_n   = sort_k;
        float sampled_sum = 0.0;
        for (int j = 0; j < sort_k; j++) {
            sampled_sum += items[j].first / sum_v;
            if (sampled_sum >= top_ps[i]) {
                sampled_n = j + 1;
                break;
            }
        }
        sampled_n = std::min(sampled_n, kMaxLogProb);

        // output
        sampled_nums[i] = sampled_n;
        for (int j = 0; j < sampled_n; j++) {
            sampled_logprobs[j] = logf(items[j].first) - logf(sampled_sum * sum_v);
            sampled_indexes[j]  = items[j].second;
        }

        logits += vocab_size;
        sampled_logprobs += kMaxLogProb;
        sampled_indexes += kMaxLogProb;
    }
}

bool checkSampledLogProb(int*   nums,
                         float* probs,
                         int*   indexes,
                         int*   expected_nums,
                         float* expected_probs,
                         int*   expected_indexes,
                         int    batch_size,
                         float  eps = 1e-5)
{
    for (int i = 0; i < batch_size; i++) {
        if (nums[i] != expected_nums[i]) {
            printf("nums, b=%d, pred=%d, expect=%d\n", i, nums[i], expected_nums[i]);
            return false;
        }
        for (int j = 0; j < nums[i]; j++) {
            if (indexes[j] != expected_indexes[j]) {
                printf("indexes, b=%d, j=%d, pred=%d, expect=%d\n", i, j, indexes[j], expected_indexes[j]);
                return false;
            }
        }
        for (int j = 0; j < nums[i]; j++) {
            if (fabs(probs[j] - expected_probs[j]) > eps) {
                printf("probs, b=%d, j=%d, pred=%f, expect=%f\n", i, j, probs[j], expected_probs[j]);
                return false;
            }
        }
        probs += kMaxLogProb;
        expected_probs += kMaxLogProb;
        indexes += kMaxLogProb;
        expected_indexes += kMaxLogProb;
        indexes++;
        expected_indexes++;
    }
    return true;
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
    unsigned long long              seed = 0;
    cudaStream_t                    stream;
    Allocator<AllocatorType::CUDA>* allocator;
    curandState_t*                  curand_states;
};

template<typename T>
class TopKSamplingKernelTest: public SamplingKernelTest<T> {

protected:
    const int end_id = 0;
    using SamplingKernelTest<T>::seed;
    using SamplingKernelTest<T>::stream;
    using SamplingKernelTest<T>::allocator;
    using SamplingKernelTest<T>::curand_states;

public:
    void runTest(SamplingKernelTestParam param)
    {
        size_t batch_size  = param.batch_size;
        size_t vocab_size  = param.vocab_size;
        size_t output_len  = param.output_len;
        size_t max_seq_len = output_len;

        uint  top_k = param.top_k;
        float top_p = param.top_p;

        // Logit values in the host of shape (batch_size x vocab_size).
        T* h_logits = new T[batch_size * vocab_size];
        T* h_probs  = new T[batch_size * vocab_size];
        T* h_lprobs = new T[batch_size * vocab_size];

        int*  h_output_ids  = new int[batch_size];
        int*  h_seq_lengths = new int[batch_size];
        bool* h_finished    = new bool[batch_size];

        float* expected_cum_lprobs = new float[batch_size];
        std::fill_n(expected_cum_lprobs, batch_size, 0);

        curandState_t* curand_states =
            reinterpret_cast<curandState_t*>(allocator->malloc(sizeof(curandState_t) * batch_size, false));
        invokeCurandInitialize(curand_states, batch_size, seed, stream);

        size_t workspace_size = 0;
        // retrieve the workspace size of the top-k sampling kernel.
        invokeTopKSampling<T>(nullptr,
                              workspace_size,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              top_k,
                              1.0f,
                              vocab_size,
                              nullptr,
                              stream,
                              batch_size,
                              nullptr);
        void* workspace = allocator->malloc(workspace_size);

        int*  end_ids     = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        int*  seq_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        bool* finished    = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));

        T*     probs         = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size));
        float* cum_lprobs    = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
        float* output_lprobs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * output_len * batch_size));
        int*   output_ids    = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batch_size));

        // Init by zero.
        deviceFill(seq_lengths, batch_size, 0);
        deviceFill(finished, batch_size, false);
        deviceFill(end_ids, batch_size, end_id);

        deviceFill(cum_lprobs, batch_size, 0.0f);
        deviceFill(output_lprobs, output_len * batch_size, 0.0f);
        deviceFill(output_ids, max_seq_len * batch_size, 0);

        for (size_t step = 0; step < output_len; ++step) {
            initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);
            computeProb(h_probs, h_logits, batch_size, vocab_size);
            cudaH2Dcpy(probs, h_probs, batch_size * vocab_size);
            invokeTopKSampling(workspace,
                               workspace_size,
                               // Note that the kernel needs vocab probs instead of
                               // log-prob if cum_log_probs or output_log_probs are
                               // provided. It's because the sampling layer already
                               // preprocesses log_prob_buf when those are provided.
                               probs,
                               output_ids + step * batch_size,
                               seq_lengths,
                               finished,
                               cum_lprobs,
                               output_lprobs + step * batch_size,
                               curand_states,
                               top_k,
                               top_p,
                               vocab_size,
                               end_ids,
                               stream,
                               batch_size,
                               nullptr);

            // Compute reference.
            cudaD2Hcpy(h_output_ids, output_ids + step * batch_size, batch_size);
            cudaD2Hcpy(h_seq_lengths, seq_lengths, batch_size);
            cudaD2Hcpy(h_finished, finished, batch_size);
            computeLogProb(h_lprobs, h_logits, batch_size, vocab_size);
            for (size_t i = 0; i < batch_size; ++i) {
                int idx = i * vocab_size + h_output_ids[i];
                expected_cum_lprobs[i] += (int)step < h_seq_lengths[i] ? (float)h_lprobs[idx] : 0.0f;
                EXPECT_EQ(h_finished[i], h_output_ids[i] == end_id);
            }
        }
        bool passed = checkResult(param.toString(), cum_lprobs, expected_cum_lprobs, batch_size);
        EXPECT_TRUE(passed);

        delete[] expected_cum_lprobs;
        delete[] h_seq_lengths;
        delete[] h_logits;
        delete[] h_lprobs;
        delete[] h_probs;
        delete[] h_output_ids;
    }

    void runBatchTest(SamplingKernelTestParam param, bool has_diff_runtime_args, bool use_skip_decode)
    {
        size_t batch_size = param.batch_size;
        size_t vocab_size = param.vocab_size;
        size_t output_len = param.output_len;
        size_t seq_len    = output_len;

        int   top_k = param.top_k;
        float top_p = param.top_p;

        int*   h_top_ks = new int[batch_size];
        float* h_top_ps = new float[batch_size];
        for (size_t i = 0; i < batch_size; ++i) {
            h_top_ks[i] = (!has_diff_runtime_args || i % 3 == 0) ? top_k : 1;
            h_top_ps[i] = (!has_diff_runtime_args || i % 3 == 0) ? top_p : 0.1 * top_p;
        }
        int max_top_k = *std::max_element(h_top_ks, h_top_ks + batch_size);

        // Logit values in the host of shape (batch_size x vocab_size).
        T* h_logits = new T[batch_size * vocab_size];
        T* h_probs  = new T[batch_size * vocab_size];
        T* h_lprobs = new T[batch_size * vocab_size];

        float* expected_cum_lprobs = new float[batch_size];

        int*  h_output_ids  = new int[batch_size];
        int*  h_seq_lengths = new int[batch_size];
        bool* h_finished    = new bool[batch_size];
        bool* h_skip_decode = new bool[batch_size];

        initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);
        std::fill_n(expected_cum_lprobs, batch_size, 0);
        for (size_t i = 0; i < batch_size; ++i) {
            h_skip_decode[i] = use_skip_decode && (i % 2 == 0);
        }

        curandState_t* curand_states =
            reinterpret_cast<curandState_t*>(allocator->malloc(sizeof(curandState_t) * batch_size, false));
        invokeCurandInitialize(curand_states, batch_size, seed, stream);

        size_t workspace_size = 0;
        // retrieve the workspace size of the top-k sampling kernel.
        invokeBatchTopKSampling<T>(nullptr,  // workspace
                                   workspace_size,
                                   nullptr,  // log_probs
                                   nullptr,  // ids
                                   nullptr,  // sequence_lengths
                                   nullptr,  // finished
                                   nullptr,  // cum_log_probs
                                   nullptr,  // output_log_probs
                                   nullptr,  // sampled_logprobs
                                   nullptr,  // sampled_indexes
                                   nullptr,  // sampled_nums
                                   nullptr,  // curandstates
                                   max_top_k,
                                   nullptr,  // top_ks
                                   1.0f,
                                   nullptr,
                                   vocab_size,
                                   nullptr,  // end_ids
                                   stream,
                                   batch_size,
                                   nullptr);
        void* workspace = allocator->malloc(workspace_size, false);

        int*   top_ks = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        float* top_ps = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));

        int*  end_ids     = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        int*  seq_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        int*  output_ids  = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * seq_len * batch_size));
        bool* finished    = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));
        bool* skip_decode = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));

        T*     probs         = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size, true));
        float* cum_lprobs    = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
        float* output_lprobs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * output_len * batch_size));

        // Initialize.
        cudaH2Dcpy(top_ks, h_top_ks, batch_size);
        cudaH2Dcpy(top_ps, h_top_ps, batch_size);
        cudaH2Dcpy(skip_decode, h_skip_decode, batch_size);

        deviceFill(end_ids, batch_size, end_id);
        deviceFill(seq_lengths, batch_size, 0);
        deviceFill(finished, batch_size, false);
        deviceFill(cum_lprobs, batch_size, 0.0f);
        deviceFill(output_lprobs, output_len * batch_size, 0.0f);
        deviceFill(output_ids, seq_len * batch_size, 0);

        for (size_t step = 0; step < output_len; ++step) {
            initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);
            computeProb(h_probs, h_logits, batch_size, vocab_size);
            cudaH2Dcpy(probs, h_probs, batch_size * vocab_size);

            invokeBatchTopKSampling(workspace,
                                    workspace_size,
                                    // Note that the kernel needs vocab probs instead of
                                    // log-prob if cum_log_probs or output_log_probs are
                                    // provided. It's because the sampling layer already
                                    // preprocesses log_prob_buf when those are provided.
                                    probs,
                                    output_ids + step * batch_size,
                                    seq_lengths,
                                    finished,
                                    cum_lprobs,
                                    output_lprobs + step * batch_size,
                                    nullptr,  // sampled_logprobs
                                    nullptr,  // sampled_indexes
                                    nullptr,  // sampled_nums
                                    curand_states,
                                    max_top_k,
                                    top_ks,
                                    1.0f,
                                    nullptr,
                                    vocab_size,
                                    end_ids,
                                    stream,
                                    batch_size,
                                    skip_decode);

            // Compute reference.
            cudaD2Hcpy(h_output_ids, output_ids + step * batch_size, batch_size);
            cudaD2Hcpy(h_seq_lengths, seq_lengths, batch_size);
            cudaD2Hcpy(h_finished, finished, batch_size);
            computeLogProb(h_lprobs, h_logits, batch_size, vocab_size);
            for (size_t i = 0; i < batch_size; ++i) {
                if (!h_skip_decode[i]) {
                    int idx = i * vocab_size + h_output_ids[i];
                    expected_cum_lprobs[i] += (int)step < h_seq_lengths[i] ? (float)h_lprobs[idx] : 0.0f;
                    EXPECT_EQ(h_finished[i], h_output_ids[i] == end_id);
                }
            }
        }
        bool passed = checkResult(param.toString(), cum_lprobs, expected_cum_lprobs, batch_size);
        EXPECT_TRUE(passed) << "Fail subtest (has_diff_runtime_args: " << has_diff_runtime_args
                            << ", skip_decode: " << use_skip_decode << ")";

        delete[] expected_cum_lprobs;
        delete[] h_seq_lengths;
        delete[] h_logits;
        delete[] h_lprobs;
        delete[] h_probs;
        delete[] h_output_ids;
        delete[] h_top_ks;
        delete[] h_skip_decode;
    }

    void runBatchTest(SamplingKernelTestParam param)
    {
        this->runBatchTest(param, false, false);
        this->runBatchTest(param, false, true);
        this->runBatchTest(param, true, false);
        this->runBatchTest(param, true, true);
    }

    void runBatchLogprobTest(SamplingKernelTestParam param)
    {
        size_t batch_size = param.batch_size;
        size_t vocab_size = param.vocab_size;
        size_t output_len = param.output_len;
        size_t seq_len    = output_len;

        int   top_k = param.top_k;
        float top_p = param.top_p;

        std::vector<int>   _h_top_ks(batch_size);
        std::vector<float> _h_top_ps(batch_size);

        int*   h_top_ks = _h_top_ks.data();
        float* h_top_ps = _h_top_ps.data();
        for (size_t i = 0; i < batch_size; ++i) {
            h_top_ks[i] = top_k;
            h_top_ps[i] = top_p;
        }
        int max_top_k = *std::max_element(h_top_ks, h_top_ks + batch_size);

        // Logit values in the host of shape (batch_size x vocab_size).
        std::vector<T> _h_logits(batch_size * vocab_size);
        T*             h_logits = _h_logits.data();
        initRandom(h_logits, batch_size * vocab_size, -500.0f, 500.0f);

        std::vector<float> expected_sampled_logprobs(batch_size * kMaxLogProb);
        std::vector<int>   expected_sampled_indexes(batch_size * kMaxLogProb);
        std::vector<int>   expected_sampled_nums(batch_size);

        // uniforms
        std::vector<float> _h_uniforms(batch_size);
        float*             h_uniforms = _h_uniforms.data();

        float*         uniforms = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
        curandState_t* curand_states =
            reinterpret_cast<curandState_t*>(allocator->malloc(sizeof(curandState_t) * batch_size, false));
        invokeCurandInitialize(curand_states, batch_size, seed, stream);
        get_curand_uniform<<<batch_size, 1, 0, stream>>>(curand_states, uniforms, batch_size);

        cudaAutoCpy(h_uniforms, uniforms, batch_size, stream);
        // revert curand_states
        invokeCurandInitialize(curand_states, batch_size, seed, stream);

        cudaStreamSynchronize(stream);
        computeSampledLogProb(h_logits,
                              h_top_ks,
                              h_top_ps,
                              h_uniforms,
                              expected_sampled_logprobs.data(),
                              expected_sampled_indexes.data(),
                              expected_sampled_nums.data(),
                              batch_size,
                              vocab_size);

        size_t workspace_size = 0;
        // retrieve the workspace size of the top-k sampling kernel.
        invokeBatchTopKSampling<T>(nullptr,  // workspace
                                   workspace_size,
                                   nullptr,  // log_probs
                                   nullptr,  // ids
                                   nullptr,  // sequence_lengths
                                   nullptr,  // finished
                                   nullptr,  // cum_log_probs
                                   nullptr,  // output_log_probs
                                   nullptr,  // sampled_logprobs
                                   nullptr,  // sampled_indexes
                                   nullptr,  // sampled_nums
                                   nullptr,  // curandstates
                                   max_top_k,
                                   nullptr,  // top_ks
                                   1.0f,
                                   nullptr,
                                   vocab_size,
                                   nullptr,  // end_ids
                                   stream,
                                   batch_size,
                                   nullptr);

        void* workspace = allocator->malloc(workspace_size, false);

        int*   top_ks = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        float* top_ps = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));

        int* output_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * seq_len * batch_size));
        T*   logits     = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size, true));

        float* sampled_logprobs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size * kMaxLogProb));
        int*   sampled_indexes  = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size * kMaxLogProb));
        int*   sampled_nums     = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));

        // Initialize.
        cudaAutoCpy(top_ks, h_top_ks, batch_size, stream);
        cudaAutoCpy(top_ps, h_top_ps, batch_size, stream);

        deviceFill(output_ids, seq_len * batch_size, 0, stream);
        cudaAutoCpy(logits, h_logits, batch_size * vocab_size, stream);

        deviceFill(sampled_logprobs, batch_size * kMaxLogProb, 0.f, stream);
        deviceFill(sampled_indexes, batch_size * kMaxLogProb, 0, stream);
        deviceFill(sampled_nums, batch_size, 0, stream);

        invokeBatchTopKSampling(workspace,  // workspace
                                workspace_size,
                                logits,                      // log_probs
                                output_ids,                  // ids
                                nullptr,                     // sequence_lengths
                                nullptr,                     // finished
                                nullptr,                     // cum_log_probs
                                nullptr,                     // output_log_probs
                                sampled_logprobs,            // sampled_logprobs
                                (uint32_t*)sampled_indexes,  // sampled_indexes
                                (uint32_t*)sampled_nums,     // sampled_nums
                                curand_states,               // curandstates
                                max_top_k,
                                top_ks,  // top_ks
                                1.0f,
                                top_ps,
                                vocab_size,
                                nullptr,  // end_ids
                                stream,
                                batch_size,
                                nullptr);

        std::vector<float> _h_sampled_logprobs(batch_size * kMaxLogProb);
        std::vector<int>   _h_sampled_indexes(batch_size * kMaxLogProb);
        std::vector<int>   _h_sampled_nums(batch_size);
        float*             h_sampled_logprobs = _h_sampled_logprobs.data();
        int*               h_sampled_indexes  = _h_sampled_indexes.data();
        int*               h_sampled_nums     = _h_sampled_nums.data();
        cudaAutoCpy(h_sampled_logprobs, sampled_logprobs, batch_size * kMaxLogProb, stream);
        cudaAutoCpy(h_sampled_indexes, sampled_indexes, batch_size * kMaxLogProb, stream);
        cudaAutoCpy(h_sampled_nums, sampled_nums, batch_size, stream);
        cudaStreamSynchronize(stream);

        bool passed = checkSampledLogProb(h_sampled_nums,
                                          h_sampled_logprobs,
                                          h_sampled_indexes,
                                          expected_sampled_nums.data(),
                                          expected_sampled_logprobs.data(),
                                          expected_sampled_indexes.data(),
                                          batch_size);

        EXPECT_TRUE(passed);
    }
};

TYPED_TEST_SUITE(TopKSamplingKernelTest, FloatType);

TYPED_TEST(TopKSamplingKernelTest, CorrectnessGreedy)
{
    this->runTest({6, 4, 1, 1, 1.0f, 1});
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessAncestral)
{
    this->runTest({6, 4, 1, 4, 1.0f, 1});
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessLargeK63)
{
    this->runTest({16, 51200, 1, 63, 1.0f, 8});
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessLargeK1024)
{
    this->runTest({16, 51200, 1, 1024, 1.0f, 8});
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessTopKTopP)
{
    this->runTest({16, 4000, 1, 63, 0.3f, 8});
};

TYPED_TEST(TopKSamplingKernelTest, NotSupportedLargerThanK1024)
{
    EXPECT_THROW(this->runTest({16, 4000, 1, 1025, 1.0f, 8}), std::domain_error);
};

TYPED_TEST(TopKSamplingKernelTest, BatchCorrectnessGreedy)
{
    this->runBatchTest({6, 4, 1, 1, 1.0f, 1});
};

TYPED_TEST(TopKSamplingKernelTest, BatchCorrectnessAncestral)
{
    this->runBatchTest({6, 4, 1, 4, 1.0f, 1});
};

TYPED_TEST(TopKSamplingKernelTest, BatchCorrectnessLargeK63)
{
    this->runBatchTest({8, 4000, 1, 63, 1.0f, 8});
};

TYPED_TEST(TopKSamplingKernelTest, BatchCorrectnessLargeK1024)
{
    this->runBatchTest({8, 4000, 1, 1024, 0.0f, 8});
};

TYPED_TEST(TopKSamplingKernelTest, BatchCorrectnessTopKTopP)
{
    this->runBatchTest({8, 4000, 1, 63, 0.3f, 8});
};

TYPED_TEST(TopKSamplingKernelTest, BatchCorrectnessTopKTopPLogprobs)
{
    this->runBatchLogprobTest({32, 8000, 1, 40, 0.8f, 1});
    this->runBatchLogprobTest({32, 8000, 1, 40, 1.0f, 1});
    this->runBatchLogprobTest({32, 8000, 1, 1024, 0.9f, 1});
};

template<typename T>
class TopPSamplingKernelTest: public SamplingKernelTest<T> {

protected:
    const int end_id = 0;
    using SamplingKernelTest<T>::seed;
    using SamplingKernelTest<T>::stream;
    using SamplingKernelTest<T>::allocator;
    using SamplingKernelTest<T>::curand_states;

public:
    void runTest(SamplingKernelTestParam param)
    {
        size_t batch_size = param.batch_size;
        size_t vocab_size = param.vocab_size;
        size_t output_len = param.output_len;
        size_t seq_len    = output_len;

        float top_p = param.top_p;

        // Logit values in the host of shape (batch_size x vocab_size).
        T* h_logits = new T[batch_size * vocab_size];
        T* h_probs  = new T[batch_size * vocab_size];
        T* h_lprobs = new T[batch_size * vocab_size];

        float* expected_cum_lprobs = new float[batch_size];
        std::fill_n(expected_cum_lprobs, batch_size, 0);

        int*  h_output_ids  = new int[batch_size];
        int*  h_seq_lengths = new int[batch_size];
        bool* h_finished    = new bool[batch_size];

        initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);

        int device;
        cudaGetDevice(&device);
        struct cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device);

        curandState_t* curand_states =
            reinterpret_cast<curandState_t*>(allocator->malloc(sizeof(curandState_t) * batch_size, false));
        invokeCurandInitialize(curand_states, batch_size, seed, stream);

        int* end_ids     = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        int* seq_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        int* output_ids  = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * seq_len * batch_size));

        bool* finished    = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));
        bool* skip_decode = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));

        T*     probs         = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size));
        float* cum_lprobs    = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
        float* output_lprobs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * output_len * batch_size));

        int* begin_offsets    = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * (batch_size + 1)));
        int* end_offsets      = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * (batch_size + 1)));
        int* topp_id_vals_buf = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size * vocab_size));

        size_t workspace_size        = 0;
        size_t cub_temp_storage_size = 0;
        // retrieve the workspace size of the top-p sampling kernel.
        invokeTopPSampling<T>(nullptr,  // workspace
                              workspace_size,
                              cub_temp_storage_size,
                              nullptr,      // output_ids
                              nullptr,      // sequence_length
                              nullptr,      // finished_buffer
                              nullptr,      // cum_log_probs
                              nullptr,      // output_log_probs
                              (T*)nullptr,  // log_probs
                              topp_id_vals_buf,
                              end_offsets,
                              begin_offsets,
                              curand_states,
                              batch_size,
                              vocab_size,
                              nullptr,
                              top_p,
                              stream,
                              &device_prop,
                              nullptr);
        void* workspace = allocator->malloc(workspace_size);

        // Initialize.
        deviceFill(end_ids, batch_size, end_id);
        deviceFill(seq_lengths, batch_size, 0);
        deviceFill(finished, batch_size, false);
        deviceFill(cum_lprobs, batch_size, 0.0f);
        deviceFill(output_lprobs, output_len * batch_size, 0.0f);
        deviceFill(output_ids, seq_len * batch_size, 0);

        for (size_t step = 0; step < output_len; ++step) {
            initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);
            computeProb(h_probs, h_logits, batch_size, vocab_size);
            cudaH2Dcpy(probs, h_probs, batch_size * vocab_size);

            invokeTopPInitialize(topp_id_vals_buf, end_offsets, begin_offsets, batch_size, vocab_size, stream);

            invokeTopPSampling<T>(workspace,
                                  workspace_size,
                                  cub_temp_storage_size,
                                  output_ids + step * batch_size,
                                  seq_lengths,
                                  finished,
                                  cum_lprobs,
                                  output_lprobs + step * batch_size,
                                  // Note that the kernel needs vocab probs instead of
                                  // log-prob if cum_log_probs or output_log_probs are
                                  // provided. It's because the sampling layer already
                                  // preprocesses log_prob_buf when those are provided.
                                  probs,
                                  topp_id_vals_buf,
                                  end_offsets,
                                  begin_offsets,
                                  curand_states,
                                  batch_size,
                                  vocab_size,
                                  end_ids,
                                  top_p,
                                  stream,
                                  &device_prop,
                                  nullptr);

            // Compute reference.
            cudaD2Hcpy(h_output_ids, output_ids + step * batch_size, batch_size);
            cudaD2Hcpy(h_seq_lengths, seq_lengths, batch_size);
            cudaD2Hcpy(h_finished, finished, batch_size);
            computeLogProb(h_lprobs, h_logits, batch_size, vocab_size);
            for (size_t i = 0; i < batch_size; ++i) {
                int idx = i * vocab_size + h_output_ids[i];
                expected_cum_lprobs[i] += (int)step < h_seq_lengths[i] ? (float)h_lprobs[idx] : 0.0f;
                EXPECT_EQ(h_finished[i], h_output_ids[i] == end_id);
            }
        }
        bool passed = checkResult(param.toString(), cum_lprobs, expected_cum_lprobs, batch_size);
        EXPECT_TRUE(passed);

        delete[] expected_cum_lprobs;
        delete[] h_seq_lengths;
        delete[] h_logits;
        delete[] h_lprobs;
        delete[] h_probs;
        delete[] h_output_ids;
    }

    void runBatchTest(SamplingKernelTestParam param, bool has_diff_runtime_args, bool use_skip_decode)
    {
        size_t batch_size = param.batch_size;
        size_t vocab_size = param.vocab_size;

        float  top_p    = param.top_p;
        float* h_top_ps = new float[batch_size];
        // Initialize runtime top k values.
        for (size_t i = 0; i < batch_size; ++i) {
            h_top_ps[i] = (!has_diff_runtime_args || i % 3 == 0) ? top_p : 0.1 * top_p;
        }
        float max_top_p = *std::max_element(h_top_ps, h_top_ps + batch_size);

        size_t output_len = param.output_len;
        size_t seq_len    = output_len;

        // Logit values in the host of shape (batch_size x vocab_size).
        T* h_logits = new T[batch_size * vocab_size];
        T* h_probs  = new T[batch_size * vocab_size];
        T* h_lprobs = new T[batch_size * vocab_size];

        float* expected_cum_lprobs = new float[batch_size];
        std::fill_n(expected_cum_lprobs, batch_size, 0);

        int*  h_output_ids  = new int[batch_size];
        int*  h_seq_lengths = new int[batch_size];
        bool* h_finished    = new bool[batch_size];
        bool* h_skip_decode = new bool[batch_size];

        initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);
        std::fill_n(expected_cum_lprobs, batch_size, 0);
        for (size_t i = 0; i < batch_size; ++i) {
            h_skip_decode[i] = use_skip_decode && (i % 2 == 0);
        }

        int device;
        cudaGetDevice(&device);
        struct cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device);

        curandState_t* curand_states =
            reinterpret_cast<curandState_t*>(allocator->malloc(sizeof(curandState_t) * batch_size, false));
        invokeCurandInitialize(curand_states, batch_size, seed, stream);

        float* top_ps = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));

        int* end_ids     = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        int* seq_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        int* output_ids  = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * seq_len * batch_size));

        bool* finished    = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));
        bool* skip_decode = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));

        T*     probs         = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size));
        float* cum_lprobs    = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
        float* output_lprobs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * output_len * batch_size));

        int* begin_offsets    = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * (batch_size + 1)));
        int* end_offsets      = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * (batch_size + 1)));
        int* topp_id_vals_buf = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size * vocab_size));

        size_t workspace_size        = 0;
        size_t cub_temp_storage_size = 0;
        // retrieve the workspace size of the top-p sampling kernel.
        invokeBatchTopPSampling<T>(nullptr,  // workspace
                                   workspace_size,
                                   cub_temp_storage_size,
                                   nullptr,      // output_ids
                                   nullptr,      // sequence_length
                                   nullptr,      // finished_buffer
                                   nullptr,      // cum_log_probs
                                   nullptr,      // output_log_probs
                                   (T*)nullptr,  // log_probs
                                   nullptr,      // sampled_logprobs
                                   nullptr,      // sampled_indexes
                                   nullptr,      // sampled_nums
                                   topp_id_vals_buf,
                                   end_offsets,
                                   begin_offsets,
                                   curand_states,
                                   batch_size,
                                   vocab_size,
                                   nullptr,
                                   max_top_p,
                                   top_ps,
                                   stream,
                                   &device_prop,
                                   nullptr);
        void* workspace = allocator->malloc(workspace_size);

        // Initialize.
        cudaH2Dcpy(top_ps, h_top_ps, batch_size);
        cudaH2Dcpy(skip_decode, h_skip_decode, batch_size);
        deviceFill(end_ids, batch_size, end_id);
        deviceFill(seq_lengths, batch_size, 0);
        deviceFill(finished, batch_size, false);
        deviceFill(cum_lprobs, batch_size, 0.0f);
        deviceFill(output_lprobs, output_len * batch_size, 0.0f);
        deviceFill(output_ids, seq_len * batch_size, 0);

        for (size_t step = 0; step < output_len; ++step) {
            initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);
            computeProb(h_probs, h_logits, batch_size, vocab_size);
            cudaH2Dcpy(probs, h_probs, batch_size * vocab_size);

            invokeTopPInitialize(topp_id_vals_buf, end_offsets, begin_offsets, batch_size, vocab_size, stream);

            invokeBatchTopPSampling<T>(workspace,
                                       workspace_size,
                                       cub_temp_storage_size,
                                       output_ids + step * batch_size,
                                       seq_lengths,
                                       finished,
                                       cum_lprobs,
                                       output_lprobs + step * batch_size,
                                       // Note that the kernel needs vocab probs instead of
                                       // log-prob if cum_log_probs or output_log_probs are
                                       // provided. It's because the sampling layer already
                                       // preprocesses log_prob_buf when those are provided.
                                       probs,
                                       nullptr,  // sampled_logprobs
                                       nullptr,  // sampled_indexes
                                       nullptr,  // sampled_nums
                                       topp_id_vals_buf,
                                       end_offsets,
                                       begin_offsets,
                                       curand_states,
                                       batch_size,
                                       vocab_size,
                                       end_ids,
                                       max_top_p,
                                       top_ps,
                                       stream,
                                       &device_prop,
                                       skip_decode);

            // Compute reference.
            cudaD2Hcpy(h_output_ids, output_ids + step * batch_size, batch_size);
            cudaD2Hcpy(h_seq_lengths, seq_lengths, batch_size);
            cudaD2Hcpy(h_finished, finished, batch_size);
            computeLogProb(h_lprobs, h_logits, batch_size, vocab_size);
            for (size_t i = 0; i < batch_size; ++i) {
                if (!h_skip_decode[i]) {
                    int idx = i * vocab_size + h_output_ids[i];
                    expected_cum_lprobs[i] += (int)step < h_seq_lengths[i] ? (float)h_lprobs[idx] : 0.0f;
                    EXPECT_EQ(h_finished[i], h_output_ids[i] == end_id);
                }
            }
        }
        bool passed = checkResult(param.toString(), cum_lprobs, expected_cum_lprobs, batch_size);
        EXPECT_TRUE(passed) << "Fail subtest (has_diff_runtime_args: " << has_diff_runtime_args
                            << ", skip_decode: " << use_skip_decode << ")";

        delete[] expected_cum_lprobs;
        delete[] h_seq_lengths;
        delete[] h_logits;
        delete[] h_lprobs;
        delete[] h_probs;
        delete[] h_output_ids;
        delete[] h_top_ps;
        delete[] h_skip_decode;
    }

    void runBatchTest(SamplingKernelTestParam param)
    {
        this->runBatchTest(param, false, false);
        this->runBatchTest(param, false, true);
        this->runBatchTest(param, true, false);
        this->runBatchTest(param, true, true);
    }

    void runBatchLogprobTest(SamplingKernelTestParam param)
    {
        size_t batch_size = param.batch_size;
        size_t vocab_size = param.vocab_size;
        size_t output_len = param.output_len;
        size_t seq_len    = output_len;

        int   top_k = param.top_k;
        float top_p = param.top_p;

        std::vector<int>   _h_top_ks(batch_size);
        std::vector<float> _h_top_ps(batch_size);

        int*   h_top_ks = _h_top_ks.data();
        float* h_top_ps = _h_top_ps.data();
        for (size_t i = 0; i < batch_size; ++i) {
            h_top_ks[i] = top_k;
            h_top_ps[i] = top_p;
        }
        float max_top_p = *std::max_element(h_top_ps, h_top_ps + batch_size);

        // Logit values in the host of shape (batch_size x vocab_size).
        std::vector<T> _h_logits(batch_size * vocab_size);
        std::vector<T> _h_probs(batch_size * vocab_size);
        T*             h_logits = _h_logits.data();
        T*             h_probs  = _h_probs.data();
        initRandom(h_logits, batch_size * vocab_size, -200.0f, 200.0f);

        std::vector<float> expected_sampled_logprobs(batch_size * kMaxLogProb);
        std::vector<int>   expected_sampled_indexes(batch_size * kMaxLogProb);
        std::vector<int>   expected_sampled_nums(batch_size);

        // uniforms
        std::vector<float> _h_uniforms(batch_size);
        float*             h_uniforms = _h_uniforms.data();

        float*         uniforms = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
        curandState_t* curand_states =
            reinterpret_cast<curandState_t*>(allocator->malloc(sizeof(curandState_t) * batch_size, false));
        invokeCurandInitialize(curand_states, batch_size, seed, stream);
        get_curand_uniform<<<batch_size, 1, 0, stream>>>(curand_states, uniforms, batch_size);

        cudaAutoCpy(h_uniforms, uniforms, batch_size, stream);
        // revert curand_states
        invokeCurandInitialize(curand_states, batch_size, seed, stream);

        cudaStreamSynchronize(stream);
        computeSampledLogProb(h_logits,
                              h_top_ks,
                              h_top_ps,
                              h_uniforms,
                              expected_sampled_logprobs.data(),
                              expected_sampled_indexes.data(),
                              expected_sampled_nums.data(),
                              batch_size,
                              vocab_size);

        int device;
        cudaGetDevice(&device);
        struct cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device);

        int*   begin_offsets    = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * (batch_size + 1)));
        int*   end_offsets      = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * (batch_size + 1)));
        int*   topp_id_vals_buf = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size * vocab_size));
        float* top_ps           = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
        int*   output_ids       = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * seq_len * batch_size));
        T*     probs            = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size, true));

        float* sampled_logprobs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size * kMaxLogProb));
        int*   sampled_indexes  = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size * kMaxLogProb));
        int*   sampled_nums     = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));

        deviceFill(sampled_logprobs, batch_size * kMaxLogProb, 0.f, stream);
        deviceFill(sampled_indexes, batch_size * kMaxLogProb, 0, stream);
        deviceFill(sampled_nums, batch_size, 0, stream);

        size_t workspace_size        = 0;
        size_t cub_temp_storage_size = 0;
        // retrieve the workspace size of the top-p sampling kernel.
        invokeBatchTopPSampling<T>(nullptr,  // workspace
                                   workspace_size,
                                   cub_temp_storage_size,
                                   nullptr,      // output_ids
                                   nullptr,      // sequence_length
                                   nullptr,      // finished_buffer
                                   nullptr,      // cum_log_probs
                                   nullptr,      // output_log_probs
                                   (T*)nullptr,  // log_probs
                                   nullptr,      // sampled_logprobs
                                   nullptr,      // sampled_indexes
                                   nullptr,      // sampled_nums
                                   topp_id_vals_buf,
                                   end_offsets,
                                   begin_offsets,
                                   curand_states,
                                   batch_size,
                                   vocab_size,
                                   nullptr,
                                   max_top_p,
                                   top_ps,
                                   stream,
                                   &device_prop,
                                   nullptr);

        void* workspace = allocator->malloc(workspace_size);

        // Initialize.
        cudaAutoCpy(top_ps, h_top_ps, batch_size, stream);
        deviceFill(output_ids, seq_len * batch_size, 0, stream);
        cudaAutoCpy(top_ps, h_top_ps, batch_size, stream);

        computeProb(h_probs, h_logits, batch_size, vocab_size);
        cudaAutoCpy(probs, h_probs, batch_size * vocab_size, stream);

        invokeTopPInitialize(topp_id_vals_buf, end_offsets, begin_offsets, batch_size, vocab_size, stream);
        invokeBatchTopPSampling<T>(workspace,
                                   workspace_size,
                                   cub_temp_storage_size,
                                   output_ids,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   // Note that the kernel needs vocab probs instead of
                                   // log-prob if cum_log_probs or output_log_probs are
                                   // provided. It's because the sampling layer already
                                   // preprocesses log_prob_buf when those are provided.
                                   probs,
                                   sampled_logprobs,            // sampled_logprobs
                                   (uint32_t*)sampled_indexes,  // sampled_indexes
                                   (uint32_t*)sampled_nums,     // sampled_nums
                                   topp_id_vals_buf,
                                   end_offsets,
                                   begin_offsets,
                                   curand_states,
                                   batch_size,
                                   vocab_size,
                                   nullptr,
                                   max_top_p,
                                   top_ps,
                                   stream,
                                   &device_prop,
                                   nullptr);

        std::vector<float> _h_sampled_logprobs(batch_size * kMaxLogProb);
        std::vector<int>   _h_sampled_indexes(batch_size * kMaxLogProb);
        std::vector<int>   _h_sampled_nums(batch_size);
        float*             h_sampled_logprobs = _h_sampled_logprobs.data();
        int*               h_sampled_indexes  = _h_sampled_indexes.data();
        int*               h_sampled_nums     = _h_sampled_nums.data();
        cudaAutoCpy(h_sampled_logprobs, sampled_logprobs, batch_size * kMaxLogProb, stream);
        cudaAutoCpy(h_sampled_indexes, sampled_indexes, batch_size * kMaxLogProb, stream);
        cudaAutoCpy(h_sampled_nums, sampled_nums, batch_size, stream);
        cudaStreamSynchronize(stream);

        bool passed = checkSampledLogProb(h_sampled_nums,
                                          h_sampled_logprobs,
                                          h_sampled_indexes,
                                          expected_sampled_nums.data(),
                                          expected_sampled_logprobs.data(),
                                          expected_sampled_indexes.data(),
                                          batch_size);

        EXPECT_TRUE(passed);
    }
};

TYPED_TEST_SUITE(TopPSamplingKernelTest, FloatType);

TYPED_TEST(TopPSamplingKernelTest, CorrectnessSmallP)
{
    this->runTest({6, 4, 1, 0, 0.2f, 1});
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeP)
{
    this->runTest({6, 4, 1, 0, 0.9f, 1});
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessAncestral)
{
    this->runTest({6, 4, 1, 0, 1.0f, 1});
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeVocabSmallP)
{
    this->runTest({32, 51200, 1, 0, 0.2f, 16});
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeVocabLargeP)
{
    this->runTest({32, 51200, 1, 0, 0.9f, 16});
};

TYPED_TEST(TopPSamplingKernelTest, BatchCorrectnessSmallP)
{
    this->runBatchTest({6, 4, 1, 0, 0.2f, 1});
};

TYPED_TEST(TopPSamplingKernelTest, BatchCorrectnessLargeP)
{
    this->runBatchTest({6, 4, 1, 0, 0.9f, 1});
};

TYPED_TEST(TopPSamplingKernelTest, BatchCorrectnessSmallP2)
{
    this->runBatchTest({8, 4000, 1, 0, 0.2f, 16});
};

TYPED_TEST(TopPSamplingKernelTest, BatchCorrectnessLargeP2)
{
    this->runBatchTest({8, 4000, 1, 0, 0.9f, 16});
};
TYPED_TEST(TopPSamplingKernelTest, BatchCorrectnessTopPLogprobs)
{
    this->runBatchLogprobTest({32, 4000, 1, 0, 0.1f, 1});
    this->runBatchLogprobTest({32, 4000, 1, 0, 0.8f, 1});
};

__global__ void generateRandomNumber(unsigned int* vals, curandState_t* states, const int batch_size)
{
    int idx = threadIdx.x;
    if (idx < batch_size) {
        vals[idx] = curand(states + idx);
    }
}

TEST(SamplingKernelTest, CurandBatchInitialize)
{
    size_t       batch_size = 127;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    curandState_t* curand_states;
    check_cuda_error(cudaMalloc(&curand_states, sizeof(curandState_t) * batch_size));
    unsigned long long* h_random_seeds = new unsigned long long[batch_size];
    const size_t        period_size    = 3;
    for (size_t i = 0; i < batch_size; ++i) {
        h_random_seeds[i] = i / period_size;
    }
    unsigned long long* d_random_seeds;
    check_cuda_error(cudaMalloc(&d_random_seeds, sizeof(unsigned long long) * batch_size));
    check_cuda_error(
        cudaMemcpy(d_random_seeds, h_random_seeds, sizeof(unsigned long long) * batch_size, cudaMemcpyHostToDevice));

    // Initialize curand states.
    invokeCurandBatchInitialize(curand_states, batch_size, d_random_seeds, stream);
    sync_check_cuda_error();

    // Generate random numbers using initialized curand states.
    unsigned int* d_rand_vals;
    unsigned int* h_rand_vals = new unsigned int[batch_size];
    check_cuda_error(cudaMalloc(&d_rand_vals, sizeof(unsigned int) * batch_size));
    generateRandomNumber<<<1, batch_size, 0, stream>>>(d_rand_vals, curand_states, batch_size);
    check_cuda_error(
        cudaMemcpyAsync(h_rand_vals, d_rand_vals, sizeof(unsigned int) * batch_size, cudaMemcpyDeviceToHost, stream));
    check_cuda_error(cudaStreamSynchronize(stream));

    // The same seed produces the same random number.
    for (size_t i = 0; i + period_size - 1 < batch_size; i += period_size) {
        for (size_t j = 1; j < period_size; ++j) {
            EXPECT_TRUE(h_rand_vals[i] == h_rand_vals[i + j])
                << fmtstr("Fail at val[%d]=%d <> val[%d]=%d", i, h_rand_vals[i], i + j, h_rand_vals[i + j]);
        }
    }

    delete h_rand_vals;
    delete h_random_seeds;
    check_cuda_error(cudaFree(d_rand_vals));
    check_cuda_error(cudaFree(d_random_seeds));
    check_cuda_error(cudaFree(curand_states));
    check_cuda_error(cudaStreamDestroy(stream));
}

}  // end of namespace
