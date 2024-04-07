#include <algorithm>  // std::min, std::max
#include <iostream>   // snprintf
#include <math.h>     // expf, log
#include <stdlib.h>   // rand
#include <string>     // std::string
#include <vector>     // std::vector

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/layers/DynamicDecodeLayer.h"
#include "src/turbomind/layers/sampling_layers/TopKSamplingLayer.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

#include "gtest_utils.h"

using namespace turbomind;

struct SamplingLayerTestParam {
    size_t batch_size;
    size_t vocab_size;
    size_t beam_width;
    size_t top_k;
    float  top_p;
    size_t output_len;

    std::string toString()
    {
        return fmtstr("SamplingLayerTestParam[batch=%ld, vocab=%ld, beam=%ld, k=%ld, p=%3.1f, output_len=%ld]",
                      batch_size,
                      vocab_size,
                      beam_width,
                      top_k,
                      top_p,
                      output_len);
    }
};

template<typename T>
void computeProb(T* probs, T* logits, int batch_size, int vocab_size)
{
    // Compute the log probability from logits.
    //   logits = batch_size x vocab_size vector.
    //   logprobs = log(softmax(logits)) (softmax along with vocab dimension)
    for (int bidx = 0; bidx < batch_size; ++bidx) {
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            sum += expf((float)logits[bidx * vocab_size + i]);
        }
        for (int i = 0; i < vocab_size; ++i) {
            int idx    = bidx * vocab_size + i;
            probs[idx] = static_cast<T>(expf((float)logits[idx]) / (sum + EPSILON));
        }
    }
}

template<typename T>
void computeLogProb(T* logprobs, T* logits, int batch_size, int vocab_size)
{
    // Compute the log probability from logits.
    //   logits = batch_size x vocab_size vector.
    //   logprobs = log(softmax(logits)) (softmax along with vocab dimension)
    for (int bidx = 0; bidx < batch_size; ++bidx) {
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            sum += expf(logits[bidx * vocab_size + i]);
        }
        for (int i = 0; i < vocab_size; ++i) {
            int idx       = bidx * vocab_size + i;
            logprobs[idx] = static_cast<T>(logf(expf(logits[idx]) / (sum + EPSILON) + EPSILON));
        }
    }
}

template<typename T>
class SamplingDecodeTest: public testing::Test {
protected:
    unsigned long long              seed           = 0;
    const static unsigned long long max_seed       = 30;
    const size_t                    batch_size     = 6;
    const size_t                    beam_width     = 1;
    const size_t                    batchxbeam     = batch_size * beam_width;
    const size_t                    vocab_size     = 8;
    const size_t                    max_input_len  = 0;  // has no effect.
    const size_t                    max_output_len = 3;
    const size_t                    max_seq_len    = max_input_len + max_output_len;
    const int                       end_id         = vocab_size - 1;
    const DataType                  data_type      = getTensorType<T>();

    // vocab size 8 & length 3
    T* test_input_logits;

    cudaStream_t                            stream;
    ft::Allocator<ft::AllocatorType::CUDA>* allocator;
    cublasHandle_t                          cublas_handle;
    cublasLtHandle_t                        cublaslt_handle;
    std::mutex*                             cublas_wrapper_mutex;
    cublasMMWrapper*                        cublas_wrapper;
    DynamicDecodeLayer<T>*                  dynamic_decode_layer;

    int*   h_output_ids;
    T*     h_logits;
    T*     h_probs;
    T*     h_log_probs;
    float* h_cum_log_probs;
    float* h_output_log_probs;

    T*     d_logits;
    int*   d_input_lengths;
    float* d_cum_log_probs;
    float* d_output_log_probs;
    int*   d_output_ids;
    int*   d_end_ids;

    void setup(unsigned long long seed = 0)
    {
        this->seed = seed;

        check_cuda_error(cudaStreamCreate(&stream));
        allocator = new Allocator<AllocatorType::CUDA>(getDevice());
        allocator->setStream(stream);

        struct cudaDeviceProp prop;
        check_cuda_error(cudaGetDeviceProperties(&prop, 0));
        check_cuda_error(cublasCreate(&cublas_handle));
        check_cuda_error(cublasLtCreate(&cublaslt_handle));
        check_cuda_error(cublasSetStream(cublas_handle, stream));
        cublasAlgoMap cublas_algo_map(GEMM_CONFIG);
        cublas_wrapper_mutex = new std::mutex();

        cublas_wrapper = new cublasMMWrapper(
            cublas_handle, cublaslt_handle, stream, &cublas_algo_map, cublas_wrapper_mutex, allocator);

        dynamic_decode_layer = new DynamicDecodeLayer<T>(vocab_size,
                                                         vocab_size,
                                                         end_id,
                                                         stream,
                                                         cublas_wrapper,
                                                         allocator,
                                                         false,   // is_free_buffer_after_forward
                                                         &prop);  // cuda_device_prop

        h_output_ids       = new int[batchxbeam];
        h_logits           = new T[batchxbeam * vocab_size];
        h_probs            = new T[batchxbeam * vocab_size];
        h_log_probs        = new T[batchxbeam * vocab_size];
        h_cum_log_probs    = new float[batchxbeam];
        h_output_log_probs = new float[max_output_len * batchxbeam];

        // prob = (0.4, 0.3, 0.2, 0.1, ...)
        test_input_logits = new T[24]{
            -0.9163,  -1.2040,  -1.6094,  -2.3026,  -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX,  // step 0
            -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -0.9163,  -1.2040,  -1.6094,  -2.3026,   // step 1
            -FLT_MAX, -FLT_MAX, -0.9163,  -1.2040,  -1.6094,  -2.3026,  -FLT_MAX, -FLT_MAX   // step 2
        };

        d_logits           = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batchxbeam * vocab_size, true));
        d_input_lengths    = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batchxbeam));
        d_cum_log_probs    = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batchxbeam));
        d_output_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * max_output_len * batchxbeam));
        d_output_ids       = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batchxbeam));
        d_end_ids          = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batchxbeam));

        // Init by zero.
        cudaMemset(d_cum_log_probs, 0, sizeof(float) * batchxbeam);
        cudaMemset(d_output_log_probs, 0, sizeof(float) * max_output_len * batchxbeam);
        cudaMemset(d_output_ids, 0, sizeof(int) * max_seq_len * batchxbeam);
        deviceFill(d_end_ids, batchxbeam, end_id, stream);
    }

    void teardown()
    {
        delete[] test_input_logits;
        delete[] h_output_ids;
        delete[] h_logits;
        delete[] h_probs;
        delete[] h_log_probs;
        delete[] h_cum_log_probs;
        delete[] h_output_log_probs;
        delete dynamic_decode_layer;
        delete cublas_wrapper;
        delete cublas_wrapper_mutex;
        delete allocator;
        check_cuda_error(cublasDestroy(cublas_handle));
        check_cuda_error(cublasLtDestroy(cublaslt_handle));
        check_cuda_error(cudaStreamDestroy(stream));
    }

    TensorMap* createInputTensors(
        int* topk, size_t topk_size, float* topp, size_t topp_size, float* temperature, float* repetition_penalty)
    {
        // construct common input tensors
        TensorMap* input_tensors = new TensorMap();
        if (topk != nullptr) {
            input_tensors->insert({"runtime_top_k", {MEMORY_CPU, TYPE_INT32, {topk_size}, topk}});
        }
        if (topp != nullptr) {
            input_tensors->insert({"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {topp_size}, topp}});
        }
        if (temperature != nullptr) {
            input_tensors->insert({"temperature", Tensor{MEMORY_CPU, TYPE_FP32, {1}, temperature}});
        }
        if (repetition_penalty != nullptr) {
            input_tensors->insert({"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, repetition_penalty}});
        }
        input_tensors->insert(
            {"logits", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size, beam_width, vocab_size}, d_logits}});
        input_tensors->insert({"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size}, nullptr}});
        input_tensors->insert({"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_len}});
        input_tensors->insert(
            {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, d_input_lengths}});
        input_tensors->insert({"end_id", Tensor{MEMORY_CPU, TYPE_INT32, {batchxbeam}, &d_end_ids}});
        input_tensors->insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, {1}, &seed}});
        return input_tensors;
    }

    TensorMap* createOutputTensors()
    {
        // construct common output tensors
        TensorMap* output_tensors = new TensorMap();
        output_tensors->insert(
            {"output_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, d_output_ids}});
        output_tensors->insert({"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, nullptr}});
        output_tensors->insert(
            {"cum_log_probs", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size * beam_width}, d_cum_log_probs}});
        output_tensors->insert(
            {"output_log_probs",
             Tensor{MEMORY_GPU, TYPE_FP32, {max_seq_len, batch_size, beam_width}, d_output_log_probs}});
        output_tensors->insert({"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, nullptr}});
        return output_tensors;
    }

    void batchH2Dcpy(T* dst, T* src, size_t m, size_t n)
    {
        for (size_t i = 0; i < m; ++i) {
            cudaH2Dcpy(dst + i * n, src, n);
        }
    }

    bool checkResult(int* d_output_ids, std::vector<std::set<int>>& expected_ids)
    {
        assert(expected_ids.size() == max_seq_len * batchxbeam);
        int* h_output_ids = new int[max_seq_len * batchxbeam];
        cudaD2Hcpy(h_output_ids, d_output_ids, max_seq_len * batchxbeam);
        int failures = 0;
        for (size_t i = 0; i < max_seq_len * batchxbeam; ++i) {
            size_t        s     = i / batchxbeam;
            size_t        b     = i % batchxbeam;
            std::set<int> expts = expected_ids.at(i);
            if (expts.count(h_output_ids[i]) == 0) {
                if (failures < 10) {
                    std::stringstream ss;
                    ss << " - Fail "
                       << " (step=" << s << ", batch=" << b << ") "
                       << "actual=" << h_output_ids[i] << ", expected";
                    for (auto& expt : expts) {
                        ss << " " << expt;
                    }
                    TM_LOG_DEBUG("%s", ss.str().c_str());
                }
                ++failures;
            }
        }
        TM_LOG_DEBUG(
            "check...%6s : failures: %d / %d", failures == 0 ? "....OK" : "FAILED", failures, max_seq_len * batchxbeam);
        delete[] h_output_ids;
        return failures == 0;
    }

public:
    void runTest(std::vector<std::set<int>> expected_output_ids,
                 int*                       top_ks,
                 size_t                     top_k_size,
                 float*                     top_ps,
                 size_t                     top_p_size,
                 float*                     temperature,
                 float*                     repetition_penalty,
                 bool                       use_local_batch = false)
    {
        size_t local_batch_size = use_local_batch ? batch_size / 3 : batch_size;
        uint   ite              = use_local_batch ? 1 : 0;
        for (unsigned long long seed = 0; seed < max_seed; ++seed) {
            this->setup(seed);
            size_t     step = max_input_len;
            TensorMap* input_tensors =
                createInputTensors(top_ks, top_k_size, top_ps, top_p_size, temperature, repetition_penalty);
            input_tensors->insert({"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}});
            input_tensors->insert({"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}});
            input_tensors->insert({"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &local_batch_size}});
            TensorMap* output_tensors = createOutputTensors();

            dynamic_decode_layer->setup(batch_size, beam_width, input_tensors);
            for (step = max_input_len; step < max_output_len; ++step) {
                // Reset by the test value since the sampling layer internally update the logit buffer.
                batchH2Dcpy(input_tensors->at("logits").getPtr<T>(),
                            test_input_logits + step * vocab_size,
                            batchxbeam,
                            vocab_size);
                dynamic_decode_layer->forward(output_tensors, input_tensors);
            }
            bool passed = checkResult(d_output_ids, expected_output_ids);
            EXPECT_TRUE(passed) << "Failed at seed " << seed;
#ifndef NDEBUG
            if (!passed) {
                TM_LOG_ERROR("actual output ids");
                printMatrix(d_output_ids, max_seq_len, batch_size, batch_size, true);
            }
#endif
            delete output_tensors;
            delete input_tensors;
            this->teardown();
        }
    }
};

TYPED_TEST_SUITE(SamplingDecodeTest, FloatType);

TYPED_TEST(SamplingDecodeTest, TopK)
{
    int                        top_k = 2;
    std::vector<std::set<int>> expected_output_ids{
        // batch
        //  0       1       2       3       4       5
        {0, 1},
        {0, 1},
        {0, 1},
        {0, 1},
        {0, 1},
        {0, 1},  // step 0
        {4, 5},
        {4, 5},
        {4, 5},
        {4, 5},
        {4, 5},
        {4, 5},  // step 1
        {2, 3},
        {2, 3},
        {2, 3},
        {2, 3},
        {2, 3},
        {2, 3}  // step 2
    };
    this->runTest(expected_output_ids, &top_k, 1, nullptr, 0, nullptr, nullptr);
}

TYPED_TEST(SamplingDecodeTest, BatchTopK)
{
    size_t                     batch_size = this->batch_size;
    int*                       top_ks     = new int[batch_size]{2, 1, 1, 2, 1, 1};
    std::vector<std::set<int>> expected_output_ids{
        // batch
        //  0    1    2       3    4    5
        {0, 1},
        {0},
        {0},
        {0, 1},
        {0},
        {0},  // step 0
        {4, 5},
        {4},
        {4},
        {4, 5},
        {4},
        {4},  // step 1
        {2, 3},
        {2},
        {2},
        {2, 3},
        {2},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, top_ks, batch_size, nullptr, 0, nullptr, nullptr);
    delete[] top_ks;
}

TYPED_TEST(SamplingDecodeTest, TopP)
{
    float                      top_p = 0.3;
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},  // step 0
        {4},
        {4},
        {4},
        {4},
        {4},
        {4},  // step 1
        {2},
        {2},
        {2},
        {2},
        {2},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, nullptr, 0, &top_p, 1, nullptr, nullptr);
}

TYPED_TEST(SamplingDecodeTest, BatchTopP)
{
    size_t                     batch_size = this->batch_size;
    float*                     top_ps     = new float[batch_size]{0.3f, 0.5f, 0.5f, 0.3f, 0.5f, 0.5f};
    std::vector<std::set<int>> expected_output_ids{
        {0},
        {0, 1},
        {0, 1},
        {0},
        {0, 1},
        {0, 1},  // step 0
        {4},
        {4, 5},
        {4, 5},
        {4},
        {4, 5},
        {4, 5},  // step 1
        {2},
        {2, 3},
        {2, 3},
        {2},
        {2, 3},
        {2, 3}  // step 2
    };
    this->runTest(expected_output_ids, nullptr, 0, top_ps, batch_size, nullptr, nullptr);
    delete[] top_ps;
}

TYPED_TEST(SamplingDecodeTest, TopKTopP)
{
    int                        top_k = 2;
    float                      top_p = 0.3;
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},  // step 0
        {4},
        {4},
        {4},
        {4},
        {4},
        {4},  // step 1
        {2},
        {2},
        {2},
        {2},
        {2},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, &top_k, 1, &top_p, 1, nullptr, nullptr);
}

TYPED_TEST(SamplingDecodeTest, BatchTopKTopP)
{
    size_t                     batch_size = this->batch_size;
    int*                       top_ks     = new int[batch_size]{2, 2, 1, 2, 2, 1};
    float                      top_p      = 0.3;
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},  // step 0
        {4},
        {4},
        {4},
        {4},
        {4},
        {4},  // step 1
        {2},
        {2},
        {2},
        {2},
        {2},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, top_ks, batch_size, &top_p, 1, nullptr, nullptr);
    delete[] top_ks;
}

TYPED_TEST(SamplingDecodeTest, TopKBatchTopP)
{
    size_t                     batch_size = this->batch_size;
    int                        top_k      = 2;
    float*                     top_ps     = new float[batch_size]{0.5, 0.3, 0.5, 0.5, 0.3, 0.5};
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0, 1},
        {0},
        {0, 1},
        {0, 1},
        {0},
        {0, 1},  // step 0
        {4, 5},
        {4},
        {4, 5},
        {4, 5},
        {4},
        {4, 5},  // step 1
        {2, 3},
        {2},
        {2, 3},
        {2, 3},
        {2},
        {2, 3}  // step 2
    };
    this->runTest(expected_output_ids, &top_k, 1, top_ps, batch_size, nullptr, nullptr);
    delete[] top_ps;
}

TYPED_TEST(SamplingDecodeTest, BatchTopKBatchTopP)
{
    size_t                     batch_size = this->batch_size;
    int*                       top_ks     = new int[batch_size]{2, 2, 0, 2, 2, 0};
    float*                     top_ps     = new float[batch_size]{0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0, 1},
        {0},
        {0, 1},
        {0, 1},
        {0},
        {0, 1},  // step 0
        {4, 5},
        {4},
        {4, 5},
        {4, 5},
        {4},
        {4, 5},  // step 1
        {2, 3},
        {2},
        {2, 3},
        {2, 3},
        {2},
        {2, 3}  // step 2
    };
    this->runTest(expected_output_ids, top_ks, batch_size, top_ps, batch_size, nullptr, nullptr);
    delete[] top_ks;
    delete[] top_ps;
}

TYPED_TEST(SamplingDecodeTest, InvalidArgsZeroTopK)
{
    size_t                     batch_size = this->batch_size;
    int                        top_k      = 0;
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},  // step 0
        {4},
        {4},
        {4},
        {4},
        {4},
        {4},  // step 1
        {2},
        {2},
        {2},
        {2},
        {2},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, &top_k, 1, nullptr, 0, nullptr, nullptr);
}

TYPED_TEST(SamplingDecodeTest, InvalidArgsZeroTopP)
{
    size_t                     batch_size = this->batch_size;
    float                      top_p      = 0;
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},  // step 0
        {4},
        {4},
        {4},
        {4},
        {4},
        {4},  // step 1
        {2},
        {2},
        {2},
        {2},
        {2},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, nullptr, 0, &top_p, 1, nullptr, nullptr);
}

TYPED_TEST(SamplingDecodeTest, InvalidArgsZeroTopKTopP)
{
    size_t                     batch_size = this->batch_size;
    int                        top_k      = 0;
    float                      top_p      = 0;
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},  // step 0
        {4},
        {4},
        {4},
        {4},
        {4},
        {4},  // step 1
        {2},
        {2},
        {2},
        {2},
        {2},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, &top_k, 1, &top_p, 1, nullptr, nullptr);
}

TYPED_TEST(SamplingDecodeTest, InvalidArgsZeroBatchTopKTopP)
{
    size_t                     batch_size = this->batch_size;
    int*                       top_ks     = new int[batch_size]{0, 0, 0, 0, 0, 0};
    float                      top_p      = 0;
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},  // step 0
        {4},
        {4},
        {4},
        {4},
        {4},
        {4},  // step 1
        {2},
        {2},
        {2},
        {2},
        {2},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, top_ks, batch_size, &top_p, 1, nullptr, nullptr);
    delete[] top_ks;
}

TYPED_TEST(SamplingDecodeTest, InvalidArgsZeroTopKBatchTopP)
{
    size_t                     batch_size = this->batch_size;
    int                        top_k      = 0;
    float*                     top_ps     = new float[batch_size]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},  // step 0
        {4},
        {4},
        {4},
        {4},
        {4},
        {4},  // step 1
        {2},
        {2},
        {2},
        {2},
        {2},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, &top_k, 1, top_ps, batch_size, nullptr, nullptr);
    delete[] top_ps;
}

TYPED_TEST(SamplingDecodeTest, InvalidArgsBatchTopKContainZero)
{
    size_t                     batch_size = this->batch_size;
    int*                       top_ks     = new int[batch_size]{2, 1, 0, 0, 2, 1};
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0, 1},
        {0},
        {0},
        {0},
        {0, 1},
        {0},  // step 0
        {4, 5},
        {4},
        {4},
        {4},
        {4, 5},
        {4},  // step 1
        {2, 3},
        {2},
        {2},
        {2},
        {2, 3},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, top_ks, batch_size, nullptr, 0, nullptr, nullptr);
    delete[] top_ks;
}

TYPED_TEST(SamplingDecodeTest, InvalidArgsBatchTopPContainZero)
{
    size_t                     batch_size = this->batch_size;
    float*                     top_ps     = new float[batch_size]{0.5f, 0.5f, 0.0f, 0.5f, 0.0f, 0.3f};
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0, 1},
        {0, 1},
        {0},
        {0, 1},
        {0},
        {0},  // step 0
        {4, 5},
        {4, 5},
        {4},
        {4, 5},
        {4},
        {4},  // step 1
        {2, 3},
        {2, 3},
        {2},
        {2, 3},
        {2},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, nullptr, 0, top_ps, batch_size, nullptr, nullptr);
    delete[] top_ps;
}

TYPED_TEST(SamplingDecodeTest, InvalidArgsBatchTopKTopPContainZero)
{
    size_t                     batch_size = this->batch_size;
    int*                       top_ks     = new int[batch_size]{2, 2, 1, 0, 2, 0};
    float                      top_p      = 0.0;
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0, 1},
        {0, 1},
        {0},
        {0},
        {0, 1},
        {0},  // step 0
        {4, 5},
        {4, 5},
        {4},
        {4},
        {4, 5},
        {4},  // step 1
        {2, 3},
        {2, 3},
        {2},
        {2},
        {2, 3},
        {2}  // step 2
    };
    this->runTest(expected_output_ids, top_ks, batch_size, &top_p, 1, nullptr, nullptr);
    delete[] top_ks;
}

TYPED_TEST(SamplingDecodeTest, InvalidArgsTopKBatchTopPContainZero)
{
    size_t                     batch_size = this->batch_size;
    int                        top_k      = 0;
    float*                     top_ps     = new float[batch_size]{0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0, 1},
        {0},
        {0},
        {0, 1},  // step 0
        {4},
        {4},
        {4, 5},
        {4},
        {4},
        {4, 5},  // step 1
        {2},
        {2},
        {2, 3},
        {2},
        {2},
        {2, 3}  // step 2
    };
    this->runTest(expected_output_ids, &top_k, 1, top_ps, batch_size, nullptr, nullptr);
    delete[] top_ps;
}

TYPED_TEST(SamplingDecodeTest, InvalidArgsBatchTopKBatchTopPContainZero)
{
    size_t                     batch_size = this->batch_size;
    int*                       top_ks     = new int[batch_size]{0, 2, 1, 2, 2, 0};
    float*                     top_ps     = new float[batch_size]{0.0, 0.3, 0.9, 0.0, 0.3, 0.5};
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0},
        {0, 1},
        {0},
        {0, 1},  // step 0
        {4},
        {4},
        {4},
        {4, 5},
        {4},
        {4, 5},  // step 1
        {2},
        {2},
        {2},
        {2, 3},
        {2},
        {2, 3}  // step 2
    };
    this->runTest(expected_output_ids, top_ks, batch_size, top_ps, batch_size, nullptr, nullptr);
    delete[] top_ks;
    delete[] top_ps;
}

TYPED_TEST(SamplingDecodeTest, LocalBatchBatchTopP)
{
    size_t                     batch_size = this->batch_size;
    float*                     top_ps     = new float[batch_size]{0.3f, 0.5f, 0.5f, 0.3f, 0.5f, 0.5f};
    std::vector<std::set<int>> expected_output_ids{
        {0},
        {0},
        {0, 1},
        {0},
        {0},
        {0},  // step 0
        {0},
        {0},
        {4, 5},
        {4},
        {0},
        {0},  // step 1
        {0},
        {0},
        {2, 3},
        {2},
        {0},
        {0}  // step 2
    };
    this->runTest(expected_output_ids, nullptr, 0, top_ps, batch_size, nullptr, nullptr, true);
    delete[] top_ps;
}

TYPED_TEST(SamplingDecodeTest, LocalBatchBatchTopKBatchTopP)
{
    size_t                     batch_size = this->batch_size;
    int*                       top_ks     = new int[batch_size]{2, 2, 0, 2, 2, 0};
    float*                     top_ps     = new float[batch_size]{0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    std::vector<std::set<int>> expected_output_ids{
        // batch
        {0},
        {0},
        {0, 1},
        {0, 1},
        {0},
        {0},  // step 0
        {0},
        {0},
        {4, 5},
        {4, 5},
        {0},
        {0},  // step 1
        {0},
        {0},
        {2, 3},
        {2, 3},
        {0},
        {0}  // step 2
    };
    this->runTest(expected_output_ids, top_ks, batch_size, top_ps, batch_size, nullptr, nullptr, true);
    delete[] top_ks;
    delete[] top_ps;
}

template<typename T>
class SamplingDecodeTest2: public FtTestBase {

public:
    void SetUp() override
    {
        FtTestBase::SetUp();
        check_cuda_error(cudaGetDeviceProperties(&prop, 0));
        check_cuda_error(cublasCreate(&cublas_handle));
        check_cuda_error(cublasLtCreate(&cublaslt_handle));
        check_cuda_error(cublasSetStream(cublas_handle, stream));
        cublas_algo_map      = new cublasAlgoMap("");
        cublas_wrapper_mutex = new std::mutex();
        cublas_wrapper       = new cublasMMWrapper(
            cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, allocator);
    }
    void TearDown() override
    {
        delete cublas_wrapper;
        delete cublas_wrapper_mutex;
        delete cublas_algo_map;
        check_cuda_error(cublasLtDestroy(cublaslt_handle));
        check_cuda_error(cublasDestroy(cublas_handle));
        FtTestBase::TearDown();
    }

protected:
    using FtTestBase::stream;
    using FtTestBase::allocator;

    struct cudaDeviceProp prop;
    cublasHandle_t        cublas_handle;
    cublasLtHandle_t      cublaslt_handle;
    cublasAlgoMap*        cublas_algo_map;
    std::mutex*           cublas_wrapper_mutex;
    cublasMMWrapper*      cublas_wrapper;

    DataType data_type = getTensorType<T>();

    size_t batch_size;
    size_t beam_width;
    size_t batchxbeam;
    size_t vocab_size;
    size_t max_input_len;
    size_t max_output_len;
    size_t max_seq_len;

    uint  top_k;
    float top_p;
    float temperature;
    float repetition_penalty;
    int   end_id;

    T*     h_logits;
    T*     h_probs;
    T*     h_log_probs;
    float* h_cum_log_probs;
    float* h_output_log_probs;
    int*   h_output_ids;

    T*     d_logits;
    int*   d_input_lengths;
    float* d_cum_log_probs;
    float* d_output_log_probs;
    int*   d_output_ids;
    int*   d_end_ids;

    void setup(SamplingLayerTestParam param)
    {
        batch_size     = param.batch_size;
        beam_width     = param.beam_width;
        batchxbeam     = batch_size * param.beam_width;
        vocab_size     = param.vocab_size;
        max_input_len  = 0;
        max_output_len = param.output_len;
        max_seq_len    = max_input_len + max_output_len;

        top_k = param.top_k;
        top_p = param.top_p;
        // use default values having no effect.
        temperature        = 1.0f;
        repetition_penalty = 1.0f;
        end_id             = 0;

        h_logits     = new T[batchxbeam * vocab_size];
        h_output_ids = new int[batchxbeam];

        d_logits        = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batchxbeam * vocab_size));
        d_input_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batchxbeam));
        d_output_ids    = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batchxbeam));
        d_end_ids       = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));

        // Init by zero.
        deviceFill(d_input_lengths, batchxbeam, 0, stream);
        deviceFill(d_output_ids, max_seq_len * batchxbeam, 0, stream);
        deviceFill(d_end_ids, batch_size, end_id);
    }

    void teardown()
    {
        delete[] h_logits;
        delete[] h_output_ids;
    }

    void runCurandTest(SamplingLayerTestParam param, bool use_local_batch, bool use_single_random_seed)
    {
        setup(param);
        const DataType data_type = getTensorType<T>();

        const size_t local_batch_size = use_local_batch ? 3 : batch_size;
        assert(batch_size % local_batch_size == 0);

        DynamicDecodeLayer<T>* dynamic_decode_layer = new DynamicDecodeLayer<T>(vocab_size,
                                                                                vocab_size,
                                                                                end_id,
                                                                                stream,
                                                                                cublas_wrapper,
                                                                                allocator,
                                                                                false,   // is_free_buffer_after_forward
                                                                                &prop);  // cuda_device_prop

        // Prepare decoding arguments
        const size_t        random_seed_size = use_single_random_seed ? 1 : batch_size;
        const size_t        period_size      = 3;
        unsigned long long* random_seed      = new unsigned long long[random_seed_size];
        for (size_t i = 0; i < random_seed_size; ++i) {
            random_seed[i] = i / period_size;
        }

        TensorMap runtime_args;
        runtime_args.insert({"random_seed", Tensor(MEMORY_CPU, TYPE_UINT64, {random_seed_size}, random_seed)});
        runtime_args.insert({"runtime_top_k", Tensor(MEMORY_CPU, TYPE_UINT32, {1}, &top_k)});
        runtime_args.insert({"runtime_top_p", Tensor(MEMORY_CPU, TYPE_FP32, {1}, &top_p)});
        dynamic_decode_layer->setup(batch_size, beam_width, &runtime_args);

        for (size_t step = max_input_len; step < max_output_len; ++step) {
            const size_t iteration_num = batch_size / local_batch_size;
            initRandom(h_logits, beam_width * vocab_size, -3.0f, 3.0f);
            tile(h_logits, batch_size, beam_width * vocab_size);
            cudaH2Dcpy(d_logits, h_logits, batchxbeam * vocab_size);

            for (uint ite = 0; ite < iteration_num; ++ite) {
                TensorMap dynamic_decode_input_tensors(
                    {{"logits", Tensor{MEMORY_GPU, data_type, {batch_size, beam_width, vocab_size}, d_logits}},
                     {"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size}, nullptr}},
                     {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                     {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_len}},
                     {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, d_input_lengths}},
                     {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                     {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &local_batch_size}},
                     {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, d_end_ids}},
                     {"random_seed", {MEMORY_CPU, TYPE_UINT64, {random_seed_size}, random_seed}},
                     {"runtime_top_k", {MEMORY_CPU, TYPE_UINT32, {1}, &top_k}},
                     {"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &top_p}}});

                // common outputs
                TensorMap dynamic_decode_output_tensors(
                    {{"output_ids",
                      Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, d_output_ids}},
                     {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, nullptr}},
                     {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, nullptr}}});

                dynamic_decode_layer->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
                sync_check_cuda_error();

                // check results.
                cudaD2Hcpy(h_output_ids,
                           dynamic_decode_output_tensors.at("output_ids").getPtrWithOffset<int>(step * batchxbeam),
                           batchxbeam);
            }
            // The same seed produces the same random number.
            for (size_t i = 0; i + period_size - 1 < batchxbeam; i += period_size) {
                for (size_t j = 1; j < period_size; ++j) {
                    EXPECT_TRUE(h_output_ids[i] == h_output_ids[i + j])
                        << fmtstr("Fail at step %u val[%d]=%d <> val[%d]=%d",
                                  step,
                                  i,
                                  h_output_ids[i],
                                  i + j,
                                  h_output_ids[i + j]);
                }
            }
        }
        delete dynamic_decode_layer;
        delete[] random_seed;
        teardown();
    }

    void runCumLogProbTest(SamplingLayerTestParam param)
    {
        setup(param);
        unsigned long long     seed                 = 43;
        const DataType         data_type            = getTensorType<T>();
        DynamicDecodeLayer<T>* dynamic_decode_layer = new DynamicDecodeLayer<T>(vocab_size,
                                                                                vocab_size,
                                                                                end_id,
                                                                                stream,
                                                                                cublas_wrapper,
                                                                                allocator,
                                                                                false,   // is_free_buffer_after_forward
                                                                                &prop);  // cuda_device_prop

        // Logit values in the host of shape ((batch_size x beam) x vocab_size) where beam = 1.
        // T* h_logits = new T[batch_size * beam_width * vocab_size];
        T*     h_probs                = new T[batch_size * beam_width * vocab_size];
        T*     h_log_probs            = new T[batch_size * beam_width * vocab_size];
        float* h_cum_log_probs        = new float[batch_size * beam_width];
        float* h_output_log_probs     = new float[max_output_len * batch_size * beam_width];
        float* expected_cum_log_probs = new float[batch_size * beam_width];
        initRandom(h_logits, batch_size * beam_width * vocab_size, -3.0f, 3.0f);
        computeProb(h_probs, h_logits, batch_size * beam_width, vocab_size);
        computeLogProb(h_log_probs, h_logits, batch_size * beam_width, vocab_size);
        std::fill_n(expected_cum_log_probs, batch_size * beam_width, 0);

        int* tiled_input_lengths_buf = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size * beam_width));
        float* cum_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size * beam_width));
        float* output_log_probs =
            reinterpret_cast<float*>(allocator->malloc(sizeof(float) * max_output_len * batch_size * beam_width));

        int* output_ids =
            reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batch_size * beam_width));
        int* h_output_ids = new int[batch_size * beam_width];

        int* end_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
        deviceFill(end_ids, batch_size, end_id);

        // Init by zero.
        cudaMemset(cum_log_probs, 0, sizeof(float) * batch_size * beam_width);
        cudaMemset(output_log_probs, 0, sizeof(float) * max_output_len * batch_size * beam_width);
        cudaMemset(output_ids, 0, sizeof(int) * max_seq_len * batch_size * beam_width);

        TensorMap input_tensors({{"random_seed", {MEMORY_CPU, TYPE_INT32, {1}, &seed}},
                                 {"runtime_top_k", {MEMORY_CPU, TYPE_UINT32, {1}, &top_k}},
                                 {"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &top_p}},
                                 {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &temperature}},
                                 {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &repetition_penalty}}});
        dynamic_decode_layer->setup(batch_size, beam_width, &input_tensors);

        for (size_t step = max_input_len; step < max_output_len; ++step) {
            uint ite = 0;
            // Reset by the test value since the sampling layer internally update the logit buffer (making it log-prob).
            cudaH2Dcpy(d_logits, h_logits, batch_size * beam_width * vocab_size);
            TensorMap dynamic_decode_input_tensors(
                {{"logits", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size, beam_width, vocab_size}, d_logits}},
                 {"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size}, nullptr}},
                 {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                 {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_len}},
                 {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, tiled_input_lengths_buf}},
                 {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                 {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &batch_size}},
                 {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids}},
                 {"random_seed", {MEMORY_CPU, TYPE_UINT64, {1}, &seed}},
                 {"runtime_top_k", {MEMORY_CPU, TYPE_UINT32, {1}, &top_k}},
                 {"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &top_p}},
                 {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &temperature}},
                 {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &repetition_penalty}}});

            // common outputs
            TensorMap dynamic_decode_output_tensors(
                {{"output_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, output_ids}},
                 {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, nullptr}},
                 {"cum_log_probs", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size * beam_width}, cum_log_probs}},
                 {"output_log_probs",
                  Tensor{MEMORY_GPU, TYPE_FP32, {max_seq_len, batch_size, beam_width}, output_log_probs}},
                 {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, nullptr}}});

            dynamic_decode_layer->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);

            TM_LOG_DEBUG("Step %2d generated ids", step);
            cudaD2Hcpy(
                h_output_ids,
                dynamic_decode_output_tensors.at("output_ids").getPtrWithOffset<int>(step * (batch_size * beam_width)),
                batch_size * beam_width);
            cudaD2Hcpy(h_cum_log_probs, cum_log_probs, batch_size * beam_width);
            cudaD2Hcpy(h_output_log_probs, output_log_probs, max_output_len * batch_size * beam_width);
            for (size_t i = 0; i < batch_size * beam_width; ++i) {
                int idx = i * vocab_size + h_output_ids[i];
                expected_cum_log_probs[i] += (float)h_log_probs[idx];
                TM_LOG_DEBUG("| step %2d batch %2d idx %7d id %6d | log-prob %9.4f (expt: %9.4f) "
                             "| cum-log-prob %9.4f (expt: %9.4f) | prob %9.4e",
                             (int)step,
                             (int)i,
                             (int)idx,
                             (int)h_output_ids[i],
                             h_output_log_probs[step * batch_size * beam_width + i],
                             (float)h_log_probs[idx],
                             h_cum_log_probs[i],
                             expected_cum_log_probs[i],
                             (float)h_probs[idx]);
            }
            TM_LOG_DEBUG("");
        }

        bool passed = checkResult(param.toString(), cum_log_probs, expected_cum_log_probs, batch_size * beam_width);
        EXPECT_TRUE(passed);

        delete[] expected_cum_log_probs;
        delete[] h_output_log_probs;
        delete[] h_cum_log_probs;
        delete[] h_log_probs;
        delete[] h_probs;

        delete dynamic_decode_layer;
    }
};

TYPED_TEST_SUITE(SamplingDecodeTest2, FloatType);

TYPED_TEST(SamplingDecodeTest2, CorrectnessSingleRandTopK)
{
    // test TopKSampling
    this->runCurandTest({113, 1201, 1, 3, 1.0f, 5}, false, true);
}

TYPED_TEST(SamplingDecodeTest2, CorrectnessSingleRandTopP)
{
    this->runCurandTest({113, 1201, 1, 0, 1.0f, 5}, false, true);
}

TYPED_TEST(SamplingDecodeTest2, CorrectnessBatchRandTopK)
{
    // test TopKSampling
    this->runCurandTest({113, 1201, 1, 3, 1.0f, 5}, false, false);
}

TYPED_TEST(SamplingDecodeTest2, CorrectnessBatchRandTopP)
{
    this->runCurandTest({113, 1201, 1, 0, 1.0f, 5}, false, false);
}

TYPED_TEST(SamplingDecodeTest2, CorrectnessBatchRandTopKLocalBatch)
{
    // test TopKSampling
    this->runCurandTest({99, 1201, 1, 3, 1.0f, 5}, true, false);
}

TYPED_TEST(SamplingDecodeTest2, CorrectnessBatchRandTopPLocalBatch)
{
    this->runCurandTest({99, 1201, 1, 0, 1.0f, 5}, true, false);
}

TYPED_TEST(SamplingDecodeTest2, CorrectnessCumLogProbTopK)
{
    this->runCumLogProbTest({99, 1201, 1, 5, 1.0f, 5});
}

TYPED_TEST(SamplingDecodeTest2, CorrectnessCumLogProbTopP)
{
    this->runCumLogProbTest({99, 1201, 1, 0, 1.0f, 5});
}
