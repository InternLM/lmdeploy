#include <algorithm>  // std::min, std::max
#include <iostream>   // snprintf
#include <math.h>     // expf, log
#include <stdlib.h>   // rand
#include <string>     // std::string
#include <vector>     // std::vector

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "src/fastertransformer/kernels/sampling_topk_kernels.h"
#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopKSamplingLayer.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

#include "tests/unittests/unittest_utils.h"

using namespace fastertransformer;

struct TestCase {
    std::string name;
    size_t      batch_size;
    size_t      vocab_size;
    size_t      beam_width;
    size_t      top_k;
    float       top_p;
    size_t      output_len;

    std::string toString()
    {
        char buf[100];
        snprintf(buf,
                 sizeof(buf),
                 "TestCase[name=%s, batch=%ld, vocab=%ld, beam=%ld, k=%ld, p=%3.1f, output_len=%ld]",
                 name.c_str(),
                 batch_size,
                 vocab_size,
                 beam_width,
                 top_k,
                 top_p,
                 output_len);
        return buf;
    }

    void print()
    {
        FT_LOG_INFO(toString());
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

/////////////////////////////////// Tests //////////////////////////////////////////

template<typename T>
void testCumLogProbComputation(TestCase tc)
{

    bool is_fp32 = std::is_same<T, float>::value;

    size_t             beam_width = tc.beam_width;
    uint               top_k      = tc.top_k;
    float              top_p      = tc.top_p;
    unsigned long long seed       = 0;
    // use default values having no effect.
    float temperature        = 1.0f;
    float len_penalty        = 0.0f;
    float repetition_penalty = 1.0f;

    size_t batch_size     = tc.batch_size;
    size_t vocab_size     = tc.vocab_size;
    int    end_id         = 3;
    size_t max_input_len  = 0;  // has no effect.
    size_t max_output_len = tc.output_len;
    size_t max_seq_len    = max_input_len + max_output_len;

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    check_cuda_error(cudaStreamCreate(&stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));

    cublasAlgoMap                   cublas_algo_map(GEMM_CONFIG);
    Allocator<AllocatorType::CUDA>* allocator = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator->setStream(stream);

    std::mutex*      cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper* cublas_wrapper =
        new cublasMMWrapper(cublas_handle, cublaslt_handle, stream, &cublas_algo_map, cublas_wrapper_mutex, allocator);

    DynamicDecodeLayer<T>* dynamic_decode_layer = new DynamicDecodeLayer<T>(vocab_size,
                                                                            vocab_size,
                                                                            end_id,
                                                                            stream,
                                                                            cublas_wrapper,
                                                                            allocator,
                                                                            false,   // is_free_buffer_after_forward
                                                                            &prop);  // cuda_device_prop

    const DataType data_type   = getTensorType<T>();
    size_t         logits_size = batch_size * beam_width * vocab_size;
    T*             logits_buf  = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * logits_size, true));

    // Logit values in the host of shape ((batch_size x beam) x vocab_size) where beam = 1.
    T*     h_logits               = new T[batch_size * beam_width * vocab_size];
    T*     h_probs                = new T[batch_size * beam_width * vocab_size];
    T*     h_log_probs            = new T[batch_size * beam_width * vocab_size];
    float* h_cum_log_probs        = new float[batch_size * beam_width];
    float* h_output_log_probs     = new float[max_output_len * batch_size * beam_width];
    float* expected_cum_log_probs = new float[batch_size * beam_width];
    initRandom(h_logits, batch_size * beam_width * vocab_size, -10.0f / vocab_size, -1.0f);
    computeProb(h_probs, h_logits, batch_size * beam_width, vocab_size);
    computeLogProb(h_log_probs, h_logits, batch_size * beam_width, vocab_size);
    memset(expected_cum_log_probs, 0, sizeof(float) * batch_size * beam_width);

#ifndef NDEBUG
    FT_LOG_DEBUG("logit values");
    printMatrixWithLimit(h_logits, batch_size * beam_width, vocab_size, vocab_size, false);
    FT_LOG_DEBUG("\nprob values");
    printMatrixWithLimit(h_probs, batch_size * beam_width, vocab_size, vocab_size, false);
    FT_LOG_DEBUG("\nlog-prob values");
    printMatrixWithLimit(h_log_probs, batch_size * beam_width, vocab_size, vocab_size, false);
#endif

    int*   tiled_input_lengths_buf = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size * beam_width));
    float* cum_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size * beam_width));
    float* output_log_probs =
        reinterpret_cast<float*>(allocator->malloc(sizeof(float) * max_output_len * batch_size * beam_width));

    int* output_ids   = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batch_size * beam_width));
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
                             {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &len_penalty}},
                             {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &repetition_penalty}}});
    dynamic_decode_layer->setup(batch_size, beam_width, &input_tensors);

    for (size_t step = max_input_len; step < max_output_len; ++step) {
        uint ite = 0;
        // Reset by the test value since the sampling layer internally update the logit buffer (making it log-prob).
        cudaH2Dcpy(logits_buf, h_logits, logits_size);
        TensorMap dynamic_decode_input_tensors(
            {{"logits", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size, beam_width, vocab_size}, logits_buf}},
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
             {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &len_penalty}},
             {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &repetition_penalty}}});

        // common outputs
        TensorMap dynamic_decode_output_tensors(
            {{"output_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, output_ids}},
             {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, nullptr}},
             {"cum_log_probs", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size * beam_width}, cum_log_probs}},
             {"output_log_probs",
              Tensor{MEMORY_GPU, TYPE_FP32, {max_seq_len, batch_size, beam_width}, output_log_probs}},
             {"parent_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, nullptr}},
             {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, nullptr}},
             // necessary for beam search.
             {"tgt_cache_indirection",
              Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width, max_output_len}, nullptr}}});

        dynamic_decode_layer->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);

        FT_LOG_DEBUG("Step %2d generated ids", step);
        cudaD2Hcpy(
            h_output_ids,
            (int*)dynamic_decode_output_tensors.at("output_ids").getPtrWithOffset(step * (batch_size * beam_width)),
            batch_size * beam_width);
        cudaD2Hcpy(h_cum_log_probs, cum_log_probs, batch_size * beam_width);
        cudaD2Hcpy(h_output_log_probs, output_log_probs, max_output_len * batch_size * beam_width);
        for (size_t i = 0; i < batch_size * beam_width; ++i) {
            int idx = i * vocab_size + h_output_ids[i];
            expected_cum_log_probs[i] += (float)h_log_probs[idx];
            FT_LOG_DEBUG("| step %2d batch %2d idx %7d id %6d | log-prob %9.4f (expt: %9.4f) "
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
        FT_LOG_DEBUG("");

#ifndef NDEBUG
        // print output ids
        for (size_t s = max_input_len; s < max_seq_len; ++s) {
            cudaD2Hcpy(
                h_output_ids,
                (int*)dynamic_decode_output_tensors.at("output_ids").getPtrWithOffset(s * (batch_size * beam_width)),
                batch_size * beam_width);
            printf("%02d ", (int)s);
            for (size_t b = 0; b < batch_size; ++b) {
                printf("%3d ", (int)h_output_ids[b]);
            }
            printf("\n");
        }
#endif
    }
    std::string tag    = tc.toString() + (is_fp32 ? " (fp32)" : " (fp16)");
    bool        passed = checkResult(tag, cum_log_probs, expected_cum_log_probs, batch_size * beam_width);
    EXPECT_TRUE(passed);

    delete[] expected_cum_log_probs;
    delete[] h_output_log_probs;
    delete[] h_cum_log_probs;
    delete[] h_logits;
    delete[] h_log_probs;
    delete[] h_probs;
    delete[] h_output_ids;

    delete dynamic_decode_layer;
    delete cublas_wrapper;
    delete allocator;
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cublasDestroy(cublas_handle));
    check_cuda_error(cublasLtDestroy(cublaslt_handle));
}

void printTensors(TensorMap* map, size_t limit = 8)
{
    FT_LOG_INFO("Tensors:");
    for (auto& kv : *map) {
        Tensor t = kv.second;
        FT_LOG_INFO(" - %-18s : %s", kv.first.c_str(), t.toString().c_str());
    }
}

template<typename T>
class SamplingDecodeTest {
private:
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

    Allocator<AllocatorType::CUDA>* allocator;
    std::mutex*                     cublas_wrapper_mutex;
    cublasMMWrapper*                cublas_wrapper;
    DynamicDecodeLayer<T>*          dynamic_decode_layer;

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
        struct cudaDeviceProp prop;
        check_cuda_error(cudaGetDeviceProperties(&prop, 0));
        cudaStream_t     stream;
        cublasHandle_t   cublas_handle;
        cublasLtHandle_t cublaslt_handle;
        check_cuda_error(cudaStreamCreate(&stream));
        check_cuda_error(cublasCreate(&cublas_handle));
        check_cuda_error(cublasLtCreate(&cublaslt_handle));
        check_cuda_error(cublasSetStream(cublas_handle, stream));
        cublasAlgoMap cublas_algo_map(GEMM_CONFIG);
        allocator = new Allocator<AllocatorType::CUDA>(getDevice());
        allocator->setStream(stream);
        cublas_wrapper_mutex = new std::mutex();
        cublas_wrapper       = new cublasMMWrapper(
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

    bool checkResult(std::string name, int* d_output_ids, std::vector<std::set<int>>& expected_ids)
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
                    ss << " - Fail " << name << " (step=" << s << ", batch=" << b << ") "
                       << "actual=" << h_output_ids[i] << ", expected";
                    for (auto& expt : expts) {
                        ss << " " << expt;
                    }
                    FT_LOG_DEBUG("%s", ss.str().c_str());
                }
                ++failures;
            }
        }
        delete[] h_output_ids;
        FT_LOG_DEBUG("check...%6s : %s (failures: %d / %d)",
                     failures == 0 ? "....OK" : "FAILED",
                     name.c_str(),
                     failures,
                     max_seq_len * batchxbeam);
        return failures == 0;
    }

    bool testSampling(std::string                name,
                      std::vector<std::set<int>> expected_output_ids,
                      int*                       top_ks,
                      size_t                     top_k_size,
                      float*                     top_ps,
                      size_t                     top_p_size,
                      float*                     temperature,
                      float*                     repetition_penalty)
    {
        FT_LOG_INFO("Test %s", name.c_str());
        std::string tag    = fmtstr("Test %s T=%s", name.c_str(), std::is_same<T, float>::value ? "fp32" : "fp16");
        bool        passed = true;
        for (unsigned long long seed = 0; seed < max_seed; ++seed) {
            this->setup(seed);
            size_t     step = max_input_len;
            uint       ite  = 0;
            TensorMap* input_tensors =
                createInputTensors(top_ks, top_k_size, top_ps, top_p_size, temperature, repetition_penalty);
            input_tensors->insert({"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}});
            input_tensors->insert({"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}});
            input_tensors->insert({"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &batch_size}});
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
            bool is_ok = checkResult(tag + fmtstr(" seed=%lld", seed), d_output_ids, expected_output_ids);
            passed &= is_ok;
#ifndef NDEBUG
            if (!is_ok) {
                FT_LOG_ERROR("actual output ids");
                printMatrix(d_output_ids, max_seq_len, batch_size, batch_size, true);
            }
#endif
            delete output_tensors;
            delete input_tensors;
            this->teardown();
        }
        FT_LOG_INFO("check...%6s : %s", passed ? "....OK" : "FAILED", tag.c_str());
        return passed;
    }

    bool testSamplingWithLocalBatch(std::string                name,
                                    std::vector<std::set<int>> expected_output_ids,
                                    int*                       top_ks,
                                    size_t                     top_k_size,
                                    float*                     top_ps,
                                    size_t                     top_p_size,
                                    float*                     temperature,
                                    float*                     repetition_penalty)
    {
        FT_LOG_INFO("Test %s", name.c_str());
        std::string tag    = fmtstr("Test %s T=%s", name.c_str(), std::is_same<T, float>::value ? "fp32" : "fp16");
        bool        passed = true;
        size_t      local_batch_size = 2;
        uint        ite              = 1;
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
            bool is_ok = checkResult(tag + fmtstr(" seed=%lld", seed), d_output_ids, expected_output_ids);
            passed &= is_ok;
#ifndef NDEBUG
            if (!is_ok) {
                FT_LOG_ERROR("actual output ids");
                printMatrix(d_output_ids, max_seq_len, batch_size, batch_size, true);
            }
#endif
            delete output_tensors;
            delete input_tensors;
            this->teardown();
        }
        FT_LOG_INFO("check...%6s : %s", passed ? "....OK" : "FAILED", tag.c_str());
        return passed;
    }

public:
    void testTopK()
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
        bool passed = this->testSampling("TopK", expected_output_ids, &top_k, 1, nullptr, 0, nullptr, nullptr);
        EXPECT_TRUE(true);
    }

    void testBatchTopK()
    {
        int*                       top_ks = new int[batch_size]{2, 1, 1, 2, 1, 1};
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
        bool passed =
            this->testSampling("BatchTopK", expected_output_ids, top_ks, batch_size, nullptr, 0, nullptr, nullptr);
        delete[] top_ks;
        EXPECT_TRUE(passed);
    }

    void testTopP()
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
        bool passed = this->testSampling("TopP", expected_output_ids, nullptr, 0, &top_p, 1, nullptr, nullptr);
        EXPECT_TRUE(true);
    }

    void testBatchTopP()
    {
        float*                     top_ps = new float[batch_size]{0.3f, 0.5f, 0.5f, 0.3f, 0.5f, 0.5f};
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
        bool passed =
            this->testSampling("BatchTopP", expected_output_ids, nullptr, 0, top_ps, batch_size, nullptr, nullptr);
        delete[] top_ps;
        EXPECT_TRUE(passed);
    }

    void testTopKTopP()
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
        bool passed = this->testSampling("TopP", expected_output_ids, &top_k, 1, &top_p, 1, nullptr, nullptr);
        EXPECT_TRUE(true);
    }

    void testBatchTopKTopP()
    {
        std::string                name   = "BatchTopKTopP";
        int*                       top_ks = new int[batch_size]{2, 2, 1, 2, 2, 1};
        float                      top_p  = 0.3;
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
        bool passed = this->testSampling(name, expected_output_ids, top_ks, batch_size, &top_p, 1, nullptr, nullptr);
        delete[] top_ks;
        EXPECT_TRUE(passed);
    }

    void testTopKBatchTopP()
    {
        std::string                name   = "TopKBatchTopP";
        int                        top_k  = 2;
        float*                     top_ps = new float[batch_size]{0.5, 0.3, 0.5, 0.5, 0.3, 0.5};
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
        bool passed = this->testSampling(name, expected_output_ids, &top_k, 1, top_ps, batch_size, nullptr, nullptr);
        delete[] top_ps;
        EXPECT_TRUE(passed);
    }

    void testBatchTopKBatchTopP()
    {
        std::string                name   = "BatchTopKBatchTopP";
        int*                       top_ks = new int[batch_size]{2, 2, 0, 2, 2, 0};
        float*                     top_ps = new float[batch_size]{0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
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
        bool passed =
            this->testSampling(name, expected_output_ids, top_ks, batch_size, top_ps, batch_size, nullptr, nullptr);
        delete[] top_ks;
        delete[] top_ps;
        EXPECT_TRUE(passed);
    }

    void testInvalidArgsZeroTopK()
    {
        std::string                name  = "InvalidArgsZeroTopK";
        int                        top_k = 0;
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
        bool passed = this->testSampling(name, expected_output_ids, &top_k, 1, nullptr, 0, nullptr, nullptr);
        EXPECT_TRUE(passed);
    }

    void testInvalidArgsZeroTopP()
    {
        std::string                name  = "InvalidArgsZeroTopP";
        float                      top_p = 0;
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
        bool passed = this->testSampling(name, expected_output_ids, nullptr, 0, &top_p, 1, nullptr, nullptr);
        EXPECT_TRUE(passed);
    }

    void testInvalidArgsZeroTopKTopP()
    {
        std::string                name  = "InvalidArgsZeroTopKTopP";
        int                        top_k = 0;
        float                      top_p = 0;
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
        bool passed = this->testSampling(name, expected_output_ids, &top_k, 1, &top_p, 1, nullptr, nullptr);
        EXPECT_TRUE(passed);
    }

    void testInvalidArgsZeroBatchTopKTopP()
    {
        std::string                name   = "InvalidArgsZeroBatchTopKTopP";
        int*                       top_ks = new int[batch_size]{0, 0, 0, 0, 0, 0};
        float                      top_p  = 0;
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
        bool passed = this->testSampling(name, expected_output_ids, top_ks, batch_size, &top_p, 1, nullptr, nullptr);
        delete[] top_ks;
        EXPECT_TRUE(passed);
    }

    void testInvalidArgsZeroTopKBatchTopP()
    {
        std::string                name   = "InvalidArgsZeroTopKBatchTopP";
        int                        top_k  = 0;
        float*                     top_ps = new float[batch_size]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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
        bool passed = this->testSampling(name, expected_output_ids, &top_k, 1, top_ps, batch_size, nullptr, nullptr);
        delete[] top_ps;
        EXPECT_TRUE(passed);
    }

    void testInvalidArgsBatchTopKContainZero()
    {
        std::string                name   = "InvalidArgsBatchTopKContainZero";
        int*                       top_ks = new int[batch_size]{2, 1, 0, 0, 2, 1};
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
        bool passed = this->testSampling(name, expected_output_ids, top_ks, batch_size, nullptr, 0, nullptr, nullptr);
        delete[] top_ks;
        EXPECT_TRUE(passed);
    }

    void testInvalidArgsBatchTopPContainZero()
    {
        std::string                name   = "InvalidArgsBatchTopPContainZero";
        float*                     top_ps = new float[batch_size]{0.5f, 0.5f, 0.0f, 0.5f, 0.0f, 0.3f};
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
        bool passed = this->testSampling(name, expected_output_ids, nullptr, 0, top_ps, batch_size, nullptr, nullptr);
        delete[] top_ps;
        EXPECT_TRUE(passed);
    }

    void testInvalidArgsBatchTopKTopPContainZero()
    {
        std::string                name   = "InvalidArgsBatchTopKTopPContainZero";
        int*                       top_ks = new int[batch_size]{2, 2, 1, 0, 2, 0};
        float                      top_p  = 0.0;
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
        bool passed = this->testSampling(name, expected_output_ids, top_ks, batch_size, &top_p, 1, nullptr, nullptr);
        delete[] top_ks;
        EXPECT_TRUE(passed);
    }

    void testInvalidArgsTopKBatchTopPContainZero()
    {
        std::string                name   = "InvalidArgsTopKBatchTopPContainZero";
        int                        top_k  = 0;
        float*                     top_ps = new float[batch_size]{0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
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
        bool passed = this->testSampling(name, expected_output_ids, &top_k, 1, top_ps, batch_size, nullptr, nullptr);
        delete[] top_ps;
        EXPECT_TRUE(passed);
    }

    void testInvalidArgsBatchTopKBatchTopPContainZero()
    {
        std::string                name   = "InvalidArgsBatchTopKBatchTopPContainZero";
        int*                       top_ks = new int[batch_size]{0, 2, 1, 2, 2, 0};
        float*                     top_ps = new float[batch_size]{0.0, 0.3, 0.9, 0.0, 0.3, 0.5};
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
        bool passed =
            this->testSampling(name, expected_output_ids, top_ks, batch_size, top_ps, batch_size, nullptr, nullptr);
        delete[] top_ks;
        delete[] top_ps;
        EXPECT_TRUE(passed);
    }

    void testLocalBatchBatchTopP()
    {
        std::string                name   = "LocalBatch_BatchTopP";
        float*                     top_ps = new float[batch_size]{0.3f, 0.5f, 0.5f, 0.3f, 0.5f, 0.5f};
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
        bool passed = this->testSamplingWithLocalBatch(
            name, expected_output_ids, nullptr, 0, top_ps, batch_size, nullptr, nullptr);
        delete[] top_ps;
        EXPECT_TRUE(passed);
    }

    void testLocalBatchBatchTopKBatchTopP()
    {
        std::string                name   = "LocalBatch_BatchTopKBatchTopP";
        int*                       top_ks = new int[batch_size]{2, 2, 0, 2, 2, 0};
        float*                     top_ps = new float[batch_size]{0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
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
        bool passed = this->testSamplingWithLocalBatch(
            name, expected_output_ids, top_ks, batch_size, top_ps, batch_size, nullptr, nullptr);
        delete[] top_ks;
        delete[] top_ps;
        EXPECT_TRUE(passed);
    }

    void testAll()
    {
        this->testTopK();
        this->testTopP();
        this->testTopKTopP();
        this->testBatchTopK();
        this->testBatchTopP();
        this->testBatchTopKTopP();
        this->testTopKBatchTopP();
        this->testBatchTopKBatchTopP();
        this->testInvalidArgsZeroTopK();
        this->testInvalidArgsZeroTopP();
        this->testInvalidArgsZeroBatchTopKTopP();
        this->testInvalidArgsZeroTopKBatchTopP();
        this->testInvalidArgsZeroTopKTopP();
        this->testInvalidArgsBatchTopKContainZero();
        this->testInvalidArgsBatchTopPContainZero();
        this->testInvalidArgsBatchTopKTopPContainZero();
        this->testInvalidArgsTopKBatchTopPContainZero();
        this->testInvalidArgsBatchTopKBatchTopPContainZero();
        this->testLocalBatchBatchTopP();
        this->testLocalBatchBatchTopKBatchTopP();
    }
};

__global__ void generateRandomNumber(unsigned int* vals, curandState_t* states, const int batch_size)
{
    int idx = threadIdx.x;
    if (idx < batch_size) {
        vals[idx] = curand(states + idx);
    }
}

template<typename T>
static inline bool isEqualInPeriod(T* vals, size_t size, size_t period_size)
{
    // The same seed produces the same random number.
    for (size_t i = 0; i + period_size - 1 < size; i += period_size) {
        for (size_t j = 1; j < period_size; ++j) {
            if (vals[i] != vals[i + j]) {
                FT_LOG_INFO(" **** *** ** * [%d] %d <> [%d] %d", i, vals[i], i + j, vals[i + j]);
                return false;
            }
        }
    }
    return true;
}

template<typename T>
static inline bool isEqualInPeriod(T* vals, size_t size, size_t period_size, size_t except)
{
    // The same seed produces the same random number.
    for (size_t i = 0; i + period_size - 1 < size; i += period_size) {
        for (size_t j = 1; j < period_size; ++j) {
            if (j != except && vals[i] != vals[i + j]) {
                FT_LOG_INFO(" **** *** ** * [%d] %d <> [%d] %d", i, vals[i], i + j, vals[i + j]);
                return false;
            }
        }
    }
    return true;
}

void testCuandBatchInitialize(const size_t batch_size)
{
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
    bool passed = isEqualInPeriod(h_rand_vals, batch_size, period_size);
    FT_LOG_INFO("CuandBatchInitTest check....... : %s", passed ? "OK" : "FAILED");
    EXPECT_TRUE(passed);

    delete h_rand_vals;
    delete h_random_seeds;

    check_cuda_error(cudaFree(d_rand_vals));
    check_cuda_error(cudaFree(d_random_seeds));
    check_cuda_error(cudaFree(curand_states));
    check_cuda_error(cudaStreamDestroy(stream));
}

template<typename T, bool SINGLE_RANDOM_SEED, bool HAS_DIFF_ARGS, bool USE_LOCAL_BATCH>
void testSamplingLayerCurandInit(TestCase tc)
{
    FT_LOG_DEBUG("testSamplingLayerCurandInit %s", tc.toString().c_str());
    const DataType data_type = getTensorType<T>();

    const size_t beam_width = 1;
    const uint   top_k      = tc.top_k;
    const float  top_p      = tc.top_p;
    // use default values having no effect.
    const float temperature        = 1.0f;
    const float len_penalty        = 0.0f;
    const float repetition_penalty = 1.0f;
    const int   end_id             = 3;

    const size_t batch_size       = tc.batch_size;
    const size_t batchxbeam       = batch_size * beam_width;
    const size_t local_batch_size = USE_LOCAL_BATCH ? 2 : batch_size;
    assert(batch_size % local_batch_size == 0);
    const size_t vocab_size     = tc.vocab_size;
    const size_t max_input_len  = 0;  // has no effect.
    const size_t max_output_len = tc.output_len;
    const size_t max_seq_len    = max_input_len + max_output_len;

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    check_cuda_error(cudaStreamCreate(&stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));
    cublasAlgoMap                   cublas_algo_map(GEMM_CONFIG);
    std::mutex*                     cublas_wrapper_mutex = new std::mutex();
    Allocator<AllocatorType::CUDA>* allocator            = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator->setStream(stream);
    cublasMMWrapper* cublas_wrapper =
        new cublasMMWrapper(cublas_handle, cublaslt_handle, stream, &cublas_algo_map, cublas_wrapper_mutex, allocator);
    DynamicDecodeLayer<T>* dynamic_decode_layer = new DynamicDecodeLayer<T>(vocab_size,
                                                                            vocab_size,
                                                                            end_id,
                                                                            stream,
                                                                            cublas_wrapper,
                                                                            allocator,
                                                                            false,   // is_free_buffer_after_forward
                                                                            &prop);  // cuda_device_prop

    T*   h_logits     = reinterpret_cast<T*>(malloc(sizeof(T) * batchxbeam * vocab_size));
    int* h_output_ids = reinterpret_cast<int*>(malloc(sizeof(int) * batchxbeam));

    T*   d_logits_buf    = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batchxbeam * vocab_size));
    int* d_input_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batchxbeam));
    int* d_output_ids    = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batchxbeam));

    // Init by zero.
    cudaMemset(d_input_lengths, 0, sizeof(int) * batchxbeam);
    cudaMemset(d_output_ids, 0, sizeof(int) * max_seq_len * batchxbeam);

    // Prepare decoding arguments
    const size_t        random_seed_size = SINGLE_RANDOM_SEED ? 1 : batch_size;
    const size_t        period_size      = 3;
    unsigned long long* random_seed      = new unsigned long long[random_seed_size];
    for (size_t i = 0; i < random_seed_size; ++i) {
        random_seed[i] = i / period_size;
    }
    const bool   has_diff_runtime_args = HAS_DIFF_ARGS;
    const size_t runtime_args_size     = has_diff_runtime_args ? batch_size : 1;
    uint*        runtime_top_k         = new uint[runtime_args_size];
    float*       runtime_top_p         = new float[runtime_args_size];
    const size_t except_idx            = 1;
    for (size_t i = 0; i < runtime_args_size; ++i) {
        runtime_top_k[i] = (top_k > 1) && (i % period_size == except_idx) ? 1 : top_k;
        runtime_top_p[i] = (i % period_size == except_idx) ? top_p * 0.1f : top_p;
    }
    int* d_end_id_buf = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    deviceFill(d_end_id_buf, batch_size, end_id);

#ifndef NDEBUG
    FT_LOG_DEBUG("Random Seeds");
    printMatrixWithLimit(random_seed, 1, random_seed_size, random_seed_size, false);
#endif

    bool passed = true;

    TensorMap runtime_args;
    runtime_args.insert({"has_diff_runtime_args", Tensor(MEMORY_CPU, TYPE_BOOL, {1}, &has_diff_runtime_args)});
    runtime_args.insert({"random_seed", Tensor(MEMORY_CPU, TYPE_UINT64, {random_seed_size}, random_seed)});
    runtime_args.insert({"runtime_top_k", Tensor(MEMORY_CPU, TYPE_INT32, {runtime_args_size}, runtime_top_k)});
    runtime_args.insert({"runtime_top_p", Tensor(MEMORY_CPU, TYPE_FP32, {runtime_args_size}, runtime_top_p)});
    runtime_args.insert({"temperature", Tensor(MEMORY_CPU, TYPE_FP32, {1}, &temperature)});
    runtime_args.insert({"len_penalty", Tensor(MEMORY_CPU, TYPE_FP32, {1}, &len_penalty)});
    runtime_args.insert({"repetition_penalty", Tensor(MEMORY_CPU, TYPE_FP32, {1}, &repetition_penalty)});
    dynamic_decode_layer->setup(batch_size, beam_width, &runtime_args);

    for (size_t step = max_input_len; step < max_output_len; ++step) {
        const size_t iteration_num = batch_size / local_batch_size;

        initRandom(h_logits, beam_width * vocab_size, -10.0f / vocab_size, -1.0f);
        tile(h_logits, batch_size, beam_width * vocab_size);
        cudaH2Dcpy(d_logits_buf, h_logits, batchxbeam * vocab_size);

#ifndef NDEBUG
        FT_LOG_DEBUG("logit values");
        printMatrixWithLimit(h_logits, batchxbeam, vocab_size, vocab_size, false);
#endif
        for (uint ite = 0; ite < iteration_num; ++ite) {
            TensorMap dynamic_decode_input_tensors(
                {{"logits", Tensor{MEMORY_GPU, data_type, {batch_size, beam_width, vocab_size}, d_logits_buf}},
                 {"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size}, nullptr}},
                 {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                 {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_len}},
                 {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, d_input_lengths}},
                 {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                 {"has_diff_runtime_args", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &has_diff_runtime_args}},
                 {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &local_batch_size}},
                 {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, d_end_id_buf}},
                 {"random_seed", {MEMORY_CPU, TYPE_UINT64, {random_seed_size}, random_seed}},
                 {"runtime_top_k", {MEMORY_CPU, TYPE_UINT32, {runtime_args_size}, runtime_top_k}},
                 {"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {runtime_args_size}, runtime_top_p}},
                 {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &temperature}},
                 {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &len_penalty}},
                 {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &repetition_penalty}}});

            // common outputs
            TensorMap dynamic_decode_output_tensors(
                {{"output_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, d_output_ids}},
                 {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, nullptr}},
                 {"parent_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, nullptr}},
                 {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, nullptr}},
                 // necessary for beam search.
                 {"tgt_cache_indirection",
                  Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width, max_output_len}, nullptr}}});

            dynamic_decode_layer->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
            sync_check_cuda_error();
#ifndef NDEBUG
            FT_LOG_DEBUG("Step %2d generated ids", step);
            printMatrix(d_output_ids, max_seq_len, batchxbeam, batchxbeam, true);
            FT_LOG_DEBUG("");
#endif
            // check results.
            cudaD2Hcpy(h_output_ids,
                       (int*)dynamic_decode_output_tensors.at("output_ids").getPtrWithOffset(step * batchxbeam),
                       batchxbeam);
        }
        bool is_ok = isEqualInPeriod(h_output_ids, batchxbeam, period_size, except_idx);
        passed &= is_ok;
    }
    std::string tag = fmtstr("%s (seed=%-6s has_diff_args=%-5s local_batch=%-5s T=%s)",
                             tc.toString().c_str(),
                             SINGLE_RANDOM_SEED ? "single" : "multi",
                             HAS_DIFF_ARGS ? "true" : "false",
                             USE_LOCAL_BATCH ? "true" : "false",
                             (std::is_same<T, float>::value ? " fp32" : " fp16"));
    FT_LOG_INFO("check...%s SamplingLayerCurandInitTest %-30s", passed ? "....OK" : "FAILED", tag.c_str());
    EXPECT_TRUE(passed);

    free(h_logits);
    free(h_output_ids);

    delete dynamic_decode_layer;
    delete runtime_top_k;
    delete runtime_top_p;
    delete random_seed;
    delete cublas_wrapper;
    delete allocator;
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cublasDestroy(cublas_handle));
    check_cuda_error(cublasLtDestroy(cublaslt_handle));
}

int main()
{
    std::vector<TestCase> test_cases{
        // TC: name / batch / vocab / beam / k / p / outlen
        TestCase{"topk", 6, 4, 1, 1, 0.0f, 4},
        TestCase{"topk", 6, 4, 1, 4, 0.0f, 4},
        TestCase{"topk", 6, 51200, 1, 31, 0.0f, 16},
        TestCase{"topk", 32, 51200, 1, 63, 0.0f, 16},
        TestCase{"topk", 32, 51200, 1, 64, 0.0f, 16},
        TestCase{"topp", 6, 4, 1, 0, 0.2f, 4},
        TestCase{"topp", 6, 4, 1, 0, 0.8f, 4},
        TestCase{"topp", 6, 4, 1, 0, 1.0f, 4},
        TestCase{"topp", 6, 51200, 1, 0, 0.8f, 16},
        TestCase{"topp", 32, 51200, 1, 0, 0.8f, 16},
        TestCase{"topp", 32, 51200, 1, 0, 1.0f, 16},
        TestCase{"topk_topp", 6, 4, 1, 1, 0.8f, 16},
        TestCase{"topk_topp", 6, 4, 1, 4, 1.0f, 16},
        TestCase{"topk_topp", 6, 51200, 1, 31, 0.8f, 16},
        TestCase{"topk_topp", 32, 51200, 1, 63, 0.8f, 16},
        TestCase{"topk_topp", 32, 51200, 1, 64, 1.0f, 16},
    };

    for (auto& tc : test_cases) {
        testCumLogProbComputation<float>(tc);
        testCumLogProbComputation<half>(tc);
    }
    FT_LOG_INFO("testCumLogProbComputation done");

    SamplingDecodeTest<float> sampling_decode_test;
    sampling_decode_test.testAll();

    testCuandBatchInitialize(127);
    FT_LOG_INFO("testCuandBatchInitialize done");

#define LAUNCH_VARIANTS(T, tc, local_batch)                                                                            \
    testSamplingLayerCurandInit<T, true, false, local_batch>(tc);                                                      \
    testSamplingLayerCurandInit<T, true, true, local_batch>(tc);                                                       \
    testSamplingLayerCurandInit<T, false, false, local_batch>(tc);                                                     \
    testSamplingLayerCurandInit<T, false, true, local_batch>(tc);
    for (auto& tc : test_cases) {
        LAUNCH_VARIANTS(float, tc, false);  // without local batch
        LAUNCH_VARIANTS(half, tc, false);
        LAUNCH_VARIANTS(float, tc, true);  // with local batch
        LAUNCH_VARIANTS(half, tc, true);
    }
#undef LAUNCH_VARIANTS
    FT_LOG_INFO("testSamplingLayerCurandInit done");

    return 0;
}
