#include <assert.h>
#include <math.h>
#include <float.h>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <sys/time.h>

#include "src/turbomind/kernels/logprob_kernels.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"

#include "tests/unittests/gtest_utils.h"

using namespace turbomind;

////////////////////////////////////////////////////////////////////////////////////

struct LogProbKernelTestParam {
    size_t max_input_length;
    size_t batch_size;
    size_t vocab_size;
    size_t beam_width;

    std::string toString() {
        return fmtstr("LogProbKernelTestParam[max_input_length=%ld, batch=%ld, vocab=%ld, beam_width=%ld]",
                      max_input_length, batch_size, vocab_size, beam_width);
    }
};

/////////////////////////////////// Unittests //////////////////////////////////////////
template<typename T>
class LogProbKernelTest : public FtTestBase {

protected:
    void computeCumLogProbs(float* cum_log_probs,
                            float* log_probs,
                            const T* logits,
                            const int* input_ids,
                            const int* input_lengths,
                            const size_t max_input_length,
                            const size_t batch_size,
                            const size_t vocab_size,
                            const size_t vocab_size_padded)
    {
        for (size_t step = 0; step < max_input_length; ++step) {
            for (size_t i = 0; i < batch_size; ++i) {
                if ((int)step == 0) {
                    if (log_probs != nullptr) {
                        log_probs[i] = 0.0f;
                    }
                    cum_log_probs[i] = 0.0f;
                }
                else if ((int)step < input_lengths[i]) {
                    size_t step_offset = (step - 1) * batch_size * vocab_size_padded;
                    const T* vec = logits + step_offset + i * vocab_size_padded;
                    float max_logits = -FLT_MAX;
                    for (size_t v = 0; v < vocab_size; ++v) {
                        float val = static_cast<float>(vec[v]);
                        if (val > max_logits) {
                            max_logits = val;
                        }
                    }
                    float sum = 0.0f;
                    for (size_t v = 0; v < vocab_size; ++v) {
                        sum += expf(static_cast<float>(vec[v]) - max_logits);
                    }
                    int token_id = input_ids[step * batch_size + i];
                    float log_prob = static_cast<float>(vec[token_id]) - max_logits - log(sum);
                    if (log_probs != nullptr) {
                        log_probs[step * batch_size + i] = log_prob;
                    }
                    cum_log_probs[i] += log_prob;
                }
            }
        }
    }

    void computeCumLogProbsBatchFirst(float* cum_log_probs,
                                      float* log_probs,
                                      const T* logits,
                                      const int* input_ids,
                                      const int* input_lengths,
                                      const size_t max_input_length,
                                      const size_t batch_size,
                                      const size_t vocab_size,
                                      const size_t vocab_size_padded)
    {
        for (size_t i = 0; i < batch_size; ++i) {
            size_t batch_offset = i * max_input_length * vocab_size_padded;
            for (size_t step = 0; step < max_input_length; ++step) {
                if ((int)step == 0) {
                    if (log_probs != nullptr) {
                        log_probs[i * max_input_length] = 0.0f;
                    }
                    cum_log_probs[i] = 0.0f;
                }
                else if ((int)step < input_lengths[i]) {
                    const T* vec = logits + batch_offset + (step - 1) * vocab_size_padded;
                    float max_logits = -FLT_MAX;
                    for (size_t v = 0; v < vocab_size; ++v) {
                        float val = static_cast<float>(vec[v]);
                        if (val > max_logits) {
                            max_logits = val;
                        }
                    }
                    float sum = 0.0f;
                    for (size_t v = 0; v < vocab_size; ++v) {
                        sum += expf(static_cast<float>(vec[v]) - max_logits);
                    }
                    int token_id = input_ids[i * max_input_length + step];
                    float log_prob = static_cast<float>(vec[token_id]) - max_logits - log(sum);
                    if (log_probs != nullptr) {
                        log_probs[i * max_input_length + step] = log_prob;
                    }
                    cum_log_probs[i] += log_prob;
                }
            }
        }
    }

public:

    void runTest(LogProbKernelTestParam param) {
        size_t max_input_length = param.max_input_length;
        size_t batchxbeam = param.batch_size * param.beam_width;
        size_t vocab_size = param.vocab_size;
        // Make multiple of 8 as GPT does.
        size_t vocab_size_padded = static_cast<size_t>(ceil(vocab_size / 8.f) * 8);

        // input values
        T* h_logits = new T[max_input_length * batchxbeam * vocab_size];
        int* h_input_ids = new int[max_input_length * batchxbeam];
        int* h_input_lengths = new int[batchxbeam];

        // output buffers
        float* expected_cum_log_probs = new float[batchxbeam];

        // initialize host buffers
        initRandom(h_logits, max_input_length * batchxbeam * vocab_size, -10.0f / vocab_size, -1.0f);
        initRandomInt(h_input_ids, max_input_length * batchxbeam, 0, vocab_size);
        initRandomInt(h_input_lengths, batchxbeam, 1, max_input_length + 1);
        memset(expected_cum_log_probs, 0, sizeof(float) * batchxbeam);

        // device buffers
        T* d_logits = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * max_input_length * batchxbeam * vocab_size));
        int *d_input_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_input_length * batchxbeam));
        int *d_input_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batchxbeam));
        float* d_cum_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batchxbeam));

        // initialize device buffers
        cudaH2Dcpy(d_logits, h_logits, max_input_length * batchxbeam * vocab_size);
        cudaH2Dcpy(d_input_ids, h_input_ids, max_input_length * batchxbeam);
        cudaH2Dcpy(d_input_lengths, h_input_lengths, batchxbeam);
        deviceFill(d_cum_log_probs, batchxbeam, 0.0f);

        size_t workspace_size = sizeof(float) * max_input_length * batchxbeam;
        void* workspace = allocator->malloc(workspace_size);
        invokeLogProbFromLogits(d_cum_log_probs,
                                d_logits,
                                d_input_ids,
                                d_input_lengths,
                                max_input_length,
                                batchxbeam,
                                vocab_size,
                                vocab_size_padded,
                                workspace,
                                workspace_size,
                                stream,
                                false);
        computeCumLogProbs(expected_cum_log_probs,
                           nullptr,
                           h_logits,
                           h_input_ids,
                           h_input_lengths,
                           max_input_length,
                           batchxbeam,
                           vocab_size,
                           vocab_size_padded);
        bool passed = checkResult(param.toString(), d_cum_log_probs, expected_cum_log_probs, batchxbeam);
        EXPECT_TRUE(passed);

        FT_LOG_DEBUG("free host buffers");
        delete[] expected_cum_log_probs;
        delete[] h_input_lengths;
        delete[] h_input_ids;
        delete[] h_logits;
    }

    void runBatchFirstTest(LogProbKernelTestParam param) {
        size_t max_input_length = param.max_input_length;
        size_t batchxbeam = param.batch_size * param.beam_width;
        size_t vocab_size = param.vocab_size;
        // Make multiple of 8 as GPT does.
        size_t vocab_size_padded = static_cast<size_t>(ceil(vocab_size / 8.f) * 8);

        // input values
        T* h_logits = new T[max_input_length * batchxbeam * vocab_size_padded];
        int* h_input_ids = new int[max_input_length * batchxbeam];
        int* h_input_lengths = new int[batchxbeam];

        // output buffers
        float* expected_cum_log_probs = new float[batchxbeam];

        // initialize host buffers
        initRandom(h_logits, max_input_length * batchxbeam * vocab_size_padded, -10.0f / vocab_size, -1.0f);
        initRandomInt(h_input_ids, max_input_length * batchxbeam, 0, vocab_size);
        initRandomInt(h_input_lengths, batchxbeam, 1, max_input_length + 1);
        memset(expected_cum_log_probs, 0, sizeof(float) * batchxbeam);

        // device buffers
        T* d_logits =
            reinterpret_cast<T*>(allocator->malloc(sizeof(T) * max_input_length * batchxbeam * vocab_size_padded));
        int *d_input_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_input_length * batchxbeam));
        int *d_input_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batchxbeam));
        float* d_cum_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batchxbeam));

        // initialize device buffers
        cudaH2Dcpy(d_logits, h_logits, max_input_length * batchxbeam * vocab_size_padded);
        cudaH2Dcpy(d_input_ids, h_input_ids, max_input_length * batchxbeam);
        cudaH2Dcpy(d_input_lengths, h_input_lengths, batchxbeam);
        check_cuda_error(cudaMemset(d_cum_log_probs, 0, sizeof(float) * batchxbeam));

        size_t workspace_size = sizeof(float) * max_input_length * batchxbeam;
        void* workspace = allocator->malloc(workspace_size);
        invokeLogProbFromLogits(d_cum_log_probs,
                                d_logits,
                                d_input_ids,
                                d_input_lengths,
                                max_input_length,
                                batchxbeam,
                                vocab_size,
                                vocab_size_padded,
                                workspace,
                                workspace_size,
                                stream,
                                true);

        computeCumLogProbsBatchFirst(expected_cum_log_probs,
                                    nullptr,
                                    h_logits,
                                    h_input_ids,
                                    h_input_lengths,
                                    max_input_length,
                                    batchxbeam,
                                    vocab_size,
                                    vocab_size_padded);
        std::string tag = param.toString() + (std::is_same<T, float>::value ? " (fp32)" : " (fp16)");
        bool passed = checkResult(tag.c_str(), d_cum_log_probs, expected_cum_log_probs, batchxbeam);
        EXPECT_TRUE(passed);

        delete[] expected_cum_log_probs;
        delete[] h_input_lengths;
        delete[] h_input_ids;
        delete[] h_logits;
    }

};


TYPED_TEST_SUITE(LogProbKernelTest, FloatAndHalfTypes);

TYPED_TEST(LogProbKernelTest, SingleStep)
{
    this->runTest({1, 32, 16, 1});
}

TYPED_TEST(LogProbKernelTest, AccumLongStep129)
{
    this->runTest({129, 8, 50211, 1});
}

TYPED_TEST(LogProbKernelTest, AccumLongStep1023)
{
    this->runTest({1023, 8, 5001, 1});
}

TYPED_TEST(LogProbKernelTest, AccumLongStep4096)
{
    this->runTest({4096, 8, 5001, 1});
}

TYPED_TEST(LogProbKernelTest, BatchFirstSingleStep)
{
    this->runBatchFirstTest({1, 32, 16, 1});
}

TYPED_TEST(LogProbKernelTest, BatchFirstAccumLongStep129)
{
    this->runBatchFirstTest({129, 8, 50211, 1});
}

TYPED_TEST(LogProbKernelTest, BatchFirstAccumLongStep1023)
{
    this->runBatchFirstTest({1023, 8, 5001, 1});
}

TYPED_TEST(LogProbKernelTest, BatchFirstAccumLongStep4096)
{
    this->runBatchFirstTest({4096, 8, 5001, 1});
}
