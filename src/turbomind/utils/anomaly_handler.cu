
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"
#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <optional>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace turbomind {

static std::optional<float> parse_float(const std::string& s, const std::string& key)
{
    if (auto pos = s.find(key); pos != std::string::npos) {
        float value{};
        if (sscanf(s.c_str() + pos + key.size(), "%f", &value) != EOF) {
            return value;
        }
    }
    return {};
}

template<class T, int BLOCK_SIZE>
__global__ void CountAndFixAnormaly(
    T* data, int64_t size, unsigned long long* n_inf, unsigned long long* n_nan, T pinf_val, T ninf_val, T nan_val)
{
    int inf_count{};
    int nan_count{};

    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
        auto x = static_cast<float>(data[i]);
        if (isinf(x)) {
            ++inf_count;
            data[i] = x > 0.f ? pinf_val : ninf_val;
        }
        else if (isnan(x)) {
            ++nan_count;
            data[i] = nan_val;
        }
    }

    typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    if (n_inf) {
        inf_count = BlockReduce(temp_storage).Sum(inf_count);
        if (threadIdx.x == 0) {
            atomicAdd(n_inf, inf_count);
        }
    }

    // Wait for last use of `temp_storage`
    __syncthreads();

    if (n_nan) {
        nan_count = BlockReduce(temp_storage).Sum(nan_count);
        if (threadIdx.x == 0) {
            atomicAdd(n_nan, nan_count);
        }
    }
}

template<class T, int BLOCK_SIZE>
__global__ void FixLogitsAnomaly(T*   logits,  //
                                 int* is_anomaly,
                                 int  vocab_size,
                                 int  batch_size,
                                 int  fallback)
{
    const int bi = blockIdx.x;

    T* ptr = logits + vocab_size * bi;

    int count = 0;

    // Accumulate per thread anomaly count
    for (int i = threadIdx.x; i < vocab_size; i += BLOCK_SIZE) {
        const float val = static_cast<float>(ptr[i]);
        count += static_cast<int>(isnan(val) || isinf(val));
    }

    // If anything goes wrong
    int error = __syncthreads_or(count);

    if (!error) {
        return;
    }

    // Clear all logits
    for (int i = threadIdx.x; i < vocab_size; i += BLOCK_SIZE) {
        ptr[i] = T(0.f);
    }

    // Set the fallback token
    if (fallback % BLOCK_SIZE == threadIdx.x) {
        // Ideally we want INF here, but it leads to `INF - INF -> NaN` in the sampling kernels
        // Setting other logits to -INF has similar problem when banning bad words (same -INF)
        ptr[fallback] = T(65504.f);  // Maximum finite value of half
    }

    if (threadIdx.x == 0 && is_anomaly) {
        is_anomaly[bi] = 1;
    }
}

struct AnomalyHandler::Impl {

    Impl()
    {
        GlobalInit();

        if (g_level) {
            d_count_.resize(max_entries * 2);
            h_count_.resize(d_count_.size());
        }
    }

    // Process level initialization from environment variable
    static void GlobalInit()
    {
        [[maybe_unused]] static const auto _ = []() -> bool {
            const auto var = std::getenv("TM_ANOMALY_HANDLER");
            if (!var) {
                return false;
            }
            const std::string str{var};

            const auto level = parse_float(str, "level=");
            if (level) {
                g_level = static_cast<int>(*level);
            }

            TM_LOG_WARNING("[AnomalyHandler] level: %d", g_level);

            if (!g_level) {
                return {};
            }

            const auto pos_inf = parse_float(str, "pinf=");
            if (pos_inf) {
                g_pinf_val_ = *pos_inf;
                TM_LOG_WARNING("[AnomalyHandler] +INF -> %f", g_pinf_val_);
            }

            const auto neg_inf = parse_float(str, "ninf=");
            if (neg_inf) {
                g_ninf_val_ = *neg_inf;
                TM_LOG_WARNING("[AnomalyHandler] -INF -> %f", g_ninf_val_);
            }

            if (!pos_inf && !neg_inf) {
                if (const auto flush_inf = parse_float(str, "inf=")) {
                    g_pinf_val_ = *flush_inf;
                    g_ninf_val_ = -g_pinf_val_;
                    TM_LOG_WARNING("[AnomalyHandler] +INF -> %f", g_pinf_val_);
                    TM_LOG_WARNING("[AnomalyHandler] -INF -> %f", g_ninf_val_);
                }
            }

            if (const auto nan = parse_float(str, "nan=")) {
                g_nan_val_ = *nan;
                TM_LOG_WARNING("[AnomalyHandler] NaN -> %f", g_nan_val_);
            }

            const auto fallback = parse_float(str, "fallback=");
            if (fallback) {
                g_fallback = *fallback;
                TM_LOG_WARNING("[AnomalyHandler] fallback -> %d", g_fallback);
            }

            return {};
        }();
    }

    void Init(int rank, int vocab_size, int fallback, int max_batch_size, cudaStream_t stream)
    {
        if (g_level) {
            rank_       = rank;
            stream_     = stream;
            vocab_size_ = vocab_size;

            max_batch_size_ = max_batch_size;

            d_is_anomaly_.resize(max_batch_size);
            h_is_anomaly_.resize(max_batch_size);

            fallback_ = g_fallback;

            // When fallback is not set from env
            if (fallback_ == -1) {
                fallback_ = fallback;
                TM_LOG_WARNING("[AnomalyHandler] fallback: %d", fallback_);
            }

            FT_CHECK(0 <= fallback_);
            FT_CHECK(fallback_ < vocab_size);

            TM_LOG_WARNING("[AnomalyHandler] max_batch_size: %d", max_batch_size);
            TM_LOG_WARNING("[AnomalyHandler] vocab_size: %d", vocab_size);
        }
    }

    void Summarize(std::function<void(const int*, int)> handler)
    {
        if (g_level) {
            check_cuda_error(cudaMemcpyAsync(h_count_.data(),
                                             d_count_.data().get(),
                                             sizeof(size_type) * info_.size() * 2,
                                             cudaMemcpyDefault,
                                             stream_));

            check_cuda_error(cudaMemcpyAsync(h_is_anomaly_.data(),
                                             d_is_anomaly_.data().get(),
                                             sizeof(int) * batch_size_,
                                             cudaMemcpyDefault,
                                             stream_));

            check_cuda_error(cudaStreamSynchronize(stream_));

            for (size_t i = 0; i < info_.size(); ++i) {
                const auto& n_inf = h_count_[i * 2];
                const auto& n_nan = h_count_[i * 2 + 1];
                if (n_inf || n_nan) {
                    TM_LOG_WARNING("[AnomalyHandler][rank=%d] (%s) INF: %s, NaN: %s",
                                   rank_,
                                   info_[i].c_str(),
                                   std::to_string(n_inf).c_str(),
                                   std::to_string(n_nan).c_str());
                }
            }

            handler(h_is_anomaly_.data(), batch_size_);
        }
    }

    void Reset()
    {
        if (g_level) {
            if (!info_.empty()) {
                std::fill_n(h_count_.data(), info_.size() * 2, 0);
                check_cuda_error(
                    cudaMemsetAsync(d_count_.data().get(), 0, sizeof(size_type) * info_.size() * 2, stream_));
                info_.clear();
            }

            if (batch_size_) {
                std::fill_n(h_is_anomaly_.data(), batch_size_, 0);
                check_cuda_error(cudaMemsetAsync(d_is_anomaly_.data().get(), 0, sizeof(int) * batch_size_, stream_));
                batch_size_ = 0;
            }
        }
    }

    template<class T>
    void invokeCountAndFixAnomaly(T* data, int64_t size, const std::string& key, int level)
    {
        if (g_level && level <= g_level) {
            FT_CHECK(size >= 0);

            constexpr int block = 512;
            const int     grid  = (size + block - 1) / block;

            auto idx = info_.size();
            auto ptr = d_count_.data().get() + idx * 2;

            info_.push_back(key);

            FT_CHECK(info_.size() <= max_entries);

            CountAndFixAnormaly<T, block><<<grid, block, 0, stream_>>>(data,  //
                                                                       size,
                                                                       ptr,
                                                                       ptr + 1,
                                                                       g_pinf_val_,
                                                                       g_ninf_val_,
                                                                       g_nan_val_);

            sync_check_cuda_error();
        }
    }

    template<class T>
    void invokeFixLogitsAnomaly(T* logits, int batch_size, int level)
    {
        if (g_level && level <= g_level) {
            FT_CHECK(batch_size <= max_batch_size_);

            batch_size_ = batch_size;

            constexpr int block = 256;

            FixLogitsAnomaly<T, block><<<batch_size, block, 0, stream_>>>(logits,  //
                                                                          d_is_anomaly_.data().get(),
                                                                          vocab_size_,
                                                                          batch_size,
                                                                          fallback_);

            sync_check_cuda_error();
        }
    }

    static int   g_level;
    static int   g_fallback;
    static float g_pinf_val_;
    static float g_ninf_val_;
    static float g_nan_val_;

    cudaStream_t stream_{};
    int          rank_{};
    int          vocab_size_{};
    int          fallback_{};
    int          max_batch_size_{};

    ////////////////////////////////////////////////////////////////////////////////
    /// Members below has SINGLE iteration validity and must be cleared in `Reset`

    // Datum for tracing anomalies
    thrust::device_vector<size_type> d_count_;
    thrust::host_vector<size_type>   h_count_;
    std::vector<std::string>         info_;

    // Datum for fixing logits
    thrust::device_vector<int> d_is_anomaly_;
    thrust::host_vector<int>   h_is_anomaly_;
    int                        batch_size_{};
};

int   AnomalyHandler::Impl::g_level     = 0;
int   AnomalyHandler::Impl::g_fallback  = -1;
float AnomalyHandler::Impl::g_pinf_val_ = INFINITY;
float AnomalyHandler::Impl::g_ninf_val_ = -INFINITY;
float AnomalyHandler::Impl::g_nan_val_  = NAN;

AnomalyHandler::AnomalyHandler(): impl_{new Impl{}} {}

AnomalyHandler::~AnomalyHandler() = default;

AnomalyHandler& AnomalyHandler::instance()
{
    thread_local AnomalyHandler inst{};
    return inst;
}

void AnomalyHandler::Init(int rank, int vocab_size, int fallback, int max_batch_size, cudaStream_t stream) noexcept
{
    impl_->Init(rank, vocab_size, fallback, max_batch_size, stream);
}

void AnomalyHandler::Summarize(std::function<void(const int*, int)> handler)
{
    impl_->Summarize(handler);
}

void AnomalyHandler::Reset()
{
    impl_->Reset();
}

template<class T>
void AnomalyHandler::CountAndFix(T* data, int64_t size, std::string key, int level)
{
    return impl_->invokeCountAndFixAnomaly(data, size, key, level);
}

template void AnomalyHandler::CountAndFix(float*, int64_t, std::string, int);
template void AnomalyHandler::CountAndFix(half*, int64_t, std::string, int);
#ifdef ENABLE_BF16
template void AnomalyHandler::CountAndFix(__nv_bfloat16*, int64_t, std::string, int);
#endif

template<class T>
void AnomalyHandler::FixLogits(T* logits, int batch_size, int level)
{
    impl_->invokeFixLogitsAnomaly(logits, batch_size, level);
}

template void AnomalyHandler::FixLogits(float*, int, int);
template void AnomalyHandler::FixLogits(half*, int, int);
#ifdef ENABLE_BF16
template void AnomalyHandler::FixLogits(__nv_bfloat16*, int, int);
#endif

}  // namespace turbomind
