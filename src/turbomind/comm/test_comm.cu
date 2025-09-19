// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <sstream>
#include <thread>

// #include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/utils/cuda_utils.h"

using namespace turbomind::comm;
using turbomind::data_type_v;
using turbomind::check;
using turbomind::myAssert;
using std::vector;

[[maybe_unused]] static constexpr bool is_ncu = 0;

struct Context {

    cudaStream_t stream;

    cudaEvent_t ev_start;
    cudaEvent_t ev_end;

    std::vector<void*> buffers;

    template<class F>
    float exec(F func)
    {
        check_cuda_error(cudaStreamSynchronize(stream));
        check_cuda_error(cudaEventRecord(ev_start, stream));

        func(stream);

        check_cuda_error(cudaEventRecord(ev_end, stream));
        check_cuda_error(cudaEventSynchronize(ev_end));
        float ms{};
        check_cuda_error(cudaEventElapsedTime(&ms, ev_start, ev_end));
        return ms;
    }

    template<class T>
    T* malloc(size_t count)
    {
        T* data;
        check_cuda_error(cudaMallocAsync(&data, sizeof(T) * count, stream));
        buffers.push_back(data);
        return data;
    }

    template<class T>
    void copy_n(const T* src, size_t count, T* dst)
    {
        check_cuda_error(cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDefault, stream));
    }

    void sync()
    {
        check_cuda_error(cudaStreamSynchronize(stream));
    }

    Context(int device_id)
    {
        check_cuda_error(cudaSetDevice(device_id));
        check_cuda_error(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check_cuda_error(cudaEventCreate(&ev_start));
        check_cuda_error(cudaEventCreate(&ev_end));
    }
    ~Context()
    {
        for (auto& p : buffers) {
            cudaFreeAsync(p, stream);
            p = {};
        }
        cudaStreamSynchronize(stream);
        cudaEventDestroy(ev_end);
        cudaEventDestroy(ev_start);
        cudaStreamDestroy(stream);
    }
};

struct TestComm {
    std::vector<HostComm>   h_comm_;
    std::vector<DeviceComm> d_comm_;
    std::vector<HostComm>   h_split_;
    std::vector<int>        d_split_;

    int              warmup_;
    int              iters_;
    std::vector<int> tokens_;
    size_t           max_tokens_;

    static auto Init(int n_ranks, int split, const std::string& backend)
    {

        std::unique_ptr<HostGroupId> group_id = CreateHostGroupId({});
        std::string                  group_id_data;
        if (1) {  // master
            group_id->Initialize();
            std::stringstream ss;
            group_id->Export(ss);
            group_id_data = ss.str();
        }

        std::vector<DeviceComm> d_comm(n_ranks);
        std::vector<HostComm>   h_comm(n_ranks);
        std::vector<int>        d_split(n_ranks);
        std::vector<HostComm>   h_split(n_ranks);

        auto init = [&](int rank) {
            // initialize host communicators
            std::stringstream            ss(group_id_data);
            std::unique_ptr<HostGroupId> host_id = CreateHostGroupId({});
            host_id->Import(ss);
            h_comm[rank] = host_id->CreateCommunicator(n_ranks, rank);

            // initialize device communicators
            cudaSetDevice(rank);
            d_comm[rank] = CreateDeviceCommunicator(backend, n_ranks, rank, h_comm[rank]);

            // split communicators
            if (split) {
                h_split[rank] = h_comm[rank]->Split(rank / split, 0);
                d_split[rank] = d_comm[rank]->Split(rank / split, 0, 0);
            }
            else {
                h_split[rank] = h_comm[rank];
                d_split[rank] = 0;
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < n_ranks; ++i) {
            threads.emplace_back(init, i);
        }
        for (auto& t : threads) {
            t.join();
        }

        return std::make_tuple(h_comm, std::move(d_comm), h_split, d_split);
    }

    void Run(int hidden_dim, int vocab_size, int tp, int warmup, int iters, std::vector<int> tokens)
    {
        int device_num{};
        cudaGetDeviceCount(&device_num);

        std::cout << "Device count: " << device_num << "\n";

        if (tp < 0) {
            tp = device_num;
        }

        std::tie(h_comm_, d_comm_, h_split_, d_split_) = Init(device_num, 4, "cuda-ipc");

        TM_CHECK_GT(h_comm_.size(), 0);
        TM_CHECK_GT(d_comm_.size(), 0);

        warmup_ = warmup;
        iters_  = iters;
        tokens_ = tokens;

        max_tokens_ = *std::max_element(tokens_.begin(), tokens_.end());

        const int g = 0;

        TestAllReduce<half>(hidden_dim, 0);
        // TestAllreduceResidualBiasRMSnorm<half>(hidden_dim, g);
        // TestAllreduceResidualBiasRMSnormEx<half>(hidden_dim, 0, 0);
        // TestAllreduceResidualBiasRMSnormEx<half>(hidden_dim, 1, 0);
        // TestAllreduceResidualBiasRMSnormEx<half>(hidden_dim, 0, 1);
        // TestAllGather<half>(hidden_dim / tp, g);  // tp embedding
        // TestAllGather<half>(vocab_size / tp, g);
        // TestBroadcast<half>(32768, g);
    }

    template<class T>
    void TestAllReduce(size_t dim, int group = 0)
    {
        const auto dtype = data_type_v<T>;

        const int tp_size = d_comm_[0]->n_ranks(group);
        const int dp_size = d_comm_.size() / tp_size;

        //    dp         tp           dim
        std::vector<std::vector<std::vector<T>>> data(dp_size);
        //    dp         dim
        std::vector<std::vector<T>> ref_data(dp_size);

        for (int i = 0; i < dp_size; ++i) {
            data[i].resize(tp_size);
            ref_data[i].resize(max_tokens_ * dim);
        }

        auto func = [&](int index, DeviceComm& d_comm, HostComm& h_comm) {
            const int rank    = d_comm->rank(group);
            const int n_ranks = d_comm->n_ranks(group);
            const int g_rank  = d_comm->rank(0);
            const int d       = g_rank / n_ranks;

            const size_t max_count = max_tokens_ * dim;

            std::mt19937                  gen{(unsigned)index};
            std::uniform_int_distribution dist{0, 31};  // 5 mantissa bits
            if (g_rank == 0) {
                std::cout << "preparing data ... " << std::flush;
            }
            data[d][rank].resize(max_count);
            for (size_t i = 0; i < max_count; ++i) {
                data[d][rank][i] = T(dist(gen));
            }
            h_comm->Sync();
            const size_t slice = (max_count + n_ranks - 1) / n_ranks;
            for (int r = 0; r < n_ranks; ++r) {
                for (size_t i = rank * slice; i < (rank + 1) * slice && i < max_count; ++i) {
                    ref_data[d][i] += data[d][r][i];
                }
            }
            h_comm->Sync();
            if (g_rank == 0) {
                std::cout << "done.\n";
            }

            Context ctx{g_rank};

            T* d_data = ctx.malloc<T>(max_count);

            T* d_tmp = (T*)d_comm->Allocate(sizeof(T) * max_count);
            d_comm->Register(d_tmp, sizeof(T) * max_count);

            ctx.copy_n(data[d][rank].data(), max_count, d_data);

            [[maybe_unused]] auto verify = [&](auto count) {
                std::vector<T> res(count);
                ctx.copy_n(d_tmp, count, res.data());
                ctx.sync();
                size_t diff = 0;
                for (size_t i = 0; i < count; ++i) {
                    auto& x = res[i];
                    auto& y = ref_data[d][i];
                    diff += x != y;
                    if (diff == 1) {
                        printf("%d: %f vs %f\n", (int)i, (float)x, (float)y);
                    }
                }
                if (diff) {
                    printf("[rank %d] count = %d, diff = %lu\n", g_rank, (int)count, diff);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    std::abort();
                }
            };

            std::vector<float> deltas;
            for (const auto& n : tokens_) {
                const size_t count = (size_t)n * dim;
                auto&        delta = deltas.emplace_back();
                h_comm->Sync();
                for (int i = 0; i < warmup_ + iters_; ++i) {
                    ctx.copy_n(d_data, count, d_tmp);
                    auto ms = ctx.exec([&](auto stream) {  //
                        d_comm->AllReduceSum(d_tmp, d_tmp, count, dtype, group, stream);
                    });
                    if (i >= warmup_) {
                        delta += ms;
                    }
                    // verify(count);
                }
                verify(count);
            }

            if (g_rank == 0) {
                SummaryHeader("allreduce", dim, n_ranks);
                for (size_t i = 0; i < tokens_.size(); ++i) {
                    const float  avg   = deltas[i] / iters_;
                    const size_t count = tokens_[i] * dim;
                    const float  algbw = sizeof(T) * count / 1e9f / avg * 1000.f;
                    const float  busbw = algbw * (2 * (n_ranks - 1)) / n_ranks;
                    SummaryEntry(tokens_[i], count, sizeof(T), avg, algbw, busbw);
                }
            }

            d_comm->Deregister(d_tmp);
            d_comm->Free(d_tmp);
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < d_comm_.size(); ++i) {
            threads.emplace_back(func, i, std::ref(d_comm_[i]), std::ref(h_comm_[i]));
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    template<class T>
    void TestAllreduceResidualBiasRMSnorm(size_t dim, int group)
    {
        vector<T> weight(dim);
        vector<T> bias(dim);

        constexpr float eps      = 1e-5;
        constexpr bool  has_bias = true;

        std::cout << "preparing data ... " << std::flush;

        {
            std::mt19937                  gen{};
            std::uniform_int_distribution dist{0, 31};  // 5 mantissa bits
            for (size_t i = 0; i < dim; ++i) {
                weight[i] = T(dist(gen));
            }
            if (has_bias) {
                for (size_t i = 0; i < dim; ++i) {
                    bias[i] = T(dist(gen));
                }
            }
        }

        const auto dtype = data_type_v<T>;

        const int tp_size = d_comm_[0]->n_ranks(group);
        const int dp_size = d_comm_.size() / tp_size;
        // dp    tp     dim
        vector<vector<vector<T>>> src_data(dp_size);
        // dp    dim
        vector<vector<T>> ref_data(dp_size);
        vector<vector<T>> src_res(dp_size);
        vector<vector<T>> ref_res(dp_size);

        for (int i = 0; i < dp_size; ++i) {
            src_data[i].resize(tp_size);
            ref_data[i].resize(max_tokens_ * dim);
            src_res[i].resize(max_tokens_ * dim);
            ref_res[i].resize(max_tokens_ * dim);
        }

        auto func = [&](int index, DeviceComm& d_comm, HostComm& h_comm) {
            const int rank    = d_comm->rank(group);
            const int n_ranks = d_comm->n_ranks(group);
            const int g_rank  = d_comm->rank(0);
            const int d       = g_rank / n_ranks;

            const size_t max_count = max_tokens_ * dim;

            std::mt19937                  gen{(unsigned)index};
            std::uniform_int_distribution dist{0, 31};  // 5 mantissa bits

            src_data[d][rank].resize(max_count);
            for (size_t i = 0; i < max_count; ++i) {
                src_data[d][rank][i] = T(dist(gen));
            }
            h_comm->Sync();
            const size_t slice = (max_tokens_ + n_ranks - 1) / n_ranks;
            for (size_t t = rank * slice; t < (rank + 1) * slice && t < max_tokens_; ++t) {
                for (int r = 0; r < n_ranks; ++r) {
                    for (size_t i = 0; i < dim; ++i) {
                        ref_data[d][t * dim + i] += src_data[d][r][t * dim + i];
                    }
                }
                float sum = 0.f;
                for (size_t i = 0; i < dim; ++i) {
                    const size_t idx = t * dim + i;
                    src_res[d][idx]  = T(dist(gen));
                    ref_res[d][idx]  = src_res[d][idx] + ref_data[d][idx] + bias[i];  // r' <- r + (h + b)
                    sum += (float)ref_res[d][idx] * (float)ref_res[d][idx];
                }
                sum = 1 / (sqrtf(sum / dim) + eps);
                for (size_t i = 0; i < dim; ++i) {
                    const size_t idx = t * dim + i;
                    float        tmp = (float)ref_res[d][idx];
                    ref_data[d][idx] = tmp * sum * (float)weight[i];  // h' <- norm(r) * w
                }
            }
            h_comm->Sync();
            if (g_rank == 0) {
                std::cout << "done.\n";
            }

            Context ctx{g_rank};

            T* d_bias   = ctx.malloc<T>(dim);
            T* d_weight = ctx.malloc<T>(dim);

            T* d_data    = ctx.malloc<T>(max_count);
            T* d_res     = ctx.malloc<T>(max_count);
            T* d_tmp_res = ctx.malloc<T>(max_count);

            T* d_tmp_data = (T*)d_comm->Allocate(sizeof(T) * max_count);
            d_comm->Register(d_tmp_data, sizeof(T) * max_count);

            ctx.copy_n(src_data[d][rank].data(), max_count, d_data);
            ctx.copy_n(src_res[d].data(), max_count, d_res);
            ctx.copy_n(bias.data(), dim, d_bias);
            ctx.copy_n(weight.data(), dim, d_weight);

            [[maybe_unused]] auto verify = [&](auto token_num) {
                const size_t count = (size_t)token_num * dim;
                vector<T>    h_data(count);
                vector<T>    h_res(count);
                ctx.copy_n(d_tmp_data, count, h_data.data());
                ctx.copy_n(d_tmp_res, count, h_res.data());
                ctx.sync();
                const size_t slice    = (token_num + n_ranks - 1) / n_ranks * dim;
                const size_t first    = rank * slice;
                const size_t last     = std::min(first + slice, count);
                size_t       res_diff = 0;
                for (size_t i = first; i < last; ++i) {
                    auto& x       = h_res[i];
                    auto& y       = ref_res[d][i];
                    int   is_diff = !(x == y);
                    if (!res_diff && is_diff) {
                        printf("[rank %d], %ld: %f vs %f\n", g_rank, i - first, (float)x, (float)y);
                    }
                    res_diff += is_diff;
                }
                float data_diff = 0;
                for (size_t i = 0; i < count; ++i) {
                    float diff = (float)h_data[i] - (float)ref_data[d][i];
                    data_diff += std::abs(diff);
                }
                data_diff /= count;
                if (res_diff || data_diff > 0.1f || std::isnan(data_diff)) {
                    printf("[rank %d] count = %d, res_diff = %lu, data_diff = %f\n",
                           g_rank,
                           (int)token_num,
                           res_diff,
                           data_diff);
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    std::abort();
                }
                else if (g_rank == 0) {
                    printf("[rank %d] count = %d, data_diff = %f\n", g_rank, (int)token_num, data_diff);
                }
            };

            vector<float> deltas;
            for (const auto& n : tokens_) {
                const size_t count = (size_t)n * dim;
                auto&        delta = deltas.emplace_back();
                h_comm->Sync();
                for (int i = 0; i < warmup_ + iters_; ++i) {
                    ctx.copy_n(d_data, count, d_tmp_data);
                    ctx.copy_n(d_res, count, d_tmp_res);
                    auto ms = ctx.exec([&](auto stream) {  //
                        d_comm->AllreduceResidualBiasRMSnorm(d_tmp_data,
                                                             d_tmp_res,
                                                             has_bias ? d_bias : nullptr,
                                                             d_weight,
                                                             eps,
                                                             dim,
                                                             n,
                                                             dtype,
                                                             group,
                                                             stream);
                    });
                    if (i >= warmup_) {
                        delta += ms;
                    }
                    // verify(n);
                }
                verify(n);
            }

            d_comm->Deregister(d_tmp_data);
            d_comm->Free(d_tmp_data);

            if (g_rank == 0) {
                SummaryHeader("allreduce | rmsnorm", dim, n_ranks);
                for (size_t i = 0; i < tokens_.size(); ++i) {
                    const float  avg   = deltas[i] / iters_;
                    const size_t count = tokens_[i] * dim;
                    const float  algbw = sizeof(T) * count / 1e9f / avg * 1000.f;
                    const float  busbw = algbw * (2 * (n_ranks - 1)) / n_ranks;
                    SummaryEntry(tokens_[i], count, sizeof(T), avg, algbw, busbw);
                }
            }
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < d_comm_.size(); ++i) {
            threads.emplace_back(func, i, std::ref(d_comm_[i]), std::ref(h_comm_[i]));
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    template<class T>
    void TestAllGather(size_t dim, int group)
    {
        const auto dtype = data_type_v<T>;

        const int tp_size = d_comm_[0]->n_ranks(group);
        const int dp_size = d_comm_.size() / tp_size;

        vector<vector<vector<T>>> data(dp_size);

        for (int i = 0; i < dp_size; ++i) {
            data[i].resize(tp_size);
        }

        auto func = [&](int index, DeviceComm& d_comm, HostComm& h_comm) {
            const int rank    = d_comm->rank(group);
            const int n_ranks = d_comm->n_ranks(group);
            const int g_rank  = d_comm->rank(0);
            const int d       = g_rank / n_ranks;

            const size_t max_count = max_tokens_ * dim;

            if (h_comm->rank() == 0) {
                std::cout << "preparing data ... " << std::flush;
            }
            std::mt19937                  gen{(unsigned)index};
            std::uniform_int_distribution dist{0, 100};
            data[d][rank].resize(max_count);
            for (size_t i = 0; i < max_count; ++i) {
                data[d][rank][i] = T(dist(gen));
            }
            h_comm->Sync();
            if (h_comm->rank() == 0) {
                std::cout << "done.\n";
            }

            Context ctx{g_rank};

            T* d_data = ctx.malloc<T>(max_count);

            T* d_tmp = (T*)d_comm->Allocate(sizeof(T) * max_count * n_ranks);
            d_comm->Register(d_tmp, sizeof(T) * max_count * n_ranks);

            ctx.copy_n(data[d][rank].data(), max_count, d_data);

            [[maybe_unused]] auto verify = [&](int64_t count) {
                auto           total_count = count * n_ranks;
                std::vector<T> res(total_count);
                ctx.copy_n(d_tmp, total_count, res.data());
                ctx.sync();
                size_t diff = 0;
                for (int r = 0; r < n_ranks; ++r) {
                    for (auto i = 0; i < count; ++i) {
                        auto& x = res[r * count + i];
                        auto& y = data[d][r][i];
                        diff += (x != y);
                        if (diff == 1) {
                            printf("%d: %f vs %f\n", (int)i, (float)x, (float)y);
                        }
                    }
                }
                if (diff) {
                    printf("[rank %d] count = %d, diff = %lu\n", g_rank, (int)count, diff);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    std::abort();
                }
            };

            std::vector<float> deltas;
            for (const auto& n : tokens_) {
                const size_t count = (size_t)n * dim;  // dim = hidden_dim / tp
                auto&        delta = deltas.emplace_back();
                h_comm->Sync();
                for (int i = 0; i < warmup_ + iters_; ++i) {
                    check_cuda_error(cudaMemsetAsync(d_tmp, 0, sizeof(T) * count * n_ranks, ctx.stream));
                    ctx.copy_n(d_data, count, d_tmp + rank * count);
                    auto ms = ctx.exec([&](auto stream) {  //
                        if (d_comm->Query(kHasAllGather2D) && 0) {
                            d_comm->AllGather2D(
                                d_tmp + rank * count, d_tmp, dim, count, dim, n, dtype, {1, 1}, group, stream);
                        }
                        else {
                            d_comm->AllGather(d_tmp + rank * count, d_tmp, count, dtype, group, stream);
                        }
                    });
                    if (i >= warmup_) {
                        delta += ms;
                    }
                    // verify(count);
                }
                verify(count);
            }

            if (g_rank == 0) {
                SummaryHeader("allgather", dim, n_ranks);
                for (size_t i = 0; i < tokens_.size(); ++i) {
                    const float  avg   = deltas[i] / iters_;
                    const size_t count = n_ranks * tokens_[i] * dim;
                    const float  algbw = sizeof(T) * count / 1e9f / avg * 1000.f;
                    const float  busbw = algbw * (n_ranks - 1) / n_ranks;

                    SummaryEntry(tokens_[i], count, sizeof(T), avg, algbw, busbw);
                }
            }

            d_comm->Deregister(d_tmp);
            d_comm->Free(d_tmp);
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < d_comm_.size(); ++i) {
            threads.emplace_back(func, i, std::ref(d_comm_[i]), std::ref(h_comm_[i]));
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    template<class T>
    void TestBroadcast(size_t dim, int group)
    {
        const auto dtype = data_type_v<T>;

        const int tp_size = d_comm_[0]->n_ranks(group);
        const int dp_size = d_comm_.size() / tp_size;

        constexpr int root = 0;

        vector<vector<T>> data(dp_size);

        auto func = [&](int index, DeviceComm& d_comm, HostComm& h_comm) {
            const int rank    = d_comm->rank(group);
            const int n_ranks = d_comm->n_ranks(group);
            const int g_rank  = d_comm->rank(0);
            const int d       = g_rank / n_ranks;

            const size_t max_count = max_tokens_ * dim;

            if (h_comm->rank() == root) {
                std::cout << "preparing data ... " << std::flush;
                std::mt19937                  gen{(unsigned)index};
                std::uniform_int_distribution dist{0, 100};
                data[d].resize(max_count);
                for (size_t i = 0; i < max_count; ++i) {
                    data[d][i] = T(dist(gen));
                }
                std::cout << "done.\n";
            }

            h_comm->Sync();

            Context ctx{g_rank};

            T* d_data = ctx.malloc<T>(max_count);

            T* d_tmp = (T*)d_comm->Allocate(sizeof(T) * max_count);
            d_comm->Register(d_tmp, sizeof(T) * max_count);

            if (rank == root) {
                ctx.copy_n(data[d].data(), max_count, d_data);
            }

            [[maybe_unused]] auto verify = [&](int64_t count) {
                auto           total_count = count;
                std::vector<T> res(total_count);
                ctx.copy_n(d_tmp, total_count, res.data());
                ctx.sync();
                size_t diff = 0;
                for (auto i = 0; i < count; ++i) {
                    auto& x = res[i];
                    auto& y = data[d][i];
                    diff += (x != y);
                    if (diff == 1) {
                        printf("%d: %f vs %f\n", (int)i, (float)x, (float)y);
                    }
                }
                if (diff) {
                    printf("[rank %d] count = %d, diff = %lu\n", g_rank, (int)count, diff);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    std::abort();
                }
            };

            std::vector<float> deltas;
            for (const auto& n : tokens_) {
                const size_t count = (size_t)n * dim;  // dim = hidden_dim / tp
                auto&        delta = deltas.emplace_back();
                h_comm->Sync();
                for (int i = 0; i < warmup_ + iters_; ++i) {
                    check_cuda_error(cudaMemsetAsync(d_tmp, 0, sizeof(T) * count, ctx.stream));
                    if (rank == root) {
                        ctx.copy_n(d_data, count, d_tmp);
                    }
                    auto ms = ctx.exec([&](auto stream) {  //
                        d_comm->Broadcast(d_tmp, d_tmp, count, dtype, 0, group, stream);
                    });
                    if (i >= warmup_) {
                        delta += ms;
                    }
                    // verify(count);
                }
                verify(count);
            }

            if (g_rank == 0) {
                SummaryHeader("broadcast", dim, n_ranks);
                for (size_t i = 0; i < tokens_.size(); ++i) {
                    const float  avg   = deltas[i] / iters_;
                    const size_t count = tokens_[i] * dim;
                    const float  algbw = sizeof(T) * count / 1e9f / avg * 1000.f;
                    const float  busbw = algbw;
                    SummaryEntry(tokens_[i], count, sizeof(T), avg, algbw, busbw);
                }
            }

            d_comm->Deregister(d_tmp);
            d_comm->Free(d_tmp);
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < d_comm_.size(); ++i) {
            threads.emplace_back(func, i, std::ref(d_comm_[i]), std::ref(h_comm_[i]));
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    template<class T>
    void TestAllreduceResidualBiasRMSnormEx(size_t dim, int group0, int group1)
    {
        const int tp_size_0 = d_comm_.at(0)->n_ranks(group0);
        const int tp_size_1 = d_comm_.at(0)->n_ranks(group1);
        const int dp_size_0 = d_comm_.size() / tp_size_0;
        const int dp_size_1 = d_comm_.size() / tp_size_1;

        const int inner_tp = std::gcd(tp_size_0, tp_size_1);

        const auto dtype = data_type_v<T>;

        std::mt19937                  gen{};
        std::uniform_int_distribution dist{0, 31};  // 5 mantissa bits

        TM_LOG_INFO("dp_size_0 %d, tp_size_0 %d", dp_size_0, tp_size_0);
        TM_LOG_INFO("dp_size_1 %d, tp_size_1 %d", dp_size_1, tp_size_1);
        TM_LOG_INFO("inner_tp %d", inner_tp);

        vector tokens = tokens_;
        for (auto& x : tokens) {
            x = (x + dp_size_0 - 1) / dp_size_0;
        }
        std::sort(tokens.begin(), tokens.end());
        tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
        const size_t max_tokens = tokens.back();

        vector<T> ref_data(dp_size_0 * max_tokens * dim);
        vector<T> src_res(ref_data.size());
        vector<T> ref_res(ref_data.size());

        vector<T> weight(dim);
        vector<T> bias(dim);

        constexpr float eps      = 1e-5;
        constexpr bool  has_bias = true;

        std::cout << "preparing data ... " << std::flush;

        for (size_t i = 0; i < dim; ++i) {
            weight[i] = T(dist(gen));
        }

        if (has_bias) {
            for (size_t i = 0; i < dim; ++i) {
                bias[i] = T(dist(gen));
            }
        }

        std::vector<std::vector<T>> src_data(tp_size_0);
        for (int r = 0; r < tp_size_0; ++r) {
            src_data[r].resize(ref_data.size());
            for (size_t i = 0; i < ref_data.size(); ++i) {
                src_data[r][i] = T(dist(gen));
            }
        }

        for (size_t i = 0; i < src_res.size(); ++i) {
            src_res[i] = T(dist(gen));
        }

        for (int r = 0; r < tp_size_0; ++r) {
            for (size_t i = 0; i < ref_data.size(); ++i) {
                ref_data[i] += src_data[r][i];
            }
        }

        for (size_t i = 0; i < dp_size_0 * max_tokens; ++i) {
            float sum = 0.f;
            for (size_t d = 0; d < dim; ++d) {
                size_t idx   = i * dim + d;
                ref_res[idx] = src_res[idx] + ref_data[idx] + bias[d];  // r' <- r + (h + b)
                sum += (float)ref_res[idx] * (float)ref_res[idx];
            }
            sum = 1 / (sqrtf(sum / dim) + eps);
            for (size_t d = 0; d < dim; ++d) {
                size_t idx    = i * dim + d;
                ref_data[idx] = (float)ref_res[idx] * sum * (float)weight[d];  // h' <- norm(r) * w
            }
        }

        std::cout << "done" << std::endl;

        auto func = [&](int index, DeviceComm& d_comm, HostComm& h_comm) {
            const int g_rank    = d_comm->rank(0);
            const int g_n_ranks = d_comm->n_ranks(0);
            const int dp_rank_0 = g_rank / tp_size_0;
            const int dp_rank_1 = g_rank / tp_size_1;
            const int tp_rank_0 = d_comm->rank(group0);
            const int tp_rank_1 = d_comm->rank(group1);
            const int local_id  = g_rank / inner_tp;  // which local partition this rank belongs to

            // TM_LOG_INFO("g_rank %d, dp_rank_0 %d, tp_rank_0 %d, dp_rank_1 %d, tp_rank_1 %d, local_id %d",
            //             g_rank,
            //             dp_rank_0,
            //             tp_rank_0,
            //             dp_rank_1,
            //             tp_rank_1,
            //             local_id);

            const size_t max_count = max_tokens * dim;

            Context ctx{g_rank};

            T* d_bias    = ctx.malloc<T>(dim);
            T* d_weight  = ctx.malloc<T>(dim);
            T* d_data    = ctx.malloc<T>(max_count);
            T* d_res     = ctx.malloc<T>(max_count);
            T* d_tmp_res = ctx.malloc<T>(max_count);

            T* d_tmp_data = (T*)d_comm->Allocate(sizeof(T) * dp_size_0 * max_count);
            d_comm->Register(d_tmp_data, sizeof(T) * dp_size_0 * max_count);

            ctx.copy_n(bias.data(), dim, d_bias);
            ctx.copy_n(weight.data(), dim, d_weight);

            [[maybe_unused]] auto verify = [&](auto n) {
                const size_t dst_tokens = n / dp_size_1 * dp_size_0;
                const size_t dst_count  = dst_tokens * dim;
                vector<T>    h_data(dst_count);
                ctx.copy_n(d_tmp_data + dp_rank_1 * dst_count, dst_count, h_data.data());
                const size_t local_tokens = (size_t)n / dp_size_1;
                const size_t local_count  = local_tokens * dim;
                const size_t slice        = (local_tokens + inner_tp - 1) / inner_tp * dim;
                const size_t first        = std::min(local_count, g_rank % inner_tp * slice);
                const size_t last         = std::min(local_count, first + slice);
                vector<T>    h_res(last - first);
                ctx.copy_n(d_tmp_res + first, h_res.size(), h_res.data());
                ctx.sync();
                size_t res_diff = 0;
                for (size_t i = first; i < last; ++i) {
                    auto& val  = h_res[i - first];
                    auto& ref  = ref_res[local_id * local_count + i];
                    int   diff = !(val == ref);
                    if (res_diff < 5 && diff) {
                        printf("[rank %d], %ld: %f vs %f\n", g_rank, i - first, (float)val, (float)ref);
                    }
                    res_diff += diff;
                }
                float data_diff = 0;
                for (size_t i = 0; i < dst_count; ++i) {
                    float diff = (float)h_data[i] - (float)ref_data[dp_rank_1 * dst_count + i];
                    data_diff += std::abs(diff);
                }
                data_diff /= dst_count;
                if (res_diff || data_diff > 0.1f || std::isnan(data_diff)) {
                    printf(
                        "[rank %d] count = %d, res_diff = %lu, data_diff = %f\n", g_rank, (int)n, res_diff, data_diff);
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    std::abort();
                }
                else if (tp_rank_1 == 0) {
                    printf("[rank %d] count = %d, data_diff = %f\n", g_rank, (int)n, data_diff);
                }
            };

            std::vector<std::pair<int, float>> stats;
            for (const auto& n : tokens) {
                if (n % dp_size_1) {
                    if (g_rank == 0) {
                        TM_LOG_INFO("Skipped %d", n);
                    }
                    continue;
                }
                // const int src_token_num = n;
                // const int dst_token_num = n / dp_size_1 * dp_size_0;
                const size_t count       = (size_t)n * dim;
                const size_t local_count = count / dp_size_1;
                std::vector  local_token_nums(dp_size_0 * dp_size_1, n / dp_size_1);
                ctx.copy_n(src_data[tp_rank_0].data() + dp_rank_0 * count, count, d_data);
                ctx.copy_n(src_res.data() + local_id * local_count, local_count, d_res);
                auto& [_, delta] = stats.emplace_back(n * dp_size_0, 0.f);
                h_comm->Sync();
                for (int i = 0; i < warmup_ + iters_; ++i) {
                    ctx.copy_n(d_data, count, d_tmp_data + dp_rank_0 * count);
                    ctx.copy_n(d_res, local_count, d_tmp_res);
                    auto ms = ctx.exec([&](auto stream) {  //
                        d_comm->AllreduceResidualBiasRMSnormEx(d_tmp_data,
                                                               d_tmp_res,
                                                               has_bias ? d_bias : nullptr,
                                                               d_weight,
                                                               eps,
                                                               dim,
                                                               dtype,
                                                               group0,
                                                               group1,
                                                               local_token_nums.data(),
                                                               stream);
                    });
                    if (i >= warmup_) {
                        delta += ms;
                    }
                    // verify(n);
                }
                verify(n);
            }

            d_comm->Deregister(d_tmp_data);
            d_comm->Free(d_tmp_data);

            if (g_rank == 0) {
                SummaryHeader("rs | rmsnorm | ag", dim, g_n_ranks);
                for (const auto& [num, ms] : stats) {
                    const float  avg    = ms / iters_;
                    const size_t count  = num * dim;
                    const float  algbw  = sizeof(T) * count / 1e9f / avg * 1000.f;
                    const float  factor = (tp_size_0 + tp_size_1 - 2) / (float)g_n_ranks;
                    const float  busbw  = algbw * factor;
                    // g_n_ranks;
                    SummaryEntry(num, count, sizeof(T), avg, algbw, busbw);
                }
            }
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < d_comm_.size(); ++i) {
            threads.emplace_back(func, i, std::ref(d_comm_[i]), std::ref(h_comm_[i]));
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    void SummaryHeader(const char* name, int dim, int world_size)
    {
        printf("[%s] dim %d tp %d warmup %d iters %d\n", name, dim, world_size, warmup_, iters_);
        printf("%15s%15s%15s%15s%15s%15s\n", "num", "count", "size", "time", "algbw", "busbw");
        printf("%15s%15s%15s%15s%15s%15s\n", "(tokens)", "(elements)", "(MB)", "(us)", "(GB/s)", "(GB/s)");
    }

    void SummaryEntry(int num, size_t count, size_t elem_size, float time, float algbw, float busbw)
    {
        float mb_size = count * elem_size / (1024.f * 1024);
        printf("%15d%15ld%15.2f%15.3f%15.3f%15.3f\n", num, count, mb_size, time * 1e3f, algbw, busbw);
    }
};

int main(int argc, char* argv[])
{

    TestComm test;

    test.Run(2048,  //
             128000,
             -1,
             10,
             10000,
             //   {1024});
             //   {1024, 2048, 4096, 8192});
             // {512});
             //    {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128});
             //  {2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128});
             //  {128, 256, 512, 1024, 2048, 4096, 8192});
             //  {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 6144, 8192});
             //   {8192, 16384, 32768});
             //  {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 8192});
             {1,   2,   4,   6,   8,   12,   16,   24,   32,   48,   64,   96,   128,
              192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 16384});

    return 0;
}
