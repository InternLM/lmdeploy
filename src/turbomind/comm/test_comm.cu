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

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "src/turbomind/comm/barrier.h"
#include "src/turbomind/comm/comm.h"
#include "src/turbomind/comm/host.h"
#include "src/turbomind/utils/Tensor.h"

using namespace turbomind::comm;

[[maybe_unused]] static constexpr bool is_ncu = 0;

struct Context {

    cudaStream_t stream;

    cudaEvent_t ev_start;
    cudaEvent_t ev_end;

    std::vector<void*> buffers;

    template<class F>
    float exec(F func)
    {
        cudaStreamSynchronize(stream);
        cudaEventRecord(ev_start, stream);

        func(stream);

        cudaEventRecord(ev_end, stream);
        cudaEventSynchronize(ev_end);
        float ms{};
        cudaEventElapsedTime(&ms, ev_start, ev_end);
        return ms;
    }

    template<class T>
    T* malloc(size_t count)
    {
        T* data;
        cudaMallocAsync(&data, sizeof(T) * count, stream);
        buffers.push_back(data);
        return data;
    }

    template<class T>
    void copy_n(const T* src, size_t count, T* dst)
    {
        cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDefault, stream);
    }

    Context(int device_id)
    {
        cudaSetDevice(device_id);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);
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

    std::vector<std::unique_ptr<Comm>>     comm_;
    std::vector<std::shared_ptr<HostComm>> host_comm_;

    int              warmup_;
    int              iters_;
    std::vector<int> tokens_;
    size_t           max_tokens_;

    static auto Init(int world_size, const std::string& backend)
    {

        std::unique_ptr<HostGroupId> group_id = CreateHostGroupId({});
        std::string                  group_id_data;
        if (1) {  // master
            group_id->Initialize();
            std::stringstream ss;
            group_id->Export(ss);
            group_id_data = ss.str();
        }

        std::vector<std::unique_ptr<Comm>>     comm(world_size);
        std::vector<std::shared_ptr<HostComm>> host_comm(world_size);

        auto init = [&](int rank) {
            // initialize host communicators
            std::stringstream            ss(group_id_data);
            std::unique_ptr<HostGroupId> host_id = CreateHostGroupId({});
            host_id->Import(ss);
            host_comm[rank] = host_id->CreateHostCommunicator(rank, world_size);

            // initialize device communicators
            cudaSetDevice(rank);
            comm[rank] = CreateCommunicator(backend, rank, world_size, host_comm[rank]);
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < world_size; ++i) {
            threads.emplace_back(init, i);
        }
        for (auto& t : threads) {
            t.join();
        }

        return std::make_pair(host_comm, std::move(comm));
    }

    void Run(int hidden_dim, int vocab_size, int tp, int warmup, int iters, std::vector<int> tokens)
    {
        int device_num{};
        cudaGetDeviceCount(&device_num);

        std::cout << "Device count: " << device_num << "\n";

        if (tp < 0) {
            tp = device_num;
        }

        // barrier_.emplace(device_num);

        std::tie(host_comm_, comm_) = Init(device_num, "native");

        warmup_ = warmup;
        iters_  = iters;
        tokens_ = tokens;

        max_tokens_ = *std::max_element(tokens_.begin(), tokens_.end());

        TestAllReduce<half>(hidden_dim);
        TestAllreduceResidualBiasRMSnorm<half>(hidden_dim);
        TestAllGather<half>(hidden_dim / tp);  // tp embedding
        // TestAllGather<float>(vocab_size / tp);
    }

    template<class T>
    void TestAllReduce(size_t dim)
    {
        std::mt19937                       gen{};
        std::uniform_int_distribution<int> dist{0, 31};  // 5 mantissa bits
        std::vector<std::vector<T>>        data;
        std::vector<T>                     ref_data(max_tokens_ * dim);

        std::cout << "preparing data ... " << std::flush;

        for (int i = 0; i < (int)comm_.size(); ++i) {
            auto& rank_data = data.emplace_back(ref_data.size());
            for (size_t j = 0; j < rank_data.size(); ++j) {
                rank_data[j] = T(dist(gen));
                ref_data[j] += rank_data[j];
            }
        }

        std::cout << "done.\n";

        auto func = [&](Comm& comm, HostComm& h_comm) {
            const int    rank = comm.rank();
            Context      ctx{rank};
            const size_t max_count   = ref_data.size();
            T*           d_rank_data = ctx.malloc<T>(max_count);
            T*           d_tmp       = (T*)comm.Allocate(sizeof(T) * max_count);
            comm.Register(d_tmp, sizeof(T) * max_count);
            ctx.copy_n(data[rank].data(), max_count, d_rank_data);

            [[maybe_unused]] auto verify = [&](auto count) {
                std::vector<T> res(count);
                ctx.copy_n(d_tmp, count, res.data());
                cudaStreamSynchronize(ctx.stream);
                size_t diff = 0;
                for (size_t i = 0; i < count; ++i) {
                    diff += res[i] != ref_data[i];
                    if (diff == 1) {
                        printf("%d: %f vs %f\n", (int)i, (float)res[i], (float)ref_data[i]);
                    }
                }
                if (diff) {
                    printf("[rank %d] count = %d, diff = %lu\n", rank, (int)count, diff);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    std::abort();
                }
            };

            std::vector<float> deltas;
            for (const auto& n : tokens_) {
                const size_t count = (size_t)n * dim;
                auto&        delta = deltas.emplace_back();

                // barrier_->arrive_and_wait();
                h_comm.Sync();

                const std::vector<size_t> counts(comm.world_size(), count / comm.world_size());

                for (int i = 0; i < warmup_ + iters_; ++i) {
                    ctx.copy_n(d_rank_data, count, d_tmp);
                    auto ms = ctx.exec([&](auto stream) {  //
                        if (is_ncu && i == warmup_) {
                            h_comm.Sync();
                            // barrier_->arrive_and_wait();
                            if (rank == 0) {
                                cudaProfilerStart();
                            }
                            h_comm.Sync();
                            // barrier_->arrive_and_wait();
                        }

                        comm.AllReduceSum(d_tmp, d_tmp, count, stream);

                        // const auto type = turbomind::getTensorType<T>();
                        // comm.ReduceScatterAsym(d_tmp, d_tmp, counts.data(), type, stream);
                        // comm.AllGatherAsym(d_tmp, d_tmp, counts.data(), type, stream);

                        if (is_ncu && i == warmup_) {
                            h_comm.Sync();
                            // barrier_->arrive_and_wait();
                            if (rank == 0) {
                                cudaProfilerStop();
                            }
                            h_comm.Sync();
                            // barrier_->arrive_and_wait();
                        }
                    });
                    if (i >= warmup_) {
                        delta += ms;
                    }
                    // verify(count);
                }
                verify(count);
            }

            if (rank == 0) {
                SummaryHeader("allreduce", dim, comm.world_size());
                for (size_t i = 0; i < tokens_.size(); ++i) {
                    const float  avg   = deltas[i] / iters_;
                    const size_t count = tokens_[i] * dim;
                    const float  algbw = sizeof(T) * count / 1e9f / avg * 1000.f;
                    const float  busbw = algbw * (2 * (comm.world_size() - 1)) / comm.world_size();
                    SummaryEntry(tokens_[i], count, sizeof(T), avg, algbw, busbw);
                }
            }

            comm.Deregister(d_tmp);
            comm.Free(d_tmp);
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < comm_.size(); ++i) {
            threads.emplace_back(func, std::ref(*comm_[i]), std::ref(*host_comm_[i]));
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    template<class T>
    void TestAllreduceResidualBiasRMSnorm(size_t dim)
    {
        std::mt19937                       gen{};
        std::uniform_int_distribution<int> dist{0, 31};  // 5 mantissa bits
        std::vector<std::vector<T>>        data;
        std::vector<T>                     ref_data(max_tokens_ * dim);
        std::vector<T>                     residual(max_tokens_ * dim);
        std::vector<T>                     ref_residual(max_tokens_ * dim);
        std::vector<T>                     weight(dim);
        std::vector<T>                     bias(dim);
        constexpr float                    eps      = 1e-5;
        constexpr bool                     has_bias = true;

        std::cout << "preparing data ... " << std::flush;

        for (size_t i = 0; i < dim; ++i) {
            weight[i] = T(dist(gen));
        }

        if (has_bias) {
            for (size_t i = 0; i < dim; ++i) {
                bias[i] = T(dist(gen));
            }
        }

        for (int i = 0; i < (int)comm_.size(); ++i) {
            auto& rank_data = data.emplace_back(ref_data.size());
            for (size_t j = 0; j < rank_data.size(); ++j) {
                rank_data[j] = T(dist(gen));
                ref_data[j] += rank_data[j];  // sum over all ranks
            }
        }

        for (size_t i = 0; i < max_tokens_; ++i) {
            float sum = 0.f;
            for (size_t d = 0; d < dim; ++d) {
                const size_t index  = i * dim + d;
                residual[index]     = T(dist(gen));
                ref_residual[index] = residual[index] + ref_data[index] + bias[d];  // r' <- r + (h + b)
                sum += (float)ref_residual[index] * (float)ref_residual[index];
            }
            sum = rsqrtf(sum / dim + eps);
            for (size_t d = 0; d < dim; ++d) {
                const size_t index = i * dim + d;
                float        tmp   = (float)ref_residual[index];
                ref_data[index]    = tmp * sum * (float)weight[d];  // h' <- norm(r) * w
            }
        }

        std::cout << "done.\n";

        auto func = [&](Comm& comm, HostComm& h_comm) {
            const int    rank = comm.rank();
            Context      ctx{rank};
            const size_t max_count   = ref_data.size();
            T*           d_rank_data = ctx.malloc<T>(max_count);
            T*           d_residual  = ctx.malloc<T>(max_count);
            T*           d_bias      = ctx.malloc<T>(dim);
            T*           d_weight    = ctx.malloc<T>(dim);
            T*           d_tmp_res   = ctx.malloc<T>(max_count);
            T*           d_tmp_data  = (T*)comm.Allocate(sizeof(T) * max_count);

            comm.Register(d_tmp_data, sizeof(T) * max_count);

            ctx.copy_n(data[rank].data(), max_count, d_rank_data);
            ctx.copy_n(residual.data(), max_count, d_residual);
            ctx.copy_n(bias.data(), dim, d_bias);
            ctx.copy_n(weight.data(), dim, d_weight);

            [[maybe_unused]] auto verify = [&](auto token_num) {
                const size_t   count = (size_t)token_num * dim;
                std::vector<T> h_data(count);
                std::vector<T> h_res(count);
                ctx.copy_n(d_tmp_data, count, h_data.data());
                ctx.copy_n(d_tmp_res, count, h_res.data());
                cudaStreamSynchronize(ctx.stream);
                const int    world_size = comm.world_size();
                const size_t slice      = (token_num + world_size - 1) / world_size * dim;
                const size_t first      = rank * slice;
                const size_t last       = std::min(first + slice, count);
                size_t       res_diff   = 0;
                for (size_t i = first; i < last; ++i) {
                    int is_diff = !(h_res[i] == ref_residual[i]);
                    if (!res_diff && is_diff) {
                        printf("[rank %d], %d: %f vs %f\n",
                               rank,
                               (int)(i - first),
                               (float)h_res[i],
                               (float)ref_residual[i]);
                    }
                    res_diff += is_diff;
                }
                float data_diff = 0;
                for (size_t i = 0; i < count; ++i) {
                    float diff = (float)h_data[i] - (float)ref_data[i];
                    data_diff += std::abs(diff);
                }
                data_diff /= count;
                if (rank == 0) {
                    printf("[rank %d] count = %d, data_diff = %f\n", rank, (int)token_num, data_diff);
                }
                if (res_diff || data_diff > 0.1f || std::isnan(data_diff)) {
                    printf("[rank %d] count = %d, res_diff = %lu, data_diff = %f\n",
                           rank,
                           (int)token_num,
                           res_diff,
                           data_diff);
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    std::abort();
                }
            };

            std::vector<float> deltas;
            for (const auto& n : tokens_) {
                const size_t count = (size_t)n * dim;
                auto&        delta = deltas.emplace_back();
                h_comm.Sync();
                for (int i = 0; i < warmup_ + iters_; ++i) {

                    ctx.copy_n(d_rank_data, count, d_tmp_data);
                    ctx.copy_n(d_residual, count, d_tmp_res);

                    auto ms = ctx.exec([&](auto stream) {  //
                        if (is_ncu && i == warmup_) {
                            h_comm.Sync();
                            if (rank == 0) {
                                cudaProfilerStart();
                            }
                            h_comm.Sync();
                        }
                        comm.AllreduceResidualBiasRMSnorm(
                            d_tmp_data, d_tmp_res, has_bias ? d_bias : nullptr, d_weight, eps, dim, n, stream);
                        // comm.AllreduceResidualBiasRMSnormEx(d_tmp_data,
                        //                                     d_tmp_res,
                        //                                     has_bias ? d_bias : nullptr,
                        //                                     d_weight,
                        //                                     eps,
                        //                                     dim,
                        //                                     n,
                        //                                     turbomind::getTensorType<T>(),
                        //                                     comm.world_size(),
                        //                                     comm.world_size(),
                        //                                     stream);
                        if (is_ncu && i == warmup_) {
                            h_comm.Sync();
                            if (rank == 0) {
                                cudaProfilerStop();
                            }
                            h_comm.Sync();
                        }
                    });
                    if (i >= warmup_) {
                        delta += ms;
                    }
                    // verify(n);
                }
                verify(n);
            }

            comm.Deregister(d_tmp_data);
            comm.Free(d_tmp_data);

            if (rank == 0) {
                SummaryHeader("allreduce | rmsnorm", dim, comm.world_size());
                for (size_t i = 0; i < tokens_.size(); ++i) {
                    const float  avg   = deltas[i] / iters_;
                    const size_t count = tokens_[i] * dim;
                    const float  algbw = sizeof(T) * count / 1e9f / avg * 1000.f;
                    const float  busbw = algbw * (2 * (comm.world_size() - 1)) / comm.world_size();
                    SummaryEntry(tokens_[i], count, sizeof(T), avg, algbw, busbw);
                }
            }
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < comm_.size(); ++i) {
            threads.emplace_back(func, std::ref(*comm_[i]), std::ref(*host_comm_[i]));
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    template<class T>
    void TestAllGather(size_t dim)
    {
        std::mt19937                       gen{};
        std::uniform_int_distribution<int> dist{0, 100};
        std::vector<std::vector<T>>        data;

        std::cout << "preparing data ... " << std::flush;

        for (int i = 0; i < (int)comm_.size(); ++i) {
            auto& rank_data = data.emplace_back(max_tokens_ * dim);
            for (size_t j = 0; j < rank_data.size(); ++j) {
                rank_data[j] = T(dist(gen));
            }
        }

        std::cout << "done.\n";

        auto func = [&](Comm& comm, HostComm& h_comm) {
            const int    rank       = comm.rank();
            const int    world_size = comm.world_size();
            Context      ctx{rank};
            const size_t max_count   = max_tokens_ * dim;
            T*           d_rank_data = ctx.malloc<T>(max_count);
            T*           d_tmp       = (T*)comm.Allocate(sizeof(T) * max_count * world_size);
            comm.Register(d_tmp, sizeof(T) * max_count * world_size);
            ctx.copy_n(data[rank].data(), max_count, d_rank_data);
            [[maybe_unused]] auto verify = [&](int64_t count) {
                auto           total_count = count * world_size;
                std::vector<T> res(total_count);
                ctx.copy_n(d_tmp, total_count, res.data());
                cudaStreamSynchronize(ctx.stream);
                size_t diff = 0;
                for (int r = 0; r < world_size; ++r) {
                    for (auto i = 0; i < count; ++i) {
                        diff += res[r * count + i] != data[r][i];
                        if (diff == 1) {
                            printf("%d: %f vs %f\n", (int)i, (float)res[r * count + i], (float)data[r][i]);
                        }
                    }
                }
                if (diff) {
                    printf("[rank %d] count = %d, diff = %lu\n", rank, (int)count, diff);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    std::abort();
                }
            };

            std::vector<float> deltas;
            for (const auto& n : tokens_) {
                const size_t count = (size_t)n * dim;  // dim = hidden_dim / tp
                auto&        delta = deltas.emplace_back();

                // barrier_->arrive_and_wait();
                h_comm.Sync();

                const std::vector<size_t> counts(world_size, count);

                for (int i = 0; i < warmup_ + iters_; ++i) {
                    cudaMemsetAsync(d_tmp, 0, sizeof(T) * count * comm.world_size(), ctx.stream);
                    ctx.copy_n(d_rank_data, count, d_tmp + rank * count);
                    auto ms = ctx.exec([&](auto stream) {  //
                        if (comm.Query(kHasAllGather2D)) {
                            comm.AllGather2D(d_tmp + rank * count, d_tmp, dim, count, dim, n, {1, 1}, stream);
                        }
                        else {
                            comm.AllGather(d_tmp + rank * count, d_tmp, count, 0, stream);
                            // comm.AllGatherAsym(d_tmp, d_tmp, counts.data(), turbomind::getTensorType<T>(), stream);
                        }
                    });
                    if (i >= warmup_) {
                        delta += ms;
                    }
                    // verify(count);
                }

                verify(count);
            }

            if (rank == 0) {
                SummaryHeader("allgather", dim, comm.world_size());
                for (size_t i = 0; i < tokens_.size(); ++i) {
                    const float  avg   = deltas[i] / iters_;
                    const size_t count = comm.world_size() * tokens_[i] * dim;
                    const float  algbw = sizeof(T) * count / 1e9f / avg * 1000.f;
                    const float  busbw = algbw * (comm.world_size() - 1) / comm.world_size();
                    SummaryEntry(tokens_[i], count, sizeof(T), avg, algbw, busbw);
                }
            }

            comm.Deregister(d_tmp);
            comm.Free(d_tmp);
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < comm_.size(); ++i) {
            threads.emplace_back(func, std::ref(*comm_[i]), std::ref(*host_comm_[i]));
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
#if 0
    const int                N     = 8;
    auto                     state = std::make_shared<HostComm::State>(N);
    std::vector<std::thread> threads;
    for (int r = 0; r < N; ++r) {
        threads.emplace_back([&, r] {
            HostComm comm(N, r, state);
            int      group    = 0;
            // group             = comm.Split(r / (N / 2), 0);
            group             = comm.Split(r % 4, 0);
            auto         tick = std::chrono::steady_clock::now();
            volatile int a;
            volatile int b;
            for (int i = 0; i < 1; ++i) {
                a      = Allreduce<RedOp::kSum>(comm, r, group);
                auto v = Allgather(comm, r, group);
                b      = std::accumulate(v.begin(), v.end(), 0);
                for (int j = 0; j < N; ++j) {
                    comm.Sync();
                    if (j == r) {
                        std::cout << a << " " << b << std::endl;
                    }
                }
            }
            auto tock = std::chrono::steady_clock::now();

            for (int i = 0; i < N; ++i) {
                comm.Sync();
                if (i == r) {
                    std::cout << std::chrono::duration<float, std::milli>(tock - tick).count() << std::endl;
                }
            }
        });
    }
    std::cout << "main thread waiting.\n";
    for (auto& t : threads) {
        t.join();
    }
    return 0;
#endif

    TestComm test;

    test.Run(8192,  //
             128000,
             -1,
             10,
             100,
             //   {1024, 2048, 4096, 8192});
             // {512});
             //  {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128});
             //  {128, 256, 512, 1024, 2048, 4096, 8192});
             //  {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 6144, 8192});
             //   {8192, 16384, 32768});
             //  {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024});
             {1,   2,   4,   6,   8,   12,   16,   24,   32,   48,   64,   96,  128,
              192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192});

    return 0;
}
