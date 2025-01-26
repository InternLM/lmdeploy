
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <thread>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/utils/nvtx_utils.h"

using namespace turbomind;

static constexpr bool is_ncu = 1;

struct Context {

    cudaStream_t stream;

    cudaEvent_t ev_start;
    cudaEvent_t ev_end;

    std::vector<void*> buffers;

    template<class F>
    float exec(F func)
    {
        // cudaStreamSynchronize(stream);
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

    std::vector<std::unique_ptr<Comm>> comm_;

    int              warmup_;
    int              iters_;
    std::vector<int> tokens_;
    size_t           max_tokens_;

    std::optional<Barrier> barrier_;

    void Run(int hidden_dim, int vocab_size, int tp, int warmup, int iters, std::vector<int> tokens)
    {
        int device_num{};
        cudaGetDeviceCount(&device_num);

        if (tp < 0) {
            tp = device_num;
        }

        barrier_.emplace(device_num);

        std::vector<int> devices(device_num);
        std::iota(devices.begin(), devices.end(), 0);

        // comm_ = CreateNcclComm(devices);
        comm_ = CreateCustomComm(devices);

        warmup_ = warmup;
        iters_  = iters;
        tokens_ = tokens;

        max_tokens_ = *std::max_element(tokens_.begin(), tokens_.end());

        TestAllReduce<half>(hidden_dim);
        // TestAllGather<half>(hidden_dim / tp);
        // TestAllGather<float>(vocab_size / tp);
    }

    template<class T>
    void TestAllReduce(size_t dim)
    {
        std::mt19937                       gen{};
        std::uniform_int_distribution<int> dist{0, 31};  // 5 mantissa bits
        std::vector<std::vector<T>>        data;
        std::vector<T>                     ref_data(max_tokens_ * dim);

        for (int i = 0; i < (int)comm_.size(); ++i) {
            auto& rank_data = data.emplace_back(ref_data.size());
            for (size_t j = 0; j < rank_data.size(); ++j) {
                rank_data[j] = T(dist(gen));
                ref_data[j] += rank_data[j];
            }
        }

        auto func = [&](Comm& comm) {
            const int    rank = comm.rank();
            Context      ctx{rank};
            const size_t max_count   = ref_data.size();
            T*           d_rank_data = ctx.malloc<T>(max_count);
            // T*           d_tmp       = ctx.malloc<T>(max_count);

            T* d_tmp{};
            cudaMalloc(&d_tmp, sizeof(T) * max_count);
            comm.RegisterBuffer(d_tmp, sizeof(T) * max_count);

            ctx.copy_n(data[rank].data(), max_count, d_rank_data);

            auto verify = [&](auto count) {
                std::vector<T> res(count);
                ctx.copy_n(d_tmp, count, res.data());
                cudaStreamSynchronize(ctx.stream);
                size_t diff = 0;
                for (size_t i = 0; i < count; ++i) {
                    diff += res[i] != ref_data[i];
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
                for (int i = 0; i < warmup_ + iters_; ++i) {
                    ctx.copy_n(d_rank_data, count, d_tmp);
                    auto ms = ctx.exec([&](auto stream) {  //
                        if (is_ncu && i == warmup_) {
                            barrier_->arrive_and_wait();
                            if (rank == 0) {
                                cudaProfilerStart();
                            }
                            barrier_->arrive_and_wait();
                        }

                        comm.AllReduceSum(d_tmp, d_tmp, count, stream);

                        if (is_ncu && i == warmup_) {
                            barrier_->arrive_and_wait();
                            if (rank == 0) {
                                cudaProfilerStop();
                            }
                            barrier_->arrive_and_wait();
                        }
                    });
                    if (i >= warmup_) {
                        delta += ms;
                    }
                    // verify(count);
                }
                // verify(count);
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
        };

        std::vector<std::thread> threads;
        for (auto& comm : comm_) {
            threads.emplace_back(func, std::ref(*comm));
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

        for (int i = 0; i < (int)comm_.size(); ++i) {
            auto& rank_data = data.emplace_back(max_tokens_ * dim);
            for (size_t j = 0; j < rank_data.size(); ++j) {
                rank_data[j] = T(dist(gen));
            }
        }

        auto func = [&](Comm& comm) {
            const int    rank = comm.rank();
            Context      ctx{rank};
            const size_t max_count   = max_tokens_ * dim;
            T*           d_rank_data = ctx.malloc<T>(max_count);
            // T*           d_tmp       = ctx.malloc<T>(max_count * comm.world_size());
            // cudaStreamSynchronize(ctx.stream);
            T* d_tmp{};
            cudaMalloc(&d_tmp, sizeof(T) * max_count * comm.world_size());
            comm.RegisterBuffer(d_tmp, sizeof(T) * max_count * comm.world_size());

            ctx.copy_n(data[rank].data(), max_count, d_rank_data);

            std::vector<float> deltas;
            for (const auto& n : tokens_) {
                const size_t count = (size_t)n * dim;
                auto&        delta = deltas.emplace_back();

                barrier_->arrive_and_wait();

                for (int i = 0; i < warmup_ + iters_; ++i) {
                    ctx.copy_n(d_rank_data, count, d_tmp + rank * count);
                    auto ms = ctx.exec([&](auto stream) {  //
                        comm.AllGather(d_tmp + rank * count, d_tmp, count, stream);
                    });
                    if (i >= warmup_) {
                        delta += ms;
                    }
                }

                size_t diff = 0;
                for (int r = 0; r < comm.world_size(); ++r) {
                    std::vector<T> res(count);
                    ctx.copy_n(d_tmp + r * count, count, res.data());
                    cudaStreamSynchronize(ctx.stream);
                    for (size_t i = 0; i < count; ++i) {
                        diff += res[i] != data[r][i];
                    }
                }
                if (diff) {
                    printf("[rank %d] diff = %lu\n", comm.rank(), diff);
                }
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
        };

        std::vector<std::thread> threads;
        for (auto& comm : comm_) {
            threads.emplace_back(func, std::ref(*comm));
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

    test.Run(8192,  //
             128000,
             -1,
             10,
             100,
             //   {8192});
             //   {1, 8, 16, 64, 128});
             //  {128, 256, 512, 1024, 2048, 4096, 8192});
             {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 6144, 8192});

    return 0;
}