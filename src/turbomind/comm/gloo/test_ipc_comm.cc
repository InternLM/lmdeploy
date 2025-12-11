// Copyright (c) OpenMMLab. All rights reserved.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>

#include "src/turbomind/comm/host_comm.h"

using namespace turbomind::comm;

#define TEST_TRIVIALLY_COPYABLE 1

// #define SKIP_SERIALIZE 0 // useless now

// const std::string backend = "";
const std::string backend = "hybrid";
// const std::string backend = "gloo";

struct Store {
    std::string hostname_;
    std::string port_;
    int         nnodes_;
    int         node_rank_;
    std::string py_script_;
    std::string py_file_path_ = "/tmp/start_tcp_store.py";

    std::thread thread_;

    Store(const std::string& hostname, const std::string& port, int nnodes, int node_rank):
        hostname_(hostname), port_(port), nnodes_(nnodes), node_rank_(node_rank)
    {

        int pid = getpid();

        // clang-format off
    py_script_ =
"import psutil\n"
"import os\n"
"import time\n"
"from torch.distributed import TCPStore\n"
"store = TCPStore(host_name='" + hostname_ + "',\n"
"                 port=" + port_ + ",\n"
"                 world_size=" + std::to_string(nnodes_) + ",\n"
"                 is_master=" + (node_rank_ == 0 ? "True" : "False") + ")\n"
"while True:\n"
"    time.sleep(1)\n"
"    if not psutil.pid_exists(" + std::to_string(pid) + "):\n"
"        break\n"
"    if not os.path.exists('/tmp/start_tcp_store.py'):\n"
"        break\n";

        // clang-format on
        std::ofstream py_file(py_file_path_);
        py_file << py_script_;
        py_file.close();

        std::string env_addr = "LMDEPLOY_DIST_INIT_ADDR=" + hostname_;
        std::string env_port = "LMDEPLOY_DIST_INIT_PORT=" + port_;
        setenv("LMDEPLOY_DIST_INIT_ADDR", hostname_.c_str(), 1);
        setenv("LMDEPLOY_DIST_INIT_PORT", port_.c_str(), 1);

        start();
        // wait a moment for the store to start.
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }

    ~Store()
    {
        stop();
    }

    void start()
    {
        const std::string cmd = ("python " + py_file_path_);
        thread_               = std::thread([](const std::string& cmd) { int result = system(cmd.c_str()); }, cmd);
    }

    void stop()
    {
        int r = system("rm /tmp/start_tcp_store.py");
        thread_.join();
    }
};

struct TestGlooComm {
    std::string hostname_;
    std::string port_;
    int         nnodes_;
    int         node_rank_;
    int         n_ranks_per_node_;

    std::vector<HostComm> h_comm_;

    TestGlooComm(const std::string& host, const std::string& port, int nnodes, int node_rank, int n_ranks_per_node):
        hostname_(host), port_(port), nnodes_(nnodes), node_rank_(node_rank), n_ranks_per_node_(n_ranks_per_node)
    {
        h_comm_.resize(n_ranks_per_node_);
    }

    void init()
    {
        std::unique_ptr<HostGroupId> group_id = CreateHostGroupId(backend);
        std::string                  group_id_data;
        if (1) {  // master
            group_id->Initialize();
            std::stringstream ss;
            group_id->Export(ss);
            group_id_data = ss.str();
        }

        auto init = [&](int rank) {
            // initialize host communicators
            std::stringstream            ss(group_id_data);
            std::unique_ptr<HostGroupId> host_id = CreateHostGroupId(backend);
            host_id->Import(ss);
            h_comm_[rank % n_ranks_per_node_] =
                host_id->CreateCommunicator(n_ranks_per_node_ * nnodes_, rank, node_rank_);
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < n_ranks_per_node_; ++i) {
            threads.emplace_back(init, n_ranks_per_node_ * node_rank_ + i);
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    void test_broadcast()
    {
        const int count = 10;

        auto fun = [&](HostComm& comm, int rank) {
            for (int r = 0; r < comm->n_ranks(); ++r) {

#if TEST_TRIVIALLY_COPYABLE
                std::vector<int> data(count);
#else
                std::shared_ptr<std::vector<int>> data_ptr = std::make_shared<std::vector<int>>(count);
                int*                              data     = data_ptr->data();
#endif

                for (int i = 0; i < count; ++i) {
                    data[i] = i + rank * count;  // i + rank * count
                }

#if TEST_TRIVIALLY_COPYABLE
                Broadcast(comm, data.data(), count, r);
#else
                Broadcast(comm, data_ptr, r);
                data = data_ptr->data();
#endif
                // check result
                for (int i = 0; i < count; ++i) {
                    int expected = i + r * count;
                    if (data[i] != expected) {
                        printf("Rank %d: Broadcast failed at root %d, index %d, got %d, expected %d\n",
                               rank,
                               r,
                               i,
                               data[i],
                               expected);
                    }
                }
            }
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < n_ranks_per_node_; ++i) {
            threads.emplace_back(fun, std::ref(h_comm_[i]), n_ranks_per_node_ * node_rank_ + i);
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    void test_allgather()
    {
        const int count = 40;

        auto fun = [&](HostComm& comm, int rank) {

#if TEST_TRIVIALLY_COPYABLE
            std::vector<int> data(count * comm->n_ranks());
            for (int i = 0; i < count; ++i) {
                data[i + count * comm->rank()] = i + rank * count;  // i + rank * count
            }
#else
            std::vector<std::shared_ptr<std::vector<int>>> data_ptrs(comm->n_ranks());
            data_ptrs[comm->rank()] = std::make_shared<std::vector<int>>(count);
            int* data = data_ptrs[comm->rank()]->data();
            for (int i = 0; i < count; ++i) {
                data[i] = i + rank * count;  // i + rank * count
            }
#endif

#if TEST_TRIVIALLY_COPYABLE
            AllGather(comm, data.data(), count);
            for (int r = 0; r < comm->n_ranks(); ++r) {
                for (int j = 0; j < count; ++j) {
                    int expected = j + r * count;
                    if (data[j + r * count] != expected) {
                        printf("Rank %d: AllGather failed, index %d, got %d, expected %d\n",
                               rank,
                               j + r * count,
                               data[j + r * count],
                               expected);
                    }
                }
            }
#else
            AllGather(comm, data_ptrs.data(), 1);
            for (int r = 0; r < comm->n_ranks(); ++r) {
                data = data_ptrs[r]->data();
                for (int j = 0; j < count; ++j) {
                    int expected = j + r * count;
                    if (data[j] != expected) {
                        printf("Rank %d: AllGather failed, index %d, got %d, expected %d\n",
                               rank,
                               j + r * count,
                               data[j],
                               expected);
                    }
                }
            }
#endif
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < n_ranks_per_node_; ++i) {
            threads.emplace_back(fun, std::ref(h_comm_[i]), n_ranks_per_node_ * node_rank_ + i);
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    void test_allreduce()
    {
        const int count = 10;

        auto fun = [&](HostComm& comm, int rank) {
            std::vector<int> data(count);
            for (int i = 0; i < count; ++i) {
                data[i] = i + rank * count;  // i + rank * count
            }

            AllReduce(comm, data.data(), count, RedOp::kSum);
            for (int j = 0; j < count; ++j) {
                int expected{};
                for (int r = 0; r < comm->n_ranks(); ++r) {
                    expected += j + r * count;
                }
                if (data[j] != expected) {
                    printf("Rank %d: AllReduce failed, index %d, got %d, expected %d\n", rank, j, data[j], expected);
                }
            }
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < n_ranks_per_node_; ++i) {
            threads.emplace_back(fun, std::ref(h_comm_[i]), n_ranks_per_node_ * node_rank_ + i);
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    void test_perf()
    {
        const long  kMinDurationNs   = 2e9;  // 2 second
        const long  kWarmupIter      = 5;    // warmup iter
        const float kItersMultiplier = 1.2;

        std::vector<int> count = {1024, 262144, 524288, 1048576, 2097152, 4194304, 67108864};
        //                              1M,     2M,     4M,      8M,      16M,     256M

        if (node_rank_ == 0) {
            printf("%10s %10s %10s %10s %11s %18s %10s\n",
                   "size(MB)",
                   "elements",
                   "avg(us)",
                   "p50(us)",
                   "p99(us)",
                   "bandwidth(GB/s)",
                   "iterations");
        }

        auto fun = [&](HostComm& comm, int rank, int n) {

#if TEST_TRIVIALLY_COPYABLE
            std::vector<int> data(n);
#else
            std::shared_ptr<std::vector<int>> sptr;
            if (rank == 0) {
                sptr = std::make_shared<std::vector<int>>(n);
            }
#endif

            std::vector<int64_t> times;

            auto job = [&](int n_iters) {
                times.clear();
                int64_t total = 0;
                int64_t ns    = 0;
                comm->Sync();
                for (int i = 0; i < n_iters; ++i) {
                    auto start = std::chrono::high_resolution_clock::now();
#if TEST_TRIVIALLY_COPYABLE
                    Broadcast(comm, data.data(), n, 0);
#else
                    Broadcast(comm, sptr, 0);
#endif
                    auto    now = std::chrono::high_resolution_clock::now();
                    int64_t ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
                    total += ns;
                    times.push_back(ns);
                }
                Broadcast(comm, total, 0);
                return total;
            };

            auto warmup_dur = job(kWarmupIter) / kWarmupIter;
            auto iter       = (int)std::max(kMinDurationNs / warmup_dur * 0.5f, 100.f);

            while (1) {
                auto dur = job(iter);
                std::sort(times.begin(), times.end());

                if (rank == 0) {
                    size_t bytes = n * sizeof(int);
                    int    p50   = std::min(times.size() / 2, times.size() - 1);
                    int    p99   = std::min((int)(times.size() * 0.99), (int)times.size() - 1);
                    printf("%10.5f %10d %10lld %10lld %10lld %18.3f %10lld\n",
                           bytes / 1024.f / 1024.f,
                           n,
                           static_cast<long long>(dur / 1e3f / iter),
                           static_cast<long long>(times[p50] / 1e3f),
                           static_cast<long long>(times[p99] / 1e3f),
                           (bytes * iter) / (dur / 1e9f) / (1024 * 1024 * 1024),
                           static_cast<long long>(iter));
                }

                if (dur >= kMinDurationNs) {
                    break;
                }
                iter = std::max(iter * kItersMultiplier, iter + 1.f);
            }
        };

        for (auto n : count) {
            std::vector<std::thread> threads;
            for (size_t i = 0; i < n_ranks_per_node_; ++i) {
                threads.emplace_back(fun, std::ref(h_comm_[i]), n_ranks_per_node_ * node_rank_ + i, n);
            }
            for (auto& t : threads) {
                t.join();
            }
        }
    }
};

// ./test_gloo_comm <nnodes> <node_rank> <n_ranks_per_node> <init_addr>
int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <nnodes> <node_rank> <n_ranks_per_node> <init_addr>" << std::endl;
        return -1;
    }

    int nnodes           = std::atoi(argv[1]);
    int node_rank        = std::atoi(argv[2]);
    int n_ranks_per_node = std::atoi(argv[3]);

    const std::string init_addr = argv[4];
    auto              pos       = init_addr.find(":");
    const std::string host      = init_addr.substr(0, pos);
    const std::string port      = init_addr.substr(pos + 1);

    Store store(host, port, nnodes, node_rank);

    {
        TestGlooComm test(host, port, nnodes, node_rank, n_ranks_per_node);
        test.init();

        test.test_broadcast();
        test.test_allgather();
        test.test_allreduce();

        // test.test_perf();
    }

    return 0;
}
