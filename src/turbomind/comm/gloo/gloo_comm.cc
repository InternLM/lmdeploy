// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <mutex>

#include <gloo/allgather.h>
#include <gloo/allreduce.h>
#include <gloo/barrier.h>
#include <gloo/broadcast.h>
#include <gloo/common/utils.h>
#include <gloo/context.h>
#include <gloo/math.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/tcp/attr.h>
#include <gloo/transport/tcp/device.h>

#include "src/turbomind/comm/gloo/tcp_store.h"
#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind::comm {

const char* GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME";
const char  STORE_INFO_DELIM       = ',';

std::shared_ptr<::gloo::transport::Device> createGlooDevice()
{
    ::gloo::transport::tcp::attr attr;
    if (auto ifname = std::getenv(GLOO_SOCKET_IFNAME_ENV); ifname) {
        attr.iface = ifname;
    }
    else {
        attr.hostname = ::gloo::getHostname();
    }
    return ::gloo::transport::tcp::CreateDevice(attr);
}

class Store: public ::gloo::rendezvous::PrefixStore {
public:
    explicit Store(const std::string& host, int port, const std::string& prefix):
        host_(host), port_(port), ::gloo::rendezvous::PrefixStore(prefix, nullptr)
    {
        store_ = std::make_shared<TCPStore>(host_, port_);
    };

    ~Store() = default;

    std::shared_ptr<Store> New(const std::string& prefix)
    {
        std::string new_prefix = prefix + "/" + prefix_;
        return std::make_shared<Store>(host_, port_, new_prefix);
    }

public:
    std::string host_;
    int         port_;

    using ::gloo::rendezvous::PrefixStore::store_;
    using ::gloo::rendezvous::PrefixStore::prefix_;
};

class GlobalStoreFactory {
public:
    static GlobalStoreFactory& Instance()
    {
        static GlobalStoreFactory instance;
        return instance;
    }

    std::string New()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        TM_CHECK(std::getenv("LMDEPLOY_DIST_INIT_ADDR") != nullptr) << "LMDEPLOY_DIST_INIT_ADDR not set";
        TM_CHECK(std::getenv("LMDEPLOY_DIST_INIT_PORT") != nullptr) << "LMDEPLOY_DIST_INIT_PORT not set";

        std::string host = std::getenv("LMDEPLOY_DIST_INIT_ADDR");
        int         port = std::stoi(std::getenv("LMDEPLOY_DIST_INIT_PORT"));

        std::stringstream ss;
        ss << host << STORE_INFO_DELIM << port << STORE_INFO_DELIM << prefix_++;
        return ss.str();
    }

    std::shared_ptr<Store> Load(const std::string& info)
    {
        std::stringstream        ss(info);
        std::vector<std::string> keys;
        std::string              local;
        while (getline(ss, local, STORE_INFO_DELIM)) {
            keys.push_back(std::move(local));
        }
        FT_CHECK(keys.size() == 3);

        std::string host   = keys[0];
        int         port   = stoi(keys[1]);
        std::string prefix = keys[2];

        return std::make_shared<Store>(host, port, prefix);
    }

private:
    GlobalStoreFactory() {}

    std::mutex mutex_;
    int        prefix_{0};
};

typedef void (*ReduceFunc)(void*, const void*, const void*, size_t);

struct GlooCommImpl: public IpcHostCommImpl {

    struct SplitInfo {
        int color;
        int rank;

        bool operator<(const SplitInfo& other) const
        {
            return (color < other.color) || (color == other.color && rank < other.rank);
        }

        bool operator==(const SplitInfo& other) const
        {
            return (color == other.color) && (rank == other.rank);
        }
    };

    GlooCommImpl(std::shared_ptr<Store> store, int n_ranks, int rank):
        store_{std::move(store)}, rank_{rank}, n_ranks_{n_ranks}
    {
        device_  = createGlooDevice();
        context_ = std::make_shared<::gloo::rendezvous::Context>(rank_, n_ranks_);
        context_->connectFullMesh(store_, device_);
    }

    ~GlooCommImpl() {}

    char* create_buffer(size_t size) override
    {
        if (buffer_ == nullptr || size > buffer_->byte_size()) {
            buffer_ = std::make_shared<::turbomind::core::Buffer_<uint8_t>>(size, kCPU);
        }
        return static_cast<char*>(buffer_->raw_data());
    }

    int rank() const override
    {
        return rank_;
    }

    int n_ranks() const override
    {
        return n_ranks_;
    }

    bool is_same_process() const override
    {
        return false;
    }

    std::shared_ptr<HostCommImpl> Split(int color, int key) override
    {
        // don't know why key was set to 0
        auto vec  = comm::AllGather(this, SplitInfo{color, rank_});
        auto last = std::stable_partition(vec.begin(), vec.end(), [&](auto x) {  //
            return x.color == color;
        });
        vec.erase(last, vec.end());
        std::stable_sort(vec.begin(), vec.end(), [](auto& a, auto& b) {  //
            return a < b;
        });

        auto new_prefix  = std::to_string(color) + ":" + std::to_string(n_split_++);
        auto new_store   = store_->New(new_prefix);
        int  new_n_ranks = vec.size();
        int  new_rank    = std::find(vec.begin(), vec.end(), SplitInfo{color, rank_}) - vec.begin();
        return std::make_shared<GlooCommImpl>(new_store, new_n_ranks, new_rank);
    }

    void Sync(bool blocking) override
    {
        ::gloo::BarrierOptions opts(context_);
        opts.setTimeout(kTimeOut);
        ::gloo::barrier(opts);
    }

    void Broadcast(void* data, int count, DataType dtype, int root, copy_fn copy) override
    {
        ::gloo::BroadcastOptions opts(context_);
        opts.setRoot(root);
        opts.setTimeout(kTimeOut);
        opts.setOutput((char*)data, count * byte_size(dtype));
        ::gloo::broadcast(opts);
    }

    void AllGather(void* data, int count, DataType dtype, copy_fn copy) override
    {
        ::gloo::AllgatherOptions opts(context_);
        opts.setTimeout(kTimeOut);
        opts.setOutput((char*)data, count * byte_size(dtype) * n_ranks_);
        ::gloo::allgather(opts);
    }

    static ReduceFunc getReduceFunc(DataType dtype, RedOp red_op)
    {

        auto dispatch_op = [&](auto t) -> ReduceFunc {
            using T = decltype(t);
            switch (red_op) {
                case RedOp::kSum:
                    return ::gloo::sum<T>;
                case RedOp::kMax:
                    return ::gloo::max<T>;
                case RedOp::kMin:
                    return ::gloo::min<T>;
                default:
                    return {};
            }
        };

        auto dispatch = [&]() -> ReduceFunc {
            switch (dtype) {
                case kInt32:
                    return dispatch_op(int32_t{});
                case kInt64:
                    return dispatch_op(int64_t{});
                case kUint32:
                    return dispatch_op(uint32_t{});
                case kUint64:
                    return dispatch_op(uint64_t{});
                default:
                    return {};
            }
        };

        if (auto fn = dispatch()) {
            return fn;
        }
        else {
            throw std::runtime_error("not implemented");
            return {};
        }
    }

    void AllReduce(void* data, int count, DataType dtype, RedOp red_op) override
    {
        ::gloo::AllreduceOptions opts(context_);
        opts.setTimeout(kTimeOut);
        opts.setReduceFunction(getReduceFunc(dtype, red_op));
        switch (dtype) {
            case kInt32:
                opts.setOutput((int32_t*)data, count);
                break;
            case kInt64:
                opts.setOutput((int64_t*)data, count);
                break;
            case kUint32:
                opts.setOutput((uint32_t*)data, count);
                break;
            case kUint64:
                opts.setOutput((uint64_t*)data, count);
                break;
            default:
                throw std::runtime_error("not implemented");
        }
        ::gloo::allreduce(opts);
    }

    static constexpr std::chrono::milliseconds kTimeOut = std::chrono::milliseconds(1000 * 60 * 30);  // 30 minutes

    int                                          n_split_{};
    std::shared_ptr<::gloo::transport::Device>   device_;
    std::shared_ptr<::gloo::rendezvous::Context> context_;
    std::shared_ptr<Store>                       store_;
    int                                          rank_;
    int                                          n_ranks_;

    // reduce the overhead caused by repeatedly creating buffers.
    std::shared_ptr<::turbomind::core::Buffer_<uint8_t>> buffer_;
};

class GlooGroupId: public HostGroupId {

    void Initialize() override
    {
        info_ = GlobalStoreFactory::Instance().New();
        TM_LOG_INFO("[TM][COMM] GlooGroupId=%s", info_.c_str());
    }

    void Export(std::ostream& os) override
    {
        os << info_;
    }

    void Import(std::istream& is) override
    {
        std::stringstream ss;
        ss << is.rdbuf();
        info_ = ss.str();
    }

    HostComm CreateCommunicator(int n_ranks, int rank, int node_rank = 0) override
    {
        FT_CHECK(info_ != "");
        auto impl = std::make_shared<GlooCommImpl>(GlobalStoreFactory::Instance().Load(info_), n_ranks, rank);
        return std::static_pointer_cast<HostCommImpl>(impl);
    }

private:
    std::string                                info_;  // ip,port,prefix
    std::shared_ptr<::gloo::rendezvous::Store> store_;
};

std::unique_ptr<HostGroupId> CreateGlooGroupId()
{
    return std::make_unique<GlooGroupId>();
}

}  // namespace turbomind::comm
