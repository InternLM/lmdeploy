
#include "src/turbomind/engine/model_executor.h"

#include <memory>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/anomaly_handler.h"

// #include "dbg.h"

namespace turbomind {

using std::shared_ptr;
using std::unique_ptr;

struct ModelExecutor::Impl {

    LanguageModel& model_;
    LlamaLinear&   linear_;

    const int device_id_;

    Queue<unique_ptr<BatchData>>& inbound_;
    Queue<unique_ptr<BatchData>>& outbound_;

    std::thread internal_thread_;

    void InternalThreadEntry()
    {
        check_cuda_error(cudaSetDevice(device_id_));

        Stream    stream  = Stream::create();
        Allocator h_alloc = Allocator(kCPU);
        Allocator d_alloc = Allocator(stream, false);

        AnomalyHandler::instance().Init(0, 1000, 0, 1000, stream.handle());

        core::ContextGuard ctx{stream, h_alloc, d_alloc};

        unique_ptr<BatchData> d;

        while (inbound_.pop(d)) {
            TM_CHECK_NOTNULL(d);
            core::Context::stream().Wait(d->ready);
            Run(*d);
            d->done.Record(core::Context::stream());
            outbound_.push(std::move(d));
        }
    }

    void Run(BatchData& d)
    {
        auto batch = &d;

        BatchCopy copy;
        TensorMap env{{"batch", d.buf()}, {"copy", copy.buf()}};

        model_.Run(BatchOp::kPrepare, d.phase, env);
        // dbg(copy);
        copy.Run();

        model_.Run(BatchOp::kForward, d.phase, env);

        model_.Run(BatchOp::kUnprep, d.phase, env);
        // dbg(copy);
        copy.Run();

        // TM_CHECK(0);
        AnomalyHandler::instance().Summarize([](...) {});
        AnomalyHandler::instance().Reset();
    }

    Impl(LanguageModel&                model,
         Context&                      context,
         int                           device_id,
         Queue<unique_ptr<BatchData>>& inbound,
         Queue<unique_ptr<BatchData>>& outbound):
        model_{model}, linear_{*context.linear}, device_id_{device_id}, inbound_{inbound}, outbound_{outbound}
    {
    }

    ~Impl()
    {
        if (internal_thread_.joinable()) {
            internal_thread_.join();
        }
    }

    void Start()
    {
        internal_thread_ = std::thread(&Impl::InternalThreadEntry, this);
    }
};

ModelExecutor::~ModelExecutor() = default;

ModelExecutor::ModelExecutor()                         = default;
ModelExecutor::ModelExecutor(ModelExecutor&&) noexcept = default;
ModelExecutor& ModelExecutor::operator=(ModelExecutor&&) noexcept = default;

ModelExecutor::ModelExecutor(LanguageModel&                model,
                             Context&                      context,
                             int                           device_id,
                             Queue<unique_ptr<BatchData>>& inbound,
                             Queue<unique_ptr<BatchData>>& outbound):
    impl_{std::make_unique<Impl>(model, context, device_id, inbound, outbound)}
{
}

void ModelExecutor::Start()
{
    return impl_->Start();
}

}  // namespace turbomind
