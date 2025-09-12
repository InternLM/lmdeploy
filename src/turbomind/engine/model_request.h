

#pragma once

#include <memory>

#include <xgrammar/xgrammar.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/gateway.h"

namespace turbomind {

class ModelRequest {
public:
    virtual ~ModelRequest() = default;

    ModelRequest(Gateway* gateway, DataType data_type, int session_len, int vocab_size, int hidden_dim);

    // Cancel running request
    void Cancel();

    // Reset the channel to uninitailized state, calls `notify` when done
    void End(std::function<void(int)> cb, uint64_t session_id);

    struct InputParam {
        std::shared_ptr<TensorMap> tensors;

        SessionParam     session;
        GenerationConfig gen_cfg;

        bool stream_output;
        bool enable_metrics;
    };

    struct OutputParam {
        std::shared_ptr<TensorMap>          tensors;
        std::shared_ptr<AtomicRequestState> state;
        std::shared_ptr<RequestMetrics>     metrics;
    };

    OutputParam Forward(InputParam param, std::function<void()> cb);
    void        setGrammar(const xgrammar::CompiledGrammar& grammar);

protected:
    Gateway* const gateway_;

    const DataType data_type_;

    const int session_len_;
    const int hidden_dim_;
    const int vocab_size_;

    uint64_t session_id_;

    std::weak_ptr<Request> request_;

    std::shared_ptr<TensorMap>                 inputs_;
    std::shared_ptr<TensorMap>                 outputs_;
    std::shared_ptr<xgrammar::CompiledGrammar> grammar_;
};

}  // namespace turbomind
