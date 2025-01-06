

#pragma once

#include <memory>

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

class ModelRequest {
public:
    virtual ~ModelRequest() = default;

    ModelRequest(Gateway* gateway, DataType data_type, int session_len, int vocab_size, int hidden_dim);

    // Cancel running request
    void Cancel();

    // Reset the channel to uninitailized state, calls `notify` when done
    void End(std::function<void(int)> cb, uint64_t session_id);

    using TensorMap_ = std::unordered_map<std::string, ManagedTensor>;

    struct InputParam {
        std::shared_ptr<TensorMap_> tensors;

        SessionParam     session;
        GenerationConfig gen_cfg;

        bool stream_output;
    };

    struct OutputParam {
        std::shared_ptr<TensorMap_>         tensors;
        std::shared_ptr<AtomicRequestState> state;
    };

    OutputParam Forward(InputParam param, std::function<void()> cb);

protected:
    Gateway* const gateway_;

    const DataType data_type_;

    const int session_len_;
    const int hidden_dim_;
    const int vocab_size_;

    uint64_t session_id_;

    std::weak_ptr<Request> request_;

    std::shared_ptr<TensorMap_> inputs_;   // owned by caller
    std::shared_ptr<TensorMap_> outputs_;  // owned by `this`
};

}  // namespace turbomind
