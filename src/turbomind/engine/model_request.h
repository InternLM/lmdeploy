

#pragma once

#include <memory>

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

class ModelRequest {
public:
    virtual ~ModelRequest() = default;

    ModelRequest(Gateway* gateway, int session_len, int vocab_size);

    // Cancel running request
    void Cancel();

    // Reset the channel to uninitailized state, calls `notify` when done
    void End(std::function<void(int)> cb);

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
    Gateway* gateway_;

    uint64_t session_id_;

    int session_len_;
    int vocab_size_;

    std::weak_ptr<Request> request_;

    std::shared_ptr<TensorMap_> inputs_;   // owned by caller
    std::shared_ptr<TensorMap_> outputs_;  // owned by `this`
};

}  // namespace turbomind
