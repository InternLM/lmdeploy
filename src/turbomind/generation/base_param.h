

#pragma once

namespace turbomind {

class BaseGenerationParam {
public:
    explicit BaseGenerationParam(int max_batch_size, int vocab_size, int vocab_size_padded):
        max_batch_size_{max_batch_size}, vocab_size_{vocab_size}, vocab_size_padded_{vocab_size_padded}
    {
    }

protected:
    int max_batch_size_;
    int vocab_size_;
    int vocab_size_padded_;
};

}  // namespace turbomind
