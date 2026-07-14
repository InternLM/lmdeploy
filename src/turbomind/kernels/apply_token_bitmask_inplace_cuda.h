#include "src/turbomind/core/tensor.h"

namespace turbomind {
void ApplyTokenBitmaskInplace(core::Tensor                logits,
                              core::Tensor                bitmask,
                              std::optional<core::Tensor> indices = std::nullopt);
}
