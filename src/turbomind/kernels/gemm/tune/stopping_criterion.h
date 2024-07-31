
#include "src/turbomind/kernels/gemm/tune/stats.h"
#include <memory>
#include <string>
#include <vector>

namespace turbomind::gemm {

class StoppingCriterion {
public:
    virtual ~StoppingCriterion()                 = default;
    virtual bool should_stop(const Stats& stats) = 0;
};

std::unique_ptr<StoppingCriterion> CreateStoppingCriterion(int max_iter, float max_ms);

}  // namespace turbomind::gemm
