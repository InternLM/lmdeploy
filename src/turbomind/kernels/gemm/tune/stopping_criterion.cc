#include "src/turbomind/kernels/gemm/tune/stopping_criterion.h"
#include <memory>

namespace turbomind::gemm {

namespace stopping_criterions {

class Optimistic: public StoppingCriterion {
public:
    Optimistic(int max_iter, float max_ms)
    {
        max_iter_ = max_iter ? max_iter : std::numeric_limits<int>::max();
        max_ms_   = max_ms > 0 ? max_ms : std::numeric_limits<float>::infinity();
    }
    bool should_stop(const Stats& stats) override
    {
        return stats.count() >= max_iter_ || stats.sum() >= max_ms_;
    }

private:
    int   max_iter_;
    float max_ms_;
};

}  // namespace stopping_criterions

std::unique_ptr<StoppingCriterion> CreateStoppingCriterion(int max_iter, float max_ms)
{
    return std::make_unique<stopping_criterions::Optimistic>(max_iter, max_ms);
}

}  // namespace turbomind::gemm