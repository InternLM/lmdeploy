#include "src/turbomind/kernels/gemm/weight_plan.h"

#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

int WeightPlan::gate_up(ActivationType act_type, int projection_n)
{
    epilogue_      = Epilogue::kNone;
    output_format_ = family_->output_format(Epilogue::kNone);
    Epilogue epilogue = Epilogue::kNone;
    const int block = family_->gate_up(act_type, epilogue);
    if (!block || projection_n % block) {
        return 0;
    }
    epilogue_      = epilogue;
    output_format_ = family_->output_format(epilogue);
    return block;
}

void WeightPlan::pack(LinearWeight& linear, cudaStream_t stream) const
{
    family_->Pack(linear, bridge_, stream);
    linear.family        = family_;
    linear.input_format  = family_->input_format();
    linear.epilogue      = epilogue_;
    linear.output_format = output_format_;
}

}  // namespace turbomind::gemm
