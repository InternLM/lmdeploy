#pragma once

#include <array>

#include "src/turbomind/kernels/gemm/family.h"

namespace turbomind {
class LinearWeight;
}

namespace turbomind::gemm {

struct WeightQuery {
    DataFormat weight_format;
    DataType   data_type{};
    DataType   input_dtype{kNull};   // Preference only; it reorders otherwise eligible families.
    DataType   output_dtype{kNull};  // Hard requirement; kNull requires data_type output.
    bool       grouped{};
};

class WeightPlan {
public:
    const Family& family() const noexcept
    {
        return *family_;
    }

    std::array<int, 4> shape_constraints() const noexcept
    {
        return {family_->min_k(), family_->min_n(), family_->align_k(), family_->align_n()};
    }

    int  gate_up(ActivationType act_type, int projection_n);
    void pack(LinearWeight& linear, cudaStream_t stream) const;

private:
    friend class Gemm;

    const Family* family_{};
    WeightBridge  bridge_{};
    Epilogue      epilogue_{Epilogue::kNone};
    DataFormat    output_format_{};
};

}  // namespace turbomind::gemm
