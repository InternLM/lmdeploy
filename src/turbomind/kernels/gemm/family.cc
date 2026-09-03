#include "src/turbomind/kernels/gemm/family.h"

#include <utility>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/quantization.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

Family::Family(std::uint32_t id,
               int           priority,
               DataFormat    input_format,
               DataFormat    output_format,
               int           align_k,
               int           align_n,
               int           min_k,
               int           min_n,
               bool          requires_packing,
               bool          grouped,
               std::optional<WeightBridge> (*supports)(const DataFormat&, bool),
               void (*pack)(LinearWeight&, const WeightBridge&, cudaStream_t),
               int        gate_up_block,
               DataFormat fused_output,
               bool       is_graph_compatible,
               OutputSpec (*output_spec)(core::Layout, const DataFormat&, Epilogue)):
    id{id},
    priority{priority},
    input_format_{std::move(input_format)},
    output_format_{std::move(output_format)},
    align_k_{align_k},
    align_n_{align_n},
    min_k_{min_k},
    min_n_{min_n},
    requires_packing_{requires_packing},
    grouped_{grouped},
    supports_{supports},
    pack_{pack},
    gate_up_block_{gate_up_block},
    fused_output_{std::move(fused_output)},
    is_graph_compatible_{is_graph_compatible},
    output_spec_{output_spec}
{
    TM_CHECK(supports_);
    TM_CHECK(pack_);
    TM_CHECK(output_spec_);
}

DataType Family::data_type() const
{
    if (!input_format_.is_quantized()) {
        return input_format_.dtype;
    }
    TM_CHECK(!output_format_.is_quantized());
    return output_format_.dtype;
}

Epilogue Family::supported_epilogues() const
{
    return gate_up_block_ ? Epilogue::kGatedSilu : Epilogue::kNone;
}

DataFormat Family::output_format(Epilogue epilogue) const
{
    if (epilogue == Epilogue::kNone) {
        return output_format_;
    }
    TM_CHECK(gate_up_block_ && epilogue == Epilogue::kGatedSilu);
    return fused_output_;
}

std::optional<WeightBridge>
Family::supports(const DataFormat& weight_format, DataType data_type, DataType output_dtype, bool grouped) const
{
    if (grouped && !grouped_) {
        return std::nullopt;
    }
    if (WeightPackEnv() == 0 && requires_packing_) {
        return std::nullopt;
    }
    if (WeightPackEnv() == 1 && !requires_packing_) {
        return std::nullopt;
    }
    if (!input_format_.is_quantized() && input_format_.dtype != data_type) {
        return std::nullopt;
    }
    if (output_dtype == kNull) {
        output_dtype = data_type;
    }
    if (!output_format_.is_quantized() && output_format_.dtype != output_dtype) {
        return std::nullopt;
    }
    return supports_(weight_format, grouped);
}

void Family::Pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream) const
{
    pack_(linear, bridge, stream);
}

void Family::ConvertInput(Tensor& A,
                          Tensor& U,
                          const LinearWeight&,
                          const Tensor& input,
                          const Tensor& input_scales,
                          cudaStream_t stream) const
{
    if (!input_format_.is_quantized()) {
        TM_CHECK_EQ(input.dtype(), input_format_.dtype);
        A = input;
        U = input_scales;
        return;
    }
    if (input.dtype() == input_format_.dtype) {
        TM_CHECK(input_scales);
        A = input;
        U = input_scales;
        return;
    }
    TM_CHECK_EQ(input.dtype(), data_type());
    QuantizeSymm(A, U, input, stream);
}

int Family::gate_up(ActivationType act_type, Epilogue& epilogue) const
{
    if (gate_up_block_ && act_type == ActivationType::kSilu) {
        epilogue = Epilogue::kGatedSilu;
        return gate_up_block_;
    }
    epilogue = Epilogue::kNone;
    return 0;
}

}  // namespace turbomind::gemm
