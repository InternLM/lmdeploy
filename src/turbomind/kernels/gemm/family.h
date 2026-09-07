#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <optional>

#include "src/turbomind/core/data_format.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/gemm/output_spec.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind {
class LinearWeight;
}

namespace turbomind::gemm {

inline core::Layout apply_output_epilogue(core::Layout layout, Epilogue epilogue)
{
    if (epilogue == Epilogue::kGatedSilu) {
        auto shape = layout.shape();
        TM_CHECK_EQ(shape.back() % 2, 0);
        shape.back() /= 2;
        return core::Layout{std::move(shape)};
    }
    TM_CHECK(epilogue == Epilogue::kNone);
    return layout;
}

inline OutputSpec plain_output_spec(core::Layout layout, const DataFormat& format, Epilogue epilogue)
{
    TM_CHECK(!format.scales.present());

    OutputSpec spec;
    spec.layout = apply_output_epilogue(std::move(layout), epilogue);
    spec.dtype  = format.dtype;
    return spec;
}

struct WeightBridge {
    int2     replicate_scales{1, 1};
    DataType convert_scales{kNull};
    DataType convert_zeros{kNull};

    explicit operator bool() const noexcept
    {
        return replicate_scales.x != 1 || replicate_scales.y != 1 || convert_scales != kNull || convert_zeros != kNull;
    }
};

class Family {
public:
    const std::uint32_t id;
    const int           priority;

    const DataFormat& input_format() const noexcept
    {
        return input_format_;
    }

    int align_k() const noexcept
    {
        return align_k_;
    }

    int align_n() const noexcept
    {
        return align_n_;
    }

    int min_k() const noexcept
    {
        return min_k_;
    }

    int min_n() const noexcept
    {
        return min_n_;
    }

    bool grouped() const noexcept
    {
        return grouped_;
    }

    bool is_graph_compatible() const noexcept
    {
        return is_graph_compatible_;
    }

    DataType data_type() const;

    Epilogue   supported_epilogues() const;
    DataFormat output_format(Epilogue epilogue) const;
    OutputSpec output_spec(core::Layout output_layout, Epilogue epilogue) const
    {
        return output_spec_(std::move(output_layout), output_format(epilogue), epilogue);
    }

    std::optional<WeightBridge>
    supports(const DataFormat& weight_format, DataType data_type, DataType output_dtype, bool grouped) const;

    void Pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream) const;

    void ConvertInput(Tensor&             A,
                      Tensor&             U,
                      const LinearWeight& weight,
                      const Tensor&       input,
                      const Tensor&       input_scales,
                      cudaStream_t        stream) const;

    int gate_up(ActivationType, Epilogue& epilogue) const;

    Family(std::uint32_t id,
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
           int        gate_up_block                                             = 0,
           DataFormat fused_output                                              = {},
           bool       is_graph_compatible                                       = true,
           OutputSpec (*output_spec)(core::Layout, const DataFormat&, Epilogue) = plain_output_spec);

private:
    DataFormat input_format_;
    DataFormat output_format_;
    int        align_k_{};
    int        align_n_{};
    int        min_k_{};
    int        min_n_{};
    bool       requires_packing_{};
    bool       grouped_{};
    std::optional<WeightBridge> (*supports_)(const DataFormat&, bool){};
    void (*pack_)(LinearWeight&, const WeightBridge&, cudaStream_t){};
    int        gate_up_block_{};
    DataFormat fused_output_{};
    bool       is_graph_compatible_{};
    OutputSpec (*output_spec_)(core::Layout, const DataFormat&, Epilogue){};
};

}  // namespace turbomind::gemm
