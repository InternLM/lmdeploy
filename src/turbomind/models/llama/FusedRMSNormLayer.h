#pragma once

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include <memory>

namespace turbomind {

struct FusedRMSNormLayerParam {
    int                   ep_size;
    size_t                hidden_units;
    float                 rmsnorm_eps;
    int                   attn_tp_group;
    comm::DeviceCommImpl* d_comm;
};

struct FusedRMSNormLayerForwardParam {
    // for tp mode
    Tensor                  global_hidden_states;
    Tensor                  local_hidden_states;
    Tensor                  local_residual;
    const std::vector<int>& local_token_nums;
    int                     global_token_num;

    // for ep mode
    Tensor               partial_hidden_states;
    Tensor               partial_local_residual;
    std::vector<int>&    token_nums;
    std::vector<size_t>& counts;
};

enum class FusedRMSNormLayerStage : int
{
    kAttn,
    kFfn,
};

class FusedRMSNormLayer {
public:
    virtual void forward(FusedRMSNormLayerForwardParam& param,
                         const Tensor&                  weight,
                         const Tensor&                  bias,
                         FusedRMSNormLayerStage         stage) = 0;

    virtual ~FusedRMSNormLayer() = default;
};

class FusedRMSNormLayerTp: public FusedRMSNormLayer {
public:
    FusedRMSNormLayerTp(const FusedRMSNormLayerParam& param): param_(param) {}

    ~FusedRMSNormLayerTp() = default;

    void forward(FusedRMSNormLayerForwardParam& param,
                 const Tensor&                  weight,
                 const Tensor&                  bias,
                 FusedRMSNormLayerStage         stage) override
    {
        // AllReduceResidualRMSNorm
        const int group0 = stage == FusedRMSNormLayerStage::kAttn ? param_.attn_tp_group : 0;
        const int group1 = stage == FusedRMSNormLayerStage::kAttn ? 0 : param_.attn_tp_group;

        const auto dtype = param.global_hidden_states.dtype();

        const auto stream = core::Context::stream().handle();

        if (0) {}
        else if (group0 || group1) {
            param_.d_comm->AllreduceResidualBiasRMSnormEx(param.global_hidden_states.raw_data(),
                                                          param.local_residual.data_or((void*)nullptr),
                                                          bias.data_or((void*)nullptr),
                                                          weight.raw_data(),
                                                          param_.rmsnorm_eps,
                                                          param_.hidden_units,
                                                          dtype,
                                                          group0,
                                                          group1,
                                                          param.local_token_nums.data(),
                                                          stream);
            sync_check_cuda_error();
        }
        else if (param_.d_comm) {
            param_.d_comm->AllreduceResidualBiasRMSnorm(param.global_hidden_states.raw_data(),
                                                        param.local_residual.data_or((void*)nullptr),
                                                        bias.data_or((void*)nullptr),
                                                        weight.raw_data(),
                                                        param_.rmsnorm_eps,
                                                        param_.hidden_units,
                                                        param.global_token_num,
                                                        dtype,
                                                        0,
                                                        stream);
            sync_check_cuda_error();
        }
        else {
            invokeResidualBiasRMSNorm(param.global_hidden_states.raw_data(),
                                      param.local_residual.data_or((void*)nullptr),
                                      weight.raw_data(),
                                      bias.data_or((void*)nullptr),
                                      dtype,
                                      param_.hidden_units,
                                      param.global_token_num,
                                      param_.rmsnorm_eps,
                                      stream);
            sync_check_cuda_error();
        }
    }

private:
    FusedRMSNormLayerParam param_;
};

class FusedRMSNormLayerEp: public FusedRMSNormLayer {
public:
    FusedRMSNormLayerEp(const FusedRMSNormLayerParam& param): param_(param) {}

    void forward(FusedRMSNormLayerForwardParam& param,
                 const Tensor&                  weight,
                 const Tensor&                  bias,
                 FusedRMSNormLayerStage         stage) override
    {
        const auto stream = core::Context::stream().handle();

        if (stage == FusedRMSNormLayerStage::kAttn) {
            param_.d_comm->ReduceScatterV(param.local_hidden_states.data_or((void*)nullptr),  //
                                          param.partial_hidden_states.data_or((void*)nullptr),
                                          param.counts.data(),
                                          param.local_hidden_states.dtype(),
                                          param_.attn_tp_group,
                                          stream);
            sync_check_cuda_error();
        }

        invokeResidualBiasRMSNorm(param.partial_hidden_states.data_or((void*)nullptr),
                                  param.partial_local_residual.data_or((void*)nullptr),
                                  weight.raw_data(),
                                  bias.data_or((void*)nullptr),
                                  param.partial_hidden_states.dtype(),
                                  param_.hidden_units,
                                  param.token_nums[param_.d_comm->rank(param_.attn_tp_group)],
                                  param_.rmsnorm_eps,
                                  stream);
        sync_check_cuda_error();

        if (stage == FusedRMSNormLayerStage::kFfn) {
            param_.d_comm->AllGatherV(param.partial_hidden_states.data_or((void*)nullptr),
                                      param.local_hidden_states.data_or((void*)nullptr),
                                      param.counts.data(),
                                      param.local_hidden_states.dtype(),
                                      param_.attn_tp_group,
                                      stream);
            sync_check_cuda_error();
        }
    }

    ~FusedRMSNormLayerEp() = default;

private:
    FusedRMSNormLayerParam param_;
};

inline std::unique_ptr<FusedRMSNormLayer> CreateFusedRMSNormLayer(const FusedRMSNormLayerParam& param)
{
    if (param.ep_size > 1) {
        return std::make_unique<FusedRMSNormLayerEp>(param);
    }
    return std::make_unique<FusedRMSNormLayerTp>(param);
}

}  // namespace turbomind
