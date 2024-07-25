# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import distributed as dist
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager

from ..awq_modules import LinearW4A16Builder, LinearW4A16Impl


def wq_gemm_forward(
    x,
    qweight,
    qzeros,
    scales,
    w_bit=4,
    group_size=128,
    bias=None,
    out_features=0,
):
    """wq gemm forward."""
    from awq.modules.linear.gemm import awq_ext
    out_shape = x.shape[:-1] + (out_features, )
    input_dtype = x.dtype
    if input_dtype != torch.float16:
        x = x.half()

    FP16_MATMUL_HEURISTIC_CONDITION = x.size(0) * x.size(1) >= 1024

    if FP16_MATMUL_HEURISTIC_CONDITION:
        # TODO: remove event wait if awq kernel set stream
        default_stream = torch.cuda.default_stream()
        event_def = torch.cuda.Event()
        event_def.record()
        event_def.wait(default_stream)
        out = awq_ext.dequantize_weights_cuda(qweight, scales, qzeros, 0, 0, 0,
                                              False)
        event_def = torch.cuda.Event()
        event_def.record(default_stream)
        event_def.wait()
        out = torch.matmul(x, out)
    else:
        x = x.flatten(0, -2)
        if not x.is_contiguous():
            x = x.contiguous()
        out = awq_ext.gemm_forward_cuda(x, qweight, scales, qzeros, 8)

    out = out + bias if bias is not None else out
    out = out.reshape(out_shape)

    # always want 3D tensor if tensor is 2D
    if len(out.shape) == 2:
        out = out.unsqueeze(0)

    if input_dtype != torch.float16:
        out = out.to(dtype=input_dtype)
    return out


class AwqLinearW4A16Impl(LinearW4A16Impl):

    def __init__(self, mod: nn.Module):
        super().__init__()
        from awq.modules.linear.gemm import AWQ_INSTALLED
        assert AWQ_INSTALLED
        self.qweight = mod.qweight
        self.qzeros = mod.qzeros
        self.scales = mod.scales
        self.w_bit = mod.w_bit
        self.group_size = mod.group_size
        self.bias = mod.bias
        self.in_features = mod.in_features
        self.out_features = mod.out_features

    def forward(self, x, all_reduce: bool = False):
        out_features = self.scales.size(1)
        out = wq_gemm_forward(x, self.qweight, self.qzeros, self.scales,
                              self.w_bit, self.group_size, self.bias,
                              out_features)
        if all_reduce:
            dist.all_reduce(out)
        return out


class AwqLinearW4A16Builder(LinearW4A16Builder):

    @staticmethod
    def build(mod: nn.Module, ctx_mgr: StepContextManager = None):
        return AwqLinearW4A16Impl(mod)
