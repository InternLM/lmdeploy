# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn


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
    from awq.modules.linear.gemm import AWQ_INSTALLED, dequantize_gemm
    out_shape = x.shape[:-1] + (out_features, )
    input_dtype = x.dtype
    if input_dtype != torch.float16:
        x = x.half()

    if AWQ_INSTALLED:
        from awq.modules.linear.gemm import awq_ext
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024

        if FP16_MATMUL_HEURISTIC_CONDITION:
            # TODO: remove event wait if awq kernel set stream
            default_stream = torch.cuda.default_stream()
            event_def = torch.cuda.Event()
            event_def.record()
            event_def.wait(default_stream)
            out = awq_ext.dequantize_weights_cuda(qweight, scales, qzeros, 0,
                                                  0, 0, False)
            event_def = torch.cuda.Event()
            event_def.record(default_stream)
            event_def.wait()
            out = torch.matmul(x, out)
        else:
            x = x.flatten(0, -2)
            if not x.is_contiguous():
                x = x.contiguous()
            out = awq_ext.gemm_forward_cuda(x, qweight, scales, qzeros, 8)
    else:
        out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
        out = torch.matmul(x, out)

    out = out + bias if bias is not None else out
    out = out.reshape(out_shape)

    # always want 3D tensor if tensor is 2D
    if len(out.shape) == 2:
        out = out.unsqueeze(0)

    if input_dtype != torch.float16:
        out = out.to(dtype=input_dtype)
    return out


class PatchedWQLinear_GEMM(nn.Module):

    def forward(self, x):
        """forward."""
        out_features = self.scales.size(1)
        return wq_gemm_forward(x, self.qweight, self.qzeros, self.scales,
                               self.w_bit, self.group_size, self.bias,
                               out_features)
