# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.hc_prepost import HCPrePostBuildSpec
from lmdeploy.pytorch.models.patch import get_build_model_context


class HcPrePost(nn.Module):
    """DeepSeek-V4 hyper-connection pre/post reduction wrapper."""

    def __init__(self, hc_mult: int, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.impl = get_backend().build_op(
            HCPrePostBuildSpec(hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.impl.pre(x, hc_fn, hc_scale, hc_base, norm_eps, x.dtype)

    def pre_reduce(self, x: torch.Tensor, pre: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return self.impl.pre_reduce(x, pre, out_dtype)

    def post_expand(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                    comb: torch.Tensor) -> torch.Tensor:
        return self.impl.post_expand(x, residual, post, comb)
