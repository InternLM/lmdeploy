# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.kernels.fused_moe import fused_moe

from ..kernels import apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd
from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        rowwise_parallelize_linear)


class PatchedDeepseekV2Attention(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in [
                'q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj'
        ]:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        for mod_name in ['o_proj']:
            rowwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        pass


class PatchedDeepseekV2MoE(nn.Module):

    def _update_model_fn(self):
        """update model."""
        num_experts = len(self.experts)

        def __get_meta():
            exp = self.experts[0]
            ffn_dim = exp.gate_proj.weight.size(0)
            hidden_dim = exp.down_proj.weight.size(0)
            dtype = exp.gate_proj.weight.dtype
            device = exp.gate_proj.weight.device
            return ffn_dim, hidden_dim, dtype, device

        def __copy_assign_param(param, weight):
            """copy assign."""
            weight.copy_(param.data)
            param.data = weight

        ffn_dim, hidden_dim, dtype, device = __get_meta()

        gate_up_weights = torch.empty(num_experts,
                                      ffn_dim * 2,
                                      hidden_dim,
                                      device=device,
                                      dtype=dtype)
        down_weights = torch.empty(num_experts,
                                   hidden_dim,
                                   ffn_dim,
                                   device=device,
                                   dtype=dtype)

        for exp_id, exp in enumerate(self.experts):
            __copy_assign_param(exp.gate_proj.weight,
                                gate_up_weights[exp_id, :ffn_dim])
            __copy_assign_param(exp.up_proj.weight, gate_up_weights[exp_id,
                                                                    ffn_dim:])
            __copy_assign_param(exp.down_proj.weight, down_weights[exp_id])

        torch.cuda.empty_cache()

        self.register_buffer('gate_up_weights', gate_up_weights)
        self.register_buffer('down_weights', down_weights)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs

    def moe_infer(self, x, topk_ids, topk_weight):
        """moe infer."""
        ret = fused_moe(x,
                        self.gate_up_weights,
                        self.down_weights,
                        topk_weight,
                        topk_ids,
                        topk=self.num_experts_per_tok,
                        renormalize=False)
        return ret
