# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import distributed as dist
from torch import nn

from lmdeploy.pytorch.kernels.fused_moe import fused_moe


# from https://huggingface.co/microsoft/Phi-3.5-MoE-instruct/blob/482a9ba0eb0e1fa1671e3560e009d7cec2e5147c/modeling_phimoe.py#L883 # noqa: E501
def sparsemixer(scores, top_k, jitter_eps):
    assert top_k == 2
    final_multipliers = scores.new_empty((scores.shape[0], top_k))
    final_experts = torch.empty_like(final_multipliers)
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    # apply mask
    masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
    selected_experts = max_ind

    final_experts[:, 0:1] = max_ind
    # compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    final_multipliers[:, 0:1] = masked_gates.gather(dim=-1,
                                                    index=selected_experts)
    # masked out first expert
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float('-inf'),
    )
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1,
                                                           keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    final_experts[:, 1:2] = max_ind
    # apply mask
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold,
                                                  float('-inf'))
    selected_experts_top2 = max_ind
    # compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    final_multipliers[:, 1:2] = masked_gates_top2.gather(
        dim=-1, index=selected_experts_top2)
    return final_multipliers, final_experts


class PatchedPhiMoESparseMoeBlock(nn.Module):

    def _update_model_fn(self):
        """update model."""
        num_experts = self.num_experts

        def __get_meta():
            exp = self.experts[0]
            ffn_dim = exp.w1.weight.size(0)
            hidden_dim = exp.w2.weight.size(0)
            dtype = exp.w1.weight.dtype
            device = exp.w1.weight.device
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
            __copy_assign_param(exp.w1.weight,
                                gate_up_weights[exp_id, :ffn_dim])
            __copy_assign_param(exp.w3.weight, gate_up_weights[exp_id,
                                                               ffn_dim:])
            __copy_assign_param(exp.w2.weight, down_weights[exp_id])

        torch.cuda.empty_cache()

        self.register_buffer('gate_up_weights', gate_up_weights)
        self.register_buffer('down_weights', down_weights)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """rewrite moe forward."""

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = sparsemixer(
            router_logits,
            top_k=2,
            jitter_eps=self.router_jitter_noise,
        )
        out_states = fused_moe(hidden_states,
                               self.gate_up_weights,
                               self.down_weights,
                               routing_weights,
                               selected_experts,
                               topk=2,
                               renormalize=False)

        out_states = out_states.reshape(batch_size, sequence_length, -1)
        return out_states, router_logits
