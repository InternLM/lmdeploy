# Copyright (c) OpenMMLab. All rights reserved.
import gc
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
                'q_a_proj', 'kv_a_proj_with_mqa', 'q_a_layernorm',
                'kv_a_layernorm'
        ]:
            with loader.prefix_context(mod_name):
                loader.load_model_weights(getattr(self, mod_name),
                                          rank=rank,
                                          world_size=world_size,
                                          device=device)

        for mod_name in ['q_b_proj', 'kv_b_proj']:
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

    def _update_model_fn(self):
        """update model."""
        qk_nope_head_dim = self.qk_nope_head_dim
        v_head_dim = self.v_head_dim

        def __update_pe(mod, head_dim, pe_dim_offset):
            weight = mod.weight.data
            # (num_heads, q_head_dim, input_dim)
            weight = weight.unflatten(0, (-1, head_dim))
            # (num_heads, nope_head_dim, input_dim)
            w_pe = weight[:, pe_dim_offset:]
            # (num_heads, nope_head_dim//2, 2, input_dim)
            new_w_pe = w_pe.unflatten(1, (-1, 2))
            # (num_heads, nope_head_dim, input_dim)
            new_w_pe = new_w_pe.transpose(1, 2).flatten(1, 2)
            weight[:, pe_dim_offset:] = new_w_pe

        # prevent shuffle before apply rotary embedding
        __update_pe(self.q_b_proj, self.q_head_dim, qk_nope_head_dim)
        kv_dim = self.kv_lora_rank + self.qk_rope_head_dim
        __update_pe(self.kv_a_proj_with_mqa, kv_dim, self.kv_lora_rank)

        kv_b_proj = self.kv_b_proj
        w_kc, w_vc = kv_b_proj.weight.unflatten(
            0, (-1, qk_nope_head_dim + v_head_dim)).split(
                [qk_nope_head_dim, v_head_dim], dim=1)

        self.register_parameter('w_kc',
                                torch.nn.Parameter(w_kc, requires_grad=False))
        w_vc = w_vc.transpose(1, 2).contiguous()
        self.register_parameter('w_vc',
                                torch.nn.Parameter(w_vc, requires_grad=False))

        delattr(self, 'kv_b_proj')
        gc.collect()

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        world_size: int = 1,
    ):
        """forward impl."""
        context = self.context.context
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.q_seq_length
        q_start_loc = context.q_start_loc
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length
        max_kv_seq_length = context.max_kv_seq_length
        num_heads = self.num_heads // world_size
        q_len = hidden_states.size(1)

        def __qkv_proj(hidden_states):
            """qkv proj."""
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
            q = q.view(q_len, num_heads, self.q_head_dim)
            # q_pe: (q_len, num_heads, qk_rope_head_dim)
            q_nope, q_pe = torch.split(
                q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # q_nope: (q_len, num_heads, kv_lora_rank)
            q_nope_out = q_nope.new_empty(q_len, num_heads, self.kv_lora_rank)
            torch.bmm(q_nope.transpose(0, 1),
                      self.w_kc,
                      out=q_nope_out.transpose(0, 1))
            q_nope = q_nope_out

            compressed_kv = self.kv_a_proj_with_mqa(hidden_states[0, :, None])
            # compressed_kv: (q_len, 1, kv_lora_rank)
            # k_pe: (q_len, 1, qk_rope_head_dim)
            compressed_kv, k_pe = torch.split(
                compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim],
                dim=-1)
            # kv_heads == 1
            compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())

            return q_nope, q_pe, k_pe, compressed_kv

        def __rotary_emb_fn(q_pe, k_pe, out_q_pe, out_k_pe):
            """rope."""
            if not hasattr(context, '_cos'):
                cos, sin = self.rotary_emb(q_pe, seq_len=max_kv_seq_length)
                context._cos = cos
                context._sin = sin
            else:
                cos = context._cos
                sin = context._sin

            apply_rotary_pos_emb(q_pe,
                                 k_pe,
                                 cos,
                                 sin,
                                 position_ids,
                                 context.position_ids_1d,
                                 q_embed=out_q_pe,
                                 k_embed=out_k_pe)
            return out_q_pe, out_k_pe

        q_nope, q_pe, k_pe, compressed_kv = __qkv_proj(hidden_states)

        nope_size = q_nope.size(-1)
        pe_size = q_pe.size(-1)
        query_states = k_pe.new_empty(q_len, num_heads, nope_size + pe_size)
        key_states = k_pe.new_empty(q_len, 1, nope_size + pe_size)
        query_states[..., :nope_size] = q_nope
        key_states[..., :nope_size] = compressed_kv
        value_states = compressed_kv

        __rotary_emb_fn(q_pe, k_pe, query_states[..., nope_size:],
                        key_states[..., nope_size:])

        fill_kv_cache(
            key_states,
            value_states[..., :0],
            past_key_value[0],
            past_key_value[0][..., :0],
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
        )

        attn_output = query_states[..., :nope_size]
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[0][..., :nope_size],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
            sm_scale=self.softmax_scale,
            shared_kv=True,
        )

        # (num_heads, q_len, v_head_dim)
        attn_bmm_out = attn_output.new_empty(q_len, num_heads, self.v_head_dim)
        torch.bmm(attn_output.transpose(0, 1),
                  self.w_vc,
                  out=attn_bmm_out.transpose(0, 1))
        # (1, q_len, o_proj_input)
        attn_output = attn_bmm_out.flatten(-2, -1)[None]

        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

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
        """rewrite of forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            world_size=world_size,
        )


def _div_up(a, b):
    """div up."""
    return (a + b - 1) // b


class PatchedDeepseekV2MoE(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""

        def __load_mlp(exp_id, exp):
            """load mlp."""
            with loader.prefix_context(f'experts.{exp_id}'):
                loader.load_model_weights(
                    exp,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    load_only=True,
                )

        def __drop_mlp(exp_id, exp):
            """drop mlp."""
            for name, _ in exp.named_parameters(recurse=True):
                loader.pop(f'experts.{exp_id}.{name}')

        num_experts = len(self.experts)
        exp_per_rank = _div_up(num_experts, world_size)
        first_exp = rank * exp_per_rank
        last_exp = min(num_experts, first_exp + exp_per_rank)
        for exp_id, exp in enumerate(self.experts):
            if first_exp <= exp_id < last_exp:
                __load_mlp(exp_id, exp)
            else:
                __drop_mlp(exp_id, exp)
        self.experts = self.experts[first_exp:last_exp]
        with loader.prefix_context('gate'):
            loader.load_model_weights(self.gate,
                                      rank=rank,
                                      world_size=world_size,
                                      device=device)

        if self.config.n_shared_experts is not None:
            with loader.prefix_context('shared_experts'):
                loader.load_model_weights(self.shared_experts,
                                          rank=rank,
                                          world_size=world_size,
                                          device=device)

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

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, _ = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx,
                           topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts.forward(identity)
        return y

    def moe_infer(self, x, topk_ids, topk_weight):
        """moe infer."""
        world_size = 1
        rank = 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        exp_per_rank = self.gate_up_weights.size(0)
        expert_offset = rank * exp_per_rank
        ret = fused_moe(x,
                        self.gate_up_weights,
                        self.down_weights,
                        topk_weight,
                        topk_ids,
                        topk=self.num_experts_per_tok,
                        expert_offset=expert_offset,
                        num_experts=world_size * exp_per_rank,
                        renormalize=False)
        return ret
