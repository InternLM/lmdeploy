# file: qwen3_next_ascend.py
import torch
from torch import nn
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


# from lmdeploy.pytorch.kernels.triton_ops.fla.triton_utils import init_device_properties_triton
from lmdeploy.pytorch.kernels.triton_ops import causal_conv1d_fn, causal_conv1d_update_npu
from lmdeploy.pytorch.kernels.triton_ops import chunk_gated_delta_rule
from lmdeploy.pytorch.kernels.triton_ops import fused_sigmoid_gating_delta_rule_update
from lmdeploy.pytorch.kernels.triton_ops import RMSNormGated
from lmdeploy.pytorch.kernels.triton_ops import fused_recurrent_gated_delta_rule


# from .triton_ops.layernorm_gurad import RMSNormGated


class AscendGatedDeltaMeta:

    def __init__(self, state_ids: torch.Tensor, attn_metadata: Any):
        self.is_decoding = attn_metadata.is_decoding
        self.cu_seqlens = attn_metadata.q_start_loc

        # state_ids, fill invalid state with state_ids[0]
        self.valid_state = state_ids >= 0
        self.state_ids = torch.where(self.valid_state, state_ids, state_ids[0])
        self.state_ids = self.state_ids.clamp(0)
        self.has_initial_state = attn_metadata.has_initial_state
        

class AscendConv1dImpl:
    def __init__(self, activation='silu'):
        self.activation = activation

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, conv_state: torch.Tensor,
                 gated_delta_meta):
        """
        x: (B, L, D)
        weight: (D, 1, K) from lmdeploy, but vllm often expects (D, K) squeezed.
        conv_state: (B, K, D)
            prefill的 conv_state: (B, D, K) 底层是否连续都可以
            decoding中 conv_state: (B, D, K) 底层必须非连续 否则assert无法通过 并且底层算子报错
            因此在cache engine中修改conv_state的初始shape为(B, K, D) decoding阶段只需一个转置即可变为非连续
        """
        is_decoding = gated_delta_meta.is_decoding
        
        # vllm causal_conv1d typically expects weight as (D, K) for the kernel call
        weight_reshaped = weight.squeeze(1)
        x = x.squeeze(0)
        
        if is_decoding:
            
            out = causal_conv1d_update_npu(
                x,
                conv_state.transpose(1, 2),
                weight_reshaped,
                bias,
                self.activation,
                conv_state_indices=gated_delta_meta.state_ids, # Explicit state passed
                validate_data=True,
            )
            return out.unsqueeze(0), conv_state
        else:
            # Prefill Step
            
            # x_trans = x.transpose(1, 2)
            # x_trans = x_trans.squeeze(0)
            
            # query_start_loc needed for variable length handling inside the op
            query_start_loc = gated_delta_meta.cu_seqlens
            
            # By modifying the kernel code, variable has_initial_state can be eliminated.
            out = causal_conv1d_fn(
                x.t(),
                weight_reshaped,
                bias,
                activation=self.activation,
                conv_states=conv_state.transpose(1, 2),
                has_initial_state=gated_delta_meta.has_initial_state, 
                cache_indices=gated_delta_meta.state_ids,
                query_start_loc=query_start_loc
            )
            # out = out.transpose(1, 2)
            return out.t().unsqueeze(0), conv_state


class AscendGatedDeltaImpl:
    def __init__(self, use_qk_l2norm_in_kernel: bool = True):
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

    def __call__(self, query, key, value, g, beta, recurrent_state, gated_delta_meta):
        is_decoding = gated_delta_meta.is_decoding

        if is_decoding:         
            out, last_state = fused_recurrent_gated_delta_rule(
                q=query,    # torch.Size([1, 256, 8, 128])
                k=key,
                v=value,
                g=g,        # torch.Size([1, 256, 8])
                beta=beta,
                initial_state=recurrent_state,  # torch.Size([256, 8, 128, 128])
                inplace_final_state=True,
                ssm_state_indices=gated_delta_meta.state_ids,
                cu_seqlens=gated_delta_meta.cu_seqlens,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel
            )
            return out, last_state
        else:
            initial_state = recurrent_state[gated_delta_meta.state_ids]
            initial_state[~gated_delta_meta.has_initial_state, ...] = 0
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                q=query,        # torch.Size([1, seqlen, 8, 128])
                k=key,            # torch.Size([1, seqlen, 8, 128])
                v=value,        # torch.Size([1, seqlen, 8, 128])
                g=g,                # torch.Size([1, seqlen, 8])
                beta=beta,          # # torch.Size([1, seqlen, 8])
                initial_state=initial_state,  # torch.Size([seqlen, 8, 128, 128])
                output_final_state=True,
                cu_seqlens=gated_delta_meta.cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
            )
            recurrent_state[gated_delta_meta.state_ids] = last_recurrent_state.to(recurrent_state.dtype)
                
            return core_attn_out, last_recurrent_state


def build_rmsnorm_gated_ascend(hidden_size: int, eps=1e-6, device=None):
    """Wrapper to use vllm RMSNormGated on Ascend"""
    return RMSNormGated(hidden_size, eps=eps, norm_before_gate=True, device=device)