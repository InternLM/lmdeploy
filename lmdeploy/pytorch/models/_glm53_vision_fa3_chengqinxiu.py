"""Opt-in, inference-only TP1 vision FA3; language attention is untouched."""
from __future__ import annotations
from functools import lru_cache
import importlib
import inspect
import os
import torch
from torch import nn

ENV_NAME='LMDEPLOY_GLM53_VISION_FA3'
ENABLED=os.getenv(ENV_NAME,'0').strip().lower() in {'1','true','yes','on'}


@lru_cache(maxsize=1)
def public_function():
    module=importlib.import_module('flash_attn_interface')
    function=module.flash_attn_varlen_func
    required={'seqused_q','seqused_k','qv','q_descale','k_descale','v_descale','attention_chunk',
        'num_splits','pack_gqa','sm_margin','return_attn_probs'}
    if not required.issubset(inspect.signature(function).parameters):
        raise RuntimeError('Unqualified FA3 public varlen API')
    return function


def _fa3_forward(query,key,value,q_start_loc,q_seqlens,max_q_seqlen,scale,public=None):
    """Owner-provided contiguous vision packing; no GPU index/scalar CPU reads.

    Actual Glm5NextVisionAttention supplies starts=cu[:-1], lengths=diff(cu).
    This helper is not an adapter for arbitrary gapped/paged attention metadata.
    """
    cu=torch.cat((q_start_loc,(q_start_loc[-1]+q_seqlens[-1]).reshape(1)))
    call=public_function() if public is None else public
    return call(query,key,value,cu,cu,max_q_seqlen,max_q_seqlen,
        seqused_q=None,seqused_k=None,softmax_scale=scale,causal=False,
        qv=None,q_descale=None,k_descale=None,v_descale=None,window_size=(-1,-1),
        attention_chunk=0,softcap=0.0,num_splits=1,pack_gqa=None,
        deterministic=os.getenv('XTUNER_DETERMINISTIC')=='true',
        sm_margin=int(os.getenv('XTUNER_SM_MARGIN','0')),return_attn_probs=False)


class VisionFA3Attention(nn.Module):
    """Retain the original module for unsupported cases and exact flagoff C0."""
    def __init__(self,original):
        super().__init__();self.original=original
        self.scale=original.impl.scale
        self.calls=0;self.native_fallbacks=0

    def eligible(self,q,k,v,q_start_loc,q_seqlens,kv_start_loc,kv_seqlens,max_q_seqlen):
        return (not self.training and q.device.type=='cuda' and q.dtype==k.dtype==v.dtype==torch.bfloat16
            and q.device==k.device==v.device==q_start_loc.device==q_seqlens.device
            and q.ndim==k.ndim==v.ndim==3 and q.shape==k.shape==v.shape
            and q.shape[0]>0 and q.shape[1:]==(16,64)
            and not any(x.requires_grad for x in (q,k,v))
            and q_start_loc.dtype==q_seqlens.dtype==torch.int32
            and q_start_loc.ndim==q_seqlens.ndim==1 and q_start_loc.numel()==q_seqlens.numel()>0
            and kv_start_loc is None and kv_seqlens is None
            and isinstance(max_q_seqlen,int) and max_q_seqlen>0)

    def forward(self,query,key,value,q_start_loc,q_seqlens,kv_start_loc=None,kv_seqlens=None,max_q_seqlen=None):
        original_kwargs=dict(q_start_loc=q_start_loc,q_seqlens=q_seqlens,kv_start_loc=kv_start_loc,
            kv_seqlens=kv_seqlens,max_q_seqlen=max_q_seqlen)
        if not self.eligible(query,key,value,q_start_loc,q_seqlens,kv_start_loc,kv_seqlens,max_q_seqlen):
            self.native_fallbacks+=1
            return self.original(query,key,value,**original_kwargs)
        # Vision eager/prefill diagnostic only. Retain original graph ownership.
        if torch.cuda.is_current_stream_capturing():
            self.native_fallbacks+=1
            return self.original(query,key,value,**original_kwargs)
        output=_fa3_forward(query,key,value,q_start_loc,q_seqlens,max_q_seqlen,self.scale)
        assert output.shape==query.shape and output.dtype==query.dtype and output.device==query.device
        self.calls+=1
        return output


def build_vision_flash_attention(original,enabled=None):
    enabled=ENABLED if enabled is None else enabled
    if not enabled:return original
    impl=original.impl
    if not (impl.num_heads==impl.num_kv_heads==16 and impl.head_dim==impl.v_head_dim==64
            and impl.causal is False and impl.scale==.125
            and impl.sliding_window is None and impl.logit_softcapping in (None,0,0.0)):
        return original
    return VisionFA3Attention(original)
