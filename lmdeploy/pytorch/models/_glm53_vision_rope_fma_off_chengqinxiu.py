"""Vision-only opt-in: unchanged native RoPE with FP fusion disabled at launch.

This clones a Python function's global binding, never the global JIT object or
its arithmetic. FP32 computation remains in the native kernel's tile registers.
"""
from __future__ import annotations

import functools
import importlib
import os
import types

import torch
from torch import nn

ENV_NAME='LMDEPLOY_GLM53_VISION_ROPE_FMA_OFF'
ENABLED=os.getenv(ENV_NAME,'0').strip().lower() in {'1','true','yes','on'}
KERNEL_NAME='apply_rotary_pos_emb_qk_kernel'


class NoFusionKernelLaunch:
    def __init__(self,original):
        self.original=original
        self.calls=0
        self.last_launch={}

    def __getitem__(self,grid):
        original_launch=self.original[grid]
        def launch(*args,**kwargs):
            assert 'enable_fp_fusion' not in kwargs, 'Native wrapper unexpectedly changed fusion ownership'
            assert kwargs['ENABLE_FP32_COMPUTE'] is True
            assert kwargs['COMPLEX'] is False
            assert kwargs['num_warps']==2 and kwargs['num_stages']==1
            unchanged=dict(kwargs)
            kwargs['enable_fp_fusion']=False
            self.last_launch=dict(grid=list(grid),native_kwargs=unchanged,enable_fp_fusion=False)
            result=original_launch(*args,**kwargs)
            self.calls+=1
            return result
        return launch


def bind_no_fusion_function(native=None):
    if native is None:
        native=importlib.import_module('lmdeploy.pytorch.kernels.cuda.apply_rotary_pos_emb').apply_rotary_pos_emb
    assert isinstance(native,types.FunctionType) and KERNEL_NAME in native.__globals__
    proxy=NoFusionKernelLaunch(native.__globals__[KERNEL_NAME])
    local_globals=dict(native.__globals__)
    local_globals[KERNEL_NAME]=proxy
    bound=types.FunctionType(native.__code__,local_globals,native.__name__,native.__defaults__,native.__closure__)
    bound.__kwdefaults__=native.__kwdefaults__
    functools.update_wrapper(bound,native)
    assert bound.__code__ is native.__code__
    assert all(bound.__globals__[key] is value for key,value in native.__globals__.items() if key!=KERNEL_NAME)
    assert native.__globals__[KERNEL_NAME] is proxy.original
    return bound,proxy


class VisionRoPEFmaOff(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.original=original
        self.native_function,self.launch_proxy=bind_no_fusion_function()
        self.calls=0
        self.native_fallbacks=0

    def eligible(self,q,k,cos,sin,inplace,complex_mode):
        return (not self.training and not inplace and not complex_mode
            and q.device.type=='cuda' and q.device==k.device==cos.device==sin.device
            and q.dtype==k.dtype==torch.bfloat16
            and q.ndim==k.ndim==3 and q.shape==k.shape and q.shape[0]>0 and q.shape[1:]==(16,64)
            and cos.ndim==sin.ndim==2 and cos.shape==sin.shape==(q.shape[0],64)
            and cos.dtype in (torch.float32,torch.bfloat16) and sin.dtype==cos.dtype
            and cos.is_contiguous() and sin.is_contiguous()
            and not any(tensor.requires_grad for tensor in (q,k,cos,sin)))

    def forward(self,query,key,cos,sin,inplace=True,complex_mode=False):
        if not self.eligible(query,key,cos,sin,inplace,complex_mode):
            self.native_fallbacks+=1
            return self.original(query,key,cos,sin,inplace=inplace,complex_mode=complex_mode)
        if torch.cuda.is_current_stream_capturing():
            self.native_fallbacks+=1
            return self.original(query,key,cos,sin,inplace=inplace,complex_mode=complex_mode)
        result=self.native_function(query,key,cos,sin,complex_mode=False,enable_fp32_compute=True)
        assert all(out.shape==source.shape and out.dtype==source.dtype and out.device==source.device
                   for out,source in zip(result,(query,key)))
        self.calls+=1
        return result


def build_vision_rope_fma_off(original,enabled=None):
    enabled=ENABLED if enabled is None else enabled
    if not enabled:
        return original
    if not getattr(getattr(original,'impl',None),'enable_fp32_compute',False):
        return original
    return VisionRoPEFmaOff(original)
