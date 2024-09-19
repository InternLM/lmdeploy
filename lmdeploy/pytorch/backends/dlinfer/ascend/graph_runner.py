# Copyright (c) OpenMMLab. All rights reserved.
from importlib import import_module
from typing import Any, Dict, List, Tuple

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ...graph_runner import GraphRunner

logger = get_logger('lmdeploy')

class AscendGraphRunner(GraphRunner):
    """ascend graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig,
                 cache_config: CacheConfig, backend_config: BackendConfig,
                 device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config,
                         device)

        self.enable_graph = self.check_enable_graph()
        if self.enable_graph:
            self.patch_kernels_custom_op()
            self.patch_kvcache_static_shape()
            self.model = torch.compile(self.model, fullgraph=True, dynamic=True, backend='atbgraph')

    def check_enable_graph(self):
        """check enable graph."""
        return not self.backend_config.eager_mode

    def patch_kernels_custom_op(self):
        from dlinfer.graph.custom_op import register_custom_op
        ascend_kernels_module = import_module("lmdeploy.pytorch.kernels.ascend")
        ascend_backends_module = import_module("lmdeploy.pytorch.backends.ascend")

        # prefill_attention
        module_str = "pagedattention"
        paged_attn_module = getattr(ascend_kernels_module, module_str)
        func_str = "prefill_attention"
        prefill_attn_origin = getattr(paged_attn_module, func_str)
        prefill_attn_registered = register_custom_op(
            f"lmdeploy::{func_str}", ["attn_output"]
        )(prefill_attn_origin)
        setattr(paged_attn_module, func_str, prefill_attn_registered)

        # apply_rotary_pos_emb
        def apply_rotary_emb_abstract_impl(q, k, cos, sin, q_out, k_out):
            result = [q, k]
            if q_out is not None:
                result[0] = q_out
            if k_out is not None:
                result[1] = k_out
            return tuple(result)
        module_str = "apply_rotary_emb"
        apply_rotary_emb_module = getattr(ascend_backends_module, module_str)
        func_str = "apply_rotary_pos_emb"
        apply_rotary_pos_emb_origin = getattr(apply_rotary_emb_module, func_str)
        apply_rotary_pos_emb_registered = register_custom_op(
            f"lmdeploy::{func_str}",
            impl_abstract_func=apply_rotary_emb_abstract_impl
        )(apply_rotary_pos_emb_origin)
        setattr(apply_rotary_emb_module, func_str, apply_rotary_pos_emb_registered)  

    def patch_kvcache_static_shape(self):
        import torch._dynamo as dynamo
        from torch.utils._pytree import tree_map
        cache_engine_module = import_module("lmdeploy.pytorch.engine.cache_engine")
        class_str = "CacheEngine"
        cache_engine_class = getattr(cache_engine_module, class_str)
        func_str = "allocate_gpu_cache"
        allocate_gpu_cache_origin = getattr(cache_engine_class, func_str)
        def allocate_gpu_cache_mark_static(self):
            gpu_cache = allocate_gpu_cache_origin(self)
            tree_map(lambda x: dynamo.mark_static(x), gpu_cache)
            return gpu_cache
        setattr(cache_engine_class, func_str, allocate_gpu_cache_mark_static)
