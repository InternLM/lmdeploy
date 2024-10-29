# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from importlib import import_module

import torch
import torch.distributed

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

        self.supported_model = ['Llama3-8B', 'Llama2-7B', 'Qwen2-7B']
        self.enable_graph = self.check_enable_graph()
        if self.enable_graph:
            import dlinfer.graph
            dlinfer.graph.config.enable_graph_mode = True
            self.patch_kernels_custom_op()
            self.patch_kvcache_static_shape()
            self.model = torch.compile(self.model,
                                       fullgraph=True,
                                       dynamic=True,
                                       backend='atbgraph')

    def check_enable_graph(self):
        """check enable graph."""
        # eager_mode
        if self.backend_config.eager_mode:
            return False
        # tp
        if torch.distributed.is_initialized():
            warnings.warn(
                "Graph mode of device_type 'ascend' only supports tp=1 "
                'for now, fallback to eager mode', RuntimeWarning)
            return False

        warnings.warn(
            '\n\n'
            '**********************************************************\n'
            '  The following models were tested in graph mode of\n'
            "  device_type 'ascend' when tp=1:\n"
            f"  {', '.join(self.supported_model)}\n"
            '  Other LLaMa-like models may work in graph mode, please\n'
            '  check the result yourself!\n'
            '  If graph mode does not work correctly with your model,\n'
            '  please use eager mode instead.\n'
            '**********************************************************\n\n',
            RuntimeWarning)

        return True

    def patch_kernels_custom_op(self):
        from dlinfer.graph.custom_op import register_custom_op
        dlinfer_kernels_module = import_module(
            'lmdeploy.pytorch.kernels.dlinfer')
        dlinfer_backends_module = import_module(
            'lmdeploy.pytorch.backends.dlinfer')

        # prefill_attention
        module_str = 'pagedattention'
        paged_attn_module = getattr(dlinfer_kernels_module, module_str)
        func_str = 'prefill_attention'
        prefill_attn_origin = getattr(paged_attn_module, func_str)
        prefill_attn_registered = register_custom_op(
            f'lmdeploy::{func_str}', ['attn_output'])(prefill_attn_origin)
        setattr(paged_attn_module, func_str, prefill_attn_registered)

        # apply_rotary_pos_emb
        def apply_rotary_emb_abstract_impl(q, k, cos, sin, q_out, k_out):
            result = [q, k]
            if q_out is not None:
                result[0] = q_out
            if k_out is not None:
                result[1] = k_out
            return tuple(result)

        module_str = 'apply_rotary_emb'
        apply_rotary_emb_module = getattr(dlinfer_backends_module, module_str)
        func_str = 'apply_rotary_pos_emb'
        apply_rotary_pos_emb_origin = getattr(apply_rotary_emb_module,
                                              func_str)
        apply_rotary_pos_emb_registered = register_custom_op(
            f'lmdeploy::{func_str}',
            impl_abstract_func=apply_rotary_emb_abstract_impl)(
                apply_rotary_pos_emb_origin)
        setattr(apply_rotary_emb_module, func_str,
                apply_rotary_pos_emb_registered)

    def patch_kvcache_static_shape(self):
        import torch._dynamo as dynamo
        from torch.utils._pytree import tree_map
        cache_engine_module = import_module(
            'lmdeploy.pytorch.engine.cache_engine')
        class_str = 'CacheEngine'
        cache_engine_class = getattr(cache_engine_module, class_str)
        func_str = 'allocate_gpu_cache'
        allocate_gpu_cache_origin = getattr(cache_engine_class, func_str)

        def allocate_gpu_cache_mark_static(self):
            gpu_cache = allocate_gpu_cache_origin(self)
            tree_map(lambda x: dynamo.mark_static(x), gpu_cache)
            return gpu_cache

        setattr(cache_engine_class, func_str, allocate_gpu_cache_mark_static)
