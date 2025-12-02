# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from importlib import import_module
from typing import List

import torch
import torch.distributed
import torch_npu

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext
from lmdeploy.pytorch.models.chatglm2 import ChatGLMForConditionalGeneration
from lmdeploy.pytorch.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from lmdeploy.pytorch.models.qwen2_vl import Qwen2VLForConditionalGeneration
from lmdeploy.utils import get_logger

from ...graph_runner import GraphRunner
from .op_backend import SocVersion

ACL_FORMAT_ND = 2

logger = get_logger('lmdeploy')


class AscendGraphRunner(GraphRunner):
    """Ascend graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                 backend_config: BackendConfig, device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config, device)

        self.enable_graph = self.check_enable_graph()
        if self.enable_graph:
            import dlinfer.graph
            dlinfer.graph.config.enable_graph_mode = True
            self.patch_kvcache_static_shape()
            if hasattr(self.model, 'language_model'):
                self.model.language_model = torch.compile(self.model.language_model,
                                                          fullgraph=True,
                                                          dynamic=True,
                                                          backend='atbgraph')
            elif (isinstance(self.model, Qwen2VLForConditionalGeneration)
                  or isinstance(self.model, Qwen2_5_VLForConditionalGeneration)):
                self.model.model = torch.compile(self.model.model, fullgraph=True, dynamic=True, backend='atbgraph')
            elif isinstance(self.model, ChatGLMForConditionalGeneration):
                self.model.transformer.encoder = torch.compile(self.model.transformer.encoder,
                                                               fullgraph=True,
                                                               dynamic=True,
                                                               backend='atbgraph')
            else:
                self.model = torch.compile(self.model, fullgraph=True, dynamic=True, backend='atbgraph')

    def check_enable_graph(self):
        """Check enable graph."""
        # eager_mode
        if self.backend_config.eager_mode:
            return False

        warnings.warn(
            '\n\n'
            '************************************************************\n'
            '  Graph mode is an experimental feature. We currently\n'
            '  support both dense and Mixture of Experts (MoE) models\n'
            '  with bf16 and fp16 data types.\n'
            '  If graph mode does not function correctly with your model,\n'
            '  please consider using eager mode as an alternative.\n'
            '************************************************************\n\n', RuntimeWarning)

        # tp
        if torch.distributed.is_initialized():
            torch._inductor.config.compile_threads = 1
            return True

        return True

    def patch_kvcache_static_shape(self):
        import torch._dynamo as dynamo
        from torch.utils._pytree import tree_map
        cache_engine_module = import_module('lmdeploy.pytorch.engine.cache_engine')
        class_str = 'CacheEngine'
        cache_engine_class = getattr(cache_engine_module, class_str)
        func_str = 'allocate_gpu_cache'
        allocate_gpu_cache_origin = getattr(cache_engine_class, func_str)

        def allocate_gpu_cache_mark_static(self):
            gpu_cache = allocate_gpu_cache_origin(self)
            tree_map(lambda x: dynamo.mark_static(x), gpu_cache)
            return gpu_cache

        setattr(cache_engine_class, func_str, allocate_gpu_cache_mark_static)

    def _convert_kv_format(self, past_key_values: List[List[torch.Tensor]]) -> None:
        """Convert key/value caches to ACL_FORMAT_ND format if needed."""
        # Check format of first KV cache
        if torch_npu.get_npu_format(past_key_values[0][0]) == ACL_FORMAT_ND:
            return

        # Convert all KV caches to ACL_FORMAT_ND
        for layer_kv in past_key_values:
            key_cache, value_cache = layer_kv
            torch_npu.npu_format_cast(key_cache, ACL_FORMAT_ND)
            torch_npu.npu_format_cast(value_cache, ACL_FORMAT_ND)

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare inputs."""
        if self.enable_graph and SocVersion.is_Ascend910():
            self._convert_kv_format(past_key_values)
        return self.model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

    def get_capture_batch_sizes(self) -> List[int]:
        """Capture batch sizes."""
        # TODO: disable warmup now.
        return []
