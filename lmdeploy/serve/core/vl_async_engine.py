# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig
from lmdeploy.utils import get_logger

from .async_engine import AsyncEngine

logger = get_logger('lmdeploy')


class VLAsyncEngine(AsyncEngine):
    """Visual Language Async inference engine."""

    def __init__(self,
                 model_path: str,
                 backend: Literal['turbomind', 'pytorch'] = 'turbomind',
                 backend_config: TurbomindEngineConfig | PytorchEngineConfig | None = None,
                 vision_config: VisionConfig | None = None,
                 **kwargs) -> None:
        from lmdeploy.serve.processors import MultimodalProcessor
        from lmdeploy.utils import try_import_deeplink
        from lmdeploy.vl.engine import ImageEncoder

        if backend == 'pytorch':
            try_import_deeplink(backend_config.device_type)
        if backend_config and backend_config.enable_prefix_caching and not backend_config.disable_vision_encoder:
            from lmdeploy.archs import get_model_arch
            arch, _ = get_model_arch(model_path)
            _prefix_cache_archs = {'Qwen3_5ForConditionalGeneration', 'Qwen3_5MoeForConditionalGeneration'}
            if arch not in _prefix_cache_archs:
                backend_config.enable_prefix_caching = False
                logger.warning('Prefix caching is disabled for VL models (not yet verified for this architecture)')
        self.vl_encoder = ImageEncoder(model_path, backend, vision_config, backend_config=backend_config)
        super().__init__(model_path, backend=backend, backend_config=backend_config, **kwargs)
        # Update prompt_processor to support multimodal processing
        self.prompt_processor = MultimodalProcessor(self.tokenizer,
                                                    self.chat_template,
                                                    vl_encoder=self.vl_encoder,
                                                    backend=backend)
        if self.model_name == 'base':
            raise RuntimeError(
                'please specify chat template as guided in https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html#set-chat-template'  # noqa: E501
            )

    def close(self):
        if hasattr(self, 'vl_encoder'):
            del self.vl_encoder
            super().close()
