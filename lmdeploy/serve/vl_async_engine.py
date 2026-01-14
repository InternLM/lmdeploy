# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal, Optional, Union

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.utils import get_logger, try_import_deeplink
from lmdeploy.vl.engine import ImageEncoder

logger = get_logger('lmdeploy')


class VLAsyncEngine(AsyncEngine):
    """Visual Language Async inference engine."""

    def __init__(self,
                 model_path: str,
                 backend: Literal['turbomind', 'pytorch'] = 'turbomind',
                 backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
                 vision_config: Optional[VisionConfig] = None,
                 **kwargs) -> None:
        if backend == 'pytorch':
            try_import_deeplink(backend_config.device_type)
        if backend_config and backend_config.enable_prefix_caching:
            backend_config.enable_prefix_caching = False
            logger.warning('Prefix caching is disabled since LMDeploy hasn\'t support in on VL models yet')
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
