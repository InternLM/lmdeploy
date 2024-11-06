# Copyright (c) OpenMMLab. All rights reserved.
import os

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.utils import get_logger, get_model

from ..devices import get_device_manager
from .engine import Engine
from .model_agent import build_model_agent

logger = get_logger('lmdeploy')


class SpeculativeDecodingEngine(Engine):

    def __init__(self,
                 model_path: str,
                 speculative_model: str = None,
                 engine_config: PytorchEngineConfig = None,
                 trust_remote_code: bool = True) -> None:
        super().__init__(model_path, engine_config, trust_remote_code)

        if not os.path.exists(speculative_model):
            speculative_model = get_model(speculative_model,
                                          engine_config.download_dir,
                                          engine_config.revision)
        self.speculative_model = speculative_model

        with get_device_manager().context(self.device_context):
            self.speculative_model_agent = build_model_agent(
                speculative_model,
                cache_config=self.cache_config,
                backend_config=self.backend_config,
                trust_remote_code=trust_remote_code,
                tp=self.tp,
                dtype=engine_config.dtype,
                custom_module_map=engine_config.custom_module_map)
