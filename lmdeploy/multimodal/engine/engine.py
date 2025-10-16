# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.utils import get_logger

from ..models.builder import load_mm_model
from .model_agent import build_model_agent
from .post_process import PostProcessor
from .pre_process import PreProcessor

logger = get_logger('lmdeploy')


class MultiModalEngine():
    """The multi-modal async inference engine of lmdeploy."""

    def __init__(self,
                 model_path: str,
                 tokenizer: object,
                 engine_config: PytorchEngineConfig = None,
                 trust_remote_code: bool = True) -> None:
        # make sure engine config exist
        if engine_config is None:
            engine_config = PytorchEngineConfig()
        self.engine_config = copy.deepcopy(engine_config)
        self.tokenizer = tokenizer

        # build model
        self.model = load_mm_model(model_path, backend_config=self.engine_config)

        # build model agent
        self.model_agent = build_model_agent(self.model)
        self.model_agent.init()

        # init pre / post processor
        self.post_processor = PostProcessor(self.model_agent)
        self.pre_processor = PreProcessor(self.model_agent, self.post_processor)

    def start_loop(self):
        """Start async loops."""
        # invoked in api server start up event, where we already have running event loop started by uvicorn.run()
        # therefore we don't create a new event loop manually, simply start loops for each module
        self.pre_processor.start_loop()
        self.post_processor.start_loop()
        self.model_agent.start_loop()

    def close(self):
        """Close the engine and release resources."""
        self.pre_processor.close()
        self.post_processor.close()
        self.model_agent.close()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        tokenizer: object,
                        engine_config: PytorchEngineConfig = None,
                        trust_remote_code: bool = True,
                        **kwargs):
        """Create a MultiModalEngine instance."""
        return cls(model_path=pretrained_model_name_or_path,
                   tokenizer=tokenizer,
                   engine_config=engine_config,
                   trust_remote_code=trust_remote_code)

    async def encode(self, messages, session_id: int):
        """Async encode."""
        future = asyncio.Future()

        # future will later be set in post-processor
        self.pre_processor.process(session_id, messages, future)

        return await future
