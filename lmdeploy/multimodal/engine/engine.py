# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
from typing import Dict, List, Optional

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.disagg.conn.engine_conn import EngineP2PConnection
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeConnectionRequest, DistServeDropConnectionRequest,
                                                   DistServeInitRequest)
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
                 chat_template: object,
                 tokenizer: object,
                 engine_config: PytorchEngineConfig = None,
                 trust_remote_code: bool = True) -> None:
        # make sure engine config exist
        if engine_config is None:
            engine_config = PytorchEngineConfig()
        self.engine_config = copy.deepcopy(engine_config)
        self.chat_template = chat_template
        self.tokenizer = tokenizer

        # build model
        self.model = load_mm_model(model_path, backend_config=self.engine_config)

        # build model agent
        self.model_agent = build_model_agent(self.model)
        self.model_agent.init()

        # init pre / post processor
        self.post_processor = PostProcessor(self.model_agent)
        self.pre_processor = PreProcessor(self.model_agent, self.post_processor)

        self.engine_conn = EngineP2PConnection(self)

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
                        chat_template: object,
                        tokenizer: object,
                        engine_config: PytorchEngineConfig = None,
                        trust_remote_code: bool = True,
                        **kwargs):
        """Create a MultiModalEngine instance."""
        return cls(model_path=pretrained_model_name_or_path,
                   chat_template=chat_template,
                   tokenizer=tokenizer,
                   engine_config=engine_config,
                   trust_remote_code=trust_remote_code)

    async def encode(self, messages, session_id: int):
        """Async encode."""
        future = asyncio.Future()

        # future will later be set in post-processor
        self.pre_processor.process(session_id, messages, future)

        return await future

    # TODO: change this, put into pre-processor?
    async def wrap_for_pytorch(
        self,
        messages: List[Dict],
        chat_template,
        tokenizer,
        sequence_start,
        tools: Optional[List[object]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> List[Dict]:
        """
        Args:
            messages (List[Dict]): a list of message, which is supposed to be
                the output of `preprocess`
        Returns:
            a dict which will be passed to pytorch engine_instance's forward.
            The dict is like the following:
            Dict(
                'prompt': 'the prompt after applying chat template'
                'input_ids': [],
                'multimodal': {
                    'pixel_values': torch.Tensor,
                    ...
                ]
            )
        """
        result = self.model.to_pytorch(messages,
                                       chat_template,
                                       tokenizer,
                                       sequence_start,
                                       tools=tools,
                                       enable_thinking=enable_thinking)
        # clear data
        for i, message in enumerate(messages):
            if isinstance(message['content'], List):
                messages[i]['preprocess'] = None
        return result

    async def p2p_initialize(self, init_request: DistServeInitRequest):
        return await self.engine_conn.p2p_initialize(init_request)

    def p2p_connect(self, conn_request: DistServeConnectionRequest):
        return self.engine_conn.p2p_connect(conn_request)

    async def p2p_drop_connect(self, drop_conn_request: DistServeDropConnectionRequest):
        return self.engine_conn.p2p_drop_connect(drop_conn_request)
