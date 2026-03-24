# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from unittest.mock import MagicMock, patch

from lmdeploy.messages import PytorchEngineConfig, ResponseType


def test_on_add_message_disable_vision_rejects_multimodal():
    """Multimodal input with disable_vision_encoder must error, not strip
    inputs."""
    from lmdeploy.pytorch.engine.engine import Engine
    from lmdeploy.pytorch.engine.request import Request, RequestType, Response

    engine = Engine.__new__(Engine)
    engine.engine_config = PytorchEngineConfig(disable_vision_encoder=True)
    engine.input_processor = object()
    engine.scheduler = MagicMock()
    engine.scheduler.sessions = {1: MagicMock()}
    engine.req_manager = MagicMock()
    engine._add_message = MagicMock()

    resp = Response(type=ResponseType.SUCCESS, sender_id=0, event=asyncio.Event())
    req = Request(
        type=RequestType.ADD_MESSAGE,
        sender_id=0,
        data={
            'session_id': 1,
            'token_ids': [1, 2, 3],
            'input_multimodals': [{'image': []}],
            'response': True,
        },
        resp=resp,
    )

    captured = []

    def capture_response(req_manager, resp, resp_type, data, err_msg):
        captured.append((resp_type, err_msg))

    with patch('lmdeploy.pytorch.engine.engine.response_reqs', side_effect=capture_response):
        engine._on_add_message([req])

    assert len(captured) == 1
    assert captured[0][0] == ResponseType.INTERNAL_ENGINE_ERROR
    assert 'disable_vision_encoder=True' in captured[0][1]
    engine._add_message.assert_not_called()
