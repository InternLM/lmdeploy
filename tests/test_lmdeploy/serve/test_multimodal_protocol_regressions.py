# Copyright (c) OpenMMLab. All rights reserved.
import json

import pytest

from lmdeploy.serve.openai.errors import create_error_response
from lmdeploy.serve.processors.multimodal import MultimodalProcessor


def test_openai_error_envelope_preserves_legacy_fields():
    response = create_error_response(400, 'Invalid image detail.', param='messages')
    body = json.loads(response.body)
    assert response.status_code == 400
    assert body['error']['message'] == 'Invalid image detail.'
    assert body['error']['type'] == 'invalid_request_error'
    assert body['error']['param'] == 'messages'
    assert isinstance(body['error']['code'], str)
    assert body['message'] == body['error']['message']
    assert body['code'] == 400


@pytest.mark.parametrize('detail', ['invalid', 42, False, [], {}])
def test_image_detail_is_validated_before_media_loading(detail):
    processor = MultimodalProcessor(None, None)
    messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {
        'url': 'https://example.com/image.png', 'detail': detail}}]}]
    with pytest.raises(ValueError, match='detail'):
        processor._has_multimodal_input(messages)


@pytest.mark.parametrize('detail', [None, 'auto', 'low', 'high', 'original'])
def test_supported_image_detail_values_still_route_to_vision(detail):
    processor = MultimodalProcessor(None, None)
    assert processor._has_multimodal_input([{'role': 'user', 'content': [{
        'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA', 'detail': detail}}]}])
