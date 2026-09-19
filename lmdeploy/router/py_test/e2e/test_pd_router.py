# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import requests


@pytest.mark.e2e
def test_lmdeploy_pd_chat(e2e_pd_cluster):
    response = requests.post(
        f'{e2e_pd_cluster.url}/v1/chat/completions',
        json={
            'model': e2e_pd_cluster.model,
            'messages': [{'role': 'user', 'content': 'Say PD works.'}],
            'max_tokens': 8,
            'temperature': 0.0,
            'stream': False,
        },
        timeout=180,
    )
    assert response.status_code == 200, response.text
    choice = response.json()['choices'][0]
    assert isinstance(choice['message']['content'], str)


@pytest.mark.e2e
def test_lmdeploy_pd_streaming(e2e_pd_cluster):
    response = requests.post(
        f'{e2e_pd_cluster.url}/v1/chat/completions',
        json={
            'model': e2e_pd_cluster.model,
            'messages': [{'role': 'user', 'content': 'Say PD streaming works.'}],
            'max_tokens': 8,
            'temperature': 0.0,
            'stream': True,
        },
        stream=True,
        timeout=180,
    )
    assert response.status_code == 200, response.text
    chunks = [line for line in response.iter_lines() if line]
    assert chunks
    assert any(line == b'data: [DONE]' for line in chunks)
