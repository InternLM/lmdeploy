# Copyright (c) OpenMMLab. All rights reserved.
"""Integration tests for extended v1/chat/completions parameters.

Serves Qwen/Qwen3-0.6B once as a session-scoped fixture, then runs all
test cases against it.

Run:
  pytest tests/test_lmdeploy/serve/openai/chat_completion/chat_completion_r3.py -v
"""
import json
import subprocess
import sys
import time

import pytest
import requests

MODEL = "Qwen/Qwen3-0.6B"
BASE_URL = "http://127.0.0.1:23377"
SERVER_PORT = 23377

@pytest.fixture(scope="session")
def api_server():
    """Start lmdeploy api_server once for the entire test session."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "lmdeploy",
            "serve", "api_server", MODEL,
            "--server-name", "127.0.0.1",
            "--server-port", str(SERVER_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to be ready
    for _ in range(120):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    else:
        proc.terminate()
        proc.wait(timeout=10)
        pytest.skip("API server failed to start within 120s")

    yield BASE_URL

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

@pytest.fixture(scope="session")
def input_ids():
    """Token IDs used as input_ids for /generate-style requests."""
    # The following token_ids indicate:
    # """
    # <|im_start|>user
    # Repeat the word 'hello' three times.<|im_end|>
    # <|im_start|>assistant
    # """
    return [151644, 872, 198, 38718, 279, 3409, 364, 14990, 6, 2326, 3039, 13, 151645, 198, 151644, 77091, 198]


@pytest.fixture(scope="session")
def image_data():
    """Image URL used for image_data tests."""
    return ["https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg"]


def _chat(base_url, payload):
    return requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=60)


def test_basic_chat(api_server):
    """Standard chat completions still works."""
    resp = _chat(api_server, {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_completion_tokens": 8,
        "temperature": 0.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0


def test_input_ids(api_server, input_ids):
    """Use input_ids instead of messages."""
    resp = _chat(api_server, {
        "model": MODEL,
        "messages": "",
        "input_ids": input_ids,
        "max_completion_tokens": 8,
        "temperature": 0.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0


def test_input_ids_with_image_data(api_server, input_ids, image_data):
    """Use input_ids + image_data."""
    resp = _chat(api_server, {
        "model": MODEL,
        "messages": "",
        "input_ids": input_ids,
        "image_data": image_data,
        "max_completion_tokens": 8,
        "temperature": 0.0,
    })
    # May succeed or fail depending on whether the model supports VLM;
    # just check it doesn't crash the server
    assert resp.status_code in (200, 400)


def test_return_routed_experts(api_server):
    """Request routed_experts in response."""
    resp = _chat(api_server, {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 8,
        "return_routed_experts": True,
        "temperature": 0.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    has_routed_experts = "routed_experts" in data
    if has_routed_experts and data["routed_experts"] is not None:
        assert isinstance(data["routed_experts"], (list, str))


def test_return_routed_experts_default_off(api_server):
    """Without return_routed_experts, response has no routed_experts or it's None."""
    resp = _chat(api_server, {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 8,
        "temperature": 0.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("routed_experts") is None


def test_output_token_logprobs(api_server):
    """Check output_token_logprobs in response choices."""
    resp = _chat(api_server, {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 8,
        "temperature": 0.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    choice = data["choices"][0]
    assert "output_token_logprobs" in choice
    logprobs = choice["output_token_logprobs"]
    if logprobs is not None:
        assert isinstance(logprobs, list)
        if len(logprobs) > 0:
            entry = logprobs[0]
            assert len(entry) == 2
            assert isinstance(entry[0], (int, float))
            assert isinstance(entry[1], int)


def test_messages_priority_over_input_ids(api_server, input_ids):
    """input_ids is rejected when messages is non-empty."""
    resp = _chat(api_server, {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "input_ids": input_ids,
        "max_completion_tokens": 32,
        "temperature": 0.0,
    })
    assert resp.status_code != 200


def test_image_data_rejected_with_nonempty_messages(api_server, image_data):
    """image_data is rejected when messages is non-empty."""
    resp = _chat(api_server, {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "image_data": image_data,
        "max_completion_tokens": 32,
        "temperature": 0.0,
    })
    assert resp.status_code != 200


def test_streaming_with_routed_experts(api_server):
    """Streaming with return_routed_experts."""
    resp = requests.post(f"{api_server}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 8,
        "return_routed_experts": True,
        "stream": True,
        "temperature": 0.0,
    }, stream=True, timeout=60)
    assert resp.status_code == 200

    chunks = []
    for line in resp.iter_lines():
        line = line.decode("utf-8")
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            chunks.append(chunk)

    assert len(chunks) > 0
    final_chunk = chunks[-1]
    if "routed_experts" in final_chunk and final_chunk["routed_experts"] is not None:
        assert isinstance(final_chunk["routed_experts"], (list, str))


def test_streaming_output_token_logprobs(api_server):
    """Streaming with output_token_logprobs in chunks (requires logprobs enabled)."""
    resp = requests.post(f"{api_server}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 8,
        "logprobs": True,
        "top_logprobs": 1,
        "stream": True,
        "temperature": 0.0,
    }, stream=True, timeout=60)
    assert resp.status_code == 200

    chunks_with_logprobs = 0
    for line in resp.iter_lines():
        line = line.decode("utf-8")
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            for choice in chunk.get("choices", []):
                if choice.get("output_token_logprobs") is not None:
                    chunks_with_logprobs += 1

    assert chunks_with_logprobs > 0
