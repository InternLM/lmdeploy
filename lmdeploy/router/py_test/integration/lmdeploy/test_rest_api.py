"""LMDeploy REST compatibility tests routed through lmdeploy-router.

The cases are adapted from LMDeploy's ``autotest/interface/restful`` suite.
They require two externally managed Hybrid LMDeploy API servers; see
``docs/lmdeploy_support_guide.md`` for the environment variables.
"""

import json
import time

import pytest
import requests


pytestmark = [pytest.mark.integration, pytest.mark.lmdeploy]


def _post(config, base_url, path, payload, *, stream=False):
    return requests.post(
        f"{base_url}{path}",
        json=payload,
        stream=stream,
        timeout=config.request_timeout,
    )


def _chat(config, base_url, payload, *, stream=False):
    return _post(config, base_url, "/v1/chat/completions", payload, stream=stream)


def _completion(config, base_url, payload, *, stream=False):
    return _post(config, base_url, "/v1/completions", payload, stream=stream)


def _generate(config, base_url, payload, *, stream=False):
    return _post(config, base_url, "/generate", payload, stream=stream)


def _sse_payloads(response):
    payloads = []
    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8").strip()
        if text.startswith("data: ") and text != "data: [DONE]":
            payloads.append(json.loads(text[6:]))
    return payloads


def _message_text(choice):
    message = choice.get("message") or {}
    return (message.get("reasoning_content") or "") + (message.get("content") or "")


def _assert_usage(usage):
    assert usage.get("prompt_tokens", 0) > 0
    assert usage.get("completion_tokens", 0) > 0
    assert usage.get("total_tokens", 0) > 0


def _assert_chat_response(output, model):
    assert output.get("id") is not None
    assert output.get("object") == "chat.completion"
    assert output.get("model") == model
    _assert_usage(output.get("usage", {}))
    choices = output.get("choices", [])
    assert len(choices) == 1
    choice = choices[0]
    assert choice.get("index") == 0
    assert choice.get("finish_reason") in {"stop", "length"}
    assert (choice.get("message") or {}).get("role") == "assistant"
    assert _message_text(choice)


def _assert_completion_response(output, model):
    assert output.get("id") is not None
    assert output.get("object") == "text_completion"
    assert output.get("model") == model
    _assert_usage(output.get("usage", {}))
    choices = output.get("choices", [])
    assert len(choices) == 1
    assert choices[0].get("index") == 0
    assert choices[0].get("finish_reason") in {"stop", "length"}
    assert choices[0].get("text")


def _assert_generate_response(output):
    assert isinstance(output.get("text"), str) and output["text"]
    assert isinstance(output.get("output_ids"), list) and output["output_ids"]
    assert isinstance(output.get("meta_info"), dict)


def _router_url(lmdeploy_router):
    return lmdeploy_router.url


def test_models(lmdeploy_config, lmdeploy_router):
    response = requests.get(f"{_router_url(lmdeploy_router)}/v1/models", timeout=10)
    response.raise_for_status()
    payload = response.json()
    assert payload.get("object") == "list"
    assert any(
        item.get("id") == lmdeploy_config.model for item in payload.get("data", [])
    )


def test_health(lmdeploy_router):
    response = requests.get(f"{_router_url(lmdeploy_router)}/health", timeout=10)
    response.raise_for_status()
    assert "healthy" in response.text.lower()


def test_list_workers(lmdeploy_config, lmdeploy_router):
    response = requests.get(f"{_router_url(lmdeploy_router)}/list_workers", timeout=10)
    response.raise_for_status()
    assert set(response.json().get("urls", [])) == set(lmdeploy_config.backend_urls)


def test_chat_basic(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "max_tokens": 30,
        "temperature": 0.01,
    }
    response = _chat(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    _assert_chat_response(response.json(), lmdeploy_config.model)


def test_chat_streaming(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "messages": [{"role": "user", "content": "介绍你自己"}],
        "max_tokens": 30,
        "temperature": 0.01,
        "stream": True,
    }
    response = _chat(
        lmdeploy_config, _router_url(lmdeploy_router), payload, stream=True
    )
    response.raise_for_status()
    chunks = _sse_payloads(response)
    assert len(chunks) >= 2
    choices = [chunk["choices"][0] for chunk in chunks if chunk.get("choices")]
    assert choices
    assert choices[-1].get("finish_reason") in {"stop", "length"}
    assert all(choice.get("index") == 0 for choice in choices)


@pytest.mark.parametrize("stop", ["6", ["goodbye", "farewell", "bye"]])
def test_chat_stopwords(lmdeploy_config, lmdeploy_router, stop):
    payload = {
        "model": lmdeploy_config.model,
        "messages": [{"role": "user", "content": "Count to 10: 1, 2, 3, 4, 5, 6"}],
        "max_tokens": 100,
        "stop": stop,
        "temperature": 0.01,
    }
    response = _chat(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    output = response.json()
    _assert_chat_response(output, lmdeploy_config.model)
    text = _message_text(output["choices"][0])
    stops = [stop] if isinstance(stop, str) else stop
    assert all(item not in text for item in stops)


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_chat_token_limit(lmdeploy_config, lmdeploy_router, field):
    payload = {
        "model": lmdeploy_config.model,
        "messages": [{"role": "user", "content": "介绍人工智能"}],
        field: 5,
        "temperature": 0.01,
    }
    response = _chat(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    output = response.json()
    _assert_chat_response(output, lmdeploy_config.model)
    assert output["choices"][0]["finish_reason"] == "length"


def test_chat_input_ids_token_in_out(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "messages": [],
        "input_ids": list(lmdeploy_config.token_ids),
        "max_tokens": 20,
        "temperature": 0.01,
        "return_token_ids": True,
    }
    response = _chat(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    output = response.json()
    _assert_chat_response(output, lmdeploy_config.model)
    assert output["choices"][0].get("output_ids")


def test_chat_input_ids_matches_direct_backend(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "messages": [],
        "input_ids": list(lmdeploy_config.token_ids),
        "max_tokens": 15,
        "temperature": 0,
        "return_token_ids": True,
    }
    direct = []
    for backend_url in lmdeploy_config.backend_urls:
        response = _chat(lmdeploy_config, backend_url, payload)
        if response.status_code == 200:
            direct.append(response.json())
    assert direct

    response = _chat(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    routed = response.json()["choices"][0]
    matches = [
        item["choices"][0]
        for item in direct
        if _message_text(item["choices"][0]) == _message_text(routed)
    ]
    assert matches
    assert any(item.get("output_ids") == routed.get("output_ids") for item in matches)


def test_chat_temperature_zero_is_deterministic(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
        "max_tokens": 10,
        "temperature": 0,
    }
    first = _chat(lmdeploy_config, _router_url(lmdeploy_router), payload)
    second = _chat(lmdeploy_config, _router_url(lmdeploy_router), payload)
    first.raise_for_status()
    second.raise_for_status()
    assert _message_text(first.json()["choices"][0]) == _message_text(
        second.json()["choices"][0]
    )


def test_chat_logprobs(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
        "temperature": 0.01,
        "logprobs": True,
        "top_logprobs": 5,
    }
    response = _chat(lmdeploy_config, _router_url(lmdeploy_router), payload)
    if response.status_code == 400:
        pytest.skip("LMDeploy backend was not started with raw logprobs enabled")
    response.raise_for_status()
    content = response.json()["choices"][0].get("logprobs", {}).get("content")
    assert content


def test_chat_invalid_model(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": "model-that-does-not-exist",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
    }
    response = _chat(lmdeploy_config, _router_url(lmdeploy_router), payload)
    assert response.status_code in {400, 404, 500}


def test_chat_matches_direct_backend(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "messages": [{"role": "user", "content": "什么是人工智能?用一句话回答"}],
        "max_tokens": 20,
        "temperature": 0,
    }
    direct_texts = {
        _message_text(_chat(lmdeploy_config, url, payload).json()["choices"][0])
        for url in lmdeploy_config.backend_urls
    }
    response = _chat(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    assert _message_text(response.json()["choices"][0]) in direct_texts


def test_completion_basic(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "prompt": "Shanghai is",
        "max_tokens": 16,
        "temperature": 0.01,
    }
    response = _completion(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    _assert_completion_response(response.json(), lmdeploy_config.model)


def test_completion_streaming(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "prompt": "Shanghai is",
        "max_tokens": 16,
        "temperature": 0.01,
        "stream": True,
    }
    response = _completion(
        lmdeploy_config, _router_url(lmdeploy_router), payload, stream=True
    )
    response.raise_for_status()
    chunks = _sse_payloads(response)
    assert len(chunks) >= 2
    choices = [chunk["choices"][0] for chunk in chunks if chunk.get("choices")]
    assert choices[-1].get("finish_reason") in {"stop", "length"}


def test_completion_stopword(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "prompt": "Shanghai is",
        "max_tokens": 200,
        "stop": " city",
        "temperature": 0.01,
    }
    response = _completion(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    assert " city" not in response.json()["choices"][0]["text"]


def test_completion_max_tokens(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "prompt": "介绍人工智能",
        "max_tokens": 5,
        "temperature": 0.01,
    }
    response = _completion(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_completion_matches_direct_backend(lmdeploy_config, lmdeploy_router):
    payload = {
        "model": lmdeploy_config.model,
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0,
    }
    direct_texts = {
        _completion(lmdeploy_config, url, payload).json()["choices"][0]["text"]
        for url in lmdeploy_config.backend_urls
    }
    response = _completion(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    assert response.json()["choices"][0]["text"] in direct_texts


def test_generate_basic(lmdeploy_config, lmdeploy_router):
    payload = {"prompt": "The sky is", "max_tokens": 5, "temperature": 0.01}
    response = _generate(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    _assert_generate_response(response.json())


def test_generate_input_ids(lmdeploy_config, lmdeploy_router):
    payload = {
        "input_ids": list(lmdeploy_config.token_ids),
        "max_tokens": 10,
        "temperature": 0.01,
    }
    response = _generate(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    _assert_generate_response(response.json())


def test_generate_streaming(lmdeploy_config, lmdeploy_router):
    payload = {
        "prompt": "Count to 10: 1, 2, 3,",
        "max_tokens": 8,
        "temperature": 0.01,
        "stream": True,
    }
    response = _generate(
        lmdeploy_config, _router_url(lmdeploy_router), payload, stream=True
    )
    response.raise_for_status()
    chunks = _sse_payloads(response)
    assert len(chunks) >= 2
    assert "".join(chunk.get("text", "") for chunk in chunks).strip()
    assert any(chunk.get("output_ids") for chunk in chunks)


def test_generate_stop_token_ids(lmdeploy_config, lmdeploy_router):
    payload = {
        "prompt": "Hello world",
        "max_tokens": 50,
        "stop_token_ids": [13],
        "temperature": 0.01,
    }
    response = _generate(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    _assert_generate_response(response.json())


def test_generate_session_id(lmdeploy_config, lmdeploy_router):
    session_id = int(time.time()) % 100000
    first = _generate(
        lmdeploy_config,
        _router_url(lmdeploy_router),
        {"prompt": "First:", "session_id": session_id, "max_tokens": 5},
    )
    second = _generate(
        lmdeploy_config,
        _router_url(lmdeploy_router),
        {"prompt": "Second:", "session_id": session_id, "max_tokens": 5},
    )
    first.raise_for_status()
    second.raise_for_status()


def test_generate_rejects_prompt_with_input_ids(lmdeploy_config, lmdeploy_router):
    response = _generate(
        lmdeploy_config,
        _router_url(lmdeploy_router),
        {"prompt": "Hello", "input_ids": [1, 2, 3], "max_tokens": 5},
    )
    assert response.status_code in {400, 422}


def test_generate_rejects_empty_prompt(lmdeploy_config, lmdeploy_router):
    response = _generate(
        lmdeploy_config,
        _router_url(lmdeploy_router),
        {"prompt": "", "max_tokens": 5},
    )
    assert response.status_code in {400, 422}


def test_generate_matches_direct_backend(lmdeploy_config, lmdeploy_router):
    payload = {
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0,
    }
    direct = []
    for backend_url in lmdeploy_config.backend_urls:
        response = _generate(lmdeploy_config, backend_url, payload)
        response.raise_for_status()
        direct.append(response.json())

    response = _generate(lmdeploy_config, _router_url(lmdeploy_router), payload)
    response.raise_for_status()
    routed = response.json()
    matches = [item for item in direct if item["text"] == routed["text"]]
    assert matches
    assert any(item.get("output_ids") == routed.get("output_ids") for item in matches)
