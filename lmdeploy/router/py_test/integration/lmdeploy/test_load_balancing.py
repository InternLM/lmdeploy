"""Load-balancing policy tests against two real LMDeploy workers."""

from collections import Counter

import pytest
import requests


pytestmark = [pytest.mark.integration, pytest.mark.lmdeploy]

POLICIES = [
    "random",
    "round_robin",
    "power_of_two",
    "consistent_hash",
    "cache_aware",
]


def _chat(config, base_url, payload):
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=config.request_timeout,
    )
    response.raise_for_status()
    return response.json()


def _message_text(response):
    message = response["choices"][0]["message"]
    return (message.get("reasoning_content") or "") + (message.get("content") or "")


def _numeric_id(response):
    try:
        return int(response["id"])
    except (KeyError, TypeError, ValueError) as exc:
        pytest.fail(
            "load-balancing attribution requires LMDeploy's numeric per-worker response id"
        )
        raise exc


def _baseline_ids(config, payload):
    baselines = {
        url: _numeric_id(_chat(config, url, payload)) for url in config.backend_urls
    }
    for _ in range(100):
        values = list(baselines.values())
        if max(values) - min(values) >= 10:
            return baselines
        first = config.backend_urls[0]
        baselines[first] = _numeric_id(_chat(config, first, payload))
    pytest.fail(
        f"could not separate LMDeploy worker response-id baselines: {baselines}"
    )


def _attribute_worker(response, baselines):
    response_id = _numeric_id(response)
    exact = [url for url, baseline in baselines.items() if response_id == baseline + 1]
    if len(exact) == 1:
        baselines[exact[0]] = response_id
        return exact[0]

    candidates = [
        (response_id - baseline, url)
        for url, baseline in baselines.items()
        if response_id > baseline
    ]
    if not candidates:
        pytest.fail(
            f"could not attribute response id {response_id}; baselines={baselines}"
        )
    _, worker = min(candidates)
    baselines[worker] = response_id
    return worker


def _direct_outputs(config, payload):
    return {url: _chat(config, url, payload) for url in config.backend_urls}


@pytest.mark.parametrize("policy", POLICIES)
def test_lmdeploy_policy_routes_messages_and_input_ids(
    policy,
    lmdeploy_config,
    lmdeploy_router_factory,
):
    router = lmdeploy_router_factory.start(policy=policy)
    try:
        warmup = {
            "model": lmdeploy_config.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
        baselines = _baseline_ids(lmdeploy_config, warmup)
        served = Counter()

        message_payload = {
            "model": lmdeploy_config.model,
            "messages": [
                {"role": "user", "content": "What is 2+2? Answer in one word."}
            ],
            "max_tokens": 30,
            "temperature": 0,
            "stream": False,
        }
        direct_messages = _direct_outputs(lmdeploy_config, message_payload)
        baselines.update(
            {url: _numeric_id(output) for url, output in direct_messages.items()}
        )
        message_workers = []
        for _ in range(10):
            output = _chat(lmdeploy_config, router.url, message_payload)
            worker = _attribute_worker(output, baselines)
            served[worker] += 1
            message_workers.append(worker)
            assert _message_text(output) == _message_text(direct_messages[worker])

        token_payload = {
            "model": lmdeploy_config.model,
            "messages": [],
            "input_ids": list(lmdeploy_config.token_ids),
            "max_tokens": 30,
            "temperature": 0,
            "return_token_ids": True,
            "stream": False,
        }
        direct_tokens = _direct_outputs(lmdeploy_config, token_payload)
        baselines.update(
            {url: _numeric_id(output) for url, output in direct_tokens.items()}
        )
        token_workers = []
        for _ in range(10):
            output = _chat(lmdeploy_config, router.url, token_payload)
            worker = _attribute_worker(output, baselines)
            served[worker] += 1
            token_workers.append(worker)
            assert _message_text(output) == _message_text(direct_tokens[worker])
            assert output["choices"][0].get("output_ids")
            assert (
                output["choices"][0]["output_ids"]
                == direct_tokens[worker]["choices"][0]["output_ids"]
            )

        if policy in {"random", "power_of_two"}:
            assert all(served[url] > 0 for url in lmdeploy_config.backend_urls), served
        elif policy == "round_robin":
            assert all(
                current != previous
                for previous, current in zip(message_workers, message_workers[1:])
            ), message_workers
        elif policy == "consistent_hash":
            assert len(set(message_workers)) == 1, message_workers
            assert len(set(token_workers)) == 1, token_workers
        elif policy == "cache_aware":
            assert len(set(token_workers)) == 1, token_workers

            growing_workers = []
            for length in range(3, 8):
                growing_payload = {
                    **token_payload,
                    "input_ids": list(range(1, length + 1)),
                    "max_tokens": 2,
                }
                output = _chat(lmdeploy_config, router.url, growing_payload)
                growing_workers.append(_attribute_worker(output, baselines))
            assert len(set(growing_workers)) == 1, growing_workers
    finally:
        lmdeploy_router_factory.stop(router)
