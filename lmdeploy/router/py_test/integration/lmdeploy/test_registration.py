"""LMDeploy-compatible dynamic registration tests with real workers."""

import time

import pytest
import requests


pytestmark = [pytest.mark.integration, pytest.mark.lmdeploy]


def _add_node(router_url, worker_url, model, role):
    return requests.post(
        f"{router_url}/nodes/add",
        json={
            "url": worker_url,
            "status": {"models": [model], "role": role},
        },
        timeout=30,
    )


def _wait_for_inference(config, router_url, timeout=30):
    deadline = time.monotonic() + timeout
    last_response = None
    while time.monotonic() < deadline:
        last_response = requests.post(
            f"{router_url}/v1/chat/completions",
            json={
                "model": config.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 2,
                "temperature": 0,
            },
            timeout=config.request_timeout,
        )
        if last_response.status_code == 200:
            return last_response
        time.sleep(0.5)
    pytest.fail(
        "registered worker never became routable: "
        + (last_response.text[:500] if last_response is not None else "no response")
    )


def _assert_status_entry(status, worker_url, model, role):
    entry = status[worker_url]
    assert entry["role"] == role
    assert model in entry["models"]
    assert entry["unfinished"] == 0
    assert entry["latency"] == []
    assert entry["speed"] is None


def _assert_pd_unsupported_endpoints(config, router_url):
    generate = requests.post(
        f"{router_url}/generate",
        json={"prompt": "Hello", "max_tokens": 2},
        timeout=config.request_timeout,
    )
    assert generate.status_code == 501

    responses = requests.post(
        f"{router_url}/v1/responses",
        json={"model": config.model, "input": "Hello", "max_output_tokens": 2},
        timeout=config.request_timeout,
    )
    assert responses.status_code == 501


def test_hybrid_dynamic_registration_is_idempotent_and_routable(
    lmdeploy_config,
    lmdeploy_router_factory,
):
    router = lmdeploy_router_factory.start(worker_urls=())
    worker_url = lmdeploy_config.backend_urls[0]
    try:
        first = _add_node(router.url, worker_url, lmdeploy_config.model, role=1)
        first.raise_for_status()
        second = _add_node(router.url, worker_url, lmdeploy_config.model, role=1)
        second.raise_for_status()

        status = requests.get(f"{router.url}/nodes/status", timeout=10)
        status.raise_for_status()
        _assert_status_entry(status.json(), worker_url, lmdeploy_config.model, role=1)

        workers = requests.get(f"{router.url}/list_workers", timeout=10)
        workers.raise_for_status()
        assert workers.json().get("urls") == [worker_url]
        _wait_for_inference(lmdeploy_config, router.url)

        removed = requests.post(
            f"{router.url}/nodes/remove",
            json={"url": worker_url},
            timeout=10,
        )
        removed.raise_for_status()
        assert requests.get(f"{router.url}/nodes/status", timeout=10).json() == {}

        removed_again = requests.post(
            f"{router.url}/nodes/remove",
            json={"url": worker_url},
            timeout=10,
        )
        removed_again.raise_for_status()
    finally:
        lmdeploy_router_factory.stop(router)


def test_regular_router_rejects_prefill_role(
    lmdeploy_config,
    lmdeploy_router_factory,
):
    router = lmdeploy_router_factory.start(worker_urls=())
    try:
        response = _add_node(
            router.url,
            lmdeploy_config.backend_urls[0],
            lmdeploy_config.model,
            role=2,
        )
        assert response.status_code == 400
    finally:
        lmdeploy_router_factory.stop(router)


def test_pd_dynamic_registration_and_inference(
    lmdeploy_config,
    lmdeploy_router_factory,
):
    if not lmdeploy_config.pd_prefill_url or not lmdeploy_config.pd_decode_url:
        pytest.skip(
            "set LMDEPLOY_PD_PREFILL_URL and LMDEPLOY_PD_DECODE_URL for real PD testing"
        )

    router = lmdeploy_router_factory.start(lmdeploy_pd=True)
    try:
        prefill = _add_node(
            router.url,
            lmdeploy_config.pd_prefill_url,
            lmdeploy_config.model,
            role=2,
        )
        decode = _add_node(
            router.url,
            lmdeploy_config.pd_decode_url,
            lmdeploy_config.model,
            role=3,
        )
        prefill.raise_for_status()
        decode.raise_for_status()

        status = requests.get(f"{router.url}/nodes/status", timeout=10)
        status.raise_for_status()
        status_payload = status.json()
        _assert_status_entry(
            status_payload,
            lmdeploy_config.pd_prefill_url,
            lmdeploy_config.model,
            role=2,
        )
        _assert_status_entry(
            status_payload,
            lmdeploy_config.pd_decode_url,
            lmdeploy_config.model,
            role=3,
        )

        duplicate_prefill = _add_node(
            router.url,
            lmdeploy_config.pd_prefill_url,
            lmdeploy_config.model,
            role=2,
        )
        duplicate_prefill.raise_for_status()
        assert len(requests.get(f"{router.url}/nodes/status", timeout=10).json()) == 2

        hybrid = _add_node(
            router.url,
            lmdeploy_config.backend_urls[0],
            lmdeploy_config.model,
            role=1,
        )
        assert hybrid.status_code == 400

        _wait_for_inference(lmdeploy_config, router.url, timeout=120)
        _assert_pd_unsupported_endpoints(lmdeploy_config, router.url)
    finally:
        lmdeploy_router_factory.stop(router)


def test_pd_static_registration_and_inference(
    lmdeploy_config,
    lmdeploy_router_factory,
):
    if not lmdeploy_config.pd_prefill_url or not lmdeploy_config.pd_decode_url:
        pytest.skip(
            "set LMDEPLOY_PD_PREFILL_URL and LMDEPLOY_PD_DECODE_URL for real PD testing"
        )

    router = lmdeploy_router_factory.start(
        lmdeploy_pd=True,
        prefill_urls=(lmdeploy_config.pd_prefill_url,),
        decode_urls=(lmdeploy_config.pd_decode_url,),
    )
    try:
        assert set(
            requests.get(f"{router.url}/list_workers", timeout=10).json()["urls"]
        ) == {
            lmdeploy_config.pd_prefill_url,
            lmdeploy_config.pd_decode_url,
        }
        _wait_for_inference(lmdeploy_config, router.url, timeout=120)
        _assert_pd_unsupported_endpoints(lmdeploy_config, router.url)
    finally:
        lmdeploy_router_factory.stop(router)
