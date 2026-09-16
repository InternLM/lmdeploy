import pytest
import requests

from py_test.e2e.conftest import _discover_model


@pytest.mark.e2e
def test_regular_router_serving(
    e2e_router_only_rr, e2e_primary_worker
):
    router_url = e2e_router_only_rr.url
    response = requests.post(
        f"{router_url}/add_worker",
        params={"url": e2e_primary_worker.url},
        timeout=30,
    )
    response.raise_for_status()

    model = _discover_model(e2e_primary_worker.url)
    response = requests.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say router works."}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=120,
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert isinstance(choice["message"]["content"], str)


@pytest.mark.e2e
def test_regular_router_streaming(
    e2e_router_only_rr, e2e_primary_worker
):
    router_url = e2e_router_only_rr.url
    response = requests.post(
        f"{router_url}/add_worker",
        params={"url": e2e_primary_worker.url},
        timeout=30,
    )
    response.raise_for_status()
    model = _discover_model(e2e_primary_worker.url)

    response = requests.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say streaming works."}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    assert response.status_code == 200, response.text
    chunks = [line for line in response.iter_lines() if line]
    assert chunks
    assert any(line == b"data: [DONE]" for line in chunks)
