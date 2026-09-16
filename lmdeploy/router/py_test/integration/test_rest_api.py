import pytest
import requests


@pytest.mark.integration
def test_health_models_and_inference_endpoints(mock_workers, router_manager):
    _, worker_urls, _ = mock_workers(n=2)
    router = router_manager.start_router(
        worker_urls=worker_urls,
        policy="round_robin",
    )

    health = requests.get(f"{router.url}/health", timeout=5)
    assert health.status_code == 200
    assert health.text == "All servers healthy"

    models = requests.get(f"{router.url}/v1/models", timeout=5)
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "mock"

    completion = requests.post(
        f"{router.url}/v1/completions",
        json={
            "model": "mock",
            "prompt": "hello",
            "max_tokens": 1,
            "stream": False,
        },
        timeout=5,
    )
    assert completion.status_code == 200
    assert completion.json()["choices"][0]["text"] == "ok"

    chat_completion = requests.post(
        f"{router.url}/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "stream": False,
        },
        timeout=5,
    )
    assert chat_completion.status_code == 200
    assert chat_completion.json()["choices"][0]["text"] == "ok"
