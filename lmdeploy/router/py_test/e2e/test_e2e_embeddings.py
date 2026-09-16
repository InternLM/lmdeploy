import pytest
import requests


@pytest.mark.e2e
def test_embeddings_basic(e2e_router_only_rr, e2e_embedding_worker, e2e_embedding_model):
    router_url = e2e_router_only_rr.url
    response = requests.post(
        f"{router_url}/add_worker",
        params={"url": e2e_embedding_worker.url},
        timeout=30,
    )
    response.raise_for_status()

    response = requests.post(
        f"{router_url}/v1/embeddings",
        json={
            "model": e2e_embedding_model,
            "input": ["the quick brown fox", "jumps over the lazy dog"],
        },
        timeout=120,
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert len(data) == 2
    assert all(item["embedding"] for item in data)
