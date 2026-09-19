# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import requests


@pytest.mark.integration
def test_pd_dynamic_registration_and_request_routing(mock_workers, router_manager):
    _, prefill_urls, _ = mock_workers(n=1)
    _, decode_urls, decode_ids = mock_workers(n=2)
    router = router_manager.start_router(
        policy='round_robin',
        lmdeploy_pd_disaggregation=True,
        prefill_urls=[],
        decode_urls=[],
    )

    router_manager.add_lmdeploy_node(router.url, prefill_urls[0], role=2)
    for decode_url in decode_urls:
        router_manager.add_lmdeploy_node(router.url, decode_url, role=3)

    nodes = router_manager.lmdeploy_nodes(router.url)
    assert nodes[prefill_urls[0]]['role'] == 2
    assert all(nodes[url]['role'] == 3 for url in decode_urls)

    selected_decode_ids = set()
    for request_seq in range(4):
        response = requests.post(
            f'{router.url}/v1/completions',
            json={
                'model': 'mock',
                'prompt': f'hello-{request_seq}',
                'max_tokens': 2,
            },
            timeout=10,
        )
        assert response.status_code == 200
        worker_id = response.headers.get('X-Worker-Id')
        assert worker_id in decode_ids
        selected_decode_ids.add(worker_id)

    assert len(selected_decode_ids) == 2

    router_manager.remove_lmdeploy_node(router.url, decode_urls[0], role=3)
    response = requests.post(
        f'{router.url}/v1/completions',
        json={'model': 'mock', 'prompt': 'after-removal', 'max_tokens': 1},
        timeout=10,
    )
    assert response.status_code == 200
    assert response.headers.get('X-Worker-Id') == decode_ids[1]
