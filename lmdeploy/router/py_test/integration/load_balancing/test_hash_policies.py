# Copyright (c) OpenMMLab. All rights reserved.
import collections

import pytest
import requests


@pytest.mark.integration
@pytest.mark.parametrize('policy', ['consistent_hash', 'rendezvous_hash'])
def test_hash_policy_routes_sessions_consistently_and_distributes_sessions(
    policy, mock_workers, router_manager
):
    _, worker_urls, worker_ids = mock_workers(n=3)
    router = router_manager.start_router(
        worker_urls=worker_urls,
        policy=policy,
    )

    selected = []
    for session_seq in range(30):
        session_id = f'session-{session_seq}'
        for _ in range(2):
            response = requests.post(
                f'{router.url}/v1/completions',
                json={
                    'model': 'mock',
                    'prompt': 'hello',
                    'user': session_id,
                },
                timeout=5,
            )
            assert response.status_code == 200
            worker_id = response.headers.get('X-Worker-Id')
            assert worker_id in worker_ids
            selected.append((session_id, worker_id))

    by_session = collections.defaultdict(set)
    for session_id, worker_id in selected:
        by_session[session_id].add(worker_id)
    assert all(
        len(worker_ids_for_session) == 1
        for worker_ids_for_session in by_session.values()
    )
    assert len(set(frozenset(worker_ids) for worker_ids in by_session.values())) >= 2
