import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

from fastapi import APIRouter

from lmdeploy.serve.openai.endpoints.management import register
from lmdeploy.serve.openai.protocol import UpdateParamsRequest


class _Context:

    def __init__(self, *, sleeping_tags, empty_init=False):
        self.async_engine = SimpleNamespace(
            backend='pytorch',
            sleeping_tags=sleeping_tags,
            backend_config=SimpleNamespace(empty_init=empty_init),
            engine=SimpleNamespace(update_params=self._update_params),
        )
        self.health_monitor = None
        self.allow_terminate_by_client = False
        self.enable_abort_handling = False
        self.calls = []

    def _update_params(self, request):
        self.calls.append(request)


def _endpoint(context, path):
    router = APIRouter()
    register(router, context)
    return next(route.endpoint for route in router.routes if route.path == path)


def test_management_import_does_not_load_pytorch_engine():
    code = (
        'import sys\n'
        'import lmdeploy.serve.openai.endpoints.management\n'
        "assert 'lmdeploy.pytorch.engine.engine' not in sys.modules\n")

    subprocess.run([sys.executable, '-c', code], check=True)


def test_update_weights_rejects_when_kv_cache_is_available():
    context = _Context(sleeping_tags=set())
    endpoint = _endpoint(context, '/update_weights')

    response = endpoint(UpdateParamsRequest(serialized_named_tensors='payload'))

    assert response.status_code == 409
    assert 'KV cache' in json.loads(response.body)['message']
    assert context.calls == []


def test_update_weights_runs_after_only_weights_are_woken():
    context = _Context(sleeping_tags={'kv_cache'})
    endpoint = _endpoint(context, '/update_weights')
    request = UpdateParamsRequest(serialized_named_tensors='payload')

    response = endpoint(request)

    assert response.status_code == 200
    assert context.calls == [request]


def test_update_weights_rejects_before_weights_are_woken():
    context = _Context(sleeping_tags={'weights', 'kv_cache'})
    endpoint = _endpoint(context, '/update_weights')

    response = endpoint(UpdateParamsRequest(serialized_named_tensors='payload'))

    assert response.status_code == 409
    assert context.calls == []


def test_update_weights_allows_empty_init_loading():
    context = _Context(sleeping_tags={'weights', 'kv_cache'}, empty_init=True)
    endpoint = _endpoint(context, '/update_weights')

    response = endpoint(UpdateParamsRequest(serialized_named_tensors='payload'))

    assert response.status_code == 200
    assert len(context.calls) == 1


def test_distributed_update_rejects_when_kv_cache_is_available():
    context = _Context(sleeping_tags=set())
    endpoint = _endpoint(context, '/update_weights_from_distributed')

    async def _run():
        from lmdeploy.serve.openai.protocol import UpdateWeightsFromDistributedRequest

        return await endpoint(UpdateWeightsFromDistributedRequest(
            names=[], dtypes=[], shapes=[], group_name='group'))

    response = asyncio.run(_run())

    assert response.status_code == 409
    assert 'KV cache' in json.loads(response.body)['message']
