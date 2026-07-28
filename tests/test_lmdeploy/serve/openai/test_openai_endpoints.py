# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for OpenAI endpoint router assembly."""

from types import SimpleNamespace

from fastapi.routing import APIRoute

from lmdeploy.serve.openai.api_server import ServerContext
from lmdeploy.serve.openai.endpoints import create_openai_router


def _route_manifest(router):
    return {
        (next(iter(route.methods)), route.path, route.name, route.include_in_schema, len(route.dependencies),
         tuple(route.tags))
        for route in router.routes if isinstance(route, APIRoute)
    }


def test_openai_router_manifest():
    expected = {
        ('GET', '/v1/models', 'available_models', True, 0, ()),
        ('GET', '/health', 'health', True, 0, ()),
        ('GET', '/terminate', 'terminate', True, 0, ()),
        ('POST', '/v1/chat/completions', 'chat_completions_v1', True, 1, ()),
        ('POST', '/v1/completions', 'completions_v1', True, 1, ()),
        ('POST', '/generate', 'generate', True, 1, ()),
        ('POST', '/v1/embeddings', 'create_embeddings', True, 0, ('unsupported', )),
        ('POST', '/v1/encode', 'encode', True, 1, ()),
        ('POST', '/pooling', 'pooling', True, 1, ()),
        ('POST', '/get_ppl', 'get_ppl', True, 1, ()),
        ('POST', '/update_weights', 'update_params', True, 1, ()),
        ('POST', '/init_weights_update_group', 'init_weights_update_group', True, 1, ()),
        ('POST', '/update_weights_from_distributed', 'update_weights_from_distributed', True, 1, ()),
        ('POST', '/destroy_weights_update_group', 'destroy_weights_update_group', True, 1, ()),
        ('POST', '/sleep', 'sleep', True, 1, ()),
        ('POST', '/wakeup', 'wakeup', True, 1, ()),
        ('GET', '/is_sleeping', 'is_sleeping', True, 0, ()),
        ('GET', '/distserve/engine_info', 'engine_info', True, 0, ()),
        ('POST', '/distserve/p2p_initialize', 'p2p_initialize', True, 0, ()),
        ('POST', '/distserve/p2p_connect', 'p2p_connect', True, 0, ()),
        ('POST', '/distserve/p2p_drop_connect', 'p2p_drop_connect', True, 0, ()),
        ('POST', '/distserve/free_cache', 'free_cache', True, 0, ()),
        ('POST', '/abort_request', 'abort_request', True, 0, ()),
        ('POST', '/v1/chat/interactive', 'chat_interactive_v1', False, 1, ()),
    }
    assert _route_manifest(create_openai_router(ServerContext())) == expected


def test_router_uses_injected_server_context():
    first_context = ServerContext()
    first_context.async_engine = SimpleNamespace(
        model_name='first-model', backend_config=SimpleNamespace(adapters=['first-adapter']))
    second_context = ServerContext()
    second_context.async_engine = SimpleNamespace(
        model_name='second-model', backend_config=SimpleNamespace(adapters=[]))

    first_router = create_openai_router(first_context)
    second_router = create_openai_router(second_context)
    first_models = next(route.endpoint for route in first_router.routes if route.path == '/v1/models')
    second_models = next(route.endpoint for route in second_router.routes if route.path == '/v1/models')

    assert [model.id for model in first_models().data] == ['first-model', 'first-adapter']
    assert [model.id for model in second_models().data] == ['second-model']
