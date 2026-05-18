# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import tempfile

from lmdeploy.pytorch.disagg.config import EngineRole


def test_node_defaults():
    from lmdeploy.serve.proxy.node import Node
    node = Node(url='http://localhost:8001')
    assert node.url == 'http://localhost:8001'
    assert node.role == EngineRole.Hybrid
    assert node.models == []
    assert node.unfinished == 0
    assert node.speed is None
    assert node.cache_usage is None
    assert node.last_metrics_poll is None


def test_node_with_fields():
    from lmdeploy.serve.proxy.node import Node
    node = Node(url='http://localhost:8001', role=EngineRole.Prefill, models=['llama'], speed=100)
    assert node.role == EngineRole.Prefill
    assert node.models == ['llama']
    assert node.speed == 100


def test_node_latency_deque():
    from lmdeploy.serve.proxy.node import Node
    node = Node(url='http://localhost:8001')
    node.latency.append(0.5)
    node.latency.append(0.3)
    assert list(node.latency) == [0.5, 0.3]


def test_registry_add_and_get():
    from lmdeploy.serve.proxy.node import NodeRegistry

    async def _run():
        registry = NodeRegistry()
        await registry.add('http://a:8001', EngineRole.Hybrid, ['model-a'])
        nodes = await registry.get('model-a')
        assert len(nodes) == 1
        assert nodes[0].url == 'http://a:8001'
        assert nodes[0].models == ['model-a']

    asyncio.run(_run())


def test_registry_add_with_status():
    from lmdeploy.serve.proxy.node import Node, NodeRegistry

    async def _run():
        registry = NodeRegistry()
        status = Node(url='http://a:8001', role=EngineRole.Hybrid, models=['model-a'], speed=50)
        await registry.add('http://a:8001', EngineRole.Hybrid, ['model-a'], status=status)
        nodes = await registry.get('model-a')
        assert nodes[0].speed == 50

    asyncio.run(_run())


def test_registry_remove():
    from lmdeploy.serve.proxy.node import NodeRegistry

    async def _run():
        registry = NodeRegistry()
        await registry.add('http://a:8001', EngineRole.Hybrid, ['model-a'])
        await registry.remove('http://a:8001')
        nodes = await registry.get('model-a')
        assert len(nodes) == 0

    asyncio.run(_run())


def test_registry_list_models():
    from lmdeploy.serve.proxy.node import NodeRegistry

    async def _run():
        registry = NodeRegistry()
        await registry.add('http://a:8001', EngineRole.Hybrid, ['model-a', 'model-b'])
        await registry.add('http://b:8001', EngineRole.Hybrid, ['model-b', 'model-c'])
        models = await registry.list_models()
        assert set(models) == {'model-a', 'model-b', 'model-c'}

    asyncio.run(_run())


def test_registry_get_by_url():
    from lmdeploy.serve.proxy.node import NodeRegistry

    async def _run():
        registry = NodeRegistry()
        await registry.add('http://a:8001', EngineRole.Hybrid, ['model-a'])
        node = await registry.get_by_url('http://a:8001')
        assert node is not None
        assert node.url == 'http://a:8001'
        assert await registry.get_by_url('http://missing:8001') is None

    asyncio.run(_run())


def test_registry_get_by_role():
    from lmdeploy.serve.proxy.node import NodeRegistry

    async def _run():
        registry = NodeRegistry()
        await registry.add('http://a:8001', EngineRole.Hybrid, ['model-a'])
        await registry.add('http://b:8001', EngineRole.Prefill, ['model-a'])
        nodes = await registry.get('model-a', role=EngineRole.Prefill)
        assert len(nodes) == 1
        assert nodes[0].role == EngineRole.Prefill

    asyncio.run(_run())


def test_registry_update_cache_usage():
    from lmdeploy.serve.proxy.node import NodeRegistry

    async def _run():
        registry = NodeRegistry()
        await registry.add('http://a:8001', EngineRole.Hybrid, ['model-a'])
        await registry.update_cache_usage('http://a:8001', 0.35)
        node = await registry.get_by_url('http://a:8001')
        assert node.cache_usage == 0.35
        assert node.last_metrics_poll is not None

    asyncio.run(_run())


def test_registry_persist_and_load():
    from lmdeploy.serve.proxy.node import NodeRegistry

    async def _run():
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            config_path = f.name
        try:
            registry = NodeRegistry(config_path=config_path)
            await registry.add('http://a:8001', EngineRole.Hybrid, ['model-a'])
            await registry.persist()

            registry2 = NodeRegistry(config_path=config_path)
            await registry2.load()
            nodes = await registry2.get('model-a')
            assert len(nodes) == 1
            assert nodes[0].url == 'http://a:8001'
        finally:
            import os
            os.unlink(config_path)

    asyncio.run(_run())
