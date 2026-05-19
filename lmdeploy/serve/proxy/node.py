# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import copy
import json
import os.path as osp
import time
from collections import deque

from pydantic import BaseModel, Field

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.config import LATENCY_DEQUE_LEN
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _normalize_url(url: str) -> str:
    """Replace 0.0.0.0 with 127.0.0.1 for client connections."""
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    if parsed.hostname == '0.0.0.0':
        replaced = parsed._replace(netloc=parsed.netloc.replace('0.0.0.0', '127.0.0.1', 1))
        return urlunparse(replaced)
    return url


class Node(BaseModel):
    """A backend api_server node tracked by the proxy."""
    url: str
    role: EngineRole = EngineRole.Hybrid
    models: list[str] = Field(default_factory=list)
    unfinished: int = 0
    latency: deque = Field(default_factory=lambda: deque(maxlen=LATENCY_DEQUE_LEN))
    speed: int | None = None
    cache_usage: float | None = None
    last_metrics_poll: float | None = None

    model_config = {'arbitrary_types_allowed': True}


class NodeRegistry:
    """Thread-safe registry of backend nodes."""

    def __init__(self, config_path: str | None = None, cache_status: bool = True):
        self._nodes: dict[str, Node] = {}
        self._lock = asyncio.Lock()
        self._config_path = config_path or osp.join(osp.dirname(osp.realpath(__file__)), 'proxy_config.json')
        self._cache_status = cache_status

    async def add(self, url: str, role: EngineRole = EngineRole.Hybrid,
                  models: list[str] | None = None,
                  status: Node | None = None) -> None:
        url = _normalize_url(url)
        async with self._lock:
            if status is not None:
                if status.models:
                    self._nodes.pop(url, None)
                    self._nodes[url] = status
                    await self._persist_unlocked()
                    return
                node = status
            else:
                node = self._nodes.get(url, Node(url=url, role=role))

            if models is not None:
                node.models = models
            elif not node.models:
                try:
                    import requests

                    from lmdeploy.serve.openai.api_client import APIClient
                    client = APIClient(api_server_url=url)
                    node.models = client.available_models
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Could not fetch models from {url}: {e}. "
                                   'Registering node with empty model list.')
                    node.models = []

            self._nodes[url] = node
            await self._persist_unlocked()

    async def remove(self, url: str) -> None:
        async with self._lock:
            self._nodes.pop(url, None)
            await self._persist_unlocked()

    async def get(self, model_name: str, role: EngineRole | None = None) -> list[Node]:
        async with self._lock:
            result = []
            for node in self._nodes.values():
                if model_name in node.models:
                    if role is None or node.role == role:
                        result.append(node)
            return result

    async def get_by_url(self, url: str) -> Node | None:
        async with self._lock:
            return self._nodes.get(url)

    async def list_models(self) -> list[str]:
        async with self._lock:
            models = set()
            for node in self._nodes.values():
                models.update(node.models)
            return list(models)

    async def update_cache_usage(self, url: str, usage: float) -> None:
        async with self._lock:
            node = self._nodes.get(url)
            if node is not None:
                node.cache_usage = usage
                node.last_metrics_poll = time.time()

    async def all_nodes(self) -> list[Node]:
        async with self._lock:
            return list(self._nodes.values())

    async def get_nodes_by_role(self, role: EngineRole) -> dict[str, Node]:
        async with self._lock:
            return {url: node for url, node in self._nodes.items() if node.role == role}

    async def persist(self) -> None:
        async with self._lock:
            await self._persist_unlocked()

    async def _persist_unlocked(self) -> None:
        if not self._cache_status:
            return
        nodes = copy.deepcopy(self._nodes)
        for node in nodes.values():
            node.latency = deque(list(node.latency)[-LATENCY_DEQUE_LEN:])
        with open(self._config_path, 'w') as f:
            json.dump(
                {url: node.model_dump_json() for url, node in nodes.items()},
                f,
                indent=2,
            )

    async def load(self) -> None:
        async with self._lock:
            if not self._cache_status:
                return
            if not osp.exists(self._config_path):
                return
            if osp.getsize(self._config_path) == 0:
                return
            logger.info(f"Loading node configuration: {self._config_path}")
            with open(self._config_path) as f:
                config = json.load(f)
            self._nodes = {
                url: Node.model_validate_json(data) for url, data in config.items()
            }
