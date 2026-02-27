# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json
import os
import random
import threading
import time
from typing import TYPE_CHECKING, Dict, Optional

import aiohttp
import numpy as np
import requests

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.constants import AIOHTTP_TIMEOUT, ErrorCodes, RoutingStrategy, err_msg
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from .proxy import Status

logger = get_logger('lmdeploy')


class Connector:
    """Connector class responsible for creating and managing aiohttp
    ClientSession."""

    def __init__(self):
        self.limits = int(os.getenv('LMDEPLOY_AIOHTTP_LIMITS', 1024))
        self.limits_per_host = int(os.getenv('LMDEPLOY_AIOHTTP_LIMITS_PER_HOST', 128))
        self._session = None

    async def get_session(self) -> aiohttp.ClientSession:
        """Get the shared session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.limits,
                limit_per_host=self.limits_per_host,
                force_close=False,  # Keep connections alive
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT),
            )
        return self._session

    async def cleanup(self):
        """Cleanup resources, close session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def update(self, num_hosts: int):
        """Update the limits based on number of hosts."""
        new_limit = self.limits_per_host * num_hosts
        # Only update if the limit changed significantly
        if abs(new_limit - self.limits) > self.limits_per_host:
            self.limits = new_limit
            await self.cleanup()
            await self.get_session()


connector = Connector()


class Node:
    """Node class responsible for sending requests and receiving responses.

    A Node represents an API server and can handle concurrent requests to that server. All nodes share a common
    ClientSession managed by NodeManager for efficient connection pooling and reuse.
    """

    def __init__(self, url: str, status: 'Status'):
        """Initialize a Node.

        Args:
            url (str): The node URL.
            status (Status, optional): The node status.
        """
        self.url = url
        self.status = status

    async def _make_request(self, request: Dict, endpoint: str):
        """Make HTTP POST request to the node."""
        session = await connector.get_session()
        return await session.post(self.url + endpoint, json=request)

    async def stream_generate(self, request: Dict, endpoint: str):
        """Return a generator to handle the input request."""
        try:
            async with await self._make_request(request, endpoint) as response:
                async for line in response.content:
                    if line.strip():
                        yield line + b'\n\n'
        except (Exception, GeneratorExit, aiohttp.ClientError) as e:
            logger.error(f'Exception in stream_generate: {e}')
            yield self.handle_api_timeout()

    async def generate(self, request: Dict, endpoint: str):
        """Return the response of the input request."""
        try:
            async with await self._make_request(request, endpoint) as response:
                return await response.text()
        except Exception as e:
            logger.error(f'Exception in generate: {e}')
            return self.handle_api_timeout()

    def pre_call(self):
        """Preprocess before the request get processed."""
        self.status.unfinished += 1
        return time.time()

    def post_call(self, start: float):
        """Post process after the response finished."""
        self.status.unfinished -= 1
        self.status.latency.append(time.time() - start)

    def handle_api_timeout(self):
        """Handle the api time out."""
        logger.warning(f'api timeout: {self.url}')
        return json.dumps({
            'error_code': ErrorCodes.API_TIMEOUT.value,
            'text': err_msg[ErrorCodes.API_TIMEOUT],
        }).encode() + b'\n'


CONTROLLER_HEART_BEAT_EXPIRATION = int(os.getenv('LMDEPLOY_CONTROLLER_HEART_BEAT_EXPIRATION', 90))


def heart_beat_controller(proxy_controller):
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        logger.info('Start heart beat check')
        proxy_controller.remove_stale_nodes_by_expiration()


class NodeManager:
    """Manage all the api_servers, each of which is defined as a Node
    object."""

    def __init__(self) -> None:
        self.nodes = {}
        self.routing_strategy = RoutingStrategy.MIN_EXPECTED_LATENCY
        self._nodes_cache: Dict[EngineRole, Dict[str, Node]] = {}
        self._nodes_cache_dirty = True

        self.heart_beat_thread = threading.Thread(target=heart_beat_controller, args=(self, ), daemon=True)
        self.heart_beat_thread.start()

    def _invalidate_nodes_cache(self):
        """Mark node cache as invalid."""
        self._nodes_cache_dirty = True

    def get_nodes(self, role: EngineRole) -> Dict[str, Node]:
        """Get nodes for the specified role, using cache."""
        if self._nodes_cache_dirty or role not in self._nodes_cache:
            self._nodes_cache = {}
            for node_url, node_status in self.nodes.items():
                node_role = node_status.role
                if node_role not in self._nodes_cache:
                    self._nodes_cache[node_role] = {}
                self._nodes_cache[node_role][node_url] = Node(url=node_url, status=node_status)
            self._nodes_cache_dirty = False
        return self._nodes_cache.get(role, {})

    @property
    def hybrid_nodes(self):
        return self.get_nodes(EngineRole.Hybrid)

    @property
    def prefill_nodes(self):
        return self.get_nodes(EngineRole.Prefill)

    @property
    def decode_nodes(self):
        return self.get_nodes(EngineRole.Decode)

    async def add(self, node_url: str, status: 'Status'):
        """Add a node."""
        self.nodes[node_url] = status
        self._invalidate_nodes_cache()
        await connector.update(len(self.nodes))

    async def remove(self, node_url: str):
        """Remove a node."""
        if node_url not in self.nodes:
            raise ValueError(f'Node {node_url} does not exist')

        self.nodes.pop(node_url)
        self._invalidate_nodes_cache()
        await connector.update(len(self.nodes))

    async def terminate_node(self, node_url: str):
        """Terminate a node."""
        if node_url not in self.nodes:
            raise KeyError(f'Node {node_url} does not exist')

        self.nodes.pop(node_url)
        self._invalidate_nodes_cache()

        session = await connector.get_session()
        async with session.get(f'{node_url}/terminate', headers={'accept': 'application/json'}) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f'Failed to terminate node {node_url}, status={response.status}, msg={text}')

    async def terminate_all_nodes(self):
        """Terminate all nodes.

        Raises:
            RuntimeError: If any node termination fails.
        """
        if not self.nodes:
            return

        node_urls = list(self.nodes.keys())
        results = await asyncio.gather(*[self.terminate_node(url) for url in node_urls], return_exceptions=True)

        # Check for failures
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            failed_count = len(failures)
            total_count = len(node_urls)
            error_msg = f'Failed to terminate {failed_count}/{total_count} nodes'
            logger.error(f'{error_msg}: {[str(f) for f in failures]}')
            raise RuntimeError(error_msg)

    def remove_stale_nodes_by_expiration(self):
        """Remove stale nodes."""
        headers = {'accept': 'application/json'}
        to_be_deleted = [url for url in self.nodes.keys() if not self._check_node_health(url, headers)]
        for node_url in to_be_deleted:
            # Note: remove is async but we can't await here in sync method
            # The node will be removed from dict, but async cleanup won't happen
            if node_url in self.nodes:
                self.nodes.pop(node_url)
                self._invalidate_nodes_cache()
            logger.info(f'Removed node {node_url} due to heart beat expiration')

    def _check_node_health(self, node_url: str, headers: Dict) -> bool:
        """Check if a node is healthy."""
        try:
            response = requests.get(f'{node_url}/health', headers=headers)
            return response.status_code == 200
        except Exception:
            return False

    @property
    def model_list(self):
        """Supported model list."""
        return [model for status in self.nodes.values() for model in status.models]

    def _get_matched_nodes(self, model_name: str, role: EngineRole):
        """Get matched nodes and their speeds for the model."""
        nodes_with_speeds, speeds, nodes_without_speeds = [], [], []
        for node in self.get_nodes(role).values():
            if model_name in node.status.models:
                if node.status.speed is not None:
                    nodes_with_speeds.append(node)
                    speeds.append(node.status.speed)
                else:
                    nodes_without_speeds.append(node)

        if not nodes_with_speeds and not nodes_without_speeds:
            return None, None

        all_nodes = nodes_with_speeds + nodes_without_speeds
        avg_speed = sum(speeds) / len(speeds) if speeds else 1
        all_speeds = speeds + [avg_speed] * len(nodes_without_speeds)
        return all_nodes, all_speeds

    def get_node(self, model_name: str, role: EngineRole = EngineRole.Hybrid) -> Optional[Node]:
        """Get a node for the specified model and role."""
        if self.routing_strategy == RoutingStrategy.RANDOM:
            nodes, speeds = self._get_matched_nodes(model_name, role)
            if not nodes:
                return None
            weights = [s / sum(speeds) for s in speeds]
            return random.choices(nodes, weights=weights)[0]

        elif self.routing_strategy == RoutingStrategy.MIN_EXPECTED_LATENCY:
            nodes, speeds = self._get_matched_nodes(model_name, role)
            if not nodes:
                return None
            indexes = list(range(len(nodes)))
            random.shuffle(indexes)
            min_index = min(indexes, key=lambda i: nodes[i].status.unfinished / speeds[i])
            return nodes[min_index]

        elif self.routing_strategy == RoutingStrategy.MIN_OBSERVED_LATENCY:
            nodes, latencies = [], []
            for node in self.get_nodes(role).values():
                if model_name in node.status.models:
                    nodes.append(node)
                    latencies.append(np.mean(node.status.latency) if node.status.latency else float('inf'))
            if not nodes:
                return None
            return nodes[np.argmin(latencies)]

        else:
            raise ValueError(f'Invalid strategy: {self.routing_strategy}')

    def get_node_url(self, model_name: str, role: EngineRole = EngineRole.Hybrid) -> Optional[str]:
        """Get node URL."""
        node = self.get_node(model_name, role)
        return node.url if node else None
