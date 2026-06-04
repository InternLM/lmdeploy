# Copyright (c) OpenMMLab. All rights reserved.

import aiohttp

from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.serve.proxy.core.config import ProxyConfig
from lmdeploy.serve.proxy.dispatch.distserve import DistServeDispatcher
from lmdeploy.serve.proxy.dispatch.hybrid import HybridDispatcher
from lmdeploy.serve.proxy.metrics.load_tracker import InflightTracker
from lmdeploy.serve.proxy.registry.heartbeat import start_heartbeat
from lmdeploy.serve.proxy.registry.pool import ReplicaPool
from lmdeploy.serve.proxy.routing.selector import ReplicaSelector
from lmdeploy.serve.proxy.upstream.forwarder import UpstreamForwarder


class ProxyRuntime:
    """Wired dependencies for proxy request handling."""

    def __init__(self, config: ProxyConfig, session: aiohttp.ClientSession) -> None:
        self.config = config
        self.session = session
        self.pool = ReplicaPool(PDConnectionPool())
        start_heartbeat(self.pool)
        self.selector = ReplicaSelector(self.pool, config.routing_strategy)
        self.forwarder = UpstreamForwarder(session)
        self.tracker = InflightTracker(self.pool)
        self.hybrid = HybridDispatcher(self.selector, self.forwarder, self.tracker)
        self.distserve = DistServeDispatcher(config, self.pool, self.selector, self.forwarder, self.tracker)
