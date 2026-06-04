# Copyright (c) OpenMMLab. All rights reserved.

from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.dispatch.base import ProxyContext, safe_json_load, unavailable_model_bytes
from lmdeploy.serve.proxy.metrics.load_tracker import InflightTracker
from lmdeploy.serve.proxy.routing.selector import ReplicaSelector
from lmdeploy.serve.proxy.streaming_response import ProxyStreamingResponse
from lmdeploy.serve.proxy.upstream.forwarder import UpstreamForwarder
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class HybridDispatcher:
    """Single-replica pass-through dispatch."""

    def __init__(self, selector: ReplicaSelector, forwarder: UpstreamForwarder,
                 tracker: InflightTracker) -> None:
        self._selector = selector
        self._forwarder = forwarder
        self._tracker = tracker

    async def dispatch(self, ctx: ProxyContext):
        replica_url = self._selector.select(ctx.model, EngineRole.Hybrid)
        if not replica_url:
            return unavailable_model_bytes(ctx.model)

        logger.info(f'A request is dispatched to {replica_url}')
        start = self._tracker.start(replica_url)
        if start is None:
            return self._forwarder.api_timeout_bytes(replica_url)

        if ctx.stream:
            response = self._forwarder.forward_raw_stream(ctx.raw_request, replica_url, ctx.endpoint)
            background = BackgroundTasks()
            background.add_task(self._tracker.finish, replica_url, start)
            return ProxyStreamingResponse(response, background=background, media_type='text/event-stream')

        response = await self._forwarder.forward_raw_buffer(ctx.raw_request, replica_url, ctx.endpoint)
        self._tracker.finish(replica_url, start)
        return JSONResponse(safe_json_load(replica_url, response))
