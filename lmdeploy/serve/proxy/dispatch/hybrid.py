# Copyright (c) OpenMMLab. All rights reserved.

from fastapi.responses import JSONResponse

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.dispatch.base import (
    ProxyContext,
    model_not_found_response,
    replica_unavailable_response,
    response_from_api_exception,
    safe_json_load,
)
from lmdeploy.serve.proxy.metrics.load_tracker import InflightTracker
from lmdeploy.serve.proxy.routing.selector import ReplicaSelector
from lmdeploy.serve.proxy.streaming_response import ProxyStreamingResponse
from lmdeploy.serve.proxy.upstream.exceptions import APIServerException
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
            return model_not_found_response(ctx.model)

        logger.info(f'A request is dispatched to {replica_url}')
        start = self._tracker.start(replica_url)
        if start is None:
            return replica_unavailable_response(replica_url)

        if ctx.stream:
            response = self._forwarder.forward_raw_stream(ctx.raw_request, replica_url, ctx.endpoint)
            return ProxyStreamingResponse(
                response,
                raw_request=ctx.raw_request,
                on_complete=lambda: self._tracker.finish(replica_url, start),
                media_type='text/event-stream',
            )

        try:
            response = await self._forwarder.forward_raw_buffer(ctx.raw_request, replica_url, ctx.endpoint)
            return JSONResponse(safe_json_load(replica_url, response))
        except APIServerException as e:
            return response_from_api_exception(e)
        finally:
            self._tracker.finish(replica_url, start)
