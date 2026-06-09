# Copyright (c) OpenMMLab. All rights reserved.

from fastapi.responses import JSONResponse

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.dispatch.base import (
    ProxyContext,
    model_not_found_response,
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
        selected = self._selector.acquire(ctx.model, EngineRole.Hybrid)
        if selected is None:
            return model_not_found_response(ctx.model)

        logger.info(f'A request is dispatched to {selected.url}')
        if ctx.stream:
            response = self._forwarder.forward_raw_stream(ctx.raw_request, selected.url, ctx.endpoint)
            return ProxyStreamingResponse(
                response,
                raw_request=ctx.raw_request,
                on_complete=lambda: self._tracker.finish(selected),
                media_type='text/event-stream',
            )

        try:
            response = await self._forwarder.forward_raw_buffer(ctx.raw_request, selected.url, ctx.endpoint)
            if response is None:
                logger.info('client disconnected during proxy request; upstream cancelled')
                return
            return JSONResponse(safe_json_load(selected.url, response))
        except APIServerException as e:
            return response_from_api_exception(e)
        finally:
            self._tracker.finish(selected)
