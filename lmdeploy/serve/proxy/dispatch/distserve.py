# Copyright (c) OpenMMLab. All rights reserved.

import copy

from fastapi.responses import JSONResponse

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage
from lmdeploy.serve.proxy.core.config import ProxyConfig
from lmdeploy.serve.proxy.dispatch.base import (
    ProxyContext,
    model_not_found_response,
    replica_unavailable_response,
    response_from_api_exception,
    safe_json_load,
)
from lmdeploy.serve.proxy.metrics.load_tracker import InflightTracker
from lmdeploy.serve.proxy.registry.pool import ReplicaPool
from lmdeploy.serve.proxy.routing.selector import ReplicaSelector
from lmdeploy.serve.proxy.streaming_response import ProxyStreamingResponse
from lmdeploy.serve.proxy.upstream.exceptions import APIServerException
from lmdeploy.serve.proxy.upstream.forwarder import UpstreamForwarder
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class DistServeDispatcher:
    """Prefill-decode disaggregated dispatch."""

    def __init__(self, config: ProxyConfig, pool: ReplicaPool, selector: ReplicaSelector,
                 forwarder: UpstreamForwarder, tracker: InflightTracker) -> None:
        self._config = config
        self._pool = pool
        self._selector = selector
        self._forwarder = forwarder
        self._tracker = tracker

    @staticmethod
    def _build_prefill_request(request_dict: dict, endpoint: str) -> dict:
        prefill = copy.deepcopy(request_dict)
        prefill['stream'] = False
        prefill['with_cache'] = True
        prefill['preserve_cache'] = True
        if endpoint == '/v1/chat/completions':
            prefill['max_tokens'] = 1
            prefill['max_completion_tokens'] = 1
        else:
            prefill['max_tokens'] = 1
        return prefill

    async def dispatch(self, ctx: ProxyContext):
        request_dict = ctx.request_dict
        prefill_request = self._build_prefill_request(request_dict, ctx.endpoint)
        p_url = 'dummy:dummy'
        prefill_info: dict = {}

        if not self._config.dummy_prefill:
            p_url = self._selector.select(ctx.model, EngineRole.Prefill)
            if not p_url:
                return model_not_found_response(ctx.model)
            logger.info(f'A Prefill request is dispatched to {p_url}')
            start = self._tracker.start(p_url)
            if start is None:
                return replica_unavailable_response(p_url)
            try:
                prefill_response = await self._forwarder.forward_json_buffer(prefill_request, p_url, ctx.endpoint)
                prefill_info = safe_json_load(p_url, prefill_response)
            except APIServerException as e:
                return response_from_api_exception(e)
            finally:
                self._tracker.finish(p_url, start)

        d_url = self._selector.select(ctx.model, EngineRole.Decode)
        if not d_url:
            return model_not_found_response(ctx.model)
        logger.info(f'A Decode request is dispatched to {d_url}')

        if not self._config.dummy_prefill:
            if not self._pool.pd_connection_pool.is_connected(p_url, d_url):
                await self._pool.pd_connection_pool.connect(
                    PDConnectionMessage(
                        p_url=p_url,
                        d_url=d_url,
                        protocol=self._config.migration_protocol,
                        rdma_config=self._config.rdma_config,
                    ))

        remote_session_id = int(prefill_info.get('id')) if prefill_info.get('id') else 0
        remote_block_ids = prefill_info.get('cache_block_ids') or []
        remote_token_id = prefill_info.get('remote_token_ids')[-1] if prefill_info.get('remote_token_ids') else 0
        request_dict['migration_request'] = MigrationRequest(
            protocol=self._config.migration_protocol,
            remote_engine_id=p_url,
            remote_session_id=remote_session_id,
            remote_block_ids=remote_block_ids,
            remote_token_id=remote_token_id,
            is_dummy_prefill=self._config.dummy_prefill,
        ).model_dump(mode='json')

        start = self._tracker.start(d_url)
        if start is None:
            return replica_unavailable_response(d_url)

        if not self._config.dummy_prefill:
            self._pool.pd_connection_pool.shelf_prefill_session((p_url, d_url), prefill_info.get('id'))

        try:
            if ctx.stream:
                response = self._forwarder.forward_json_stream(request_dict, d_url, ctx.endpoint)
                resp = ProxyStreamingResponse(
                    response,
                    raw_request=ctx.raw_request,
                    on_complete=lambda: self._tracker.finish(d_url, start),
                    media_type='text/event-stream',
                )
            else:
                response = await self._forwarder.forward_json_buffer(request_dict, d_url, ctx.endpoint)
                resp = JSONResponse(safe_json_load(d_url, response))
                self._tracker.finish(d_url, start)
        except APIServerException as e:
            self._tracker.finish(d_url, start)
            if not self._config.dummy_prefill:
                self._pool.pd_connection_pool.unshelf_prefill_session((p_url, d_url), prefill_info.get('id'))
            return response_from_api_exception(e)

        if not self._config.dummy_prefill:
            self._pool.pd_connection_pool.unshelf_prefill_session((p_url, d_url), prefill_info.get('id'))
        return resp
