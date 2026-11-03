"""Miles control protocol for synchronous, same-node TP=1 PPO.

Generation remains LMDeploy's native /generate. This adapter owns admission,
transactional weight versions and memory handoff; CUDA tensors are decoded by
the existing LMDeploy loader. It does not implement in-place pauses.
"""

import asyncio
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from lmdeploy.serve.openai.protocol import UpdateParamsRequest
from lmdeploy.utils import serialize_state_dict
from lmdeploy.version import __version__


@dataclass
class MilesState:
    weight_version: str = '0'
    paused: bool = False
    updating: bool = False
    finalized: bool = False
    received_buckets: int = 0
    active: int = 0
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def drain(self):
        async with self.condition:
            self.paused = True
            await self.condition.wait_for(lambda: self.active == 0)


class MilesAdmissionMiddleware:
    def __init__(self, app, *, state, context):
        self.app, self.state, self.context = app, state, context

    async def __call__(self, scope, receive, send):
        generation_paths = {'/generate', '/v1/completions', '/v1/chat/completions'}
        if scope['type'] != 'http' or scope.get('path') not in generation_paths:
            return await self.app(scope, receive, send)
        state = self.state
        async with state.condition:
            if state.paused or state.updating or self.context.async_engine.is_sleeping:
                response = JSONResponse({'error': 'Generation is paused for a training/weight-update phase.'}, 503)
                return await response(scope, receive, send)
            state.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            async with state.condition:
                state.active -= 1
                state.condition.notify_all()


def install_miles_protocol(app, context, *, model_path):
    config = context.engine_config
    if context.async_engine.backend != 'pytorch' or config.tp != 1:
        raise ValueError('Miles compatibility currently supports only the PyTorch TP=1 engine')
    if config.logprobs_mode != 'temperature_logprobs':
        raise ValueError('Miles compatibility requires logprobs_mode="temperature_logprobs"')
    state = MilesState()
    context.miles_state = state
    app.add_middleware(MilesAdmissionMiddleware, state=state, context=context)
    router = APIRouter()
    _register_status(router, context, state, model_path)
    _register_memory(router, context, state)
    _register_weights(router, context, state)
    app.include_router(router)


def _register_status(router, context, state, model_path):
    @router.get('/health_generate')
    async def health():
        monitor = context.health_monitor
        if monitor is not None and monitor.snapshot()['status'] == 'unhealthy':
            raise HTTPException(503, 'LMDeploy engine is unhealthy')
        return {'status': 'healthy'}

    @router.get('/model_info')
    async def model_info():
        return {'model_path': model_path, 'weight_version': state.weight_version}

    @router.get('/get_server_info')
    async def server_info():
        return {
            'backend': 'lmdeploy', 'version': __version__,
            'weight_version': state.weight_version,
            'server_args': {'model_path': model_path, 'tp_size': 1, 'dp_size': 1,
                            'pp_size': 1, 'nnodes': 1, 'disaggregation_mode': 'null'},
            'internal_states': [{'weight_version': state.weight_version,
                                 'num_running_reqs': state.active}],
        }


def _register_memory(router, context, state):
    engine = context.async_engine

    @router.post('/pause_generation')
    async def pause(payload: dict):
        if payload.get('mode', 'retract') == 'in_place':
            raise HTTPException(400, 'In-place pause is not supported')
        await state.drain()
        return {'success': True}

    @router.post('/continue_generation')
    async def resume():
        if state.updating or state.finalized:
            raise HTTPException(409, 'Weights must be finalized and their version committed before resuming')
        state.paused = False
        return {'success': True}

    @router.get('/flush_cache')
    async def flush():
        if state.active:
            raise HTTPException(409, 'Cannot flush an active generation batch')
        if not engine.is_sleeping:
            # Sleep discards KV cache, including the prefix cache.
            await engine.sleep(level=1)
            engine.wakeup()
        return {'success': True}

    @router.post('/release_memory_occupation')
    async def offload(payload: dict):
        await state.drain()
        if engine.sleeping_tags != {'weights', 'kv_cache'}:
            await engine.sleep(level=2)
        return {'success': True}

    @router.post('/resume_memory_occupation')
    async def onload(payload: dict):
        requested = set(payload.get('tags') or ['weights', 'kv_cache'])
        # Graph allocations are rebuilt when LMDeploy's KV cache wakes up.
        tags = sorted(requested & engine.sleeping_tags & {'weights', 'kv_cache'})
        if tags:
            engine.wakeup(tags)
        if not engine.is_sleeping and not state.updating and not state.finalized:
            state.paused = False
        return {'success': True}


def _register_weights(router, context, state):
    engine = context.async_engine

    @router.post('/begin_weight_update')
    async def begin(payload: dict):
        if not payload.get('sync_base', True) or payload.get('selector', 'all') != 'all':
            raise HTTPException(400, 'Only full base-model updates are supported')
        if state.updating:
            raise HTTPException(409, 'A weight update is already open')
        await state.drain()
        if 'kv_cache' not in engine.sleeping_tags:
            await engine.sleep(level=2)
        if 'weights' in engine.sleeping_tags:
            engine.wakeup(['weights'])
        state.updating, state.finalized = True, False
        state.received_buckets = 0
        return {'success': True}

    @router.post('/update_weights_from_tensor')
    async def update(payload: dict):
        if not state.updating:
            raise HTTPException(409, 'No weight-update transaction is open')
        if payload.get('load_format') not in (None, 'safetensors_file'):
            raise HTTPException(400, 'Expected named tensors or a local safetensors bucket')
        engine.engine.update_params(UpdateParamsRequest(
            serialized_named_tensors=payload['serialized_named_tensors'],
            load_format=payload.get('load_format'), finished=False))
        state.received_buckets += 1
        return {'success': True}

    @router.post('/end_weight_update')
    async def end(payload: dict):
        if not state.updating or not state.received_buckets:
            raise HTTPException(409, 'Cannot finalize an empty or missing weight update')
        if payload.get('expected_lora_checksums'):
            raise HTTPException(400, 'LoRA is not supported')
        engine.engine.update_params(UpdateParamsRequest(
            serialized_named_tensors=serialize_state_dict({}), finished=True))
        engine.complete_weights_update()
        state.updating, state.finalized = False, True
        return {'success': True}

    @router.post('/update_weight_version')
    async def version(payload: dict):
        if not state.finalized or not state.paused:
            raise HTTPException(409, 'A finalized, paused weight update is required')
        state.weight_version = str(payload['new_version'])
        state.finalized = False
        return {'success': True}
