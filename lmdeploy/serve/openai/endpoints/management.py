# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import asyncio
import math
import os
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from lmdeploy.serve.openai.errors import create_error_response
from lmdeploy.serve.openai.protocol import (
    AbortRequest,
    DestroyWeightsUpdateGroupRequest,
    InitWeightsUpdateGroupRequest,
    UpdateParamsRequest,
    UpdateWeightsFromDistributedRequest,
    UpdateWeightsFromIPCRequest,
    UpdateWeightsFromIPCStatus,
)
from lmdeploy.serve.utils.server_utils import validate_json_request


def register(router: APIRouter, server_context) -> None:

    # These are deployment constraints, not an inference-framework profile.
    # Ordinary LMDeploy keeps every existing update transport available.
    protocols = frozenset(os.environ.get(
        'LMDEPLOY_WEIGHT_UPDATE_PROTOCOLS', 'legacy,ipc,distributed').split(','))
    if not protocols or not protocols <= {'legacy', 'ipc', 'distributed'}:
        raise ValueError('Invalid LMDEPLOY_WEIGHT_UPDATE_PROTOCOLS')
    require_id = os.environ.get('LMDEPLOY_REQUIRE_UPDATE_ID', '0')
    if require_id not in ('0', '1'):
        raise ValueError('LMDEPLOY_REQUIRE_UPDATE_ID must be 0 or 1')
    require_id = require_id == '1'

    def _check_update_protocol(protocol):
        if protocol not in protocols:
            return create_error_response(
                HTTPStatus.NOT_IMPLEMENTED,
                f'{protocol} weight updates are disabled by server configuration.')
        return None

    async def _apply_weights_update(owner, transfer, *, finished, require_sleep=True):
        """Serialize all transports through the same service-owned lifecycle.

        A timeout/cancellation cannot cancel copies already running on a worker.
        Keep the model unusable after any uncertain or partial write.
        """
        engine = server_context.async_engine
        try:
            timeout = float(os.environ.get('LMDEPLOY_WEIGHT_UPDATE_TIMEOUT_SECONDS', '180'))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('Weight update timeout must be positive and finite.')
            operation = engine.begin_weights_update(owner, require_sleep=require_sleep)
        except (ValueError, RuntimeError) as exc:
            return create_error_response(HTTPStatus.CONFLICT, str(exc))
        try:
            success, message = await asyncio.wait_for(transfer(), timeout=timeout)
            if not success:
                engine.fail_weights_update(message)
            elif finished:
                engine.complete_weights_update(owner, operation)
        except asyncio.TimeoutError:
            engine.fail_weights_update(f'Worker transfer exceeded {timeout} seconds')
            return create_error_response(HTTPStatus.GATEWAY_TIMEOUT, engine.weights_update_error)
        except BaseException as exc:
            engine.fail_weights_update(f'{type(exc).__name__}: {exc}')
            raise
        finally:
            engine.end_weights_update_operation(operation)
        return JSONResponse(content={'success': success, 'message': message},
                            status_code=HTTPStatus.OK if success else HTTPStatus.BAD_REQUEST)

    @router.get('/health')
    async def health() -> JSONResponse:
        """Health check."""
        error = getattr(server_context.async_engine, 'weights_update_error', None)
        if error is not None:
            return JSONResponse(dict(status='unhealthy', message=error),
                                status_code=HTTPStatus.SERVICE_UNAVAILABLE)
        monitor = server_context.health_monitor
        if monitor is None:
            data = dict(status='unhealthy',
                        message='Engine health monitor is not initialized.')
            return JSONResponse(jsonable_encoder(data),
                                status_code=HTTPStatus.SERVICE_UNAVAILABLE)
        data = monitor.snapshot()
        if data['status'] == 'unhealthy':
            data = await monitor.refresh_snapshot()
        status_code = HTTPStatus.OK if data['status'] in (
            'healthy', 'sleeping') else HTTPStatus.SERVICE_UNAVAILABLE
        return JSONResponse(jsonable_encoder(data), status_code=status_code)

    @router.get('/terminate')
    async def terminate():
        """Terminate server."""
        import signal

        if not server_context.allow_terminate_by_client:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'The server can not be terminated. Please add --allow-terminate-by-client when start the server.'
            )
        os.kill(os.getpid(), signal.SIGTERM)
        return Response(status_code=200)

    @router.post('/update_weights',
                 dependencies=[Depends(validate_json_request)])
    async def update_params(request: UpdateParamsRequest, raw_request: Request = None):
        """Load legacy serialized tensors, preserving the JSON-null response."""
        err = _check_update_protocol('legacy')
        if err is not None:
            return err

        async def transfer():
            await asyncio.to_thread(server_context.async_engine.engine.update_params, request)
            return True, 'Weights updated.'

        result = await _apply_weights_update((('legacy',), None), transfer,
                                             finished=request.finished, require_sleep=False)
        return JSONResponse(content=None) if result.status_code == HTTPStatus.OK else result

    def _check_pytorch_backend_for_disagg_weight_update():
        """Disaggregated weight-update endpoints are PyTorch-backend only for
        now."""
        backend = getattr(server_context.async_engine, 'backend', None)
        if backend != 'pytorch':
            return create_error_response(
                HTTPStatus.NOT_IMPLEMENTED,
                f'Disaggregated weight-update endpoints require backend="pytorch", got {backend!r}.'
            )
        return None

    @router.get('/update_weights_from_ipc', response_model=UpdateWeightsFromIPCStatus)
    async def get_update_weights_from_ipc_status():
        """Check whether checkpoint-engine can safely start an IPC update."""
        async_engine = server_context.async_engine
        backend = getattr(async_engine, 'backend', None)
        err = _check_update_protocol('ipc')
        if err is not None:
            return UpdateWeightsFromIPCStatus(
                ready=False, message='IPC updates are disabled by server configuration.',
                backend=backend, is_sleeping=async_engine.is_sleeping,
                sleeping_tags=sorted(async_engine.sleeping_tags))
        backend_config = getattr(async_engine, 'backend_config', None)
        device_type = getattr(backend_config, 'device_type', None)
        if backend != 'pytorch' or device_type != 'cuda':
            return UpdateWeightsFromIPCStatus(
                ready=False,
                message='checkpoint-engine IPC updates require backend="pytorch" and device_type="cuda".',
                backend=backend,
                device_type=device_type,
                is_sleeping=async_engine.is_sleeping,
                sleeping_tags=sorted(async_engine.sleeping_tags),
            )

        worker_statuses = await async_engine.engine.get_checkpoint_engine_status()
        worker_ready = bool(worker_statuses) and all(item['ready'] for item in worker_statuses)
        lifecycle_ready = async_engine.is_sleeping and 'kv_cache' in async_engine.sleeping_tags
        lifecycle_ready = lifecycle_ready and not (
            async_engine.weights_update_busy or async_engine.weights_update_pending
            or async_engine.weights_update_error is not None)
        ready = worker_ready and lifecycle_ready
        if not worker_ready:
            message = next((item['message'] for item in worker_statuses if not item['ready']),
                           'No checkpoint-engine workers are available.')
        elif not lifecycle_ready:
            message = ('Engine must be sleeping with kv_cache unavailable. For online updates call '
                       'POST /sleep, then POST /wakeup?tags=weights.')
        else:
            message = 'checkpoint-engine IPC receiver is ready.'
        versions = {item['checkpoint_engine_version'] for item in worker_statuses
                    if item['checkpoint_engine_version'] is not None}
        version = next(iter(versions)) if len(versions) == 1 else ','.join(sorted(versions)) or None
        return UpdateWeightsFromIPCStatus(
            ready=ready,
            message=message,
            backend=backend,
            device_type=device_type,
            checkpoint_engine_version=version,
            is_sleeping=async_engine.is_sleeping,
            sleeping_tags=sorted(async_engine.sleeping_tags),
            device_uuids=[item['device_uuid'] for item in worker_statuses],
            worker_ranks=[item['rank'] for item in worker_statuses],
        )

    @router.post('/update_weights_from_ipc',
                 dependencies=[Depends(validate_json_request)])
    async def update_weights_from_ipc(request: UpdateWeightsFromIPCRequest,
                                      raw_request: Request = None):
        """Receive weights from checkpoint-engine through CUDA IPC."""
        err = _check_update_protocol('ipc') or _check_pytorch_backend_for_disagg_weight_update()
        if err is not None:
            return err
        async_engine = server_context.async_engine
        device_type = getattr(async_engine.backend_config, 'device_type', None)
        if device_type != 'cuda':
            return create_error_response(
                HTTPStatus.NOT_IMPLEMENTED,
                f'checkpoint-engine IPC updates require device_type="cuda", got {device_type!r}.')

        return await _apply_weights_update(
            (('ipc',), None), lambda: async_engine.engine.update_weights_from_ipc(request, None),
            finished=True)

    @router.post('/init_weights_update_group',
                 dependencies=[Depends(validate_json_request)])
    async def init_weights_update_group(request: InitWeightsUpdateGroupRequest,
                                        raw_request: Request = None):
        """Initialize the torch.distributed process group used by an external
        trainer to broadcast weights into this rollout engine."""
        err = _check_pytorch_backend_for_disagg_weight_update()
        if err is not None:
            return err
        success, message = await server_context.async_engine.engine.init_weights_update_group(
            request)
        content = {'success': success, 'message': message}
        return JSONResponse(
            content=content,
            status_code=200 if success else HTTPStatus.BAD_REQUEST)

    @router.post(
        '/update_weights_from_distributed',
        dependencies=[Depends(validate_json_request)],
        description=('Receive a bucket of weights through a previously initialized weights-\n'
                     'update group and load them into the running model.'))
    async def update_weights_from_distributed(
            request: UpdateWeightsFromDistributedRequest,
            raw_request: Request = None):
        """Receive a bucket of weights through a previously initialized
        weights- update group and load them into the running model."""
        err = _check_pytorch_backend_for_disagg_weight_update()
        if err is not None:
            return err
        async_engine = server_context.async_engine
        err = _check_update_protocol('distributed')
        if err is not None:
            return err
        if require_id and not request.update_id:
            return create_error_response(HTTPStatus.CONFLICT, 'This server requires an explicit update_id.')
        owner = (request.group_name, request.update_id)
        aligned = len(request.names) == len(request.dtypes) == len(request.shapes)
        # Older senders may finalize with an empty bucket. Permit that only for
        # their already-open implicit transaction, never to initialize a model
        # or claim a new explicit transaction completed without any tensors.
        empty_finish = (request.update_id is None and request.finished
                        and async_engine.weights_update_pending
                        and async_engine.weights_update_owner == owner)
        if not aligned or (not request.names and not empty_finish):
            return create_error_response(HTTPStatus.CONFLICT, 'A nonempty, aligned weight bucket is required.')
        return await _apply_weights_update(
            owner, lambda: async_engine.engine.update_weights_from_distributed(request),
            finished=request.finished)

    @router.post('/destroy_weights_update_group',
                 dependencies=[Depends(validate_json_request)])
    async def destroy_weights_update_group(
            request: DestroyWeightsUpdateGroupRequest,
            raw_request: Request = None):
        """Tear down a previously initialized weights-update group."""
        err = _check_pytorch_backend_for_disagg_weight_update()
        if err is not None:
            return err
        success, message = await server_context.async_engine.engine.destroy_weights_update_group(
            request)
        content = {'success': success, 'message': message}
        return JSONResponse(
            content=content,
            status_code=200 if success else HTTPStatus.BAD_REQUEST)

    @router.post('/sleep', dependencies=[Depends(validate_json_request)])
    async def sleep(raw_request: Request = None):
        level = raw_request.query_params.get('level', '1')
        try:
            level = int(level)
        except (TypeError, ValueError):
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'The "level" query parameter must be an integer.')
        if level not in (1, 2):
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'The "level" query parameter must be 1 or 2.')
        async_engine = server_context.async_engine
        try:
            await async_engine.sleep(level)
        except RuntimeError as exc:
            return create_error_response(HTTPStatus.CONFLICT, str(exc))
        return Response(status_code=200)

    @router.post('/wakeup', dependencies=[Depends(validate_json_request)])
    async def wakeup(raw_request: Request = None):
        tags = raw_request.query_params.getlist('tags')
        tags = tags or None
        try:
            server_context.async_engine.wakeup(tags)
        except RuntimeError as exc:
            return create_error_response(HTTPStatus.CONFLICT, str(exc))
        return Response(status_code=200)

    @router.get('/is_sleeping')
    async def is_sleeping():
        engine = server_context.async_engine
        return JSONResponse(content={'is_sleeping': engine.is_sleeping,
                                     'weights_update_pending': engine.weights_update_pending,
                                     'weights_content_ready': engine.weights_content_ready,
                                     'weights_update_owner': engine.weights_update_owner,
                                     'weights_update_error': engine.weights_update_error})

    @router.post('/abort_request')
    async def abort_request(request: AbortRequest,
                            raw_request: Request = None):
        """Abort an ongoing request."""
        if not server_context.enable_abort_handling:
            return Response(
                status_code=501,
                content=
                'This server does not support abort requests. Enable with --enable-abort-handling flag.'
            )

        if request.abort_all:
            await server_context.async_engine.stop_all_session()
        else:
            session = server_context.find_session(request.session_id)
            if session is None:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    f'Session {request.session_id} not found.')
            await session.async_abort()
            session_mgr = server_context.session_manager
            session_mgr.remove(session)
        return Response(status_code=200)
