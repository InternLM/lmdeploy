# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

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

    @router.get('/health')
    async def health() -> JSONResponse:
        """Health check."""
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
    def update_params(request: UpdateParamsRequest,
                      raw_request: Request = None):
        """Update weights for the model."""
        server_context.async_engine.engine.update_params(request)
        return JSONResponse(content=None)

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
        topology_keys = ('world_size', 'tp', 'dp', 'dp_rank', 'ep')
        topologies = {tuple(item[key] for key in topology_keys) for item in worker_statuses}
        topology_ready = len(topologies) == 1
        topology = dict(zip(topology_keys, next(iter(topologies)))) if topology_ready else {}
        lifecycle_ready = async_engine.is_sleeping and 'kv_cache' in async_engine.sleeping_tags
        ready = worker_ready and topology_ready and lifecycle_ready
        if not worker_ready:
            message = next((item['message'] for item in worker_statuses if not item['ready']),
                           'No checkpoint-engine workers are available.')
        elif not topology_ready:
            message = 'LMDeploy workers reported inconsistent distributed topology.'
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
            **topology,
        )

    @router.post('/update_weights_from_ipc',
                 dependencies=[Depends(validate_json_request)])
    async def update_weights_from_ipc(request: UpdateWeightsFromIPCRequest,
                                      raw_request: Request = None):
        """Receive weights from checkpoint-engine through CUDA IPC."""
        err = _check_pytorch_backend_for_disagg_weight_update()
        if err is not None:
            return err
        async_engine = server_context.async_engine
        device_type = getattr(async_engine.backend_config, 'device_type', None)
        if device_type != 'cuda':
            return create_error_response(
                HTTPStatus.NOT_IMPLEMENTED,
                f'checkpoint-engine IPC updates require device_type="cuda", got {device_type!r}.')

        reject_reason = None
        if not async_engine.is_sleeping or 'kv_cache' not in async_engine.sleeping_tags:
            reject_reason = ('Engine is not prepared for a checkpoint-engine update. Call POST /sleep and '
                             'POST /wakeup?tags=weights before updating online.')
        success, message = await async_engine.engine.update_weights_from_ipc(request, reject_reason)
        if success:
            async_engine.complete_weights_update()
        return JSONResponse(
            content={'success': success, 'message': message},
            status_code=HTTPStatus.OK if success else HTTPStatus.BAD_REQUEST)

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
        success, message = await server_context.async_engine.engine.update_weights_from_distributed(
            request)
        content = {'success': success, 'message': message}
        return JSONResponse(
            content=content,
            status_code=200 if success else HTTPStatus.BAD_REQUEST)

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
        await async_engine.sleep(level)
        return Response(status_code=200)

    @router.post('/wakeup', dependencies=[Depends(validate_json_request)])
    async def wakeup(raw_request: Request = None):
        tags = raw_request.query_params.getlist('tags')
        tags = tags or None
        server_context.async_engine.wakeup(tags)
        return Response(status_code=200)

    @router.get('/is_sleeping')
    async def is_sleeping():
        is_sleeping = server_context.async_engine.is_sleeping
        return JSONResponse(content={'is_sleeping': is_sleeping})

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
