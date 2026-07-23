# Copyright (c) OpenMMLab. All rights reserved.

from http import HTTPStatus

import requests
from fastapi import APIRouter, Depends, HTTPException, Request

from lmdeploy.serve.proxy.core.replica import ReplicaLoad, ReplicaRegistration
from lmdeploy.serve.proxy.registry.pool import ReplicaNotFoundError
from lmdeploy.serve.proxy.runtime import ProxyRuntime
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def get_runtime(request: Request) -> ProxyRuntime:
    return request.app.state.runtime


def _success(message: str) -> dict[str, str]:
    return {'message': message}


def _discover_models(url: str) -> list[str]:
    """Fetch model ids from replica GET /v1/models."""
    from lmdeploy.serve.openai.api_client import APIClient

    try:
        models = APIClient(api_server_url=url).available_models
    except requests.exceptions.RequestException as e:
        logger.error(f'failed to probe {url}/v1/models: {e}')
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f'Failed to reach replica at {url}.',
        ) from e
    if not models:
        logger.error(f'replica {url} returned no models from /v1/models')
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f'No models returned from {url}/v1/models.',
        )
    return list(models)


def _build_replica_load(status: ReplicaLoad | None, discovered_models: list[str]) -> ReplicaLoad:
    entry = status.model_copy(deep=True) if status is not None else ReplicaLoad()
    entry.models = discovered_models
    return entry


def _validate_declared_models(url: str, declared: list[str], discovered: list[str]) -> None:
    if not declared:
        return
    missing = set(declared) - set(discovered)
    if missing:
        detail = (f'Declared models {sorted(missing)} not found in /v1/models '
                  f'for {url}: {discovered}')
        logger.error(detail)
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=detail)


def _request_replica_terminate(url: str) -> None:
    try:
        response = requests.get(f'{url}/terminate', headers={'accept': 'application/json'})
    except requests.exceptions.RequestException as e:
        logger.error(f'exception happened when terminating replica {url}, {e}')
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f'Failed to terminate replica {url}.',
        ) from e
    if response.status_code != HTTPStatus.OK:
        logger.error(f'Failed to terminate replica {url}, '
                     f'error_code={response.status_code}, error_msg={response.text}')
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f'Failed to terminate replica {url}.',
        )


router = APIRouter()


@router.get('/nodes/status')
def replica_status(runtime: ProxyRuntime = Depends(get_runtime)):
    """Show replica status."""
    return runtime.pool.snapshot()


@router.post('/nodes/add', dependencies=[Depends(validate_json_request)])
def add_replica(registration: ReplicaRegistration, runtime: ProxyRuntime = Depends(get_runtime)):
    """Add a replica to the pool."""
    url = registration.url
    discovered_models = _discover_models(url)
    _validate_declared_models(url, (registration.status or ReplicaLoad()).models, discovered_models)
    runtime.pool.add(url, _build_replica_load(registration.status, discovered_models))
    logger.info(f'add replica {url} successfully')
    return _success('Added successfully')


@router.post('/nodes/remove', dependencies=[Depends(validate_json_request)])
def remove_replica(registration: ReplicaRegistration, runtime: ProxyRuntime = Depends(get_runtime)):
    """Remove a replica from the pool."""
    url = registration.url
    try:
        runtime.pool.remove(url)
    except ReplicaNotFoundError as e:
        logger.error(f'delete replica {url} failed since it does not exist. '
                     'May try /nodes/status to check the replica list')
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e)) from e
    logger.info(f'delete replica {url} successfully')
    return _success('Deleted successfully')


@router.post('/nodes/terminate', dependencies=[Depends(validate_json_request)])
def terminate_replica(registration: ReplicaRegistration, runtime: ProxyRuntime = Depends(get_runtime)):
    """Terminate a remote api_server and remove it from the pool."""
    url = registration.url
    try:
        runtime.pool.remove(url)
    except ReplicaNotFoundError as e:
        logger.error(f'terminating replica {url} failed since it does not exist. '
                     'May try /nodes/status to check the replica list')
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e)) from e
    _request_replica_terminate(url)
    return _success('Terminated successfully')


@router.get('/nodes/terminate_all', dependencies=[Depends(validate_json_request)])
def terminate_all_replicas(runtime: ProxyRuntime = Depends(get_runtime)):
    """Terminate all replicas."""
    urls = list(runtime.pool.snapshot().keys())
    failures: list[str] = []
    for url in urls:
        try:
            runtime.pool.remove(url)
            _request_replica_terminate(url)
        except ReplicaNotFoundError:
            continue
        except HTTPException:
            failures.append(url)
        except Exception as e:
            logger.error(f'Failed to terminate replica {url}: {e}')
            failures.append(url)
    if failures:
        logger.error(f'Failed to terminate replicas: {failures}')
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail='Failed to terminate all replicas.',
        )
    return _success('All replicas terminated successfully')
