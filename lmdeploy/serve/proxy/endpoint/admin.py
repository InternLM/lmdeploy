# Copyright (c) OpenMMLab. All rights reserved.

from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.proxy.core.replica import ReplicaRegistration
from lmdeploy.serve.proxy.runtime import ProxyRuntime
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def get_runtime(request: Request) -> ProxyRuntime:
    return request.app.state.runtime


router = APIRouter()


@router.get('/nodes/status')
def replica_status(runtime: ProxyRuntime = Depends(get_runtime)):
    """Show replica status."""
    try:
        return runtime.pool.snapshot()
    except Exception as e:
        logger.error(f'failed to get replica status: {e}')
        return False


@router.post('/nodes/add', dependencies=[Depends(validate_json_request)])
def add_replica(registration: ReplicaRegistration, runtime: ProxyRuntime = Depends(get_runtime)):
    """Add a replica to the pool."""
    try:
        res = runtime.pool.add(registration.url, registration.status)
        if res is not None:
            logger.error(f'add replica {registration.url} failed, {res}')
            return res
        logger.info(f'add replica {registration.url} successfully')
        return 'Added successfully'
    except Exception as e:
        logger.error(f'add replica {registration.url} failed: {e}')
        return 'Failed to add, please check the input url.'


@router.post('/nodes/remove', dependencies=[Depends(validate_json_request)])
def remove_replica(registration: ReplicaRegistration, runtime: ProxyRuntime = Depends(get_runtime)):
    """Remove a replica from the pool."""
    try:
        runtime.pool.remove(registration.url)
        logger.info(f'delete replica {registration.url} successfully')
        return 'Deleted successfully'
    except Exception as e:
        logger.error(f'delete replica {registration.url} failed: {e}')
        return 'Failed to delete, please check the input url.'


@router.post('/nodes/terminate', dependencies=[Depends(validate_json_request)])
def terminate_replica(registration: ReplicaRegistration, runtime: ProxyRuntime = Depends(get_runtime)):
    """Terminate a remote api_server and remove it from the pool."""
    try:
        url = registration.url
        success = runtime.pool.terminate(url)
        if not success:
            return f'Failed to terminate replica {url}'
        return 'Terminated successfully'
    except Exception as e:
        logger.error(f'Terminate replica {registration.url} failed: {e}')
        return f'Failed to terminate replica {registration.url}, please check the input url.'


@router.get('/nodes/terminate_all', dependencies=[Depends(validate_json_request)])
def terminate_all_replicas(runtime: ProxyRuntime = Depends(get_runtime)):
    """Terminate all replicas."""
    try:
        success = runtime.pool.terminate_all()
        if not success:
            return 'Failed to terminate all replicas'
        return 'All replicas terminated successfully'
    except Exception as e:
        logger.error(f'Failed to terminate all replicas: {e}')
        return 'Failed to terminate all replicas.'
