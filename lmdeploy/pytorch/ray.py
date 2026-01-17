# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
from typing import Dict, List

import ray
from ray.util.placement_group import PlacementGroup

from lmdeploy.pytorch.devices import get_device_manager
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
PG_WAIT_TIMEOUT = 1800


def get_device_str(device_type: str = None) -> str:
    """Get device str."""
    device_type = device_type or get_device_manager().current_context().device_type
    if device_type in ['cuda', 'maca']:
        device_type = 'GPU'
    elif device_type == 'ascend':
        device_type = 'NPU'
    elif device_type == 'camb':
        device_type = 'MLU'
    else:
        raise ValueError(f'Unsupported device type: {device_type}')

    return device_type


def get_resource_kwargs(device_str: str, resource_used: float = 0.01) -> Dict[str, float]:
    """Get resource kwargs."""
    if device_str == 'GPU':
        resource_kwargs = {'num_gpus': resource_used}
    elif device_str == 'NPU':
        resource_kwargs = {'resources': {device_str: resource_used}}
    else:
        raise ValueError(f'Unsupported device type: {device_str}')
    return resource_kwargs


def _wait_until_pg_ready(current_placement_group: PlacementGroup):
    """Wait until a placement group is ready.

    It prints the informative log messages if the placement group is not created within time.
    """
    # copy from vLLM
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    placement_group_specs = current_placement_group.bundle_specs

    s = time.time()
    pg_ready_ref = current_placement_group.ready()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        if len(ready) > 0:
            break

        # Exponential backoff for warning print.
        wait_interval *= 2
        logger.info(
            'Waiting for creating a placement group of specs for '
            '%d seconds. specs=%s. Check '
            '`ray status` to see if you have enough resources,'
            ' and make sure the IP addresses used by ray cluster'
            ' are the same as VLLM_HOST_IP environment variable'
            ' specified in each node if you are running on a multi-node.', int(time.time() - s), placement_group_specs)

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        raise ValueError('Cannot provide a placement group of '
                         f'{placement_group_specs=} within {PG_WAIT_TIMEOUT} seconds. See '
                         '`ray status` to make sure the cluster has enough resources.') from None


def _get_obj_store_memory(dp: int = 1):
    """Get obj store memory."""
    import psutil
    DEFAULT_OBJECT_STORE_MEMORY_PROPORTION = os.getenv('RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION', '0.3')
    DEFAULT_OBJECT_STORE_MEMORY_PROPORTION = float(DEFAULT_OBJECT_STORE_MEMORY_PROPORTION)
    DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES = os.getenv('RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES', None)
    if DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES is None:
        DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES = 80 * (10**9)
    else:
        DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES = int(DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES)
    total_mem = psutil.virtual_memory().total
    obj_store_mem = int(total_mem * DEFAULT_OBJECT_STORE_MEMORY_PROPORTION)
    obj_store_mem = min(DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES, obj_store_mem)
    if dp > 1:
        obj_store_mem = obj_store_mem // min(8, dp)
    return obj_store_mem


def init_ray_cluster(world_size: int, ray_address: str = None, dp: int = 1, device_type: str = 'cuda'):
    """Init ray cluster."""
    # modifier from vLLM
    if not ray.is_initialized():
        try:
            num_cpus = world_size
            object_store_memory = _get_obj_store_memory(dp=dp)
            ray.init(address=ray_address,
                     ignore_reinit_error=True,
                     num_cpus=num_cpus,
                     object_store_memory=object_store_memory)
        except ValueError as e:
            if e.args is not None and len(e.args) >= 1 and e.args[
                    0] == 'When connecting to an existing cluster, num_cpus and num_gpus must not be provided.':
                ray.init(address=ray_address, ignore_reinit_error=True)
            else:
                raise

    device_str = get_device_str(device_type)

    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    owned_pg = False
    if not current_placement_group:
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        if world_size > num_devices_in_cluster:
            logger.warning(
                'The number of required %ss exceeds the total '
                'number of available %ss in the placement group.', device_str, device_str)
        # Create a new placement group
        placement_group_specs: List[Dict[str, float]] = ([{device_str: 1.0} for _ in range(world_size)])

        # Pin at least one bundle to the local node.
        # This helps multi-node DP keep each dp_rank process's workers co-located with
        # the node where the process is launched.
        current_ip = ray.util.get_node_ip_address()
        placement_group_specs[0][f'node:{current_ip}'] = 0.001

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(placement_group_specs, strategy='PACK')
        _wait_until_pg_ready(current_placement_group)
        owned_pg = True

    assert current_placement_group is not None
    # Set the placement group in the parallel config
    placement_group = current_placement_group
    return placement_group, owned_pg


class RayContext:
    """Context manager for Ray."""

    def __init__(self, world_size: int, ray_address: str = None, dp: int = 1, device_type: str = 'cuda'):
        """Initialize Ray context."""
        placement_group, owned_pg = init_ray_cluster(world_size=world_size,
                                                     ray_address=ray_address,
                                                     dp=dp,
                                                     device_type=device_type)

        self.placement_group = placement_group
        self.owned_pg = owned_pg

    def get_placement_group(self):
        """Get the placement group."""
        return self.placement_group

    def shutdown(self):
        """Shutdown Ray."""
        if self.owned_pg:
            ray.util.remove_placement_group(self.placement_group)
            logger.debug('RayContext placement group removed.')

        if ray.is_initialized():
            try:
                ray.shutdown()
                logger.debug('Ray shutdown.')
            except Exception:
                logger.exception('Error during Ray shutdown.')
        else:
            logger.debug('Ray is not initialized, skipping shutdown.')
