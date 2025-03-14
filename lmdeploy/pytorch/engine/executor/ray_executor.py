# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import ray
import ray.exceptions
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, ModelConfig
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.utils import get_logger

from .base import ExecutorBase
from .base_worker import WorkerWrapperBase
from .dist_utils import find_available_port

logger = get_logger('lmdeploy')

PG_WAIT_TIMEOUT = 1800


def get_device_str():
    """get device str."""
    device_type = get_device_manager().current_context().device_type

    if device_type == 'cuda':
        device_type = 'GPU'

    return device_type


def _wait_until_pg_ready(current_placement_group: 'PlacementGroup'):
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


def init_ray_cluster(world_size: int, ray_address: str = None):
    """init ray cluster."""
    # modifier from vLLM
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)

    device_str = get_device_str()

    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if not current_placement_group:
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        if world_size > num_devices_in_cluster:
            logger.warning(
                'The number of required %ss exceeds the total '
                'number of available %ss in the placement group.', device_str, device_str)
        # Create a new placement group
        placement_group_specs: List[Dict[str, float]] = ([{device_str: 1.0} for _ in range(world_size)])

        # gcs_addr = ray.get_runtime_context().gcs_address
        # master_addr = gcs_addr.split(':')[0]
        # current_ip = master_addr
        # # This way, at least bundle is required to be created in a current
        # # node.
        # placement_group_specs[0][f'node:{current_ip}'] = 0.001

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(placement_group_specs, strategy='PACK')
        _wait_until_pg_ready(current_placement_group)

    assert current_placement_group is not None
    # Set the placement group in the parallel config
    placement_group = current_placement_group
    return placement_group


def _get_master_addr():
    """get master addr."""
    gcs_addr = ray.get_runtime_context().gcs_address
    master_addr = gcs_addr.split(':')[0]
    return master_addr


def _get_master_port():
    """get master port."""
    return find_available_port()


class RayWorkerWrapper(WorkerWrapperBase):
    """worker wrapper."""

    def __init__(
        self,
        model_path: str,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        dist_config: DistConfig,
        adapters: Dict[str, str] = None,
        device_type: str = 'cuda',
        dtype: str = 'auto',
        log_level: int = 30,
    ):
        from lmdeploy.tokenizer import Tokenizer
        tokenizer = Tokenizer(model_path).model.model
        model_config = ModelConfig.from_pretrained(model_path, dtype=dtype, dist_config=dist_config)
        super().__init__(
            model_path=model_path,
            cache_config=cache_config,
            backend_config=backend_config,
            model_config=model_config,
            dist_config=dist_config,
            adapters=adapters,
            device_type=device_type,
            tokenizer=tokenizer,
            log_level=log_level,
        )
        self.node_ip = ray.util.get_node_ip_address()

    def get_node_ip(self):
        """get worker ip."""
        return self.node_ip

    def warmup_dist(self):
        # None default CUDA_VISIBLE_DEVICES might leads to slow first time all_reduce
        # WHY?
        logger.debug('Warmup all_reduce.')
        import torch

        from lmdeploy.pytorch.distributed import all_reduce, get_dist_manager
        with get_dist_manager().context(self.dist_ctx):
            tmp = torch.empty((1, ), device='cuda')
            all_reduce(tmp)

    def pack_output(self, output: Dict):
        """pack output."""
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                output[k] = v.numpy()
        return output

    def exit(self):
        """exit actor."""
        ray.actor.exit_actor()


class RayExecutor(ExecutorBase):
    """ray executor."""

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 backend_config: BackendConfig,
                 dist_config: DistConfig,
                 tokenizer: Any,
                 adapters: Dict[str, str] = None,
                 device_type: str = 'cuda',
                 dtype: str = 'auto'):
        """initialize Executor."""
        super().__init__(model_path=model_path,
                         model_config=model_config,
                         cache_config=cache_config,
                         backend_config=backend_config,
                         dist_config=dist_config,
                         tokenizer=tokenizer,
                         adapters=adapters,
                         device_type=device_type)

        self.dp_rank = dist_config.dp_rank
        device_ctx = DeviceContext(device_type)
        with get_device_manager().context(device_ctx):
            logger.info('Init ray cluster.')
            ray_world_size = self.world_size
            if self.dp > 1:
                ray_world_size = 1
            placement_group = init_ray_cluster(ray_world_size)
        self.placement_group = placement_group

        if self.dp == 1:
            self.master_addr = _get_master_addr()
            self.master_port = _get_master_port()
        else:
            self.master_addr = os.environ.get('LMDEPLOY_DP_MASTER_ADDR', None)
            self.master_port = os.environ.get('LMDEPLOY_DP_MASTER_PORT', None)
            if self.master_addr is None or self.master_port is None:
                raise RuntimeError('DP > 1 requires "LMDEPLOY_DP_MASTER_ADDR" and "LMDEPLOY_DP_MASTER_PORT".')

        # create workerwrapper actors
        worker_kwargs = dict(
            model_path=model_path,
            cache_config=cache_config,
            backend_config=backend_config,
            dist_config=dist_config,
            adapters=adapters,
            device_type=device_type,
            dtype=dtype,
            log_level=logger.level,
        )

        logger.info('Init ray workers.')
        self.workers = self._init_workers_ray(placement_group, worker_kwargs)
        self.dag = None
        self._prefetch_task: asyncio.Task = None
        self.remote_outs: asyncio.Queue = None

        logger.info('Init distributed process group.')
        if self.dp == 1:
            ray.get([
                worker.init_process_group.remote(rank, self.master_addr, self.master_port)
                for rank, worker in enumerate(self.workers)
            ])
        else:
            ray.get(self.workers[0].init_process_group.remote(self.dp_rank, self.master_addr, self.master_port))

        if self.dist_config.world_size > 1:
            logger.info('Warming up distribute environment, this might take long time, please waiting...')
            ray.get([worker.warmup_dist.remote() for worker in self.workers])

    def collective_rpc(self, method: str, args: Tuple[Any] = None, kwargs: Dict[str, Any] = None):
        """collective rpc."""
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        return ray.get([getattr(worker, method).remote(*args, **kwargs) for worker in self.workers])

    def build_model(self):
        """build model."""
        self.collective_rpc('build_model')

    def gather_free_mem(self):
        """gather available memory."""
        return self.collective_rpc('get_free_mem')

    def set_cache_config(self, cache_config: CacheConfig):
        """set all cache config."""
        self.collective_rpc('set_cache_config', (cache_config, ))

    def set_model_config(self, model_config: ModelConfig):
        """set all model config."""
        self.collective_rpc('set_model_config', (model_config, ))

    def build_graph_runner(self):
        """build graph runner."""
        self.collective_rpc('build_graph_runner')

    def build_cache_engine(self):
        """build cache engine."""
        self.collective_rpc('build_cache_engine')

    def warmup(self):
        """build cache engine."""
        self.collective_rpc('warmup')

    def get_input_processor(self):
        """build cache engine."""
        return ray.get(self.workers[0].get_input_processor.remote())

    async def _prefetch_outputs(self):
        while True:
            outs = await self.workers[0].get_outputs.remote()
            for out in outs:
                # pack pytorch
                for k, v in out.items():
                    if isinstance(v, np.ndarray):
                        out[k] = torch.from_numpy(v)
                self.remote_outs.put_nowait(out)

    def _prefetch_task_callback(self, task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:
            logger.exception('Ray executor prefetch task failed.')

    def start(self, forward_event: asyncio.Event):
        """start engine loop."""
        self.forward_event = forward_event
        self.collective_rpc('start')

        self.remote_outs = asyncio.Queue()
        event_loop = asyncio.get_event_loop()
        self._prefetch_task = event_loop.create_task(self._prefetch_outputs())

    def stop(self):
        """stop engine loop."""
        if self.dp == 1:
            self.collective_rpc('stop_async')
        if self._prefetch_task is not None:
            self._prefetch_task.cancel()

    def release(self):
        """release."""
        if self.dp == 1:
            self.collective_rpc('release')
            try:
                self.collective_rpc('exit')
            except ray.exceptions.RayActorError as e:
                logger.debug(f'ray actor exit: {e}')
        else:
            [ray.kill(worker) for worker in self.workers]

        ray.util.remove_placement_group(self.placement_group)

    def _compile_dag(self):
        """compile dag."""
        from ray.dag.input_node import InputNode
        from ray.dag.output_node import MultiOutputNode
        with InputNode() as input_data:
            outputs = [worker.forward_async.bind(input_data) for worker in self.workers]
            output = MultiOutputNode(outputs)

        return output

    async def forward_async(self, inputs):
        """start forward."""
        # we don't need return of forward async
        if self.dag is None:
            self.dag = self._compile_dag()
        inputs = ray.put(inputs)
        ray.get(self.dag.execute(inputs))

    async def get_output_async(self):
        """get output async."""
        if self.remote_outs.qsize() > 0:
            return self.remote_outs.get_nowait()
        return await self.remote_outs.get()

    def _sort_workers(self, driver_ip: str, workers: List[RayWorkerWrapper]):
        """sort workers by ip."""
        worker_ips = ray.get([worker.get_node_ip.remote() for worker in workers])

        ip_counts: Dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        worker_ip_map = list(zip(workers, worker_ips))

        def sort_by_driver_then_worker_ip(item):
            """Sort the workers based on 3 properties:

            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = item[1]
            return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        sorted_worker_ip_map = sorted(worker_ip_map, key=sort_by_driver_then_worker_ip)
        workers = [item[0] for item in sorted_worker_ip_map]
        return workers

    def _init_workers_ray(self, placement_group: PlacementGroup, worker_kwargs: dict):
        """init worker ray."""
        device_str = get_device_str()
        bundle_indices = []
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if bundle.get(device_str, 0):
                bundle_indices.append(bundle_id)
        bundle_indices = bundle_indices[:self.world_size]

        workers = list()
        for _, bundle_id in enumerate(bundle_indices):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker = ray.remote(
                num_cpus=0,
                num_gpus=1.0,
                scheduling_strategy=scheduling_strategy,
            )(RayWorkerWrapper).remote(**worker_kwargs)
            workers.append(worker)

        driver_ip = _get_master_addr()
        workers = self._sort_workers(driver_ip, workers)

        return workers
