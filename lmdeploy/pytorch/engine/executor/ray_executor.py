# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import contextlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import ray
import ray.exceptions
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.backends.selector import init_backend
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.disagg.conn.protocol import DistServeInitRequest, DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.ray import RayContext, get_device_str
from lmdeploy.utils import get_logger, try_import_deeplink

from .base import ExecutorBase
from .base_worker import WorkerWrapperBase
from .dist_utils import find_available_port

logger = get_logger('lmdeploy')


def _get_master_addr():
    """Get master addr."""
    addr = _envs.dist_master_addr
    if addr is not None:
        return addr
    gcs_addr = ray.get_runtime_context().gcs_address
    master_addr = gcs_addr.split(':')[0]
    return master_addr


def _get_master_port():
    """Get master port."""
    port = _envs.dist_master_port
    if port is not None:
        return port
    return find_available_port()


def get_ascend_device_rank_mapping(master_addr):
    rank_table_file = _envs.ascend_rank_table_file
    if not rank_table_file:
        raise ValueError('ASCEND_RANK_TABLE_FILE_PATH is not set')
    with open(rank_table_file, 'r') as f:
        rank_table = json.load(f)
    try:
        assert master_addr == rank_table['server_list'][0]['server_id'], 'Master address does not match rank table'
        rank_mapping = {}
        worker_ips = []
        for server in rank_table['server_list']:
            node_ip = server['server_id']
            for idx, device in enumerate(server['device']):
                local_rank = idx
                global_rank = int(device['rank_id'])
                rank_mapping[global_rank] = local_rank
                worker_ips.append(node_ip)
    except Exception as e:
        logger.error(f'Parse rank table file({rank_table})  failed')
        raise e

    envs = {
        'ASCEND_RANK_TABLE_FILE_PATH': rank_table_file,
    }
    return rank_mapping, worker_ips, envs


def _update_env_cuda_alloc_conf(env_vars: Dict):
    """Update runtime env for CUDA alloc conf."""
    cuda_alloc_conf = os.getenv('PYTORCH_CUDA_ALLOC_CONF', None)
    if cuda_alloc_conf is None:
        return

    # check and update conf, skip expandable_segments
    cuda_alloc_conf = cuda_alloc_conf.split(',')
    new_cuda_alloc_conf = []
    for conf in cuda_alloc_conf:
        if 'expandable_segments' in conf:
            if 'True' in conf:
                logger.warning('"expandable_segments:True" is not supported.')
            continue
        new_cuda_alloc_conf.append(conf)
    if len(new_cuda_alloc_conf) == 0:
        new_cuda_alloc_conf = ['expandable_segments:False']
    cuda_alloc_conf = ','.join(new_cuda_alloc_conf)

    # update env_vars
    env_vars['PYTORCH_CUDA_ALLOC_CONF'] = cuda_alloc_conf


def _update_runtime_envs(runtime_env: Dict):
    """Update runtime envs."""
    new_envs = _envs.get_all_envs()
    env_vars: Dict = runtime_env.get('env_vars', {})
    env_vars.update(new_envs)
    _update_env_cuda_alloc_conf(env_vars)
    runtime_env['env_vars'] = env_vars
    return runtime_env


def _update_runtime_env_nsys(runtime_env: Dict):
    """Update runtime env for nsys."""
    nsight_env = {
        't': 'cuda,cudnn,cublas,nvtx',
        'o': "'worker_process_%p'",
        'stop-on-exit': 'true',
    }
    prefix_path = _envs.ray_nsys_output_prefix
    if prefix_path is not None:
        nsight_env['o'] = f'{prefix_path}%p'
    runtime_env['nsight'] = nsight_env
    return runtime_env


class RemoteLogger:
    """Remote logger."""

    def __init__(self):
        self._records = dict()
        self._next_handle = 0

    def start(self, msg: str):
        """Start remote log."""
        record = torch.profiler.record_function(msg)
        record.__enter__()
        handle = self._next_handle
        self._records[handle] = record
        self._next_handle += 1
        return handle

    def end(self, handle: int):
        """End remote log."""
        record = self._records.pop(handle, None)
        if record is not None:
            record.__exit__(None, None, None)


class RayWorkerWrapper(WorkerWrapperBase):
    """Worker wrapper."""

    def __init__(
        self,
        model_path: str,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        model_config: ModelConfig,
        dist_config: DistConfig,
        misc_config: MiscConfig,
        adapters: Dict[str, str] = None,
        device_type: str = 'cuda',
        dtype: str = 'auto',
        log_level: int = 30,
        specdecode_config: SpecDecodeConfig = None,
    ):
        init_backend(device_type)
        try_import_deeplink(device_type)

        super().__init__(
            model_path=model_path,
            cache_config=cache_config,
            backend_config=backend_config,
            model_config=model_config,
            dist_config=dist_config,
            misc_config=misc_config,
            adapters=adapters,
            device_type=device_type,
            log_level=log_level,
            specdecode_config=specdecode_config,
        )
        self.node_ip = ray.util.get_node_ip_address()
        self._remote_logger = RemoteLogger()

    def set_device(self, local_rank):
        """Set worker local rank."""
        torch.cuda.set_device(local_rank)

    def set_env(self, envs: Dict[str, str]):
        for key, value in envs.items():
            os.environ[key] = value

    def get_node_ip(self):
        """Get worker ip."""
        return self.node_ip

    def warmup_dist(self):
        # None default CUDA_VISIBLE_DEVICES might leads to slow first time all_reduce
        # WHY?
        logger.debug('Warmup all_reduce.')
        import torch

        from lmdeploy.pytorch.distributed import all_reduce, get_dist_manager
        with get_dist_manager().context(self.dist_ctx):
            group = self.dist_ctx.tp_group.gpu_group
            tmp = torch.empty((1, ), device='cuda')
            all_reduce(tmp, group=group)

    def pack_output(self, output: Dict):
        """Pack output."""
        return output.to_numpy()

    def remote_log_start(self, msg: str):
        """Remote log start."""
        return self._remote_logger.start(msg)

    def remote_log_end(self, handle: int):
        """Remote log end."""
        return self._remote_logger.end(handle)

    def exit(self):
        """Exit actor."""
        ray.actor.exit_actor()


class RayExecutor(ExecutorBase):
    """Ray executor."""

    def __init__(
        self,
        model_path: str,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        dist_config: DistConfig,
        misc_config: MiscConfig,
        adapters: Dict[str, str] = None,
        device_type: str = 'cuda',
        dtype: str = 'auto',
        specdecode_config: SpecDecodeConfig = None,
    ):
        """Initialize Executor."""
        super().__init__(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            dist_config=dist_config,
            misc_config=misc_config,
            adapters=adapters,
            device_type=device_type,
            specdecode_config=specdecode_config,
        )

        device_ctx = DeviceContext(device_type)
        with get_device_manager().context(device_ctx):
            logger.info('Init ray cluster.')
            attn_tp = dist_config.attn_tp
            self.ray_ctx = RayContext(attn_tp, dp=dist_config.dp, device_type=device_type)
            placement_group = self.ray_ctx.get_placement_group()
            self.placement_group = placement_group

            if self.dp == 1:
                self.master_addr = _get_master_addr()
                self.master_port = _get_master_port()
            else:
                self.master_addr = _envs.dp_master_addr
                self.master_port = _envs.dp_master_port
                if self.master_addr is None or self.master_port is None:
                    raise RuntimeError('DP > 1 requires "LMDEPLOY_DP_MASTER_ADDR" and "LMDEPLOY_DP_MASTER_PORT".')

            # create workerwrapper actors
            worker_kwargs = dict(
                model_path=model_path,
                cache_config=cache_config,
                model_config=model_config,
                backend_config=backend_config,
                dist_config=dist_config,
                misc_config=misc_config,
                adapters=adapters,
                device_type=device_type,
                dtype=dtype,
                log_level=logger.level,
                specdecode_config=specdecode_config,
            )

            logger.info('Init ray workers.')
            self.workers = self._init_workers_ray(placement_group, worker_kwargs)
            self.dag = None
            self._prefetch_task: asyncio.Task = None
            self.remote_outs: asyncio.Queue = None

            logger.info('Init distributed environment by device.')
            self.rank_offset = dist_config.dp_rank * attn_tp
            self._init_distributed_environment_by_device(device_type)

            logger.info('Init distributed process group.')
            ray.get([
                worker.init_process_group.remote(rank + self.rank_offset, self.master_addr, self.master_port)
                for rank, worker in enumerate(self.workers)
            ])

            if self.dist_config.world_size > 1:
                logger.info('Warming up distribute environment, this might take long time, please waiting...')
                ray.get([worker.warmup_dist.remote() for worker in self.workers])

    def collective_rpc(self,
                       method: str,
                       args: Tuple[Any] = None,
                       kwargs: Dict[str, Any] = None,
                       timeout: float = None):
        """Collective rpc."""
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        return ray.get([getattr(worker, method).remote(*args, **kwargs) for worker in self.workers], timeout=timeout)

    def build_model(self):
        """Build model."""
        self.collective_rpc('build_model')

    def gather_free_mem(self):
        """Gather available memory."""
        return self.collective_rpc('get_free_mem')

    def set_cache_config(self, cache_config: CacheConfig, spec_cache_config: CacheConfig = None):
        """Set all cache config."""
        self.collective_rpc('set_cache_config', (cache_config, spec_cache_config))

    def set_model_config(self, model_config: ModelConfig, spec_model_config: ModelConfig = None):
        """Set all model config."""
        self.collective_rpc('set_model_config', (model_config, spec_model_config))

    def build_graph_runner(self):
        """Build graph runner."""
        self.collective_rpc('build_graph_runner')

    def build_cache_engine(self):
        """Build cache engine."""
        self.collective_rpc('build_cache_engine')

    def update_params(self, request: Any):
        """Update params."""
        self.collective_rpc('update_params', (request, ))

    def warmup(self):
        """Build cache engine."""
        self.collective_rpc('warmup')

    def sleep(self, level: int = 1):
        """Sleep."""
        self.collective_rpc('sleep', (level, ))

    def wakeup(self, tags: Optional[List[str]] = None):
        """Wakeup."""
        if tags is None or 'kv_cache' in tags:
            self.update_configs()
        self.collective_rpc('wakeup', (tags, ))

    def get_input_processor(self):
        """Build cache engine."""
        return ray.get(self.workers[0].get_input_processor.remote())

    async def _prefetch_outputs(self):
        while True:
            outs = await self.workers[0].get_outputs.remote()
            logger.debug(f'Receive {len(outs)} outputs from worker[0].')
            for out in outs:
                out = out.to_tensor()
                self.remote_outs.put_nowait(out)

    def _prefetch_task_callback(self, task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug(f'{task.get_name()} cancelled.')
        except KeyboardInterrupt:
            logger.debug(f'{task.get_name()} KeyboardInterrupt.')
        except BaseException:
            logger.debug(f'{task.get_name()} task failed.')

    def start(self, forward_event: asyncio.Event):
        """Start engine loop."""
        self.forward_event = forward_event
        self.collective_rpc('start')

        self.remote_outs = asyncio.Queue()
        event_loop = asyncio.get_event_loop()
        logger.info('Starting async task RayPrefetchOutput loop.')
        self._prefetch_task = event_loop.create_task(self._prefetch_outputs(), name='RayExecutorPrefetchOutput')
        self._prefetch_task.add_done_callback(self._prefetch_task_callback)

    def stop(self):
        """Stop engine loop."""
        if self.dp == 1:
            self.collective_rpc('stop_async')
            logger.debug('RayExecutor workers stopped.')
        if self._prefetch_task is not None:
            self._prefetch_task.cancel()

    def release(self):
        """release."""
        if _envs.ray_timeline_enable:
            ray.timeline(_envs.ray_timeline_output_path)

        if self.dp == 1:
            try:
                self.collective_rpc('release', timeout=5.0)
                logger.debug('RayExecutor workers released.')
            except ray.exceptions.GetTimeoutError:
                logger.info('Ray release timeout, killing workers')
                [ray.kill(worker) for worker in self.workers]
        else:
            [ray.kill(worker) for worker in self.workers]

        self.ray_ctx.shutdown()

    def _compile_dag(self):
        """Compile dag."""
        from ray.dag.input_node import InputNode
        from ray.dag.output_node import MultiOutputNode
        with InputNode() as input_data:
            outputs = [worker.forward_async.bind(input_data) for worker in self.workers]
            output = MultiOutputNode(outputs)

        return output

    async def forward_async(self, inputs):
        """Start forward."""
        # we don't need return of forward async
        if self.dag is None:
            self.dag = self._compile_dag()
        inputs = ray.put(inputs)
        # make sure in order
        outs = self.dag.execute(inputs)
        ray.get(outs)

        # free ray.put inputs
        try:
            ray._private.internal_api.free(inputs)
        except Exception as e:
            logger.warning(f'Free input ref failed: {e}')

    async def get_output_async(self):
        """Get output async."""
        return await self.remote_outs.get()

    @contextlib.contextmanager
    def remote_log(self, msg: str):
        """Send log for debugging.

        Do not use it in production.
        """
        handle_ref = self.workers[0].remote_log_start.remote(msg)
        yield
        handle = ray.get(handle_ref)
        ray.get(self.workers[0].remote_log_end.remote(handle))

    def _sort_workers(self, driver_ip: str, workers: List[RayWorkerWrapper]):
        """Sort workers by ip."""
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

    def _sort_workers_by_ip(self, ips, workers: List[RayWorkerWrapper]):
        worker_ips = ray.get([worker.get_node_ip.remote() for worker in workers])

        if len(ips) != len(workers):
            raise ValueError(f'The length of the ips list does not match the workers, '
                             f'ips length: {len(ips)}, workers length: {len(workers)}')

        # Check if all elements in ips are present in worker_ips and vice versa (ignoring order)
        if set(ips) != set(worker_ips):
            raise ValueError(f'The IP addresses in the ips list do not match the worker IPs. '
                             f'ips: {ips}, worker_ips: {worker_ips}')

        worker_ip_map = list(zip(workers, worker_ips))
        ip_priority = {ip: idx for idx, ip in enumerate(ips)}

        def get_priority(ip):
            return ip_priority.get(ip)

        sorted_worker_ip_map = sorted(worker_ip_map, key=lambda x: get_priority(x[1]))
        sorted_workers = [item[0] for item in sorted_worker_ip_map]
        return sorted_workers

    def _valid_bundle_id(self, bundle_id: int):
        """Check if a bundle is valid only when self.use_external_ray=True."""
        if (not self.ray_ctx.owned_pg and _envs.ray_external_pg_bundles
                and bundle_id not in _envs.ray_external_pg_bundles):
            return False
        return True

    def _init_workers_ray(self, placement_group: PlacementGroup, worker_kwargs: dict):
        """Init worker ray."""
        device_str = get_device_str()
        bundle_indices = []
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if bundle.get(device_str, 0) and self._valid_bundle_id(bundle_id):
                bundle_indices.append(bundle_id)
        attn_tp = self.dist_config.attn_tp
        bundle_indices = bundle_indices[:attn_tp]

        workers = list()
        for _, bundle_id in enumerate(bundle_indices):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            if device_str == 'GPU':
                runtime_env = dict()
                runtime_env = _update_runtime_envs(runtime_env)
                if _envs.ray_nsys_enable:
                    runtime_env = _update_runtime_env_nsys(runtime_env)
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0.01,
                    scheduling_strategy=scheduling_strategy,
                    runtime_env=runtime_env,
                )(RayWorkerWrapper).remote(**worker_kwargs)
            else:
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={device_str: 0.01},
                    scheduling_strategy=scheduling_strategy,
                )(RayWorkerWrapper).remote(**worker_kwargs)
            workers.append(worker)
        return workers

    def _init_distributed_environment_by_device(self, device_str: str):
        """Init distributed environment."""
        driver_ip = _get_master_addr()
        if device_str == 'cuda':
            self.workers = self._sort_workers(driver_ip, self.workers)

        elif device_str == 'ascend':
            self._init_ascend_distributed_environment(driver_ip)
        elif device_str in ['camb', 'maca']:
            self.workers = self._sort_workers(driver_ip, self.workers)
            ray.get([worker.set_device.remote(idx) for idx, worker in enumerate(self.workers)])
        else:
            raise ValueError(f'Unsupported device type: {device_str}')

    def _init_ascend_distributed_environment(self, driver_ip):
        """Init ascend distributed environment."""
        rank_table_file = _envs.ascend_rank_table_file
        set_rt_visable_devices_by_ray = _envs.ascend_set_rt_visable_devices_by_ray

        if rank_table_file:
            # if rank table file is set, use it to get rank mapping, multiple nodes
            rank_mapping, worker_ips, envs = get_ascend_device_rank_mapping(driver_ip)
            self.workers = self._sort_workers_by_ip(worker_ips, self.workers)
            ray.get([worker.set_device.remote(rank_mapping[idx]) for idx, worker in enumerate(self.workers)])
            ray.get([worker.set_env.remote(envs) for worker in self.workers])
        elif not set_rt_visable_devices_by_ray:
            # if rank table file is not set, treat as single node
            # simply set device by index, this is for single node, multiple devices
            self.workers = self._sort_workers(driver_ip, self.workers)
            ray.get([worker.set_device.remote(idx + self.rank_offset) for idx, worker in enumerate(self.workers)])
        else:
            self.workers = self._sort_workers(driver_ip, self.workers)

    """ PD Disaggregation API Begin """

    def p2p_initialize(self, init_request: DistServeInitRequest):
        return self.collective_rpc('p2p_initialize', (init_request, ))

    def p2p_connect(self, remote_engine_id: str, conn_request: List[DistServeKVTransferEndpointInfo]):
        """Rdma connect."""
        return self.collective_rpc('p2p_connect', (
            remote_engine_id,
            conn_request,
        ))

    async def migrate(self, batch: MigrationExecutionBatch):
        jobs = (worker.migrate.remote(batch) for worker in self.workers)
        return await asyncio.gather(*jobs)

    """ PD Disaggregation API Begin """
