# Copyright (c) OpenMMLab. All rights reserved.
import copy
import multiprocessing as mp
import os
import random
import signal
import socket
import sys
from typing import List, Union

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.utils import get_logger

from .api_server import serve

logger = get_logger('lmdeploy')


def find_available_ports(num: int) -> List[int]:
    """Find available port."""

    def __is_port_ok(port: int):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', port))
                s.listen(1)
                return True
            except Exception:
                return False

    ports = []
    test_port = 3000
    while len(ports) < num:
        test_port += random.randint(10, 500)
        if __is_port_ok(test_port):
            ports.append(test_port)

    return ports


def get_host_ip():
    """Get host ip."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 0))
        ip = s.getsockname()[0]
        return ip


def _run_server(gpu_ids: List[int], model_path: str, **kwargs):
    """Launch a server process."""
    cuda_visible_devices = ','.join([str(_) for _ in gpu_ids])
    os.setpgrp()
    if len(gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    serve(model_path, **kwargs)


def cleanup_processes(processes: List[mp.Process]):
    """Clean up server process."""
    for process in processes:
        logger.info(f'Terminating process group {process.pid}')
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process group may already be terminated
            pass

    # Wait for processes to terminate
    for process in processes:
        process.join(timeout=15)
        if process.is_alive():
            logger.warning(f'Process {process.pid} did not terminate gracefully, forcing kill')
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    logger.info('All processes terminated')
    sys.exit(0)


def launch_server(num_nodes: int,
                  node_rank: int,
                  model_path: str,
                  backend_config: Union[PytorchEngineConfig, TurbomindEngineConfig],
                  proxy_url: str = None,
                  **kwargs):
    """Run multiple server processes in dp mode."""
    assert proxy_url is not None, 'Please launch proxy server and pass proxy_url'
    log_level = kwargs.get('log_level', 'ERROR')
    logger.setLevel(log_level)

    mp.set_start_method('spawn', force=True)
    dp = backend_config.dp
    tp = backend_config.tp
    ep = backend_config.ep
    assert dp > 1, f'only support dp > 1, but give dp={dp}'
    assert tp > 1 or ep > 1, f'only support tp > 1 or ep > 1, but given tp={tp} ep={ep}'

    num_devices = max(dp, tp, ep)
    dp_per_node = dp // num_nodes
    tp_per_dp = num_devices // dp
    http_or_https = 'https' if kwargs.get('ssl', False) else 'http'
    model_name = kwargs.get('model_name', None)
    if model_name is None:
        model_name = model_path
    server_name = get_host_ip()
    server_urls = []
    processes = []

    server_port_li = find_available_ports(dp_per_node)

    for idx in range(dp_per_node):
        backend_config_dp = copy.deepcopy(backend_config)
        dp_rank = node_rank * dp_per_node + idx
        gpu_ids_per_dp = [gid for gid in range(idx * tp_per_dp, (idx + 1) * tp_per_dp)]
        backend_config_dp.dp_rank = dp_rank
        server_port = server_port_li[idx]

        cur_server_kwargs = dict()
        cur_server_kwargs.update(kwargs)
        cur_server_kwargs['server_name'] = server_name
        cur_server_kwargs['server_port'] = server_port
        cur_server_kwargs['backend_config'] = backend_config_dp
        cur_server_kwargs['proxy_url'] = proxy_url
        url = f'{http_or_https}://{server_name}:{server_port}'
        server_urls.append(url)
        logger.info(f'create server with url={url}')
        logger.info(f'backend_config_dp={backend_config_dp} gpus={gpu_ids_per_dp}')
        proc = mp.Process(target=_run_server, args=(gpu_ids_per_dp, model_path), kwargs=cur_server_kwargs)
        proc.start()
        processes.append(proc)

    # bind signal
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_processes(processes))
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_processes(processes))
    signal.signal(signal.SIGQUIT, lambda sig, frame: cleanup_processes(processes))

    for p in processes:
        p.join()
