# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
import multiprocessing as mp
import os
import signal
import socket
import sys
import time
from typing import Dict, List, Union

import aiohttp
import requests

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.utils import get_logger

from .api_server import serve

logger = get_logger('lmdeploy')


def find_available_port() -> bool:
    """find available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        s.listen(1)
        port = s.getsockname()[1]
        return port


def get_host_ip():
    """get host ip."""
    hostname = socket.getfqdn()
    host_ip = socket.gethostbyname(hostname)
    return host_ip


def _run_server(gpu_ids: List[int], model_path: str, **kwargs):
    """launch a server process."""
    cuda_visible_devices = ','.join([str(_) for _ in gpu_ids])
    os.setpgrp()
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    serve(model_path, **kwargs)


def cleanup_processes(processes: List[mp.Process]):
    """clean up server process."""
    for process in processes:
        logger.info(f'Terminating process group {process.pid}')
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process group may already be terminated
            pass

    # Wait for processes to terminate
    for process in processes:
        if process.is_alive():
            logger.warning(f'Process {process.pid} did not terminate gracefully, forcing kill')
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    logger.info('All processes terminated')
    sys.exit(0)


def is_server_ready(server_url: str):
    """check server."""
    url = f'{server_url}/health'
    headers = {'accept': 'application/json'}
    ready = False
    try:
        response = requests.get(url, headers=headers, timeout=3)
        ready = response.status_code == 200
    except:  # noqa
        pass
    return ready


async def _warmup_server(server_url: str, request_data: Dict, endpoint: str = '/v1/chat/completions'):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(server_url + endpoint, json=request_data, timeout=10) as response:
                resp_text = await response.text()
                assert len(resp_text) > 0
                logger.info(f'Finish warm up server {server_url}. Example output = {resp_text}')
    except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:  # noqa  # yapf: disable
        logger.error(f'Failed to warm up server {server_url} with exception: {e}')


async def _warmup_all_servers(server_urls: List[str], model_name: str):
    """warmup all servers."""
    request_dict = dict(model=model_name,
                        messages=[{
                            'role': 'user',
                            'content': 'Hello! How are you?'
                        }],
                        stream=False,
                        max_tokens=32)
    tasks = [_warmup_server(url, request_dict) for url in server_urls]
    await asyncio.gather(*tasks)


def warmup_servers(server_urls: List[str], model_name: str, timeout: int = 600):
    """warmup all servers."""
    start_time = time.time()
    while time.time() - start_time <= timeout:
        time.sleep(6)
        if all([is_server_ready(url) for url in server_urls]):
            break

    asyncio.run(_warmup_all_servers(server_urls, model_name))
    logger.info(f'Finish warm up servers: {server_urls}')


def launch_server(num_nodes: int,
                  node_rank: int,
                  model_path: str,
                  backend_config: Union[PytorchEngineConfig, TurbomindEngineConfig],
                  base_gpu_id: int = 0,
                  proxy_url: str = None,
                  **kwargs):
    """Run multiple server processes."""
    assert proxy_url is not None, 'Please launch proxy server and pass proxy_url'

    mp.set_start_method('spawn')
    dp = backend_config.dp
    dp_per_node = dp // num_nodes

    processes = []
    http_or_https = 'https' if kwargs.get('ssl', False) else 'http'
    model_name = kwargs.get('model_name', None)
    if model_name is None:
        model_name = model_path
    server_name = get_host_ip()
    server_urls = []
    tp_per_dp = 1  # each dp uses one rank
    server_port_li = [find_available_port() for _ in range(dp_per_node)]

    for idx in range(dp_per_node):
        backend_config_dp = copy.deepcopy(backend_config)
        dp_rank = node_rank * dp_per_node + idx
        backend_config_dp.dp_rank = dp_rank
        server_port = server_port_li[idx]
        gpu_ids_per_dp = [base_gpu_id + gid for gid in range(idx * tp_per_dp, (idx + 1) * tp_per_dp)]
        cur_server_kwargs = dict()
        cur_server_kwargs.update(kwargs)
        cur_server_kwargs['server_name'] = server_name
        cur_server_kwargs['server_port'] = server_port
        cur_server_kwargs['backend_config'] = backend_config_dp
        cur_server_kwargs['proxy_url'] = proxy_url
        url = f'{http_or_https}://{server_name}:{server_port}'
        server_urls.append(url)
        proc = mp.Process(target=_run_server, args=(gpu_ids_per_dp, model_path), kwargs=cur_server_kwargs, daemon=True)
        proc.start()
        processes.append(proc)

    # bind signal
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_processes(processes))
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_processes(processes))
    signal.signal(signal.SIGQUIT, lambda sig, frame: cleanup_processes(processes))

    # warm up
    warmup_servers(server_urls, model_name)

    print(f'HINT:    Please open \033[93m\033[1m{proxy_url}'
          '\033[0m in a browser for detailed api'
          ' usage!!!')
    for p in processes:
        p.join()
