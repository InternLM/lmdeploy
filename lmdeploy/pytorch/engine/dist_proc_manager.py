# Copyright (c) OpenMMLab. All rights reserved.
import os
from datetime import timedelta
from typing import Any, List

from torch import distributed as dist
from torch import multiprocessing as mp

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.utils import get_logger

from ..distributed import DistContext

logger = get_logger('lmdeploy')


def broadcast_inputs(rank: int, inputs: Any, group: dist.group):
    """get input tensor parallel."""

    dist.barrier(group)

    # broadcast meta info
    if rank != 0:
        inputs = [None, None, None]
    else:
        if inputs[0] is not None:
            device_inputs = inputs[0]
            meta_inputs = device_inputs.to_device('meta')
            inputs[0] = meta_inputs

    dist.broadcast_object_list(inputs, group=group)

    if inputs[0] is None:
        return [None, None, None]

    # device cast
    if rank == 0:
        device_inputs.broadcast()
    else:
        device_inputs = inputs[0].broadcast()

    inputs[0] = device_inputs

    return inputs


def init_dist_environ(rank: int, world_size: int, nproc_per_node: int):
    """init environ."""
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank % nproc_per_node)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(nproc_per_node)


DIST_TIMEOUT = timedelta(days=35600)


class DistProcManager:
    """distributed process manager."""

    def __init__(self,
                 model_path: str,
                 tokenizer: object,
                 engine_config: PytorchEngineConfig,
                 trust_remote_code: bool = True):

        # distribute args
        node_rank = engine_config.node_rank
        nproc_per_node = engine_config.nproc_per_node
        nnodes = engine_config.nnodes
        tp = engine_config.tp
        dp = 1
        world_size = DistContext.get_world_size(tp, dp)
        if nproc_per_node is None:
            nproc_per_node = world_size // nnodes
            engine_config.nproc_per_node = nproc_per_node
        assert (nnodes * nproc_per_node == world_size), ('nnodes * nproc_per_node != world_size')

        # get global ranks, rank0 in main process
        global_rank0 = node_rank * nproc_per_node
        global_ranks = [global_rank0 + idx for idx in range(1, nproc_per_node)]

        # fields
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.engine_config = engine_config
        self.trust_remote_code = trust_remote_code
        self.global_rank0 = global_rank0
        self.world_size = world_size
        self.global_ranks = global_ranks

        # generate mp pool
        self.mp_ctx = mp.get_context('spawn')
        self.procs: List[mp.Process] = []
        self.watchdog = None

    @staticmethod
    def _find_available_port() -> bool:
        """find available port."""
        import socket
        port = 29500
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
                port += 1

    @staticmethod
    def setup_master_addr():
        """setup master addr."""
        port = DistProcManager._find_available_port()
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(port))
        addr = os.environ['MASTER_ADDR']
        port = os.environ['MASTER_PORT']
        logger.info(f'MASTER_ADDR={addr}, MASTER_PORT={port}')

    @staticmethod
    def init_process_group(rank: int, world_size: int, nproc_per_node: int):
        """init process group."""
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size, timeout=DIST_TIMEOUT)
        assert dist.is_initialized()
        init_dist_environ(rank, world_size, nproc_per_node)

    @staticmethod
    def _dist_process(rank: int,
                      world_size: int,
                      model_path: str,
                      tokenizer: object,
                      engine_config: PytorchEngineConfig,
                      trust_remote_code: bool = True):
        """distributed."""
        from .engine import Engine

        # rank == 0 should never enter this proc.
        assert rank != 0
        DistProcManager.init_process_group(
            rank,
            world_size=world_size,
            nproc_per_node=engine_config.nproc_per_node,
        )
        # engine would call `dist_run_forever()`
        # and blocked in initialize.
        Engine.from_pretrained(
            model_path,
            tokenizer=tokenizer,
            engine_config=engine_config,
            trust_remote_code=trust_remote_code,
        )

    def start(self):
        """start process."""
        from threading import Thread
        logger.debug('Starting dist processes.')
        DistProcManager.setup_master_addr()
        engine_config = self.engine_config

        for rank in self.global_ranks:
            proc = self.mp_ctx.Process(target=self._dist_process,
                                       kwargs=dict(rank=rank,
                                                   world_size=self.world_size,
                                                   model_path=self.model_path,
                                                   tokenizer=self.tokenizer,
                                                   engine_config=engine_config,
                                                   trust_remote_code=self.trust_remote_code),
                                       name=f'ProcRank{rank}',
                                       daemon=True)
            proc.start()
            self.procs.append(proc)

        DistProcManager.init_process_group(self.global_rank0,
                                           world_size=self.world_size,
                                           nproc_per_node=engine_config.nproc_per_node)

        self.watchdog = Thread(target=self._mp_watchdog, args=(1, ), daemon=True)
        self.watchdog.start()

    def terminate_all_procs(self):
        """terminate all process."""
        log_procs = []
        for idx, p in enumerate(self.procs):
            if p.is_alive():
                p.terminate()
            else:
                exitcode = p.exitcode
                if exitcode > 0:
                    # terminated exitcode < 0
                    log_procs.append((idx, exitcode))
                p.close()
        for idx, exitcode in log_procs:
            logger.error(f'TP process[{idx}] failed with exitcode={exitcode}.')
        return log_procs

    def _check_context_alive(self):
        """check context alive."""
        procs = self.procs
        failed_procs = list(idx for idx, p in enumerate(procs) if not p.is_alive())
        if len(failed_procs) == 0:
            return

        log_procs = self.terminate_all_procs()
        # TODO: not safe exit.
        exit_code = 1 if len(log_procs) > 0 else 0
        os._exit(exit_code)

    def _mp_watchdog(self, timeout: int = 1):
        """watch dog of mp context.

        Args:
            timeout: timeout
        """
        import time
        while True:
            self._check_context_alive()
            time.sleep(timeout)
