# Copyright (c) OpenMMLab. All rights reserved.
# modify from vLLM: https://github.com/vllm-project/vllm/blob/main/vllm/v1/executor/multiproc_executor.py
import asyncio
import multiprocessing.shared_memory as shared_memory
import os
import pickle
import signal
import struct
from contextlib import asynccontextmanager, contextmanager
from multiprocessing.context import SpawnContext
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.devices import DeviceContext
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.model_agent import BaseModelAgent, build_model_agent
from lmdeploy.utils import get_logger

from .base import ExecutorBase
from .dist_utils import find_available_port, init_process_group, setup_master_addr

logger = get_logger('lmdeploy')

# 1m shared memory
SHARED_BLOCK_SIZE = 1 << 20
# num shared block
NUM_SHARED_BLOCK = 32
# data size
HEAD_SIZE = 8
# block real size
SHARED_BLOCK_REAL_SIZE = SHARED_BLOCK_SIZE + HEAD_SIZE


def get_num_packages(data_size):
    """get num packages."""
    return (data_size + SHARED_BLOCK_SIZE - 1) // SHARED_BLOCK_SIZE


class Notifier:

    def __init__(self, num_receiver: int, mp_ctx: SpawnContext):
        self.events = [mp_ctx.Event() for _ in range(NUM_SHARED_BLOCK)]
        self.bar = mp_ctx.Barrier(num_receiver + 1)
        self._event_id = 0

    def _update_event_id(self):
        self._event_id = (self._event_id + 1) % NUM_SHARED_BLOCK

    def set(self):
        self.events[self._event_id].set()
        if self._event_id == NUM_SHARED_BLOCK - 1:
            self.bar.wait()
            [event.clear() for event in self.events]
            self.bar.wait()
        self._update_event_id()

    async def set_async(self):
        # not safe if we might launch multiple reqs
        event_loop = asyncio.get_event_loop()
        self.events[self._event_id].set()
        if self._event_id == NUM_SHARED_BLOCK - 1:
            await event_loop.run_in_executor(None, self.bar.wait)
            [event.clear() for event in self.events]
            self.bar.wait()
        self._update_event_id()

    @contextmanager
    def wait(self):
        self.events[self._event_id].wait()
        yield
        if self._event_id == NUM_SHARED_BLOCK - 1:
            self.bar.wait()
            self.bar.wait()
        self._update_event_id()

    @asynccontextmanager
    async def wait_async(self):
        event_loop = asyncio.get_event_loop()
        await event_loop.run_in_executor(None, self.events[self._event_id].wait)
        yield
        if self._event_id == NUM_SHARED_BLOCK - 1:
            self.bar.wait()
            self.bar.wait()
        self._update_event_id()

    def close(self):
        for event in self.events:
            event.set()
        self.bar.abort()


class SharedBuffer:
    """shared buffer."""

    def __init__(self, proc_id: int, notifier: Notifier, name: str = None):
        self.proc_id = proc_id
        self.notifier = notifier
        self.is_create = name is None
        if self.is_create:
            # double buffer
            self.shm = shared_memory.SharedMemory(create=True, size=SHARED_BLOCK_REAL_SIZE * NUM_SHARED_BLOCK)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self._buf_id = 0

        if proc_id >= 0:
            self.proc_mask = 1 << proc_id
        else:
            self.proc_mask = 0

        self.is_closed = False

    @contextmanager
    def acquire_buf(self):
        buf = self.shm.buf
        assert buf is not None
        buf_start = self._buf_id * SHARED_BLOCK_REAL_SIZE
        out_buf = buf[buf_start:buf_start + SHARED_BLOCK_REAL_SIZE]
        yield out_buf
        self._buf_id = (self._buf_id + 1) % NUM_SHARED_BLOCK

    def name(self):
        return self.shm.name

    def pack_data(self, data, receiver_mask):
        """pack data."""
        dumped_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        data_size = len(dumped_data)

        num_packs = get_num_packages(data_size)
        head = struct.pack('II', data_size, receiver_mask)

        for _ in range(num_packs):
            with self.acquire_buf() as buf:
                pac_size = min(len(dumped_data), SHARED_BLOCK_SIZE)
                packed_data = head + dumped_data[:pac_size]
                buf[:HEAD_SIZE + pac_size] = packed_data
                dumped_data = dumped_data[pac_size:]
                yield buf

    def send(self, data, receiver_mask: int = 0xff):
        """pack data."""
        for _ in self.pack_data(data, receiver_mask):
            self.notifier.set()

    async def send_async(self, data, receiver_mask: int = 0xff):
        """async pack data."""
        for _ in self.pack_data(data, receiver_mask):
            await self.notifier.set_async()

    def _receive(self):
        """unpack data."""
        with self.acquire_buf() as buf:
            head = buf[:HEAD_SIZE]
            data_size, receiver_mask = struct.unpack('II', head)
            is_receiver = ((receiver_mask & self.proc_mask) > 0)

            pac_size = min(data_size, SHARED_BLOCK_SIZE)
            remain_size = data_size - pac_size

            dumped_data = b''
            if is_receiver:
                dumped_data += buf[HEAD_SIZE:HEAD_SIZE + pac_size]

        while remain_size > 0:
            with self.notifier.wait(), self.acquire_buf() as buf:
                pac_size = min(remain_size, SHARED_BLOCK_SIZE)
                remain_size -= pac_size
                if not is_receiver:
                    continue
                dumped_data += buf[HEAD_SIZE:HEAD_SIZE + pac_size]

        if not is_receiver:
            return None
        data = pickle.loads(dumped_data)
        return data

    def receive(self):
        """unpack data."""
        with self.notifier.wait():
            return self._receive()

    async def receive_async(self):
        """async receive data."""
        async with self.notifier.wait_async():
            return self._receive()

    def close(self):
        if self.is_closed:
            return
        self.shm.close()
        if self.is_create:
            self.shm.unlink()
        self.notifier.close()
        self.is_closed = True


class MPExecutor(ExecutorBase):
    """Single node multi device Executor powered by multiprocess."""

    @classmethod
    def setup_master_addr(cls):
        """setup master addr."""
        port = find_available_port()
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(port))
        addr = os.environ['MASTER_ADDR']
        port = os.environ['MASTER_PORT']
        setup_master_addr(addr, port)

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 backend_config: BackendConfig,
                 tokenizer: Any,
                 dp: int,
                 tp: int,
                 adapters: Dict[str, str] = None,
                 device_type: str = 'cuda'):
        """initialize Executor."""
        super().__init__(model_path=model_path,
                         model_config=model_config,
                         cache_config=cache_config,
                         backend_config=backend_config,
                         tokenizer=tokenizer,
                         dp=dp,
                         tp=tp,
                         adapters=adapters,
                         device_type=device_type)

        # initialize processes.
        self.setup_master_addr()
        self.world_size = tp * dp
        mp_ctx = mp.get_context('spawn')
        self.mp_ctx = mp_ctx
        self.comm_notifier = Notifier(self.world_size, mp_ctx)
        self.comm_buf = SharedBuffer(-1, notifier=self.comm_notifier)
        self.comm_buf_name = self.comm_buf.name()

        self.procs: List[ExecutorProc] = []
        self.ret_bufs: List[SharedBuffer] = []
        for proc_id in range(self.world_size):
            proc = ExecutorProc(proc_id=proc_id, mp_ctx=mp_ctx)

            ret_notifier = Notifier(1, mp_ctx)
            ret_buf = SharedBuffer(0, notifier=ret_notifier)
            self.ret_bufs.append(ret_buf)
            proc.start(proc_id=proc_id,
                       comm_notifier=self.comm_notifier,
                       comm_buf_name=self.comm_buf_name,
                       ret_notifier=ret_notifier,
                       ret_buf_name=ret_buf.name(),
                       model_path=model_path,
                       model_config=model_config,
                       cache_config=cache_config,
                       backend_config=backend_config,
                       tokenizer=tokenizer,
                       dp=dp,
                       tp=tp,
                       adapters=adapters,
                       device_type=device_type,
                       log_level=logger.level)
            self.procs.append(proc)

        self.ret_all_mask = (1 << self.world_size) - 1
        self._prefetch_task: asyncio.Task = None
        self.remote_outs: asyncio.Queue = None

        def signal_handler(signum, frame):
            logger.error('Received custom termination signal from sub processing, exiting...')
            self.stop()
            self.release()
            os._exit(1)

        signal.signal(signal.SIGUSR1, signal_handler)

    def collective_rpc(self,
                       method: str,
                       args: Tuple[Any] = None,
                       kwargs: Dict[str, Any] = None,
                       receiver_mask: int = 0xff,
                       return_mask: int = 0):
        """collective rpc."""
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        self.comm_buf.send(
            dict(
                method=method,
                args=args,
                kwargs=kwargs,
                return_mask=return_mask,
            ),
            receiver_mask=receiver_mask,
        )

        if return_mask:
            outputs = [None] * len(self.ret_bufs)
            for proc_id, ret_buf in enumerate(self.ret_bufs):
                if bool(return_mask & (1 << proc_id)):
                    outputs[proc_id] = ret_buf.receive()
            return outputs

    async def collective_rpc_async(self,
                                   method: str,
                                   args: Tuple[Any] = None,
                                   kwargs: Dict[str, Any] = None,
                                   receiver_mask: int = 0xff,
                                   return_mask: int = 0):
        """collective rpc."""
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        self.comm_buf.send(
            dict(
                method=method,
                args=args,
                kwargs=kwargs,
                return_mask=return_mask,
            ),
            receiver_mask=receiver_mask,
        )

        if return_mask:
            outputs = [None] * len(self.ret_bufs)
            for proc_id, ret_buf in enumerate(self.ret_bufs):
                if bool(return_mask & (1 << proc_id)):
                    outputs[proc_id] = await ret_buf.receive_async()
            return outputs

    def download_models(self):
        """download model."""
        raise NotImplementedError('Not Implemented.')

    def build_model(self):
        """build model."""
        self.collective_rpc('build_model', return_mask=self.ret_all_mask)

    def gather_free_mem(self):
        """gather available memory."""
        ret = self.collective_rpc('get_free_mem', return_mask=self.ret_all_mask)
        return ret

    def set_cache_config(self, cache_config: CacheConfig):
        """set all cache config."""
        self.collective_rpc('set_cache_config', args=(cache_config, ), return_mask=self.ret_all_mask)

    def set_model_config(self, model_config: ModelConfig):
        """set all cache config."""
        self.collective_rpc('set_model_config', args=(model_config, ), return_mask=self.ret_all_mask)

    def build_graph_runner(self):
        """build graph runner."""
        self.collective_rpc('build_graph_runner', return_mask=self.ret_all_mask)

    def build_cache_engine(self):
        """build cache engine."""
        self.collective_rpc('build_cache_engine', return_mask=self.ret_all_mask)

    async def _prefetch_outputs(self):
        while True:
            outs = (await self.collective_rpc_async('get_outputs', receiver_mask=1, return_mask=1))[0]
            for out in outs:
                self.remote_outs.put_nowait(out)

    def start(self, forward_event: asyncio.Event):
        """start engine loop."""
        self.collective_rpc('start', return_mask=self.ret_all_mask)

        self.remote_outs = asyncio.Queue()
        event_loop = asyncio.get_event_loop()
        self._prefetch_task = event_loop.create_task(self._prefetch_outputs())

    async def forward_async(self, inputs):
        """start forward."""
        await self.collective_rpc_async('set_forward_inputs', args=(inputs, ))

    async def get_output_async(self):
        """get output async."""
        return await self.remote_outs.get()

    def get_input_processor(self):
        """get input processor."""
        return self.collective_rpc('get_input_processor', receiver_mask=1, return_mask=1)[0]

    def stop(self):
        """stop engine loop."""
        if self._prefetch_task is not None:
            self._prefetch_task.cancel()

    def release(self):
        """release."""
        for proc in self.procs:
            proc.close()

        for proc in self.procs:
            proc.join()

        self.comm_buf.close()
        for ret_buf in self.ret_bufs:
            ret_buf.close()


class ExecutorProc:

    def __init__(self, proc_id: int, mp_ctx: SpawnContext):
        """executor proc."""
        self.proc_id = proc_id
        self.mp_ctx = mp_ctx
        self._proc = None

    def start(self, **kwargs):
        """start proc."""
        assert self._proc is None
        proc = self.mp_ctx.Process(target=self._main_loop,
                                   kwargs=kwargs,
                                   name=f'ExecutorProc-{self.proc_id}',
                                   daemon=True)
        proc.start()
        self._proc = proc

    def close(self):
        """stop proc."""
        if self._proc is None:
            return
        if not self._proc.is_alive():
            return
        self._proc.terminate()

    def join(self):
        if self._proc is None:
            return
        self._proc.join()

    def _main_loop(
        self,
        proc_id: int,
        comm_notifier: Any,
        comm_buf_name: str,
        ret_notifier: Any,
        ret_buf_name: str,
        model_path: str,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        tokenizer: Any,
        dp: int,
        tp: int,
        adapters: Dict[str, str] = None,
        device_type: str = 'cuda',
        log_level: int = 30,
    ):
        """main loop."""
        logger.setLevel(log_level)
        torch.cuda.set_device(proc_id)

        # catch signal
        def handle_sigterm(signum, frame):
            logger.debug(f'Proc[{proc_id}] terminated.')
            exit(0)

        signal.signal(signal.SIGTERM, handle_sigterm)

        world_size = tp * dp
        init_process_group(proc_id, world_size)

        dist_ctx = DistContext.build(proc_id, tp, dp)
        device_ctx = DeviceContext(device_type=device_type)

        model_agent = build_model_agent(model_path=model_path,
                                        model_config=model_config,
                                        cache_config=cache_config,
                                        backend_config=backend_config,
                                        tokenizer=tokenizer,
                                        device_ctx=device_ctx,
                                        dist_ctx=dist_ctx,
                                        adapters=adapters)
        comm_buf = SharedBuffer(proc_id, notifier=comm_notifier, name=comm_buf_name)
        ret_buf = SharedBuffer(-1, notifier=ret_notifier, name=ret_buf_name)
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        destroy_pg = True
        try:
            event_loop.run_until_complete(
                self._main_loop_impl(proc_id, comm_buf=comm_buf, ret_buf=ret_buf, model_agent=model_agent))
        except asyncio.CancelledError:
            logger.warning(f'Proc[{proc_id}] main loop cancelled.')
            destroy_pg = False
            os.kill(os.getppid(), signal.SIGUSR1)
        except SystemExit:
            # terminated by executor
            pass
        except BaseException:
            logger.exception(f'Proc[{proc_id}] failed')
            os.kill(os.getppid(), signal.SIGUSR1)
        finally:
            logger.debug(f'Proc[{proc_id}] cleanup.')
            model_agent.stop()
            model_agent.release()
            comm_buf.close()
            ret_buf.close()
            if dist.is_initialized() and destroy_pg:
                dist.destroy_process_group()

    @staticmethod
    async def _task_wrapper(func, args: List, kwargs: Dict, need_return: bool, ret_buf: SharedBuffer):
        ret = await func(*args, **kwargs)
        if need_return:
            await ret_buf.send_async(ret)

    @staticmethod
    async def _prefetch_output_loop(model_agent: BaseModelAgent, out_que: asyncio.Queue):
        """prefetch output loop."""
        while True:
            out = await model_agent.get_output_async()
            out_que.put_nowait(out)

    @staticmethod
    async def _get_outputs(out_que: asyncio.Queue):
        """get outputs."""
        qsize = out_que.qsize()
        outs = []
        if qsize > 0:
            for _ in range(qsize):
                outs.append(out_que.get_nowait())
            return outs
        else:
            return [await out_que.get()]

    async def _main_loop_impl(self, proc_id: int, comm_buf: SharedBuffer, ret_buf: SharedBuffer,
                              model_agent: BaseModelAgent):
        """main loop."""
        proc_mask = 1 << proc_id
        event_loop = asyncio.get_event_loop()
        out_que = asyncio.Queue()
        prefetch_out_task = None
        while True:
            command = await comm_buf.receive_async()
            if command is None:
                continue
            method = command['method']
            return_mask = command.get('return_mask', True)
            need_return = bool(proc_mask & return_mask)

            if method == 'get_outputs':
                func = self._get_outputs
                args = (out_que, )
                kwargs = dict()
            else:
                func = getattr(model_agent, method, None)
                args = command.get('args', list())
                kwargs = command.get('kwargs', dict())
            call_async = asyncio.iscoroutinefunction(func)
            assert func is not None

            logger.debug(f'proc[{proc_id}] call method: <{method}>.')
            if call_async:
                event_loop.create_task(self._task_wrapper(func, args, kwargs, need_return, ret_buf))
            else:
                ret = func(*args, **kwargs)
                if method == 'start':
                    prefetch_out_task = event_loop.create_task(self._prefetch_output_loop(model_agent, out_que))
                    prefetch_out_task.set_name(f'PrefetchOut_Proc{proc_id}')
                if need_return:
                    ret_buf.send(ret)
