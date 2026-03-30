# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import atexit
import signal
from typing import TYPE_CHECKING

import torch.multiprocessing as mp

from lmdeploy.messages import PytorchEngineConfig, SpeculativeConfig
from lmdeploy.utils import get_logger

from .base import MPEngine

logger = get_logger('lmdeploy')

if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.engine import Engine


def cancel_async_tasks(loop: asyncio.AbstractEventLoop):
    """Cancel async tasks."""
    tasks = asyncio.all_tasks(loop=loop)
    for task in tasks:
        if not task.done():
            task.cancel()
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


class ZMQMPEngine(MPEngine):

    def __init__(self,
                 model_path: str,
                 engine_config: PytorchEngineConfig = None,
                 speculative_config: SpeculativeConfig = None,
                 **kwargs) -> None:
        """Initialize mp engine."""
        from .zmq_rpc import AsyncRPCClient
        self.shared_dict = None
        self.port = None
        self.proc = None
        self._start_mp_proc(model_path, engine_config, speculative_config=speculative_config, **kwargs)

        self.rpc_client = AsyncRPCClient(port=self.port)

        super().__init__()
        atexit.register(self.close)

    def _start_mp_proc(
        self,
        model_path: str,
        engine_config: PytorchEngineConfig = None,
        speculative_config: SpeculativeConfig = None,
        **kwargs,
    ):
        """Start mp proc."""
        logger.debug('Starting engine multi-process.')
        with mp.Manager() as manager:
            self.shared_dict = manager.dict()
            condition = manager.Condition()
            self.mp_ctx = mp.get_context('spawn')
            log_level = logger.level
            target_kwargs = dict(
                model_path=model_path,
                engine_config=engine_config,
                log_level=log_level,
                speculative_config=speculative_config,
            )
            target_kwargs.update(kwargs)
            self.proc = self.mp_ctx.Process(
                target=self._mp_proc,
                args=(self.shared_dict, condition),
                kwargs=target_kwargs,
                name='mp_engine_proc',
            )
            self.proc.start()
            logger.debug('Receiving rpc server port from mp process.')
            with condition:
                if 'rpc_server_port' not in self.shared_dict:
                    condition.wait()
            self.port = self.shared_dict['rpc_server_port']

    @staticmethod
    def _mp_proc(
        shared_dict: dict,
        condition: mp.Condition,
        model_path: str,
        engine_config: PytorchEngineConfig = None,
        log_level: str = 'WARNING',
        speculative_config: SpeculativeConfig = None,
        **kwargs,
    ):
        """Mp process function."""
        from lmdeploy.pytorch.engine import Engine

        from .zmq_rpc import AsyncRPCServer

        logger.setLevel(log_level)

        # create an async rpc server
        server = AsyncRPCServer()
        with condition:
            shared_dict['rpc_server_port'] = server.port
            condition.notify()

        # create engine
        if engine_config is not None:
            engine_config.enable_mp_engine = False
        engine = Engine.from_pretrained(
            model_path,
            engine_config=engine_config,
            speculative_config=speculative_config,
            **kwargs,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(ZMQMPEngine._mp_proc_async(server, engine))
        except KeyboardInterrupt:
            logger.info('Received KeyboardInterrupt, stopping mp process.')

    @staticmethod
    async def _mp_proc_async(server, engine: 'Engine'):
        """Mp process function."""
        import inspect

        from .base_worker import EngineWorkerBase

        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()

        async def shutdown(loop, signame):
            logger.info(f'MP process received signal {signame}, stopping server.')
            if current_task is not None:
                current_task.cancel()

        for signame in {'SIGINT', 'SIGTERM'}:
            sig = getattr(signal, signame)
            loop.add_signal_handler(sig, lambda signame=signame: asyncio.create_task(shutdown(loop, signame)))

        worker = EngineWorkerBase(engine)

        for name, value in inspect.getmembers(EngineWorkerBase):
            if not name.startswith('_') and inspect.isfunction(value):
                method = getattr(worker, name)
                server.register_method(name, method)

        try:
            # run server
            await server.run()
        except asyncio.CancelledError:
            logger.info('RPC Server stopping due to cancellation.')
        except Exception as e:
            logger.error(f'RPC Server stopped with exception: {e}')
        finally:
            server.stop()
            engine.close()
            try:
                await engine.wait_tasks()
            except asyncio.CancelledError:
                logger.info('Engine wait_tasks cancelled during shutdown.')
            except Exception as e:
                logger.debug(f'Engine wait_tasks failed during shutdown: {e}')

    def _collective_rpc(self, func, *args, **kwargs):
        """Collective rpc call."""
        return self.rpc_client.call(func, *args, **kwargs)

    async def _collective_rpc_async(self, func, *args, **kwargs):
        """Collective rpc call."""
        return await self.rpc_client.async_call(func, *args, **kwargs)

    async def _collective_rpc_streaming_async(self, func, *args, **kwargs):
        """Collective rpc call."""
        async for out in self.rpc_client.async_stream_call(func, *args, **kwargs):
            yield out

    def close(self) -> None:
        """Close mp engine."""
        if self.proc is None:
            return
        logger.info('Closing mp engine.')
        self.rpc_client.stop()
        self.proc.terminate()
        self.proc.join(10)
        if not self.proc.is_alive():
            self.proc.close()
        else:
            logger.warning('MP process did not terminate in time, force killing.')
            self.proc.kill()
        self.proc = None

    def start_loop(self) -> None:
        """Start mp engine loop."""
