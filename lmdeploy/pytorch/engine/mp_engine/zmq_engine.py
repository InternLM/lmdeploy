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
                 trust_remote_code: bool = False,
                 **kwargs) -> None:
        """Initialize mp engine."""
        from .zmq_rpc import AsyncRPCClient
        self.shared_dict = None
        self.port = None
        self.proc = None
        self._start_mp_proc(model_path, engine_config, speculative_config=speculative_config,
                            trust_remote_code=trust_remote_code, **kwargs)

        self.rpc_client = AsyncRPCClient(
            port=self.port,
            server_alive_callback=self._is_proc_alive,
            server_sentinel=self.proc.sentinel if self.proc is not None else None,
            server_dead_callback=self._mark_backend_dead,
        )

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
                while 'rpc_server_port' not in self.shared_dict:
                    if not self.proc.is_alive():
                        raise RuntimeError('PyTorch ZMQ engine process exited before publishing RPC server port.')
                    condition.wait(timeout=1)
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

        # try rename the process
        try:
            import ctypes
            ctypes.CDLL(None).prctl(15, b'ZMQMPEngine', 0, 0, 0)
        except Exception as e:
            logger.debug(f'Failed to rename MPEngine process: {e}')

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
        except Exception:
            logger.exception('RPC Server stopped with exception.')
        finally:
            server.stop()
            engine.close()
            try:
                await engine.wait_tasks()
            except asyncio.CancelledError:
                logger.info('Engine wait_tasks cancelled during shutdown.')
            except Exception:
                logger.exception('Engine wait_tasks failed during shutdown.')

    def _collective_rpc(self, func, *args, **kwargs):
        """Collective rpc call."""
        return self.rpc_client.call(func, *args, **kwargs)

    async def _collective_rpc_async(self, func, *args, **kwargs):
        """Collective rpc call."""
        return await self.rpc_client.async_call(func, *args, **kwargs)

    async def _collective_rpc_streaming_async(self, func: str, sess_event: asyncio.Event,  *args, **kwargs):
        """Collective rpc call."""
        startup_notify_kwarg = 'notify_add_msg_func' if func == 'instance_async_stream_infer' else None
        async for out in self.rpc_client.async_stream_call(
                func,
                sess_event,
                *args,
                streaming_startup_notify_kwarg=startup_notify_kwarg,
                **kwargs,
        ):
            yield out

    async def get_health_status(self):
        """Get backend health status."""
        if self.proc is None or not self.proc.is_alive():
            return dict(alive=False,
                        message='PyTorch ZMQ engine process is not alive.',
                        schedule_metrics=None)
        return await super().get_health_status()

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

    def _is_proc_alive(self):
        """Return whether MP process is alive."""
        try:
            return self.proc is not None and self.proc.is_alive()
        except ValueError:
            return False

    def _mark_backend_dead(self):
        """Wake local waiters when the MP backend process dies."""
        for state in getattr(self, 'session_states', {}).values():
            state.init_done.set()

    def start_loop(self) -> None:
        """Start mp engine loop."""
