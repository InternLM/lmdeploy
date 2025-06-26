# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import pickle
import signal
from contextlib import asynccontextmanager

import torch.multiprocessing as mp

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def cancel_async_tasks(loop: asyncio.AbstractEventLoop):
    """Cancel async tasks."""
    tasks = asyncio.all_tasks(loop=loop)
    for task in tasks:
        if not task.done():
            task.cancel()
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


class EngineInstancePool:
    """Engine Instance Pool."""

    def __init__(self, engine):
        from lmdeploy.pytorch.engine import Engine
        self.engine: Engine = engine
        self.num_instance = self.engine.engine_config.max_batch_size
        self.pool = None

    def create_instance_pool(self, num_instance: int):
        """Create instance pool."""
        pool = asyncio.Queue(maxsize=num_instance)
        for _ in range(num_instance):
            instance = self.engine.create_instance()
            pool.put_nowait(instance)
        return pool

    @asynccontextmanager
    async def instance(self):
        """Get an instance from the pool."""
        # lazy create pool
        if self.pool is None:
            self.pool = self.create_instance_pool(self.num_instance)
        instance = await self.pool.get()
        try:
            yield instance
        finally:
            self.pool.put_nowait(instance)

    async def async_end(self, session_id: int):
        """End the given session."""
        async with self.instance() as instance:
            return await instance.async_end(session_id)

    async def async_cancel(self, session_id: int):
        """Stop current streaming inference."""
        async with self.instance() as instance:
            return await instance.async_cancel(session_id)

    async def async_stream_infer(self, *args, **kwargs):
        """Send stream inference request."""
        async with self.instance() as instance:
            async for result in instance.async_stream_infer(*args, **kwargs):
                yield result


class MPEngine:

    def __init__(self, model_path: str, tokenizer: object, engine_config: PytorchEngineConfig = None, **kwargs) -> None:
        """Initialize mp engine."""
        from .zmq_rpc import AsyncRPCClient

        self.in_que = mp.Queue()
        self.out_que = mp.Queue()

        self.shared_dict = None
        self.port = None
        self.proc = None
        self._start_mp_proc(model_path, tokenizer, engine_config)

        self.rpc_client = AsyncRPCClient(port=self.port)

        self.engine_config = self._collective_rpc('get_engine_config')
        self.model_config = self._collective_rpc('get_model_config')

    def _start_mp_proc(self, model_path: str, tokenizer: object, engine_config: PytorchEngineConfig = None):
        """Start mp proc."""
        logger.debug('Starting engine multi-process.')
        try:
            pickle.dumps(tokenizer)
        except Exception:
            logger.warning('Failed to pickle tokenizer. It would be created in subprocess.')
            tokenizer = None
        with mp.Manager() as manager:
            self.shared_dict = manager.dict()
            condition = manager.Condition()
            self.mp_ctx = mp.get_context('spawn')
            log_level = logger.level
            self.proc = self.mp_ctx.Process(target=self._mp_proc,
                                            args=(self.shared_dict, condition),
                                            kwargs=(dict(
                                                model_path=model_path,
                                                tokenizer=tokenizer,
                                                engine_config=engine_config,
                                                log_level=log_level,
                                            )),
                                            name='mp_engine_proc',
                                            daemon=True)
            self.proc.start()
            logger.debug('Receiving rpc server port from mp process.')
            with condition:
                if 'rpc_server_port' not in self.shared_dict:
                    condition.wait()
            self.port = self.shared_dict['rpc_server_port']

    @staticmethod
    def _mp_proc(shared_dict: dict,
                 condition: mp.Condition,
                 model_path: str,
                 tokenizer: object,
                 engine_config: PytorchEngineConfig = None,
                 log_level: str = 'WARNING'):
        """Mp process function."""
        from lmdeploy.pytorch.engine import Engine
        from lmdeploy.tokenizer import Tokenizer

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
        if tokenizer is None:
            tokenizer = Tokenizer(model_path)
        engine = Engine.from_pretrained(
            model_path,
            tokenizer=tokenizer,
            engine_config=engine_config,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def _signal_handler(signum, frame):
            """Signal handler to stop the server."""
            logger.info(f'Received signal {signum}, stopping server.')
            exit(0)

        signal.signal(signal.SIGTERM, _signal_handler)
        try:
            loop.run_until_complete(MPEngine._mp_proc_async(server, engine))
        except KeyboardInterrupt:
            logger.info('Received KeyboardInterrupt, stopping mp process.')
        finally:
            server.stop()
            engine.close()
            cancel_async_tasks(loop)

    @staticmethod
    async def _mp_proc_async(server, engine):
        """Mp process function."""
        engine.start_loop()
        instance_pool = EngineInstancePool(engine)

        server.register_method('end_session', engine.end_session)
        server.register_method('get_engine_config', engine.get_engine_config)
        server.register_method('get_model_config', engine.get_model_config)
        server.register_method('p2p_initialize', engine.p2p_initialize)
        server.register_method('p2p_connect', engine.p2p_connect)
        server.register_method('instance_async_end', instance_pool.async_end)
        server.register_method('instance_async_cancel', instance_pool.async_cancel)
        server.register_method('instance_async_stream_infer', instance_pool.async_stream_infer)

        try:
            # run server
            await server.run()
        except Exception as e:
            logger.error(f'RPC Server stopped with exception: {e}')

    def _collective_rpc(self, func, *args, **kwargs):
        """Collective rpc call."""
        return self.rpc_client.call(func, *args, **kwargs)

    async def _collective_rpc_async(self, func, *args, **kwargs):
        """Collective rpc call."""
        return await self.rpc_client.async_call(func, *args, **kwargs)

    async def _collective_rpc_streaming_async(self, func, *args, **kwargs):
        """Collective rpc call."""
        return self.rpc_client.async_stream_call(func, *args, **kwargs)

    def close(self) -> None:
        """Close mp engine."""
        logger.info('Closing mp engine.')
        self.rpc_client.stop()
        self.proc.terminate()
        self.proc.join(10)

    def start_loop(self) -> None:
        """Start mp engine loop."""

    def end_session(self, session_id: int):
        """End session."""
        return self._collective_rpc('end_session', session_id)

    def p2p_initialize(self, conn_request):
        """Init rdma link."""
        return self._collective_rpc('p2p_initialize', conn_request)

    def p2p_connect(self, conn_request):
        """rdma_connect."""
        return self._collective_rpc('p2p_connect', conn_request)

    def create_instance(self, cuda_stream_id=0):
        """Create instance."""
        return MPEngineInstance(self)


class MPEngineInstance:
    """MP Engine Instance."""

    def __init__(self, engine: MPEngine):
        self.engine = engine

    async def async_end(self, session_id: int):
        """End the given session."""
        return await self.engine._collective_rpc_async('instance_async_end', session_id)

    async def async_cancel(self, session_id: int):
        """Stop current streaming inference."""
        return await self.engine._collective_rpc_async('instance_async_cancel', session_id)

    async def async_stream_infer(self, *args, **kwargs):
        """Send stream inference request."""
        generator = await self.engine._collective_rpc_streaming_async('instance_async_stream_infer', *args, **kwargs)
        async for result in generator:
            yield result
