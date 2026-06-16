# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import inspect
import pickle
from collections.abc import Callable
from uuid import uuid4

import zmq
import zmq.asyncio
from zmq.asyncio import Context

from lmdeploy.utils import get_logger

from .base_worker import EngineOutputGather

logger = get_logger('lmdeploy')


class RPCServerDeadError(RuntimeError):
    """Raised when the ZMQ MP backend process is not alive."""


class _RPCServerLiveness:
    """Track ZMQ MP backend process death."""

    def __init__(self,
                 pending: dict,
                 server_alive_callback: Callable[[], bool] | None = None,
                 server_sentinel: int | None = None,
                 server_dead_callback: Callable[[], None] | None = None):
        # Pending async RPC futures owned by AsyncRPCClient. When the backend
        # dies, every waiter in this dict must be failed so callers do not hang.
        self.pending = pending
        # Optional cheap process-alive probe, currently backed by
        # multiprocessing.Process.is_alive() in ZMQMPEngine.
        self.server_alive_callback = server_alive_callback
        # Optional multiprocessing process sentinel fd. When supported by the
        # event loop, add_reader() wakes us immediately after backend exit.
        self.server_sentinel = server_sentinel
        # Callback used by ZMQMPEngine to wake local stream-init waiters
        # before failing or dropping local session state.
        self.server_dead_callback = server_dead_callback
        # First recorded backend-death reason. Non-None means all later RPCs
        # should fail fast with RPCServerDeadError.
        self.dead_msg = None
        # The loop that owns the sentinel reader. remove_reader() must be
        # called on the same loop where add_reader() succeeded. Non-None also
        # means the sentinel is currently registered.
        self.sentinel_loop = None

    def new_dead_error(self):
        """Create a backend-dead error."""
        msg = self.dead_msg or 'PyTorch ZMQ engine process is not alive.'
        return RPCServerDeadError(msg)

    def mark_dead(self, message: str | None = None, *, log: bool = True):
        """Mark backend dead and fail waiters."""
        first_dead = self.dead_msg is None
        if first_dead:
            self.dead_msg = message or 'PyTorch ZMQ engine process is not alive.'
            if log:
                logger.error(self.dead_msg)
            if self.server_dead_callback is not None:
                try:
                    self.server_dead_callback()
                except Exception:
                    logger.exception('ZMQ MP backend-dead callback failed.')
        if self.sentinel_loop is not None and self.server_sentinel is not None:
            try:
                self.sentinel_loop.remove_reader(self.server_sentinel)
            except Exception:
                logger.debug('Failed to remove ZMQ MP backend sentinel reader.', exc_info=True)
            finally:
                self.sentinel_loop = None
        for request_id, future in list(self.pending.items()):
            if not future.done():
                future.set_exception(self.new_dead_error())
            self.pending.pop(request_id, None)

    def check_alive(self):
        """Raise if the backend is already known to be dead."""
        if self.dead_msg is not None:
            raise self.new_dead_error()
        if self.server_alive_callback is not None and not self.server_alive_callback():
            self.mark_dead()
            raise self.new_dead_error()

    def register_sentinel(self):
        """Register backend process sentinel on the current asyncio loop."""
        if self.server_sentinel is None or self.sentinel_loop is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            loop.add_reader(self.server_sentinel, self.mark_dead)
        except (NotImplementedError, RuntimeError, ValueError, OSError):
            logger.debug('Failed to register ZMQ MP backend sentinel reader.', exc_info=True)
            return
        self.sentinel_loop = loop


def _task_callback(task: asyncio.Task) -> None:
    """Raise exception on finish."""
    task_name = task.get_name()
    try:
        task.result()
    except asyncio.CancelledError:
        logger.debug(f'Task <{task_name}> cancelled.')
    except Exception:
        logger.exception(f'Task <{task_name}> failed')
    finally:
        if not task.done():
            task.cancel()


class AsyncRPCServer:

    def __init__(self):
        # Warning: DO NOT allow visit rpc server from external network
        # unauthorized access may lead to code execution vulnerability
        address = 'tcp://localhost'
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.port = self.socket.bind_to_random_port(address)
        self.methods: dict[str, Callable] = {}
        self.running = False

        # streaming
        self.stream_output = dict()
        self.stream_tasks = dict()
        self._stream_idx = 0
        self._engine_output_gather = EngineOutputGather()

        self.tasks = set()

    def get_port(self):
        return self.port

    def _get_next_stream_id(self):
        """Get next stream id."""
        self._stream_idx += 1
        return self._stream_idx

    def register_method(self, name: str, func: Callable):
        """Register method."""
        if asyncio.iscoroutinefunction(func):
            func_type = 'async'
        elif inspect.isasyncgenfunction(func):
            func_type = 'async_streaming'
        else:
            func_type = 'default'
        self.methods[name] = (func_type, func)

    def send_multipart(self, client_id: bytes, data: bytes):
        """Send multipart message to client."""
        try:
            self.socket.send_multipart([client_id, pickle.dumps(data)])
        except zmq.ZMQError as e:
            logger.error(f'Failed to send message to client[{client_id}]: {e}')

    def call_method_default(self, client_id, method: Callable, request: dict):
        request_id = request.get('request_id')
        args = request.get('args', [])
        kwargs = request.get('kwargs', {})
        try:
            result = method(*args, **kwargs)
            response = dict(success=True, request_id=request_id, result=result)
        except Exception as e:
            logger.exception(f'ZMQ MP sync RPC method failed: method={method.__name__}, request_id={request_id}.')
            response = dict(success=False, request_id=request_id, error=str(e))
        self.send_multipart(client_id, response)

    async def _method_async_task(self, client_id, request_id, method: Callable, args: tuple, kwargs: dict):
        """Call method in a task."""
        try:
            result = await method(*args, **kwargs)
            response = dict(success=True, request_id=request_id, result=result)
        except Exception as e:
            logger.exception(f'ZMQ MP async RPC method failed: method={method.__name__}, request_id={request_id}.')
            response = dict(success=False, request_id=request_id, error=str(e))
        self.send_multipart(client_id, response)

    async def _method_async_streaming_task(self, stream_id: int, request_id: int, client_id: int, method: Callable,
                                           args: tuple, kwargs: dict, startup_notify_kwarg: str | None = None):
        """Call method in a task for streaming."""

        response_sent = False

        def __send_resp():
            nonlocal response_sent
            if response_sent:
                return
            response_sent = True
            response = dict(success=True, request_id=request_id, result=stream_id)
            self.send_multipart(client_id, response)

        stream_out = dict(
            event=asyncio.Event(),
            result=None,
            stopped=False,
            pending=False,
        )
        self.stream_output[stream_id] = stream_out
        kwargs = dict(kwargs)
        if startup_notify_kwarg is not None:
            origin_notify_func = kwargs.get(startup_notify_kwarg)

            def _notify_startup():
                try:
                    if origin_notify_func is not None:
                        origin_notify_func()
                finally:
                    __send_resp()

            kwargs[startup_notify_kwarg] = _notify_startup
        else:
            __send_resp()
        try:
            generator = method(*args, **kwargs)
            async for result in generator:
                self._engine_output_gather.add(stream_id, result)
                stream_out['result'] = result
                stream_out['pending'] = True
                stream_out['event'].set()
        except Exception as e:
            logger.exception(f'ZMQ MP stream task failed: stream_id={stream_id}, request_id={request_id}, '
                             f'method={method.__name__}.')
            stream_out['error'] = e
            stream_out['event'].set()
        finally:
            if not response_sent:
                __send_resp()
            stream_out['stopped'] = True
            if not stream_out['pending']:
                stream_out['result'] = None
            stream_out['event'].set()

    async def get_stream_output(self, stream_id: int):
        """Get streaming output."""
        stream_out = self.stream_output.get(stream_id)
        if stream_out is None:
            return None, True
        event = stream_out['event']
        await event.wait()
        event.clear()
        result = stream_out['result']
        stopped = stream_out['stopped']
        stream_out['pending'] = False
        result = self._engine_output_gather.pop(stream_id, result)
        if stopped:
            self.stream_output.pop(stream_id, None)
            self.stream_tasks.pop(stream_id, None)
        if 'error' in stream_out:
            raise stream_out['error']
        return result, stopped

    def drop_stream_output(self, stream_id: int):
        """Drop an abandoned streaming call."""
        stream_out = self.stream_output.get(stream_id, None)
        task = self.stream_tasks.pop(stream_id, None)
        if task is not None and task.done():
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(f'ZMQ MP abandoned stream task state check failed: stream_id={stream_id}.')
            else:
                if exc is not None:
                    logger.error(
                        f'ZMQ MP abandoned stream task finished with exception: stream_id={stream_id}.',
                        exc_info=(type(exc), exc, exc.__traceback__))
        elif task is not None:
            task.cancel()
        if stream_out is not None:
            stream_out['stopped'] = True
            if not stream_out['pending']:
                stream_out['result'] = None
            stream_out['event'].set()
        self.stream_output.pop(stream_id, None)
        self._engine_output_gather.discard(stream_id)

    async def call_method_async(self, client_id, method: Callable, request: dict):
        """Call method async."""
        request_id = request.get('request_id')
        method_name = request.get('method')
        args = request.get('args', [])
        kwargs = request.get('kwargs', {})
        startup_notify_kwarg = request.get('streaming_startup_notify_kwarg', None)
        event_loop = asyncio.get_event_loop()
        name = f'{method_name}_{client_id}'
        if request.get('streaming', False):
            # if method is a streaming method, use a different task
            stream_id = self._get_next_stream_id()
            task = event_loop.create_task(self._method_async_streaming_task(stream_id, request_id, client_id, method,
                                                                            args, kwargs, startup_notify_kwarg),
                                          name=name)
            self.tasks.add(task)
            self.stream_tasks[stream_id] = task

            def _discard_stream_task(done_task, stream_id=stream_id):
                self.tasks.discard(done_task)
                self.stream_tasks.pop(stream_id, None)

            task.add_done_callback(_discard_stream_task)
        else:
            task = event_loop.create_task(self._method_async_task(client_id, request_id, method, args, kwargs),
                                          name=name)
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def call_and_response(self):
        """Call method."""
        # receive message: [client_id, empty, request_data]
        client_id, request_data = self.socket.recv_multipart()
        request = pickle.loads(request_data)

        method_name = request.get('method')
        logger.debug(f'call method: {method_name}')
        if method_name not in self.methods:
            request_id = request.get('request_id')
            response = dict(success=False, request_id=request_id, error=f'Method {method_name} not found')
            self.send_multipart(client_id, response)
        else:
            method_type, method = self.methods[method_name]
            if method_type in ('async', 'async_streaming'):
                await self.call_method_async(client_id, method, request)
            else:
                self.call_method_default(client_id, method, request)

    async def run(self):
        logger.info('Starting AsyncRPCServer...')
        self.running = True
        poller = zmq.asyncio.Poller()
        poller.register(self.socket, zmq.POLLIN)

        self.register_method('_asyncrpcserver_get_stream_output', self.get_stream_output)
        self.register_method('_asyncrpcserver_drop_stream_output', self.drop_stream_output)
        try:
            events = await poller.poll(timeout=10)
            while self.running:
                while self.socket in dict(events):
                    await self.call_and_response()
                    events = await poller.poll(timeout=0)
                events = await poller.poll(timeout=10)

        except zmq.ZMQError:
            logger.exception('ZMQRPCServer error')
        except Exception:
            logger.exception('AsyncRPCServer error')
        finally:
            logger.info('Stopping AsyncRPCServer...')
            self.socket.close()
            self.context.term()
            self.running = False

    def stop(self):
        self.running = False
        for task in self.tasks:
            task.cancel()


class AsyncRPCClient:

    def __init__(self,
                 port: int = 5555,
                 server_alive_callback: Callable[[], bool] | None = None,
                 server_sentinel: int | None = None,
                 server_dead_callback: Callable[[], None] | None = None):
        logger.info(f'Connecting to AsyncRPCServer on port {port}...')
        address = f'tcp://localhost:{port}'

        socket_type = zmq.DEALER

        # sync socket
        self.sync_ctx = zmq.Context()
        self.sync_socket = self.sync_ctx.socket(socket_type)
        self.sync_socket.connect(address)
        self.sync_poller = zmq.Poller()
        self.sync_poller.register(self.sync_socket, zmq.POLLIN)

        # async socket
        self.async_ctx = Context.instance()
        self.async_socket = self.async_ctx.socket(socket_type)
        self.async_socket.connect(address)

        self.pending = {}
        self._server_liveness = _RPCServerLiveness(
            self.pending,
            server_alive_callback=server_alive_callback,
            server_sentinel=server_sentinel,
            server_dead_callback=server_dead_callback,
        )
        self._listen_task = None
        self.running = False

    def _mark_server_dead(self, message: str | None = None, *, log: bool = True):
        """Mark backend dead and fail waiters."""
        self._server_liveness.mark_dead(message, log=log)

    def _check_server_alive(self):
        """Raise if the backend is already known to be dead."""
        self._server_liveness.check_alive()

    def _register_server_sentinel(self):
        """Register backend process sentinel on the current asyncio loop."""
        self._server_liveness.register_sentinel()

    def _set_reply_default(self, request_id: int, reply: dict):
        """Default reply handler for sync socket."""
        logger.debug(f'recv reply request_id: {request_id}')
        future: asyncio.Future | None = self.pending.pop(request_id, None)
        if future is None:
            logger.debug(f'reply request_id {request_id} has no pending future.')
            return
        if future.done():
            logger.debug(f'reply request_id {request_id} future is already done.')
            return
        try:
            if reply['success']:
                future.set_result(reply['result'])
            else:
                future.set_exception(Exception(reply['error']))
        except Exception as e:
            logger.exception(f'ZMQ MP set reply future failed: request_id={request_id}, error={e}.')

    def _set_reply(self, reply: dict):
        request_id = reply['request_id']
        self._set_reply_default(request_id, reply)

    def _poll_recv(self, timeout: float = 3):
        """Poll and receive message."""
        # socket.recv would block the process, use poll to avoid hanging
        while True:
            self._check_server_alive()
            sockets = dict(self.sync_poller.poll(timeout=timeout * 1000))
            if self.sync_socket in sockets:
                return self.sync_socket.recv()
            self._check_server_alive()

    def _try_start_listen(self):
        """Try to start listening on async socket."""
        self._register_server_sentinel()
        if self._listen_task is None or self._listen_task.done():
            logger.debug('Starting async listen task...')
            self._listen_task = asyncio.create_task(self.listen(), name='AsyncRPCClient.listen')
            self._listen_task.add_done_callback(_task_callback)

    def call(self, method, *args, **kwargs):
        self._check_server_alive()
        request_id = str(uuid4())
        logger.debug(f'call method: {method}, request_id: {request_id}')
        data = pickle.dumps(dict(request_id=request_id, method=method, args=args, kwargs=kwargs))
        self.sync_socket.send(data)
        self._check_server_alive()

        reply = self._poll_recv()
        reply = pickle.loads(reply)
        while reply['request_id'] != request_id:
            self._set_reply(reply)
            reply = self._poll_recv()
            reply = pickle.loads(reply)

        logger.debug(f'recv reply request_id: {request_id}')
        if reply['success']:
            return reply['result']
        else:
            raise Exception(reply['error'])

    async def _async_call_impl(self, method, streaming, *args, _streaming_startup_notify_kwarg: str | None = None,
                               **kwargs):
        self._try_start_listen()
        self._check_server_alive()
        request_id = str(uuid4())
        future = asyncio.Future()
        self.pending[request_id] = future

        logger.debug(f'call method: {method}, request_id: {request_id}')
        data = pickle.dumps(
            dict(
                request_id=request_id,
                method=method,
                args=args,
                kwargs=kwargs,
                streaming=streaming,
                streaming_startup_notify_kwarg=_streaming_startup_notify_kwarg,
            ))

        try:
            await self.async_socket.send(data)
            self._check_server_alive()
            return await future
        except asyncio.CancelledError:
            self.pending.pop(request_id, None)
            future.cancel()
            raise
        except Exception:
            self.pending.pop(request_id, None)
            raise

    async def async_call(self, method, *args, **kwargs):
        """Async call."""
        return await self._async_call_impl(method, False, *args, **kwargs)

    async def async_stream_call(self,
                                method,
                                sess_event: asyncio.Event,
                                *args,
                                streaming_startup_notify_kwarg: str | None = None,
                                **kwargs):
        """Streaming call."""
        stream_task = asyncio.create_task(
            self._async_call_impl(
                method,
                True,
                *args,
                _streaming_startup_notify_kwarg=streaming_startup_notify_kwarg,
                **kwargs,
            ))
        stream_id = None
        stopped = False

        def _mark_init_done(task: asyncio.Task):
            sess_event.set()
            if task.cancelled():
                return
            try:
                exc = task.exception()
            except Exception:
                logger.exception(f'ZMQ MP stream startup task exception check failed: method={method}.')
                return
            if exc is not None:
                logger.error(f'ZMQ MP stream startup failed before init: method={method}.',
                             exc_info=(type(exc), exc, exc.__traceback__))

        def _drop_abandoned_stream(task: asyncio.Task):

            async def _drop_stream():
                try:
                    _stream_id = await task
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception(f'ZMQ MP stream startup task failed before abandoned drop: method={method}.')
                    return

                try:
                    await self.async_call('_asyncrpcserver_drop_stream_output', _stream_id)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception(f'ZMQ MP abandoned stream drop failed: stream_id={_stream_id}, method={method}.')

            asyncio.create_task(_drop_stream(), name='AsyncRPCClient.drop_stream_output')

        stream_task.add_done_callback(_mark_init_done)
        try:
            stream_id = await asyncio.shield(stream_task)
            while not stopped:
                output, stopped = await self.async_call('_asyncrpcserver_get_stream_output', stream_id)
                if output is not None:
                    yield output
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f'ZMQ MP stream call failed: method={method}, stream_id={stream_id}.')
            raise
        finally:
            if not stopped:
                _drop_abandoned_stream(stream_task)

    async def listen(self):
        self._listen_task = asyncio.current_task()
        self.running = True
        try:
            while self.running:
                reply = await self.async_socket.recv()
                reply = pickle.loads(reply)
                self._set_reply(reply)
        except zmq.ZMQError:
            logger.exception('AsyncRPCClient listen error')
        finally:
            self.running = False
            self.close_sockets()

    def stop(self):
        """Stop the client."""
        self.running = False
        self._mark_server_dead('ZMQ RPC client stopped.', log=False)
        if self._listen_task is not None:
            self._listen_task.cancel()
        self.close_sockets()

    def close_sockets(self):
        """Close sockets."""
        self.async_socket.close()
        self.sync_socket.close()
        self.async_ctx.term()
        self.sync_ctx.term()
