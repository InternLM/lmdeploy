# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import inspect
import multiprocessing as mp
import pickle
from typing import Callable, Dict
from uuid import uuid4

import zmq
from zmq.asyncio import Context

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _task_callback(task: asyncio.Task) -> None:
    """Raise exception on finish."""
    task_name = task.get_name()
    try:
        task.result()
    except asyncio.CancelledError:
        logger.debug(f'Task <{task_name}> cancelled.')
        return
    except Exception:
        logger.exception(f'Task <{task_name}> failed')
    finally:
        if not task.done():
            task.cancel()


class AsyncRPCServer:

    def __init__(self, shared_dict: dict = None, condition: mp.Condition = None):
        address = 'tcp://*'
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.port = self.socket.bind_to_random_port(address)
        if condition is not None:
            with condition:
                shared_dict['rpc_server_port'] = self.port
                condition.notify()
        self.methods: Dict[str, Callable] = {}
        self.running = False

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
        self.socket.send_multipart([client_id, pickle.dumps(data)])

    def call_method_default(self, client_id, method: Callable, request: Dict):
        request_id = request.get('request_id')
        args = request.get('args', [])
        kwargs = request.get('kwargs', {})
        try:
            result = method(*args, **kwargs)
            response = dict(success=True, request_id=request_id, result=result)
        except Exception as e:
            response = dict(success=False, request_id=request_id, error=str(e))
        self.send_multipart(client_id, response)

    async def _method_async_task(self, client_id, request_id, method: Callable, args: tuple, kwargs: Dict):
        """Call method in a task."""
        try:
            result = await method(*args, **kwargs)
            response = dict(success=True, request_id=request_id, result=result)
        except Exception as e:
            response = dict(success=False, request_id=request_id, error=str(e))
        self.send_multipart(client_id, response)

    async def _method_async_streaming_task(self, client_id, request_id, method: Callable, args: tuple, kwargs: Dict):
        """Call method in a task for streaming."""
        try:
            generator = method(*args, **kwargs)
            result = await anext(generator)
            async for next_result in generator:
                response = dict(success=True, request_id=request_id, result=result, streaming_end=False)
                self.send_multipart(client_id, response)
                result = next_result
            response = dict(success=True, request_id=request_id, result=result, streaming_end=True)
            self.send_multipart(client_id, response)
        except Exception as e:
            response = dict(success=False, request_id=request_id, error=str(e))
            self.send_multipart(client_id, response)

    def call_method_async(self, client_id, method: Callable, request: Dict):
        """Call method async."""
        request_id = request.get('request_id')
        method_name = request.get('method')
        args = request.get('args', [])
        kwargs = request.get('kwargs', {})
        event_loop = asyncio.get_event_loop()
        name = f'{method_name}_{client_id}'
        if request.get('streaming', False):
            # if method is a streaming method, use a different task
            event_loop.create_task(self._method_async_streaming_task(client_id, request_id, method, args, kwargs),
                                   name=name)
        else:
            event_loop.create_task(self._method_async_task(client_id, request_id, method, args, kwargs), name=name)

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
                self.call_method_async(client_id, method, request)
            else:
                self.call_method_default(client_id, method, request)

    async def run(self):
        logger.info('Starting AsyncRPCServer...')
        self.running = True
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        event_loop = asyncio.get_event_loop()
        try:
            while self.running:
                events = await event_loop.run_in_executor(None, poller.poll, 100)  # 100ms timeout
                if self.socket in dict(events):
                    await self.call_and_response()
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


class StreamingOutput:
    """Streaming output."""

    def __init__(self):
        self.event = asyncio.Event()
        self.stopped = False
        self.result = None

    def set_result(self, result, stopped: bool = False):
        """Set result for streaming output."""
        self.result = result
        self.stopped = stopped
        self.event.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Awaitable for streaming output."""
        if self.stopped and not self.event.is_set():
            raise StopAsyncIteration
        await self.event.wait()
        self.event.clear()
        return self.result


class AsyncRPCClient:

    def __init__(self, port: int = 5555):
        logger.info(f'Connecting to AsyncRPCServer on port {port}...')
        address = f'tcp://localhost:{port}'

        socket_type = zmq.DEALER

        # sync socket
        self.sync_ctx = zmq.Context()
        self.sync_socket = self.sync_ctx.socket(socket_type)
        self.sync_socket.connect(address)

        # async socket
        self.async_ctx = Context.instance()
        self.async_socket = self.async_ctx.socket(socket_type)
        self.async_socket.connect(address)

        self.pending = {}
        self.stream_pending = {}
        self._listen_task = None
        self.running = False

    def _set_reply_default(self, request_id: int, reply: Dict):
        """Default reply handler for sync socket."""
        logger.debug(f'recv reply request_id: {request_id}')
        future: asyncio.Future = self.pending.pop(request_id)
        if reply['success']:
            future.set_result(reply['result'])
        else:
            future.set_exception(Exception(reply['error']))

    def _set_reply_streaming(self, request_id: int, reply: Dict):
        """Streaming reply handler for async socket."""
        logger.debug(f'recv stream reply request_id: {request_id}')
        result = self.stream_pending[request_id]
        if reply['success']:
            stopped = reply.get('streaming_end', False)
            result.set_result(reply['result'], stopped=stopped)
            if stopped:
                self.stream_pending.pop(request_id)
        else:
            result.set_result(reply['result'], True)
            self.stream_pending.pop(request_id)
            raise Exception(reply['error'])

    def _set_reply(self, reply: Dict):
        request_id = reply['request_id']
        if request_id in self.pending:
            self._set_reply_default(request_id, reply)
        elif request_id in self.stream_pending:
            self._set_reply_streaming(request_id, reply)

    def _try_start_listen(self):
        """Try to start listening on async socket."""
        if self._listen_task is None or self._listen_task.done():
            logger.debug('Starting async listen task...')
            self._listen_task = asyncio.create_task(self.listen(), name='AsyncRPCClient.listen')
            self._listen_task.add_done_callback(_task_callback)

    def call(self, method, *args, **kwargs):
        request_id = str(uuid4())
        logger.debug(f'call method: {method}, request_id: {request_id}')
        data = pickle.dumps(dict(request_id=request_id, method=method, args=args, kwargs=kwargs))
        self.sync_socket.send(data)

        reply = self.sync_socket.recv()
        reply = pickle.loads(reply)
        while reply['request_id'] != request_id:
            self._set_reply(reply)
            reply = self.sync_socket.recv()
            reply = pickle.loads(reply)

        logger.debug(f'recv reply request_id: {request_id}')
        if reply['success']:
            return reply['result']
        else:
            raise Exception(reply['error'])

    async def async_call(self, method, *args, **kwargs):
        self._try_start_listen()
        request_id = str(uuid4())
        future = asyncio.Future()
        self.pending[request_id] = future

        logger.debug(f'call method: {method}, request_id: {request_id}')
        data = pickle.dumps(dict(request_id=request_id, method=method, args=args, kwargs=kwargs))
        await self.async_socket.send(data)

        return await future

    async def async_stream_call(self, method, *args, **kwargs):
        self._try_start_listen()
        request_id = str(uuid4())
        output = StreamingOutput()
        self.stream_pending[request_id] = output

        logger.debug(f'call method: {method}, request_id: {request_id}')
        data = pickle.dumps(dict(request_id=request_id, method=method, args=args, kwargs=kwargs, streaming=True))
        await self.async_socket.send(data)
        return output

    async def listen(self):
        self._listen_task = asyncio.current_task()
        self.running = True
        while self.running:
            # # get as many replies as possible from sync socket
            # while True:
            #     try:
            #         reply = self.sync_socket.recv(zmq.NOBLOCK)
            #         reply = pickle.loads(reply)
            #         self._set_reply(reply)
            #     except zmq.Again:
            #         break
            reply = await self.async_socket.recv()
            reply = pickle.loads(reply)
            self._set_reply(reply)

    def stop(self):
        """Stop the client."""
        self.running = False
        if self._listen_task is not None:
            self._listen_task.cancel()
        self.async_socket.close()
        self.sync_socket.close()
        self.async_ctx.term()
        self.sync_ctx.term()
