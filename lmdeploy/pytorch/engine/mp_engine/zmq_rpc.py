# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import inspect
import pickle
from typing import Callable, Dict
from uuid import uuid4

import zmq
import zmq.asyncio
from zmq.asyncio import Context

from lmdeploy.utils import get_logger

from .base_worker import EngineOutputGather

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

    def __init__(self):
        # Warning: DO NOT allow visit rpc server from external network
        # unauthorized access may lead to code execution vulnerability
        address = 'tcp://localhost'
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.port = self.socket.bind_to_random_port(address)
        self.methods: Dict[str, Callable] = {}
        self.running = False

        # streaming
        self.stream_output = dict()
        self._stream_idx = 0
        self._engine_output_gather = EngineOutputGather()

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

    async def _method_async_streaming_task(self, stream_id, method: Callable, args: tuple, kwargs: Dict):
        """Call method in a task for streaming."""
        stream_out = dict(
            event=asyncio.Event(),
            result=None,
            stopped=False,
        )
        self.stream_output[stream_id] = stream_out
        try:
            generator = method(*args, **kwargs)
            async for result in generator:
                self._engine_output_gather.add(stream_id, result)
                stream_out['result'] = result
                stream_out['event'].set()
        except Exception as e:
            stream_out['error'] = e
            stream_out['event'].set()
        finally:
            stream_out['stopped'] = True

    async def get_stream_output(self, stream_id: int):
        """Get streaming output."""
        if stream_id not in self.stream_output:
            raise ValueError(f'Stream ID {stream_id} not found')
        stream_out = self.stream_output[stream_id]
        event = stream_out['event']
        await stream_out['event'].wait()
        event.clear()
        result = stream_out['result']
        stopped = stream_out['stopped']
        result = self._engine_output_gather.pop(stream_id, result)
        if stopped:
            self.stream_output.pop(stream_id)
        if 'error' in stream_out:
            raise Exception(stream_out['error'])
        return result, stopped

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
            stream_id = self._get_next_stream_id()
            event_loop.create_task(self._method_async_streaming_task(stream_id, method, args, kwargs), name=name)
            response = dict(success=True, request_id=request_id, result=stream_id)
            self.send_multipart(client_id, response)
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
        poller = zmq.asyncio.Poller()
        poller.register(self.socket, zmq.POLLIN)

        self.register_method('_asyncrpcserver_get_stream_output', self.get_stream_output)
        try:
            while self.running:
                events = await poller.poll(timeout=10)
                if self.socket in dict(events):
                    await self.call_and_response()
                else:
                    await asyncio.sleep(0)
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
        self._listen_task = None
        self.running = False

    def _set_reply_default(self, request_id: int, reply: Dict):
        """Default reply handler for sync socket."""
        logger.debug(f'recv reply request_id: {request_id}')
        future: asyncio.Future = self.pending.pop(request_id)
        try:
            if reply['success']:
                future.set_result(reply['result'])
            else:
                future.set_exception(Exception(reply['error']))
        except Exception as e:
            logger.debug(f'Set future failed with exception: {e}')

    def _set_reply(self, reply: Dict):
        request_id = reply['request_id']
        self._set_reply_default(request_id, reply)

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

    async def _async_call_impl(self, method, streaming, *args, **kwargs):
        self._try_start_listen()
        request_id = str(uuid4())
        future = asyncio.Future()
        self.pending[request_id] = future

        logger.debug(f'call method: {method}, request_id: {request_id}')
        data = pickle.dumps(dict(request_id=request_id, method=method, args=args, kwargs=kwargs, streaming=streaming))
        await self.async_socket.send(data)

        return await future

    async def async_call(self, method, *args, **kwargs):
        """Async call."""
        return await self._async_call_impl(method, False, *args, **kwargs)

    async def async_stream_call(self, method, *args, **kwargs):
        """Streaming call."""
        stream_id = await self._async_call_impl(method, True, *args, **kwargs)

        stopped = False
        while not stopped:
            output, stopped = await self.async_call('_asyncrpcserver_get_stream_output', stream_id)
            yield output

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
        if self._listen_task is not None:
            self._listen_task.cancel()
        self.close_sockets()

    def close_sockets(self):
        """Close sockets."""
        self.async_socket.close()
        self.sync_socket.close()
        self.async_ctx.term()
        self.sync_ctx.term()
