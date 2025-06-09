import asyncio
import multiprocessing as mp


class TestZMQRPC:

    def sub_proc(self, shared_dict=None, condition=None):
        from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCServer
        server = AsyncRPCServer(shared_dict, condition)

        async def streaming_method(name):
            for i in range(3):
                yield f'{name}: streaming method {i}'

        def method(name):
            return f'{name}: method'

        async def async_method(name):
            return f'{name}: async method'

        def close():
            print('close server...')
            server.stop()

        server.register_method('method', method)
        server.register_method('async_method', async_method)
        server.register_method('streaming_method', streaming_method)
        server.register_method('close', close)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        asyncio.run(server.run())

    async def async_main(self, port):
        from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCClient
        client = AsyncRPCClient(port=port)

        loop = asyncio.get_event_loop()
        _ = loop.create_task(client.listen())

        # Example usage
        result = client.call('async_method', 'test2')
        assert result == 'test2: async method'
        result = await client.async_call('method', 'test1')
        assert result == 'test1: method'

        idx = 0
        async for result in client.async_stream_call('streaming_method', 'test3'):
            assert result == f'test3: streaming method {idx}'
            idx += 1

        await client.async_call('close')

    def test_zmq_rpc(self):
        with mp.Manager() as manager:
            shared_dict = manager.dict()
            condition = manager.Condition()
            ctx = mp.get_context('spawn')
            proc = ctx.Process(target=self.sub_proc, args=(shared_dict, condition), daemon=True)
            proc.start()

            with condition:
                if 'rpc_server_port' not in shared_dict:
                    condition.wait()
            port = shared_dict['rpc_server_port']

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        asyncio.run(self.async_main(port))

        proc.join()
