# PyTorchEngine 多线程推理

自 [PR2907](https://github.com/InternLM/lmdeploy/pull/2907) 起，我们废除了 PytorchEngine 的 thread_safe 模式以保证引擎能够更高效的运行。我们鼓励用户尽可能使用**服务接口**或**协程**来实现高并发，比如：

```python
import asyncio
from lmdeploy import pipeline, PytorchEngineConfig

event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(event_loop)

model_path = 'Llama-3.2-1B-Instruct'
pipe = pipeline(model_path, backend_config=PytorchEngineConfig())

async def _gather_output():
    tasks = [
        pipe.async_batch_infer('Hakuna Matata'),
        pipe.async_batch_infer('giraffes are heartless creatures'),
    ]
    return await asyncio.gather(*tasks)

output = asyncio.run(_gather_output())
print(output[0].text)
print(output[1].text)
```

如果你确实有多线程推理的需求，那么可以进行简单的封装，来实现类似的效果。

```python
import threading
from queue import Queue
import asyncio
from lmdeploy import pipeline, PytorchEngineConfig

model_path = 'Llama-3.2-1B-Instruct'


async def _batch_infer(inque: Queue, outque: Queue, pipe):
    while True:
        if inque.empty():
            await asyncio.sleep(0)
            continue

        input = inque.get_nowait()
        output = await pipe.async_batch_infer(input)
        outque.put(output)


def server(inques, outques):
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    pipe = pipeline(model_path, backend_config=PytorchEngineConfig())
    for inque, outque in zip(inques, outques):
        event_loop.create_task(_batch_infer(inque, outque, pipe))
    event_loop.run_forever()

def client(inque, outque, message):
    inque.put(message)
    print(outque.get().text)


inques = [Queue(), Queue()]
outques = [Queue(), Queue()]

t_server = threading.Thread(target=server, args=(inques, outques))
t_client0 = threading.Thread(target=client, args=(inques[0], outques[0], 'Hakuna Matata'))
t_client1 = threading.Thread(target=client, args=(inques[1], outques[1], 'giraffes are heartless creatures'))

t_server.start()
t_client0.start()
t_client1.start()

t_client0.join()
t_client1.join()
```

> \[!WARNING\]
> 我们不鼓励这样实现，多线程会带来额外的开销，使得推理性能不稳定
