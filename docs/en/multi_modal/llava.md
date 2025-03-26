# LLaVA

LMDeploy supports the following llava series of models, which are detailed in the table below:

|                Model                 | Size | Supported Inference Engine |
| :----------------------------------: | :--: | :------------------------: |
| llava-hf/Llava-interleave-qwen-7b-hf |  7B  |     TurboMind, PyTorch     |
|       llava-hf/llava-1.5-7b-hf       |  7B  |     TurboMind, PyTorch     |
|  llava-hf/llava-v1.6-mistral-7b-hf   |  7B  |          PyTorch           |
|   llava-hf/llava-v1.6-vicuna-7b-hf   |  7B  |          PyTorch           |
|   liuhaotian/llava-v1.6-mistral-7b   |  7B  |         TurboMind          |
|   liuhaotian/llava-v1.6-vicuna-7b    |  7B  |         TurboMind          |

The next chapter demonstrates how to deploy an Llava model using LMDeploy, with [llava-hf/llava-interleave](https://huggingface.co/llava-hf/llava-interleave-qwen-7b-hf) as an example.

```{note}
PyTorch engine removes the support of original llava models after v0.6.4. Please use their corresponding transformers models instead, which can be found in https://huggingface.co/llava-hf
```

## Installation

Please install LMDeploy by following the [installation guide](../get_started/installation.md).

Or, you can go with office docker image:

```shell
docker pull openmmlab/lmdeploy:latest
```

## Offline inference

The following sample code shows the basic usage of VLM pipeline. For detailed information, please refer to [VLM Offline Inference Pipeline](./vl_pipeline.md)

```python
from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image


pipe = pipeline("llava-hf/llava-interleave-qwen-7b-hf", backend_config=TurbomindEngineConfig(cache_max_entry_count=0.5),
    gen_config=GenerationConfig(max_new_tokens=512))

image = load_image('https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg')
prompt = 'Describe the image.'
print(f'prompt:{prompt}')
response = pipe((prompt, image))
print(response)

```

More examples are listed below:

<details>
  <summary>
    <b>multi-image multi-round conversation, combined images</b>
  </summary>

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('llava-hf/llava-interleave-qwen-7b-hf', log_level='INFO')
messages = [
    dict(role='user', content=[
        dict(type='text', text='Describe the two images in detail.'),
        dict(type='image_url', image_url=dict(url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Beijing_Small.jpeg')),
        dict(type='image_url', image_url=dict(url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Chongqing_Small.jpeg'))
    ])
]
out = pipe(messages, gen_config=GenerationConfig(top_k=1))

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='What are the similarities and differences between these two images.'))
out = pipe(messages, gen_config=GenerationConfig(top_k=1))
```

</details>

## Online serving

You can launch the server by the `lmdeploy serve api_server` CLI:

```shell
lmdeploy serve api_server llava-hf/llava-interleave-qwen-7b-hf
```

You can also start the service using the aforementioned built docker image:

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:latest \
    lmdeploy serve api_server llava-hf/llava-interleave-qwen-7b-hf
```

The docker compose is another option. Create a `docker-compose.yml` configuration file in the root directory of the lmdeploy project as follows:

```yaml
version: '3.5'

services:
  lmdeploy:
    container_name: lmdeploy
    image: openmmlab/lmdeploy:latest
    ports:
      - "23333:23333"
    environment:
      HUGGING_FACE_HUB_TOKEN: <secret>
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    stdin_open: true
    tty: true
    ipc: host
    command: lmdeploy serve api_server llava-hf/llava-interleave-qwen-7b-hf
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: "all"
              capabilities: [gpu]
```

Then, you can execute the startup command as below:

```shell
docker-compose up -d
```

If you find the following logs after running `docker logs -f lmdeploy`, it means the service launches successfully.

```text
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
INFO:     Started server process [2439]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on  http://0.0.0.0:23333  (Press CTRL+C to quit)
```

The arguments of `lmdeploy serve api_server` can be reviewed in detail by `lmdeploy serve api_server -h`.

More information about `api_server` as well as how to access the service can be found from [here](api_server_vl.md)
