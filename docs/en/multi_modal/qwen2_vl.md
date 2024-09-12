# Qwen2-VL

LMDeploy supports the following Qwen-VL series of models, which are detailed in the table below:

|    Model     |  Size  | Supported Inference Engine |
| :----------: | :----: | :------------------------: |
| Qwen-VL-Chat |   -    |     TurboMind, Pytorch     |
|   Qwen2-VL   | 2B, 7B |          PyTorch           |

The next chapter demonstrates how to deploy an Qwen-VL model using LMDeploy, with [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) as an example.

## Installation

Please install LMDeploy by following the [installation guide](../get_started/installation.md), and install other packages that Qwen2-VL needs

```shell
pip install qwen_vl_utils
```

Or, you can build a docker image to set up the inference environment. If the CUDA version on your host machine is `>=12.4`, you can run:

```
git clone https://github.com/InternLM/lmdeploy.git
cd lmdeploy
docker build --build-arg CUDA_VERSION=cu12 -t openmmlab/lmdeploy:qwen2vl . -f ./docker/Qwen2VL_Dockerfile
```

Otherwise, you can go with:

```shell
docker build --build-arg CUDA_VERSION=cu11 -t openmmlab/lmdeploy:qwen2vl . -f ./docker/Qwen2VL_Dockerfile
```

## Offline inference

The following sample code shows the basic usage of VLM pipeline. For detailed information, please refer to [VLM Offline Inference Pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('Qwen/Qwen2-VL-2B-Instruct')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe((f'describe this image', image))
print(response)
```

More examples are listed below:

<details>
  <summary>
    <b>multi-image multi-round conversation, combined images</b>
  </summary>

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('Qwen/Qwen2-VL-2B-Instruct', log_level='INFO')
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

<details>
  <summary>
    <b>image resolution for performance boost</b>
  </summary>

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('Qwen/Qwen2-VL-2B-Instruct', log_level='INFO')

min_pixels = 64 * 28 * 28
max_pixels = 64 * 28 * 28
messages = [
    dict(role='user', content=[
        dict(type='text', text='Describe the two images in detail.'),
        dict(type='image_url', image_url=dict(min_pixels=min_pixels, max_pixels=max_pixels, url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Beijing_Small.jpeg')),
        dict(type='image_url', image_url=dict(min_pixels=min_pixels, max_pixels=max_pixels, url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Chongqing_Small.jpeg'))
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
lmdeploy serve api_server Qwen/Qwen2-VL-2B-Instruct
```

You can also start the service using the aforementioned built docker image:

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:qwen2vl \
    lmdeploy serve api_server Qwen/Qwen2-VL-2B-Instruct
```

The docker compose is another option. Create a `docker-compose.yml` configuration file in the root directory of the lmdeploy project as follows:

```yaml
version: '3.5'

services:
  lmdeploy:
    container_name: lmdeploy
    image: openmmlab/lmdeploy:qwen2vl
    ports:
      - "23333:23333"
    environment:
      HUGGING_FACE_HUB_TOKEN: <secret>
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    stdin_open: true
    tty: true
    ipc: host
    command: lmdeploy serve api_server Qwen/Qwen2-VL-2B-Instruct
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
