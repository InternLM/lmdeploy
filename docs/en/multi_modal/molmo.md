# Molmo

LMDeploy supports the following molmo series of models, which are detailed in the table below:

|      Model      | Size | Supported Inference Engine |
| :-------------: | :--: | :------------------------: |
| Molmo-7B-D-0924 |  7B  |         TurboMind          |
|  Molmo-72-0924  | 72B  |         TurboMind          |

The next chapter demonstrates how to deploy a molmo model using LMDeploy, with [Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) as an example.

## Installation

Please install LMDeploy by following the [installation guide](../get_started/installation.md)

## Offline inference

The following sample code shows the basic usage of VLM pipeline. For detailed information, please refer to [VLM Offline Inference Pipeline](./vl_pipeline.md)

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('allenai/Molmo-7B-D-0924')

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

pipe = pipeline('allenai/Molmo-7B-D-0924', log_level='INFO')
messages = [
    dict(role='user', content=[
        dict(type='text', text='Describe the two images in detail.'),
        dict(type='image_url', image_url=dict(url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Beijing_Small.jpeg')),
        dict(type='image_url', image_url=dict(url='https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Chongqing_Small.jpeg'))
    ])
]
out = pipe(messages, gen_config=GenerationConfig(do_sample=False))

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='What are the similarities and differences between these two images.'))
out = pipe(messages, gen_config=GenerationConfig(do_sample=False))
```

</details>

## Online serving

You can launch the server by the `lmdeploy serve api_server` CLI:

```shell
lmdeploy serve api_server allenai/Molmo-7B-D-0924
```

You can also start the service using the docker image:

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:latest \
    lmdeploy serve api_server allenai/Molmo-7B-D-0924
```

If you find the following logs, it means the service launches successfully.

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
