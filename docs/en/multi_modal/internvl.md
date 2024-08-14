# internvl

## Introduction

InternVL2 is a powerful open-source visual language model (VLM) developed by OpenGVLab, LMDeploy now supports the InternVL2 series models on both PyTorch backend and TurboMind backend \[OpenGVLab/InternVL2\]（ https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e ）Here, we use the InternVL2-26B model \[OpenGVLab/InternVL2-26B\]（ https://huggingface.co/OpenGVLab/InternVL2-26B ）As an example, illustrate how to deploy the InternVL series models using the TLMDeploy framework;

### Environment configuration and image construction

Install Docker, Docker compose, and \[compose gpu support\]（ https://docs.docker.com/compose/gpu-support/ ）And ensure that your host's CUDA version is higher than 11.8 if the host's CUDA major version is CUDA11; If the CUDA major version of the host is CUDA12, please ensure that it is higher than 12.2; If the CUDA version in the image is higher than the CUDA version of the host computer, Pytorch in the image will not be available; After completing the environment configuration, execute in the root directory of the lmdeploy project:

```shell
#build the base image, it only use to base LLM
docker build -t openmmlab/lmdeploy:latest-cu12 . -f ./docker/Dockerfile
#build the internvl need image, it can be use to VLM like intervl2
docker build -t openmmlab/lmdeploy:latest-internvl-cu12 . -f ./docker/InternVL_Dockerfile
```

### Start up

After completing the image construction, create a docker-compose.yml configuration file in the root directory of the lmdeploy project; The environment variable CUDA_VISIBLEDEVICES can control which GPUs are used on the host machine, and the -- tp parameter can control how many GPUs are used

```yaml
version: '3.5'

services:
  lmdeploy:
    container_name: lmdeploy
    image: openmmlab/lmdeploy:latest-internvl-cu12
    ports:
      - "23333:23333"
    environment:
      HUGGING_FACE_HUB_TOKEN: <secret>
      CUDA_VISIBLE_DEVICES: 0,1,2,3
    volumes:
      #- ~/.cache/huggingface:/root/.cache/huggingface
      - ./root:/root
    stdin_open: true
    tty: true
    ipc: host #for nccl,it is necessary
    command: lmdeploy serve api_server OpenGVLab/InternVL2-26B  --server-name 0.0.0.0 --server-port 23333  --tp 4 --model-name internvl2-internlm2 --cache-max-entry-count 0.5
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: "all"
              capabilities: [gpu]
```

Execute the startup command

```shell
docker-compose up -d
```

Execute Docker logs - f lmdeploy and see the folllowing logs, indicating successful startup

```shell
Fetching 33 files: 100%|█████████████████████████████████████████████████████████████████| 33/33 [00:00<00:00, 102908.57it/s]
[WARNING] gemm_config.in is not found;  using default GEMM algo
[WARNING] gemm_config.in is not found;  using default GEMM algo
[WARNING] gemm_config.in is not found;  using default GEMM algo
[WARNING] gemm_config.in is not found;  using default GEMM algo
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
HINT:    Please open  http://0.0.0.0:23333   in a browser for detailed api usage!!!
INFO:     Started server process [2439]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on  http://0.0.0.0:23333  (Press CTRL+C to quit)
```

### Testing

The following Python script can be used for Q&A to test the deployment of the internVL2 multimodal model

```python
import asyncio
from openai import OpenAI

async def main():
    client = OpenAI(api_key='YOUR_API_KEY',
                         base_url='http://0.0.0.0:23333/v1')
    model_cards = await client.models.list()._get_page()
    response = await client.chat.completions.create(
        model=model_cards.data[0].id,
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                #'content': ' provide three suggestions about time management'
                #Please list 10 places in Guangzhou that are suitable for taking girls shopping
                'content':[
                {"type": "text", "text": "What kind of clothing are the characters wearing in the picture?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://c-ssl.duitang.com/uploads/item/201707/12/20170712134209_chaxS.jpeg",
                    },
                },
            ]
            },
        ],
        temperature=0.8,
        top_p=0.8)
    print(response)

asyncio.run(main())
```
