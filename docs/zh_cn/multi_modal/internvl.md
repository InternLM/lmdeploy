# internvl

## 简介

InternVL2 是由OpenGVLab开源的一个强大的开源视觉语言模型（VLM）. LMDeploy 在PyTorch后端以及TurboMind 后端均支持 InternVL2 系列模型 [OpenGVLab/InternVL2](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e) , 这里以 InternVL2-26B 模型[OpenGVLab/InternVL2-26B](https://huggingface.co/OpenGVLab/InternVL2-26B)为例来说明如何使用通lmdeploy框架部署InternVL系列模型；

### 环境配置及镜像构建

安装docker以及docker-compose以及[compose-gpu-support](https://docs.docker.com/compose/gpu-support/),并确保您的宿主机cuda版本，若宿主机的cuda大版本为cuda11，请确保其高于11.8；若宿主机的cuda大版本为cuda12，请确保其高于12.2；若镜像内的cuda版本高于宿主机的cuda版本，镜像内的Pytorch将无法使用；完成环境配置后，在lmdeploy项目根目录执行：

```shell
#build the base image,it only use to base LLM
docker build -t openmmlab/lmdeploy:latest-cu12 . -f ./docker/Dockerfile
#build the internvl need image,it can be use to VLM like intervl2
docker build -t openmmlab/lmdeploy:latest-internvl-cu12 . -f ./docker/InternVL_Dockerfile
```

### 启动

完成镜像构建后，在lmdeploy项目根目录创建docker-compose.yml配置文件;其中环境变量CUDA_VISIBLE_DEVICES可以控制用宿主机上的哪些gpu，--tp参数可以控制使用几块gpu

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

执行启动命令

```shell
docker-compose up -d
```

执行docker logs -f lmdeploy看到以下日志，表明启动成功

```shell
Fetching 33 files: 100%|█████████████████████████████████████████████████████████████████| 33/33 [00:00<00:00, 102908.57it/s]
[WARNING] gemm_config.in is not found; using default GEMM algo
[WARNING] gemm_config.in is not found; using default GEMM algo
[WARNING] gemm_config.in is not found; using default GEMM algo
[WARNING] gemm_config.in is not found; using default GEMM algo
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
INFO:     Started server process [2439]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:23333 (Press CTRL+C to quit)
```

### 测试

可以通过以下python脚本进行问答，以测试internVL2多模态模型的部署情况

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
                #'content': '请列出广州10个适合带女生去购物的地方'
                'content':[
                {"type": "text", "text": "画面中的人物穿的是什么服饰？"},
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
