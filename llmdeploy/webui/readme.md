## Quick Start

1. 拉取镜像
   ```shell
   docker pull allentdan/llama:v1
   ```
2. 启动容器，这里用一个 13B 的llama 为例，映射两张卡到容器。
   ```shell
   docker run -v /nvme/shared_data:/shared_data -v ~/codes:/codes --name 13B -p 4567:4567 --gpus '"device=6,7"' -ti allentdan/llama:v1
   ```
3. 启动 PyTorch 服务
   ```shell
   cd /codes/llama_service/llama_service/fairscale
   torchrun --nproc_per_node 2 communication/service.py --ckpt_dir /shared_data/chatpjlm-0/v0.2.1-13B/llama --tokenizer_path /shared_data/chatpjlm-0/llamav4.model
   ```
4. 另启一个命令行终端，进docker，启动 tritonserver
   ```shell
   docker exec -ti 13B /bin/bash
   cd /codes/llama_service/llama_service/fairscale
   tritonserver --model-repository triton_model --grpc-port=4567
   ```
5. 物理机运行 webui，例如物理机的ip地址是 10.140.24.140，选用服务端口 6006
   ```shell
   python llama_service/webui/app.py --cfg configs/chatpjlm_0_v0.2.2_fairscale.py --triton_server_addr localhost:4567 --server_name 10.140.24.140 --server_port 6006
   ```

浏览器打开 10.140.24.140:6006 就可以直接使用了。
