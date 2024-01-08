# 模型服务

## 部署 [LLaMA-2](https://github.com/facebookresearch/llama) 服务

请从[这里](https://huggingface.co/meta-llama) 下载 llama2 模型，参考如下命令部署服务：

<details open>
<summary><b>7B</b></summary>

```shell
lmdeploy convert llama2 /path/to/llama-2-7b-chat-hf
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>13B</b></summary>

```shell
lmdeploy convert llama2 /path/to/llama-2-13b-chat-hf --tp 2
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>70B</b></summary>

```shell
lmdeploy convert llama2 /path/to/llama-2-70b-chat-hf --tp 8
bash workspace/service_docker_up.sh
```

</details>

## 部署 [LLaMA](https://github.com/facebookresearch/llama) 服务

请填写[这张表](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)，获取 LLaMA 模型权重

<details open>
<summary><b>7B</b></summary>

```shell
lmdeploy convert llama /path/to/llama-7b llama \
    --tokenizer_path /path/to/tokenizer/model
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>13B</b></summary>

```shell
lmdeploy convert llama /path/to/llama-13b llama \
    --tokenizer_path /path/to/tokenizer/model --tp 2
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>30B</b></summary>

```shell
lmdeploy convert llama /path/to/llama-30b llama \
    --tokenizer_path /path/to/tokenizer/model --tp 4
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>65B</b></summary>

```shell
lmdeploy convert llama /path/to/llama-65b llama \
    --tokenizer_path /path/to/tokenizer/model --tp 8
bash workspace/service_docker_up.sh
```

</details>

### 部署 [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) 服务

<details open>
<summary><b>7B</b></summary>

```shell
python3 -m pip install fschat
python3 -m fastchat.model.apply_delta \
  --base-model-path /path/to/llama-7b \
  --target-model-path /path/to/vicuna-7b \
  --delta-path lmsys/vicuna-7b-delta-v1.1

lmdeploy convert vicuna /path/to/vicuna-7b
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>13B</b></summary>

```shell
python3 -m pip install fschat
python3 -m fastchat.model.apply_delta \
  --base-model-path /path/to/llama-13b \
  --target-model-path /path/to/vicuna-13b \
  --delta-path lmsys/vicuna-13b-delta-v1.1

lmdeploy convert vicuna /path/to/vicuna-13b
bash workspace/service_docker_up.sh
```

</details>
