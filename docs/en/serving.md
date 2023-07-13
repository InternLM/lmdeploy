# Serving a model

## Serving [LLaMA](https://github.com/facebookresearch/llama)

Weights for the LLaMA models can be obtained from by filling out [this form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)

<details open>
<summary><b>7B</b></summary>

```shell
python3 -m lmdeploy.serve.turbomind.deploy llama-7B /path/to/llama-7b llama \
    --tokenizer_path /path/to/tokenizer/model
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>13B</b></summary>

```shell
python3 -m lmdeploy.serve.turbomind.deploy llama-13B /path/to/llama-13b llama \
    --tokenizer_path /path/to/tokenizer/model --tp 2
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>30B</b></summary>

```shell
python3 -m lmdeploy.serve.turbomind.deploy llama-32B /path/to/llama-30b llama \
    --tokenizer_path /path/to/tokenizer/model --tp 4
bash workspace/service_docker_up.sh
```

</details>

<details open>
<summary><b>65B</b></summary>

```shell
python3 -m lmdeploy.serve.turbomind.deploy llama-65B /path/to/llama-65b llama \
    --tokenizer_path /path/to/tokenizer/model --tp 8
bash workspace/service_docker_up.sh
```

</details>

### Serving [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)

<details open>
<summary><b>7B</b></summary>

```shell
python3 -m pip install fschat
python3 -m fastchat.model.apply_delta \
  --base-model-path /path/to/llama-7b \
  --target-model-path /path/to/vicuna-7b \
  --delta-path lmsys/vicuna-7b-delta-v1.1

python3 -m lmdeploy/serve/turbomind/deploy.py vicuna-7B /path/to/vicuna-7b hf
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

python3 -m lmdeploy/serve/turbomind/deploy.py vicuna-13B /path/to/vicuna-13b hf
bash workspace/service_docker_up.sh
```

</details>
