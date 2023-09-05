# codellama

## Introduction

[codellama](https://github.com/facebookresearch/codellama) features enhanced coding capabilities. It can generate code and natural language about code, from both code and natural language prompts (e.g., “Write me a function that outputs the fibonacci sequence”). It can also be used for code completion and debugging. It supports many of the most popular programming languages used today, including Python, C++, Java, PHP, Typescript (Javascript), C#, Bash and more.

There are three sizes (7b, 13b, 34b) as well as three flavours (base model, Python fine-tuned, and instruction tuned) released on [HuggingFace](https://huggingface.co/codellama).

| Base Model                                                                      | Python                                                                                        | Instruct                                                                                          |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf)   | [codellama/CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)   | [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)   |
| [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [codellama/CodeLlama-13b-Python-hf](https://huggingface.co/codellama/CodeLlama-13b-Python-hf) | [codellama/CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf) |
| [codellama/CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf) | [codellama/CodeLlama-34b-Python-hf](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) | [codellama/CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf) |

The correspondence between the model and capabilities is:

| models     | code completion | infilling         | instructions / chat | python specialist |
| ---------- | --------------- | ----------------- | ------------------- | ----------------- |
| Base Model | Y               | Y(7B,13B), N(34B) | N                   | N                 |
| Python     | Y               | N                 | N                   | Y                 |
| Instruct   | Y               | Y(7B,13B), N(34B) | Y                   | N                 |

## Inference

Based on the above table, download the model that meets your requirements. Execute the following command to interact with the model in the console:

```shell
# install lmdeploy
python3 -m pip install lmdeploy

# convert weight layout
python3 -m lmdeploy.serve.turbomind.deploy codellama /the/path/of/codellama/model

# interactive communicate with AI assistant in console
# --cap is among [completion, infill, instruct, python] with default value instruct
python3 -m lmdeploy.turbomind.chat ./workspace --cap <capability>
```

**Note**: lmdeploy supports copying code blocks to the console. Make sure to end input with "!!".

## Serving

### Serving with Restful API

Launch inference server by:

```shell
python3 -m lmdeploy.serve.openai.api_server ./workspace server_ip server_port --instance_num 32 --tp 1
```

Then, you can communicate with it by command line,

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
python -m lmdeploy.serve.openai.api_client restful_api_url
```

or webui,

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
# server_ip and server_port here are for gradio ui
# example: python -m lmdeploy.serve.gradio.app http://localhost:23333 localhost 6006 --restful_api True
python -m lmdeploy.serve.gradio.app restful_api_url server_ip --restful_api True
```

Refer to [restful_api.md](docs/en/restful_api.md) for more details.

### Serving with gradio

```shell
python3 -m lmdeploy.serve.gradio.app ./workspace
```
