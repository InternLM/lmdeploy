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
python3 -m pip install lmdeploy[all]

# convert weight layout
lmdeploy convert codellama /the/path/of/codellama/model
```

Then, you can communicate with codellama in consolo by following instructions in next sections

**Note**:

- minimum requirement of `transformers` is **v4.33.0**
- lmdeploy supports copying code blocks to the console. But you have to press enter, input "!!" and press enter again to end the prompt. The way to input prompt for other supported models keeps unchanged, i.e., double pressing enter.

### Completion

```shell
lmdeploy chat turbomind ./workspace --cap completion
```

### Infilling

```shell
lmdeploy chat turbomind ./workspace --cap infilling
```

The input code is supposed to have a special placeholder `<FILL>`. For example,

```
def remove_non_ascii(s: str) -> str:
    """ <FILL>
    return result
```

And the generated code piece by `turbomind.chat` is the one to be filled in `<FILL>`

### Chat

```
lmdeploy chat turbomind ./workspace --cap chat --sys-instruct "Provide answers in Python"
```

`--sys-instruct` instruction can be changed to other coding languages as long as codellama supports it

### Python specialist

```
lmdeploy chat turbomind ./workspace --cap python
```

Python fine-tuned model is highly recommended when 'python specialist' capability is required.

## Quantization

TBD

## Serving

**LMDeploy server only supports `chat` capabllity**. The res ones are going to be supported soon.

Launch inference server by:

```shell
# --instance_num: number of instances to performance inference, which can be viewed as max requests concurrency
# --tp: the number of GPUs used in tensor parallelism
lmdeploy serve api_server ./workspace --server_name ${server_ip} --server_port ${server_port} --instance_num 32 --tp 1
```

Then, you can communicate with it by command line,

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
lmdeploy serve api_client api_server_url
```

or through webui after launching gradio,

```shell
# api_server_url is what printed in api_server.py, e.g. http://localhost:23333
# server_ip and server_port here are for gradio ui
# example: lmdeploy serve gradio http://localhost:23333 --server_name localhost --server_port 6006
lmdeploy serve gradio api_server_url --server_name ${gradio_ui_ip} --server_port ${gradio_ui_port}
```

Regarding the detailed information of RESTful API, you can refer to [restful_api.md](../serving/restful_api.md).
