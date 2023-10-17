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
```

Then, you can communicate with codellama in consolo by following instructions in next sections

**Note**:

- minimum requirement of `transformers` is **v4.33.0**
- lmdeploy supports copying code blocks to the console. But you have to press enter, input "!!" and press enter again to end the prompt. The way to input prompt for other supported models keeps unchanged, i.e., double pressing enter.

### Completion

```shell
python3 -m lmdeploy.turbomind.chat ./workspace --cap completion
```

### Infilling

```shell
python3 -m lmdeploy.turbomind.chat ./workspace --cap infilling
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
python3 -m lmdeploy.turbomind.chat ./workspace --cap chat --sys-instruct "Provide answers in Python"
```

`--sys-instruct` instruction can be changed to other coding languages as long as codellama supports it

### Python specialist

```
python3 -m lmdeploy.turbomind.chat ./workspace --cap python
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
python3 -m lmdeploy.serve.openai.api_server ./workspace server_ip server_port --instance_num 32 --tp 1
```

Then, you can communicate with it by command line,

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
python -m lmdeploy.serve.openai.api_client restful_api_url
```

or through webui after launching gradio,

```shell
# restful_api_url is what printed in api_server.py, e.g. http://localhost:23333
# server_ip and server_port here are for gradio ui
# example: python -m lmdeploy.serve.gradio.app http://localhost:23333 localhost 6006 --restful_api True
python -m lmdeploy.serve.gradio.app restful_api_url server_ip --restful_api True
```

Regarding the detailed information of RESTful API, you can refer to [restful_api.md](../restful_api.md).
