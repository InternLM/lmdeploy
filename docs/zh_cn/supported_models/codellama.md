# Code Llama

## 模型介绍

[codellama](https://github.com/facebookresearch/codellama) 支持很多种编程语言，包括 Python, C++, Java, PHP, Typescript (Javascript), C#, Bash 等等。具备代码续写、代码填空、对话、python专项等 4 种能力。

它在 [HuggingFace](https://huggingface.co/codellama) 上发布了基座模型，Python模型和指令微调模型：

| 基座模型                                                                        | Python微调模型                                                                                | 指令模型                                                                                          |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf)   | [codellama/CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)   | [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)   |
| [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [codellama/CodeLlama-13b-Python-hf](https://huggingface.co/codellama/CodeLlama-13b-Python-hf) | [codellama/CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf) |
| [codellama/CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf) | [codellama/CodeLlama-34b-Python-hf](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) | [codellama/CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf) |

模型和能力的对应关系为：

| 模型           | 代码续写 | 代码填空          | 对话 | Python专项 |
| -------------- | -------- | ----------------- | ---- | ---------- |
| 基座模型       | Y        | Y(7B,13B), N(34B) | N    | N          |
| Python微调模型 | Y        | N                 | N    | Y          |
| 指令微调模型   | Y        | Y(7B,13B), N(34B) | Y    | N          |

## 推理

根据上述的模型和能力关系表，下载感兴趣的模型。执行如下的命令，把模型权重转成 turbomind 要求的格式：

```shell
# 安装 lmdeploy
python3 -m pip install lmdeploy

# 转模型格式
python3 -m lmdeploy.serve.turbomind.deploy codellama /the/path/of/codellama/model
```

接下来，可参考如下章节，在控制台与 codellama 进行交互式对话。

**注意**: `lmdeploy.turbomind.chat` 支持把代码块拷贝到控制台，务必使用"!!"结束输入

### 代码续写

```shell
python3 -m lmdeploy.turbomind.chat ./workspace --cap completion
```

**注意**: lmdeploy 支持把代码块拷贝到控制台，务必使用"!!"结束输入

### 代码填空

```shell
```

### 对话

```
python3 -m lmdeploy.turbomind.chat ./workspace --cap insturct --sys-instruct "Provide answers in Python"
```

### Python 专项

## 服务
