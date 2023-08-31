# Code Llama

## 模型介绍

[codellama](https://github.com/facebookresearch/codellama) 支持很多种编程语言，包括 Python, C++, Java, PHP, Typescript (Javascript), C#, Bash 等等。具备代码续写、代码填空、对话、python专项等 4 种能力。

它在 [HuggingFace](https://huggingface.co/codellama) 上发布了基座模型，Python模型和指令模型：

| 基座模型                                                                        | Python微调模型                                                                                | 指令模型                                                                                          |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf)   | [codellama/CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)   | [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)   |
| [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [codellama/CodeLlama-13b-Python-hf](https://huggingface.co/codellama/CodeLlama-13b-Python-hf) | [codellama/CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf) |
| [codellama/CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf) | [codellama/CodeLlama-34b-Python-hf](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) | [codellama/CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf) |

模型和能力的对应关系为：

| 能力       | 基座模型          | Python微调模型 | 指令模型          |
| ---------- | ----------------- | -------------- | ----------------- |
| 代码续写   | Y                 | Y              | Y                 |
| 代码填空   | Y(7B,13B), N(34B) | N              | Y(7B,13B), N(34B) |
| 对话       | N                 | N              | Y                 |
| Python专项 | N                 | Y              | N                 |

## 推理

根据上述的模型和能力关系表，下载感兴趣的模型。执行如下的命令，即可在控制台和模型对话：

```shell
# 安装 lmdeploy
python3 -m pip install lmdeploy

# 转模型格式
python3 -m lmdeploy.serve.turbomind.deploy codellama /the/path/of/codellama/model

# 在控制台与模型对话
# --cap 可选择 completion, infill, instruct, python
python3 -m lmdeploy.demo.codellama ./workspace --cap <capability>
```

lmdeploy 支持把代码块拷贝到控制台，务必使用"!!"结束输入。如下图所示：
