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
python3 -m pip install lmdeploy[all]

# 转模型格式
lmdeploy convert codellama /path/of/codellama/model
```

接下来，可参考如下章节，在控制台与 codellama 进行交互式对话。

**注意**:

- **transformers最低要求 v4.33.0**
- `lmdeploy.turbomind.chat` 支持把代码块拷贝到控制台，**结束输出的方式为回车，再输入"!!"，再回车**。其他非 codellama 模型，仍然是两次回车结束输入。

### 代码续写

```shell
lmdeploy chat turbomind ./workspace --cap completion
```

### 代码填空

```shell
lmdeploy chat turbomind ./workspace --cap infilling
```

输入的代码块中要包含 `<FILL>`，比如：

```
def remove_non_ascii(s: str) -> str:
    """ <FILL>
    return result
```

`turbomind.chat` 输出的代码即是要填到 `<FILL>` 中的内容

### 对话

```
lmdeploy chat turbomind ./workspace --cap chat --sys-instruct "Provide answers in Python"
```

可以把 `--sys-instruct` 的指令换成 codellama 支持的其他变成语言。

### Python 专项

```
lmdeploy chat turbomind ./workspace --cap python
```

建议这里部署 Python 微调模型

## 量化

TBD

## 服务

**目前，server 支持的是对话功能**，其余功能后续再加上。

启动 sever 的方式是：

```shell
# --instance_num: turbomind推理实例的个数。可理解为支持的最大并发数
# --tp: 在 tensor parallel时，使用的GPU数量
lmdeploy serve api_server ./workspace --server_name 0.0.0.0 --server_port ${server_port} --instance_num 32 --tp 1
```

打开 `http://{server_ip}:{server_port}`，即可访问 swagger，查阅 RESTful API 的详细信息。

你可以用命令行，在控制台与 server 通信：

```shell
# api_server_url 就是 api_server 产生的，比如 http://localhost:23333
lmdeploy serve api_client api_server_url
```

或者，启动 gradio，在 webui 的聊天对话框中，与 codellama 交流：

```shell
# api_server_url 就是 api_server 产生的，比如 http://localhost:23333
# server_ip 和 server_port 是用来提供 gradio ui 访问服务的
# 例子: lmdeploy serve gradio http://localhost:23333 --server_name localhost --server_port 6006
lmdeploy serve gradio api_server_url --server_name ${gradio_ui_ip} --server_port ${gradio_ui_port}
```

关于 RESTful API的详细介绍，请参考[这份](../serving/restful_api.md)文档。
