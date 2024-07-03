# Triton Inference Server 性能测试

Triton Inference Server(TIS) 是 LMDeploy 支持的除了 api_server 之外的另一种 serving 方式。它的性能测试方式和测试指标和 [api_server](./profile_api_server.md) 的测试方式类似。

```{note}
LMDeploy 尚未实现 Triton Inference Server 的 ensemble 推理模式，所以推理性能要比 api_server 弱。对于追求性能的用户，我们推荐使用 api_server 部署服务。
```

TIS 性能测试脚本是 `profile_serving.py`。测试之前，请安装 lmdeploy 预编译包，并下载评测脚本和测试数据集。

```shell
pip install 'lmdeploy[serve]'
git clone --depth=1 https://github.com/InternLM/lmdeploy
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## 测量指标

LMDeploy 统计首token延时（first_token_latency）、token吞吐量（tokens/s）和请求吞吐量（RPM）。

`first_token_latency` 只有在流式推理的情况下才会输出。

token吞吐量的计算公式为：

$$
吞吐量 = 生成的token数量 / 总时间
$$

请求吞吐量的计算公式为：

$$
吞吐量 = 请求数量 / 总时间
$$

总时间包括 prefill 时间

## 测量方法

我们以 [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) 为例，展示 triton inference server 的性能测试流程

### 启动服务

启动服务之前，必须先把模型转换为 turbomind 模型格式：

```shell
lmdeploy convert internlm internlm/internlm-7b --dst-path ./internlm-7b --trust-remote-code
```

然后，执行如下命令，启动服务：

```shell
bash ./internlm-7b/service_docker_up.sh
```

### 测速

```shell
python3 profile_serving.py 0.0.0.0:33337 ./internlm-7b/triton_models/tokenizer ./ShareGPT_V3_unfiltered_cleaned_split.json
```

关于 `profile_serving.py` 脚本中的参数，比如请求并发数、采样参数等等，可以通过运行命令 `python3 profile_serving.py -h` 查阅。
