# Speculative Decoding

投机解码是一种优化技术，它通过引入轻量级草稿模型来预测多个后续token，再由主模型在前向推理过程中验证并选择匹配度最高的长token序列。与标准的自回归解码相比，这种方法可使系统一次性生成多个token。

:::{note}
请注意，这是lmdeploy中的实验性功能。
:::

## 示例

请参考如下使用示例。

### Eagle 3

#### 安装依赖

安装 [flash-atten3 ](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release)

```shell
git clone --depth=1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

#### pipeline

```python
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig


if __name__ == '__main__':

    model_path = 'meta-llama/Llama-3.1-8B-Instruct'
    spec_cfg = SpeculativeConfig(
        method='eagle3',
        num_speculative_tokens=3,
        model='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
    )
    pipe = pipeline(model_path, backend_config=PytorchEngineConfig(max_batch_size=128), speculative_config=spec_cfg)
    response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
    print(response)

```

#### serving

```shell
lmdeploy serve api_server \
meta-llama/Llama-3.1-8B-Instruct \
--backend pytorch \
--server-port 24545 \
--speculative-draft-model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
--speculative-algorithm eagle3 \
--speculative-num-draft-tokens 3 \
--max-batch-size 128 \
--enable-metrics
```

### Deepseek MTP

#### 安装依赖

Install [FlashMLA](https://github.com/deepseek-ai/FlashMLA?tab=readme-ov-file#installation)

```shell
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v .
```

#### pipeline

```python
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig


if __name__ == '__main__':

    model_path = 'deepseek-ai/DeepSeek-V3'
    spec_cfg = SpeculativeConfig(
        method='deepseek_mtp',
        num_speculative_tokens=3,
    )
    pipe = pipeline(model_path,
                    backend_config=PytorchEngineConfig(tp=16, max_batch_size=128),
                    speculative_config=spec_cfg)
    response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
    print(response)
```

#### serving

```shell
lmdeploy serve api_server \
deepseek-ai/DeepSeek-V3 \
--backend pytorch \
--server-port 24545 \
--tp 16 \
--speculative-algorithm deepseek_mtp \
--speculative-num-draft-tokens 3 \
--max-batch-size 128 \
--enable-metrics
```

## 投机解码与结构化输出

投机解码（MTP）可以与[结构化输出](./structed_output.md)结合使用，使草稿模型提出的 token 也遵循语法约束（如 JSON Schema、正则表达式），从而显著提高接受率。

:::{note}
该功能支持继承自 `DeepseekMTP` 的投机方法，包括 `deepseek_mtp`、`qwen3_5_mtp` 和 `eagle3`（注意事项见下文）。仅支持 PyTorch 后端。
:::

### 工作原理

语法掩码在两个阶段分别施加：

1. **草稿模型** — 使用 fork 出的语法匹配器，逐位置串行施加掩码。每个位置的掩码依赖于前一位置接受的 token，确保草稿模型提出符合语法的 token。
2. **主模型验证** — 对主模型的 logits 进行逐位置串行的语法掩码处理。拒绝采样后，仅将接受的 token 反馈给原始（未 fork 的）语法匹配器，使其为下一步保持正确的状态。

对于 **Eagle3**，草稿词表可能与目标词表不同，因此无法直接对草稿词表的 logits 施加目标语法掩码。此时草稿模型自由提出 token，语法约束完全由主模型侧的拒绝采样来保证。

### pipeline

```python
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import GenerationConfig, SpeculativeConfig

model_path = 'deepseek-ai/DeepSeek-V3'
spec_cfg = SpeculativeConfig(method='deepseek_mtp', num_speculative_tokens=3)
pipe = pipeline(
    model_path,
    backend_config=PytorchEngineConfig(tp=16, max_batch_size=128),
    speculative_config=spec_cfg,
)

schema = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'age': {'type': 'integer'},
    },
    'required': ['name', 'age'],
}
gen_config = GenerationConfig(
    response_format=dict(type='json_schema', json_schema=dict(name='person', schema=schema)),
    max_new_tokens=256,
)
response = pipe(['请用 JSON 格式做自我介绍。'], gen_config=gen_config)
print(response)
```

### api_server

```shell
lmdeploy serve api_server \
deepseek-ai/DeepSeek-V3 \
--backend pytorch \
--server-port 24545 \
--tp 16 \
--speculative-algorithm deepseek_mtp \
--speculative-num-draft-tokens 3 \
--max-batch-size 128
```

客户端可以按照[结构化输出](./structed_output.md)文档中的方式使用 `response_format`：

```python
from openai import OpenAI

schema = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'age': {'type': 'integer'},
    },
    'required': ['name', 'age'],
}
response_format = dict(type='json_schema', json_schema=dict(name='person', schema=schema))

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:24545/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[{'role': 'user', 'content': '请用 JSON 格式做自我介绍。'}],
    response_format=response_format,
)
print(response)
```
