# Speculative Decoding

投机解码是一种优化技术，它通过引入轻量级草稿模型来预测多个后续token，再由主模型在前向推理过程中验证并选择匹配度最高的长token序列。与标准的自回归解码相比，这种方法可使系统一次性生成多个token。

> \[!NOTE\]
> 请注意，这是lmdeploy中的实验性功能。

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
