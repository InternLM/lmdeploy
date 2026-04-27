# Speculative Decoding

Speculative decoding is an optimization technique that introcude a lightweight draft model to propose multiple next tokens and then, the main model verify and choose the longest matched tokens in a forward pass. Compared with standard auto-regressive decoding, this methold lets the system generate multiple tokens at once.

> \[!NOTE\]
> This is an experimental feature in lmdeploy.

## Examples

Here are some examples.

### Eagle 3

#### Prepare

Install [flash-atten3 ](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release)

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

#### Prepare

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

## Guided Decoding with Speculative Decoding

Speculative decoding (MTP) can be combined with [structured output](./structed_output.md) so that the draft tokens proposed by the spec model also respect the grammar constraints (e.g. JSON schema, regex). This significantly improves the acceptance rate compared to running spec decoding without grammar masks.

> \[!NOTE\]
> This feature is supported for spec methods that inherit from `DeepseekMTP`, including `deepseek_mtp`, `qwen3_5_mtp`, and `eagle3` (with caveats below). Only the PyTorch backend is supported.

### How it works

The grammar mask is applied at two stages:

1. **Draft model** — forked grammar matchers are used to mask each draft position serially. Each position's mask depends on the token accepted at the previous position, ensuring the draft model proposes grammatically valid tokens.
2. **Target model verification** — position-serial grammar masking is applied to the target model's logits. After rejection sampling, only the accepted tokens are fed back to the original (un-forked) grammar matchers, keeping them in sync for the next step.

For **Eagle3**, the draft vocabulary may differ from the target vocabulary, so a grammar mask cannot be directly applied to draft-vocab logits. In this case, the draft model proposes freely and grammar constraints are enforced solely on the target side via rejection sampling.

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
response = pipe(['Introduce yourself as JSON.'], gen_config=gen_config)
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

The client can then use `response_format` as described in the [structured output](./structed_output.md) documentation:

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
    messages=[{'role': 'user', 'content': 'Introduce yourself as JSON.'}],
    response_format=response_format,
)
print(response)
```
