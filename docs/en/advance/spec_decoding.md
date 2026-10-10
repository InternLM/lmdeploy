# Speculative Decoding

Speculative decoding is an optimization technique that introcude a lightweight draft model to propose multiple next tokens and then, the main model verify and choose the longest matched tokens in a forward pass. Compared with standard auto-regressive decoding, this methold lets the system generate multiple tokens at once.

:::{note}
This is an experimental feature in lmdeploy.
:::

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
--max-batch-size 128
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
--max-batch-size 128
```

### DFlash

For DFlash, the block size is the complete draft query/target verification
window. It includes one current target token, so a block size of 8 proposes
seven new draft tokens. The block size must not exceed the maximum declared by
the draft checkpoint.

#### pipeline

```python
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig

spec_cfg = SpeculativeConfig(
    method='dflash',
    model='z-lab/Qwen3.5-35B-A3B-DFlash',
    dflash_block_size=8,
)
pipe = pipeline(
    'Qwen/Qwen3.5-35B-A3B',
    backend_config=PytorchEngineConfig(tp=2),
    speculative_config=spec_cfg,
)
```

#### serving

```shell
lmdeploy serve api_server \
Qwen/Qwen3.5-35B-A3B \
--backend pytorch \
--tp 2 \
--speculative-algorithm dflash \
--speculative-draft-model z-lab/Qwen3.5-35B-A3B-DFlash \
--speculative-dflash-block-size 8
```

When a DFlash block size is provided, it overrides
`--speculative-num-draft-tokens` by setting the number of newly proposed
tokens to `block_size - 1`.

## GLM-5.3-Flash prefix caching

GLM-5.3-Flash can reuse prefix caches with ordinary autoregressive decoding or
`deepseek_mtp`. Enable `PytorchEngineConfig.enable_prefix_caching` in either mode.
The following text-only example uses TP4; choose a TP size and cache budget that
fit your hardware. Set `speculative_config=None` to disable MTP without disabling
prefix caching.

```python
from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig

if __name__ == '__main__':
    engine = PytorchEngineConfig(
        tp=4,
        max_batch_size=4,
        enable_prefix_caching=True,
        prefix_cache_state_budget=8,
        max_prefill_token_num=512,
        language_model_only=True,
    )
    spec = SpeculativeConfig(method='deepseek_mtp', num_speculative_tokens=3)
    with pipeline('/path/to/GLM-5.3-Flash', backend_config=engine,
                  speculative_config=spec) as pipe:
        notes = 'A rectangle has four sides and four right angles. ' * 80
        prompt = notes + '\nHow many sides does a rectangle have?'
        generation = GenerationConfig(max_new_tokens=128, do_sample=False)
        cold = pipe(prompt, gen_config=generation)
        warm = pipe(prompt, gen_config=generation)
        print(cold.cached_tokens, warm.cached_tokens)
```

### Configuration limits

- Cache blocks contain 64 tokens. If setting `num_gpu_blocks` manually, count
  these blocks rather than KPool entries.
- `prefix_cache_state_budget` reserves extra checkpoint slots and uses additional
  memory. Zero adds no reserved slots; idle runtime slots may still be reused.
- With MTP, keep `prefix_cache_decode_state_interval=0`. Without MTP, positive
  values must be multiples of 64.
- PD migration is not supported.

The native FP8 CUDA path also supports DP/EP serving, for example `--tp 2 --dp 2
--ep 4`. Enable text-prefill PCG with `--piecewise-cudagraph-max-tokens 512`
and `--max-prefill-token-num 512`. Startup warmup prepares these plans; skipping
warmup leaves unprepared prefills eager. Vision-bearing chunks also fall back to
eager execution. MTP draft prefill remains eager, while supported decode steps
use the ordinary CUDA graph path. These choices are independent of prefix caching.

Inspect `Response.cached_tokens` to verify actual reuse. Changing prefill,
PCG padding, batch shapes or parallel topology can change FP8 generation trajectories; compare task accuracy and
MTP acceptance as well as cache hits rather than assuming bitwise-identical
outputs or a guaranteed speedup.

## Guided Decoding with Speculative Decoding

Speculative decoding (MTP) can be combined with [structured output](./structed_output.md) so that the draft tokens proposed by the spec model also respect the grammar constraints (e.g. JSON schema, regex). This significantly improves the acceptance rate compared to running spec decoding without grammar masks.

:::{note}
This feature is supported for spec methods that inherit from `DeepseekMTP`, including `deepseek_mtp`, `qwen3_5_mtp`, and `eagle3`. Only the PyTorch backend is supported.
:::

### How it works

The grammar mask is applied at two stages:

1. **Draft model** — forked grammar matchers are used to mask each draft position serially. Each position's mask depends on the token accepted at the previous position, ensuring the draft model proposes grammatically valid tokens.
2. **Target model verification** — position-serial grammar masking is applied to the target model's logits. After rejection sampling, only the accepted tokens are fed back to the original (un-forked) grammar matchers, keeping them in sync for the next step.

When the draft model uses a different vocabulary from the target model (e.g. Eagle 3 with a compressed draft vocabulary), the target-vocab bitmask produced by xgrammar is translated to a draft-vocab bitmask via an efficient scatter-add kernel before being applied to the draft logits.

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

if __name__ == '__main__':

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

if __name__ == '__main__':

  client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:24545/v1')
  model_name = client.models.list().data[0].id
  response = client.chat.completions.create(
      model=model_name,
      messages=[{'role': 'user', 'content': 'Introduce yourself as JSON.'}],
      response_format=response_format,
  )
  print(response)
```
