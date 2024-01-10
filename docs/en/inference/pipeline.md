# Pipeline

## Example

An example using default parameters:

```python
import lmdeploy

pipe = lmdeploy.pipeline('internlm/internlm-chat-7b')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

An example showing how to set tensor parallel num:

```python
import lmdeploy
from lmdeploy.turbomind import EngineConfig

backend_config = EngineConfig(tp=2)
pipe = lmdeploy.pipeline('internlm/internlm-chat-7b',
                         backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

An example for setting sampling parameters:

```python
import lmdeploy
from lmdeploy.messages import GenerationConfig
from lmdeploy.turbomind import EngineConfig

backend_config = EngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = lmdeploy.pipeline('internlm/internlm-chat-7b',
                         backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
print(response)
```

An example for OpenAI format prompt input:

```python
import lmdeploy
from lmdeploy.messages import GenerationConfig
from lmdeploy.turbomind import EngineConfig

backend_config = EngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = lmdeploy.pipeline('internlm/internlm-chat-7b',
                         backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
response = pipe(prompts,
                gen_config=gen_config)
print(response)
```

Below is an example for pytorch backend. Please install triton first.

```shell
pip install triton>=2.1.0
```

```python
import lmdeploy
from lmdeploy.messages import GenerationConfig
from lmdeploy.pytorch import EngineConfig

backend_config = EngineConfig(session_len=2024)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = lmdeploy.pipeline('internlm/internlm-chat-7b',
                         backend='pytorch',
                         backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
response = pipe(prompts, gen_config=gen_config)
print(response)
```

## `pipeline` API

The `pipeline` function is a higher-level API designed for users to easily instantiate and use the AsyncEngine.

### Init parameters:

| Parameter            | Type                                                 | Description                                                                                                                          | Default     |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
| model_path           | str                                                  | Path to the model. Can be a path to a local directory storing a Turbomind model, or a model_id for models hosted on huggingface.co.  | N/A         |
| model_name           | Optional\[str\]                                      | Name of the model when the model_path points to a Pytorch model on huggingface.co.                                                   | None        |
| backend              | Literal\['turbomind', 'pytorch'\]                    | Specifies the backend to use, either turbomind or pytorch.                                                                           | 'turbomind' |
| backend_config       | TurbomindEngineConfig \| PytorchEngineConfig \| None | Configuration object for the backend. It can be either TurbomindEngineConfig or PytorchEngineConfig depending on the backend chosen. | None        |
| chat_template_config | Optional\[ChatTemplateConfig\]                       | Configuration for chat template.                                                                                                     | None        |
| instance_num         | int                                                  | The number of instances to be created for handling concurrent requests.                                                              | 32          |
| tp                   | int                                                  | Number of tensor parallelunits. Will be deprecated later, please use backend_config.                                                 | 1           |
| log_level            | str                                                  | The level of logging.                                                                                                                | 'ERROR'     |

### Invocation

| Parameter Name     | Data Type                | Default Value | Description                                                                                                                                                                                                                      |
| ------------------ | ------------------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| prompts            | List\[str\]              | None          | A batch of prompts.                                                                                                                                                                                                              |
| gen_config         | GenerationConfig or None | None          | An instance of GenerationConfig. Default is None.                                                                                                                                                                                |
| do_preprocess      | bool                     | True          | Whether to pre-process the messages. Default is True, which means chat_template will be applied.                                                                                                                                 |
| request_output_len | int                      | 512           | The number of output tokens. This parameter will be deprecated. Please use the gen_config parameter instead.                                                                                                                     |
| top_k              | int                      | 40            | The number of the highest probability vocabulary tokens to keep for top-k-filtering. This parameter will be deprecated. Please use the gen_config parameter instead.                                                             |
| top_p              | float                    | 0.8           | If set to a float \< 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. This parameter will be deprecated. Please use the gen_config parameter instead. |
| temperature        | float                    | 0.8           | Used to modulate the next token probability. This parameter will be deprecated. Please use the gen_config parameter instead.                                                                                                     |
| repetition_penalty | float                    | 1.0           | The parameter for repetition penalty. 1.0 means no penalty. This parameter will be deprecated. Please use the gen_config parameter instead.                                                                                      |
| ignore_eos         | bool                     | False         | Indicator for ignoring end-of-string (eos). This parameter will be deprecated. Please use the gen_config parameter instead.                                                                                                      |

## EngineConfig (turbomind)

### Description

This class provides the configuration parameters for TurboMind backend.

### Arguments

| Parameter             | Type          | Description                                                                                              | Default |
| --------------------- | ------------- | -------------------------------------------------------------------------------------------------------- | ------- |
| model_name            | str, Optional | The name of the deployed model.                                                                          | None    |
| model_format          | str, Optional | The layout of the deployed model. Can be one of the following values: hf, llama, awq.                    | None    |
| group_size            | int           | The group size used when quantizing weights to 4-bit.                                                    | 0       |
| tp                    | int           | The number of GPU cards used in tensor parallelism.                                                      | 1       |
| session_len           | int, Optional | The maximum session length of a sequence.                                                                | None    |
| max_batch_size        | int           | The maximum batch size during inference.                                                                 | 128     |
| max_context_token_num | int           | The maximum number of tokens to be processed in each forward pass.                                       | 1       |
| cache_max_entry_count | float         | The percentage of GPU memory occupied by the k/v cache.                                                  | 0.5     |
| cache_block_seq_len   | int           | The length of a sequence in a k/v block.                                                                 | 128     |
| cache_chunk_size      | int           | The number of blocks each time the TurboMind engine tries to realloc from GPU memory.                    | -1      |
| num_tokens_per_iter   | int           | Number of tokens to be processed per iteration.                                                          | 0       |
| max_prefill_iters     | int           | Maximum prefill iterations for a single request.                                                         | 1       |
| use_context_fmha      | int           | Whether or not to use fmha in context decoding.                                                          | 1       |
| quant_policy          | int           | Set it to 4 when k/v is quantized into 8 bits.                                                           | 0       |
| rope_scaling_factor   | float         | Scaling factor used for dynamic ntk. TurboMind follows the implementation of transformer LlamaAttention. | 0.0     |
| use_dynamic_ntk       | bool          | Whether or not to use dynamic ntk.                                                                       | False   |
| use_logn_attn         | bool          | Whether or not to use logarithmic attention.                                                             | False   |

## EngineConfig (pytorch)

### Description

This class provides the configuration parameters for Pytorch backend.

### Arguments

| Parameter        | Type | Description                                                                                              | Default     |
| ---------------- | ---- | -------------------------------------------------------------------------------------------------------- | ----------- |
| model_name       | str  | Name of the given model.                                                                                 | ''          |
| tp               | int  | Tensor Parallelism.                                                                                      | 1           |
| session_len      | int  | Maximum session length.                                                                                  | None        |
| max_batch_size   | int  | Maximum batch size.                                                                                      | 128         |
| eviction_type    | str  | Action to perform when kv cache is full. Options are \['recompute', 'copy'\].                            | 'recompute' |
| prefill_interval | int  | Interval to perform prefill.                                                                             | 16          |
| block_size       | int  | Paging cache block size.                                                                                 | 64          |
| num_cpu_blocks   | int  | Number of CPU blocks. If the number is 0, cache would be allocated according to the current environment. | 0           |
| num_gpu_blocks   | int  | Number of GPU blocks. If the number is 0, cache would be allocated according to the current environment. | 0           |

## GenerationConfig

### Description

This class contains the generation parameters used by inference engines.

### Arguments

| Parameter          | Type        | Description                                                                                                           | Default |
| ------------------ | ----------- | --------------------------------------------------------------------------------------------------------------------- | ------- |
| n                  | int         | Number of chat completion choices to generate for each input message.                                                 | 1       |
| max_new_tokens     | int         | Maximum number of tokens that can be generated in chat completion.                                                    | 512     |
| top_p              | float       | Nucleus sampling, where the model considers the tokens with top_p probability mass.                                   | 1.0     |
| top_k              | int         | The model considers the top_k tokens with the highest probability.                                                    | 1       |
| temperature        | float       | Sampling temperature.                                                                                                 | 0.8     |
| repetition_penalty | float       | Penalty to prevent the model from generating repeated words or phrases. A value larger than 1 discourages repetition. | 1.0     |
| ignore_eos         | bool        | Indicator to ignore the eos_token_id or not.                                                                          | False   |
| random_seed        | int         | Seed used when sampling a token.                                                                                      | None    |
| stop_words         | List\[str\] | Words that stop generating further tokens.                                                                            | None    |
| bad_words          | List\[str\] | Words that the engine will never generate.                                                                            | None    |

## FAQs

- *RuntimeError: context has already been set*. If you got this for tp>1 in pytorch backend. Please make sure the python script has following
  ```python
  if __name__ == '__main__':
  ```
  Generally, in the context of multi-threading or multi-processing, it might be necessary to ensure that initialization code is executed only once. In this case, `if __name__ == '__main__':` can help to ensure that these initialization codes are run only in the main program, and not repeated in each newly created process or thread.
