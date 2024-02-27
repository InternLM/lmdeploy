# Inference Pipeline

In this tutorial, We will first present a list of examples to introduce the usage of `lmdeploy.pipeline`.

Then, we will describe the pipeline API in detail.

## Usage

- **An example using default parameters:**

```python
from lmdeploy import pipeline

pipe = pipeline('internlm/internlm2-chat-7b')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

In this example, the pipeline by default allocates a predetermined percentage of GPU memory for storing k/v cache. The ratio is dictated by the parameter `TurbomindEngineConfig.cache_max_entry_count`.

There have been alterations to the strategy for setting the k/v cache ratio throughout the evolution of LMDeploy. The following are the change histories:

1. `v0.2.0 <= lmdeploy <= v0.2.1`

   `TurbomindEngineConfig.cache_max_entry_count` defaults to 0.5, indicating 50% GPU **total memory** allocated for k/v cache. Out Of Memory (OOM) errors may occur if a 7B model is deployed on a GPU with memory less than 40G. If you encounter an OOM error, please decrease the ratio of the k/v cache occupation as follows:

   ```python
   from lmdeploy import pipeline, TurbomindEngineConfig

   # decrease the ratio of the k/v cache occupation to 20%
   backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

   pipe = pipeline('internlm/internlm2-chat-7b',
                   backend_config=backend_config)
   response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
   print(response)
   ```

2. `lmdeploy > v0.2.1`

   The allocation strategy for k/v cache is changed to reserve space from the **GPU free memory** proportionally. The ratio `TurbomindEngineConfig.cache_max_entry_count` has been adjusted to 0.8 by default. If OOM error happens, similar to the method mentioned above, please consider reducing the ratio value to decrease the memory usage of the k/v cache.

- **An example showing how to set tensor parallel num**:

```python
from lmdeploy import pipeline, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

- **An example for setting sampling parameters:**

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
print(response)
```

- **An example for OpenAI format prompt input:**

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
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

- **An example for streaming mode:**

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2-chat-7b',
                backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
for item in pipe.stream_infer(prompts, gen_config=gen_config):
    print(item)
```

- **Below is an example for pytorch backend. Please install triton first.**

```shell
pip install triton>=2.1.0
```

```python
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

backend_config = PytorchEngineConfig(session_len=2048)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm-chat-7b',
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

| Parameter            | Type                                                 | Description                                                                                                                          | Default                                    |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| model_path           | str                                                  | Path to the model. Can be a path to a local directory storing a Turbomind model, or a model_id for models hosted on huggingface.co.  | N/A                                        |
| model_name           | Optional\[str\]                                      | Name of the model when the model_path points to a Pytorch model on huggingface.co.                                                   | None                                       |
| backend_config       | TurbomindEngineConfig \| PytorchEngineConfig \| None | Configuration object for the backend. It can be either TurbomindEngineConfig or PytorchEngineConfig depending on the backend chosen. | None, running turbomind backend by default |
| chat_template_config | Optional\[ChatTemplateConfig\]                       | Configuration for chat template.                                                                                                     | None                                       |
| log_level            | str                                                  | The level of logging.                                                                                                                | 'ERROR'                                    |

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

### Response

| Parameter Name     | Type                                    | Description                                                                                                                                                                                                                                       |
| ------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| text               | str                                     | The text response from the server. If the output text is an empty string and the finish_reason is 'length', it means the maximum session length has been reached.                                                                                 |
| generate_token_len | int                                     | The number of tokens in the response.                                                                                                                                                                                                             |
| input_token_len    | int                                     | The number of tokens in the input prompt. Note that this may include the chat template part.                                                                                                                                                      |
| session_id         | int                                     | The ID for running a session. Basically, it refers to the index position of the input request batch.                                                                                                                                              |
| finish_reason      | Optional\[Literal\['stop', 'length'\]\] | The reason the model stopped generating tokens. This will be set to 'stop' if the model encounters a stop word; if the maximum number of tokens specified in the request is reached or the session length is reached, it will be set to 'length'. |

## TurbomindEngineConfig

### Description

This class provides the configuration parameters for TurboMind backend.

### Arguments

| Parameter             | Type          | Description                                                                                                                           | Default |
| --------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| model_name            | str, Optional | The chat template name of the deployed model, deprecated and has no effect when version > 0.2.1                                       | None    |
| model_format          | str, Optional | The layout of the deployed model. Can be one of the following values: hf, llama, awq.                                                 | None    |
| tp                    | int           | The number of GPU cards used in tensor parallelism.                                                                                   | 1       |
| session_len           | int, Optional | The maximum session length of a sequence.                                                                                             | None    |
| max_batch_size        | int           | The maximum batch size during inference.                                                                                              | 128     |
| cache_max_entry_count | float         | The percentage of GPU memory occupied by the k/v cache.                                                                               | 0.5     |
| quant_policy          | int           | Set it to 4 when k/v is quantized into 8 bits.                                                                                        | 0       |
| rope_scaling_factor   | float         | Scaling factor used for dynamic ntk. TurboMind follows the implementation of transformer LlamaAttention.                              | 0.0     |
| use_logn_attn         | bool          | Whether or not to use logarithmic attention.                                                                                          | False   |
| download_dir          | str, optional | Directory to download and load the weights, default to the default cache directory of huggingface.                                    | None    |
| revision              | str, optional | The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. | None    |

## PytorchEngineConfig

### Description

This class provides the configuration parameters for Pytorch backend.

### Arguments

| Parameter             | Type  | Description                                                                                                                           | Default     |
| --------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| model_name            | str   | The chat template name of the deployed model                                                                                          | ''          |
| tp                    | int   | Tensor Parallelism.                                                                                                                   | 1           |
| session_len           | int   | Maximum session length.                                                                                                               | None        |
| max_batch_size        | int   | Maximum batch size.                                                                                                                   | 128         |
| cache_max_entry_count | float | The percentage of free GPU memory occupied by the k/v cache.                                                                          | 0.8         |
| eviction_type         | str   | Action to perform when kv cache is full. Options are \['recompute', 'copy'\].                                                         | 'recompute' |
| prefill_interval      | int   | Interval to perform prefill.                                                                                                          | 16          |
| block_size            | int   | Paging cache block size.                                                                                                              | 64          |
| num_cpu_blocks        | int   | Number of CPU blocks. If the number is 0, cache would be allocated according to the current environment.                              | 0           |
| num_gpu_blocks        | int   | Number of GPU blocks. If the number is 0, cache would be allocated according to the current environment.                              | 0           |
| adapters              | dict  | The path configs to lora adapters.                                                                                                    | None        |
| download_dir          | str   | Directory to download and load the weights, default to the default cache directory of huggingface.                                    | None        |
| revision              | str   | The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. | None        |

## GenerationConfig

### Description

This class contains the generation parameters used by inference engines.

### Arguments

| Parameter           | Type        | Description                                                                                                           | Default |
| ------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------- | ------- |
| n                   | int         | Number of chat completion choices to generate for each input message. Currently, only 1 is supported                  | 1       |
| max_new_tokens      | int         | Maximum number of tokens that can be generated in chat completion.                                                    | 512     |
| top_p               | float       | Nucleus sampling, where the model considers the tokens with top_p probability mass.                                   | 1.0     |
| top_k               | int         | The model considers the top_k tokens with the highest probability.                                                    | 1       |
| temperature         | float       | Sampling temperature.                                                                                                 | 0.8     |
| repetition_penalty  | float       | Penalty to prevent the model from generating repeated words or phrases. A value larger than 1 discourages repetition. | 1.0     |
| ignore_eos          | bool        | Indicator to ignore the eos_token_id or not.                                                                          | False   |
| random_seed         | int         | Seed used when sampling a token.                                                                                      | None    |
| stop_words          | List\[str\] | Words that stop generating further tokens.                                                                            | None    |
| bad_words           | List\[str\] | Words that the engine will never generate.                                                                            | None    |
| min_new_tokens      | int         | The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.                               | None    |
| skip_special_tokens | bool        | Whether or not to remove special tokens in the decoding.                                                              | True    |

## FAQs

- *RuntimeError: context has already been set*. If you got this for tp>1 in pytorch backend. Please make sure the python script has following

  ```python
  if __name__ == '__main__':
  ```

  Generally, in the context of multi-threading or multi-processing, it might be necessary to ensure that initialization code is executed only once. In this case, `if __name__ == '__main__':` can help to ensure that these initialization codes are run only in the main program, and not repeated in each newly created process or thread.

- To customize a chat template, please refer to [chat_template.md](../supported_models/chat_template.md).
