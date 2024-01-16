# Context length extrapolation

Long text extrapolation refers to the ability of LLM to handle data longer than the training text during inference. TurboMind engine now support [LlamaDynamicNTKScalingRotaryEmbedding](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L178) and the implementation is consistent with huggingface.

## Usage

You can enable the context length extrapolation abality by modifying the TurbomindEngineConfig. Edit the `session_len` to the expected length and change `rope_scaling_factor` to a number no less than 1.0.

Here is an example:

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(rope_scaling_factor=2.0, session_len=160000)
pipe = pipeline('internlm/internlm2-chat-7b', backend_config=backend_config)
prompt = 'Use a long prompt to replace this sentence'
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
response = pipe(prompt, gen_config=gen_config)
print(response)
```

## Evaluation

We use several methods to evaluate the long-context-length inference ability of LMDeploy, including [passkey retrieval](#passkey-retrieval), [needle in a haystack](#needle-in-a-haystack) and computing [perplexity](#perplexity)

### Passkey Retrieval

You can try the following code to test how many times LMDeploy can retrieval the special key.

```python
import numpy as np
from lmdeploy import pipeline
from lmdeploy import TurbomindEngineConfig

session_len = 160000
backend_config = TurbomindEngineConfig(rope_scaling_factor=2.0, session_len=session_len)
pipe = pipeline('internlm/internlm2-chat-7b', backend_config=backend_config)


def passkey_retrival(session_len, n_round=5):
    # create long context input
    tok = pipe.tokenizer
    task_description = 'There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.'
    garbage = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.'

    for _ in range(n_round):
        n_times = (session_len - 1000) // len(tok.encode(garbage))
        n_garbage_prefix = np.random.randint(0, n_times)
        n_garbage_suffix = n_times - n_garbage_prefix
        garbage_prefix = ' '.join([garbage] * n_garbage_prefix)
        garbage_suffix = ' '.join([garbage] * n_garbage_suffix)
        pass_key = np.random.randint(1, 50000)
        information_line = f'The pass key is {pass_key}. Remember it. {pass_key} is the pass key.'  # noqa: E501
        final_question = 'What is the pass key? The pass key is'
        lines = [
            task_description,
            garbage_prefix,
            information_line,
            garbage_suffix,
            final_question,
        ]

        # inference
        prompt = ' '.join(lines)
        response = pipe([prompt])
        print(pass_key, response)


passkey_retrival(session_len, 5)
```

### Needle In A Haystack

[OpenCompass](https://github.com/open-compass/opencompass) offers very useful tools to perform needle-in-a-haystack evaluation. For specific instructions, please refer to the [guide](https://github.com/open-compass/opencompass/blob/main/docs/en/advanced_guides/needleinahaystack_eval.md).

### Perplexity

The following codes demonstrate how to use LMDeploy to calculate perplexity.

```python
from datasets import load_dataset
from lmdeploy import TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
import numpy as np

# load model and tokenizer
engine_config = TurbomindEngineConfig(rope_scaling_factor=2.0, session_len=160000)
engine = TurboMind.from_pretrained('internlm/internlm2-chat-7b', engine_config)
tokenizer = engine.tokenizer
generator = engine.create_instance()

# get perplexity
text = 'Use a long prompt to replace this sentence'
input_ids = tokenizer.encode(text)
loss = generator.get_ppl(input_ids)[0]
ppl = np.exp(loss)
```
