# 长文本外推

长文本外推指 LLM 推理时处理比训练文本更长数据的能力。TurboMind 引擎目前支持 [LlamaDynamicNTKScalingRotaryEmbedding](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L178), 并与 HuggingFace 的实现对齐。

## 如何使用

如果要直接加载 HuggingFace 格式的模型，可以通过修改 TurbomindEngineConfig 参数的方式赋予模型外推能力。将 `session_len` 修改为外推的长度，并将 `rope_scaling_factor` 修改为大于 1.0 的值。下面以 pipeline 接口为例，展示一个 Passkey Retrieval 的例子

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

## 评测

### 大海捞针

可使用 OpenCompass 进行测评，具体使用方法，请参考[文档](https://github.com/open-compass/opencompass/blob/main/docs/zh_cn/advanced_guides/needleinahaystack_eval.md)

### 困惑度

下面展示使用 TurboMind 计算困惑度的用法

```python
from datasets import load_dataset
from lmdeploy import TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
import numpy as np

# load model and tokenizer
engine_config = TurbomindEngineConfig(rope_scaling_factor=1.0, session_len=200000)
engine = TurboMind.from_pretrained('internlm/internlm2-chat-7b', engine_config)
tokenizer = engine.tokenizer
generator = engine.create_instance()

# get perplexity
text = 'The grass is green. The sky is blue. The sun is yellow'
input_ids = tokenizer.encode(text)
loss = generator.get_ppl(input_ids)[0]
ppl = np.exp(loss)
```
