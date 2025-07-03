# Reward Models

LMDeploy supports  reward models, which are detailed in the table below:

|      Model       |     Size      | Supported Inference Engine |
| :--------------: | :-----------: | :------------------------: |
| Qwen2.5-Math-RM  |      72B      |          PyTorch           |
| InternLM2-Reward | 1.8B, 7B, 20B |          PyTorch           |

## Offline Inference

We take `internlm/internlm2-1_8b-reward` as an example:

```python
from transformers import AutoTokenizer
from lmdeploy import pipeline, PytorchEngineConfig

model_path = "internlm/internlm2-1_8b-reward"
chat = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
    {"role": "assistant", "content": "To determine how much Janet makes from selling the duck eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid by the ducks each day.\n2. Determine how many eggs Janet eats and bakes for herself each day.\n3. Find out how many eggs are left to be sold.\n4. Calculate the revenue from selling the remaining eggs at $2 per egg.\n\nLet's start with the first step:\n\n1. Janet's ducks lay 16 eggs per day.\n\nNext, we calculate how many eggs Janet eats and bakes for herself each day:\n\n2. Janet eats 3 eggs for breakfast every morning.\n3. Janet bakes 4 eggs for her friends every day.\n\nSo, the total number of eggs Janet eats and bakes for herself each day is:\n\\[ 3 + 4 = 7 \\text{ eggs} \\]\n\nNow, we find out how many eggs are left to be sold:\n\\[ 16 - 7 = 9 \\text{ eggs} \\]\n\nFinally, we calculate the revenue from selling the remaining eggs at $2 per egg:\n\\[ 9 \\times 2 = 18 \\text{ dollars} \\]\n\nTherefore, Janet makes 18 dollars every day at the farmers' market."}
]

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

conversation_str = tokenizer.apply_chat_template(
    chat,
    tokenize=False,
    add_generation_prompt=False
)

input_ids = tokenizer.encode(
    conversation_str,
    add_special_tokens=False
)


if __name__ == '__main__':
    engine_config = PytorchEngineConfig(tp=tp)
    with pipeline(model_path, backend_config=engine_config) as pipe:
        score = pipe.get_reward_score(input_ids)
        print(f'score: {score}')
```

## Online Inference

Start the API server:

```bash
lmdeploy serve api_server internlm/internlm2-1_8b-reward --backend pytorch
```

Get the reward score from the `/pooling` API endpoint:

```
curl http://0.0.0.0:23333/pooling \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm/internlm2-1_8b-reward",
    "input": "Who are you?"
  }'
```
