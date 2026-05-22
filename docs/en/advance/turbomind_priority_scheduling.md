# TurboMind Priority Scheduling

TurboMind supports request-level priority scheduling. This is useful when one service handles different classes of traffic at the same time, such as interactive requests, background batch jobs, regular users, and high-priority users. After priority scheduling is enabled, requests with higher priority are admitted to inference first, while requests that have already started are kept whenever possible to reduce extra KV cache swapping overhead.

Priority scheduling only takes effect with the TurboMind backend. The default scheduling policy is `fifo`; in that mode, requests are still scheduled by arrival order and the `priority` field does not change the scheduling order.

## Enable Priority Scheduling

Start the API Server with `--schedule-policy priority`:

```bash
lmdeploy serve api_server internlm/internlm2_5-7b-chat \
    --backend turbomind \
    --schedule-policy priority
```

When creating a pipeline in Python, enable it through `TurbomindEngineConfig`:

```python
from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

backend_config = TurbomindEngineConfig(schedule_policy='priority')
pipe = pipeline('internlm/internlm2_5-7b-chat', backend_config=backend_config)

response = pipe(
    'Introduce LMDeploy',
    gen_config=GenerationConfig(priority=0, max_new_tokens=256),
)
```

`schedule_policy` supports the following values:

| Value      | Behavior                                                                                                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fifo`     | Default policy. Schedule requests by arrival order and preserve the original behavior.                                                                                   |
| `priority` | Schedule by request priority. Smaller values are admitted first; requests with the same priority keep FIFO order; already-started requests are kept before new requests. |

## Set Request Priority

Set request priority with `priority`. The valid range is `[0, 255]`. Smaller values have higher priority; `0` is the highest priority and `255` is the lowest priority. The default value is `0`. `priority` must be an integer; out-of-range values and strings, floats, booleans, or other non-integer types are rejected by validation.

### OpenAI-Compatible API

Both `/v1/chat/completions` and `/v1/completions` support the LMDeploy extension field `priority`.

```bash
curl http://localhost:23333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm/internlm2_5-7b-chat",
    "messages": [{"role": "user", "content": "Summarize this text"}],
    "priority": 0,
    "max_tokens": 256
  }'
```

When using the OpenAI Python SDK, pass this extension field through `extra_body`:

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')

response = client.chat.completions.create(
    model='internlm/internlm2_5-7b-chat',
    messages=[{'role': 'user', 'content': 'Summarize this text'}],
    max_tokens=256,
    extra_body={'priority': 0},
)
```

Use a larger value for normal-priority requests:

```python
response = client.chat.completions.create(
    model='internlm/internlm2_5-7b-chat',
    messages=[{'role': 'user', 'content': 'Generate a long background report'}],
    max_tokens=1024,
    extra_body={'priority': 32},
)
```

### Pipeline

For pipeline calls, use `GenerationConfig.priority`:

```python
from lmdeploy import GenerationConfig

urgent = pipe(
    'Quickly review this code',
    gen_config=GenerationConfig(priority=0, max_new_tokens=256),
)

background = pipe(
    'Generate a long offline analysis report',
    gen_config=GenerationConfig(priority=32, max_new_tokens=1024),
)
```

Priority scheduling is most visible when the service is handling concurrent requests. If there is only one synchronous request and no resource contention, changing its priority does not change the output content.

## Scheduling Semantics

After `schedule_policy='priority'` is enabled, TurboMind uses request priority in two stages:

1. Before requests enter the engine, the waiting queue admits requests with smaller `priority` values first.
2. After requests enter the engine, already-started requests are kept first. Requests that have not started are then ordered by `priority` and arrival order.

Therefore, `priority` is a non-preemptive priority policy:

- A high-priority request can overtake lower-priority requests that are still waiting.
- A new high-priority request does not preempt a lower-priority request that has already started.
- Requests with the same priority are processed in arrival order.
- A continuous stream of high-priority requests may make lower-priority requests wait longer. The current policy does not include aging, deadlines, quotas, or weighted fair scheduling.

This policy favors throughput and stability. It is a good fit for online services that need traffic differentiation while avoiding frequent preemption and KV cache swaps.

## Usage Tips

- Reserve smaller values, such as `0` or `1`, for the most urgent requests.
- Use middle values, such as `16` or `32`, for normal online requests.
- Use larger values, such as `128` or `255`, for background, offline, or delay-tolerant requests.
- If all requests use the default value `0`, scheduling is equivalent to FIFO among requests with the same priority.
- This feature only changes scheduling order. It does not change sampling parameters, output quality, or token generation rules.
