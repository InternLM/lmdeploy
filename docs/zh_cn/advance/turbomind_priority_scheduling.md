# TurboMind 优先级调度

TurboMind 支持按请求优先级进行调度。该功能适合在同一个服务中同时处理多类请求，例如交互式请求、后台批处理任务、普通用户请求和高优先级用户请求。启用后，服务会优先让优先级更高的请求进入推理，同时尽量保留已经开始执行的请求，减少 KV cache 交换带来的额外开销。

优先级调度只在 TurboMind 后端生效。默认调度策略是 `fifo`，此时请求仍按到达顺序处理，`priority` 字段不会改变调度顺序。

## 启用优先级调度

启动 API Server 时设置 `--schedule-policy priority`：

```bash
lmdeploy serve api_server internlm/internlm2_5-7b-chat \
    --backend turbomind \
    --schedule-policy priority
```

在 Python 中创建 pipeline 时，可以通过 `TurbomindEngineConfig` 启用：

```python
from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

backend_config = TurbomindEngineConfig(schedule_policy='priority')
pipe = pipeline('internlm/internlm2_5-7b-chat', backend_config=backend_config)

response = pipe(
    '介绍一下 LMDeploy',
    gen_config=GenerationConfig(priority=0, max_new_tokens=256),
)
```

`schedule_policy` 支持以下取值：

| 取值       | 行为                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------- |
| `fifo`     | 默认策略。按请求到达顺序调度，保持原有行为。                                                          |
| `priority` | 按请求优先级调度。数值更小的请求优先进入推理；同优先级请求保持 FIFO；已经开始执行的请求优先于新请求。 |

## 设置请求优先级

请求优先级通过 `priority` 设置，取值范围为 `[0, 255]`。数值越小，优先级越高；`0` 是最高优先级，`255` 是最低优先级。未设置时默认值为 `0`。`priority` 必须是整数，超出范围或使用字符串、浮点数、布尔值等类型会被校验拒绝。

### OpenAI 兼容接口

`/v1/chat/completions` 和 `/v1/completions` 都支持 LMDeploy 扩展字段 `priority`。

```bash
curl http://localhost:23333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm/internlm2_5-7b-chat",
    "messages": [{"role": "user", "content": "请总结这段文本"}],
    "priority": 0,
    "max_tokens": 256
  }'
```

使用 OpenAI Python SDK 时，可以通过 `extra_body` 传入该扩展字段：

```python
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')

response = client.chat.completions.create(
    model='internlm/internlm2_5-7b-chat',
    messages=[{'role': 'user', 'content': '请总结这段文本'}],
    max_tokens=256,
    extra_body={'priority': 0},
)
```

普通优先级请求可以使用更大的数值：

```python
response = client.chat.completions.create(
    model='internlm/internlm2_5-7b-chat',
    messages=[{'role': 'user', 'content': '后台生成一份长报告'}],
    max_tokens=1024,
    extra_body={'priority': 32},
)
```

### Pipeline

Pipeline 调用中使用 `GenerationConfig.priority`：

```python
from lmdeploy import GenerationConfig

urgent = pipe(
    '帮我快速检查这段代码',
    gen_config=GenerationConfig(priority=0, max_new_tokens=256),
)

background = pipe(
    '生成一份较长的离线分析报告',
    gen_config=GenerationConfig(priority=32, max_new_tokens=1024),
)
```

优先级调度主要在服务并发处理多个请求时体现效果。单个同步请求没有其他请求竞争资源时，设置不同优先级不会改变输出内容。

## 调度语义

启用 `schedule_policy='priority'` 后，TurboMind 会在两个阶段使用请求优先级：

1. 请求进入 engine 前，等待队列优先取出 `priority` 数值更小的请求。
2. 请求进入 engine 后，已经开始执行的请求优先保留；尚未开始执行的请求再按 `priority` 和到达顺序排序。

因此，`priority` 是非抢占式优先级策略：

- 高优先级请求可以超过仍在等待的低优先级请求。
- 高优先级新请求不会抢占已经开始执行的低优先级请求。
- 同优先级请求按到达顺序处理。
- 持续到来的高优先级请求可能让低优先级请求等待更久；当前策略不包含 aging、deadline、配额或加权公平调度。

这种策略更偏向吞吐和稳定性，适合希望区分请求等级，同时避免频繁抢占和 KV cache swap 的在线服务。

## 使用建议

- 为最紧急的请求保留较小的优先级，例如 `0` 或 `1`。
- 为普通在线请求使用中间值，例如 `16` 或 `32`。
- 为后台、离线或可延迟请求使用更大的值，例如 `128` 或 `255`。
- 如果所有请求都使用默认值 `0`，调度效果等价于同优先级 FIFO。
- 该功能只影响调度顺序，不改变采样参数、输出质量或 token 生成规则。
