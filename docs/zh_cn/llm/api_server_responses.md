# OpenAI Responses 兼容接口

LMDeploy 提供了轻量级的 OpenAI Responses 兼容接口，方便接入使用新版 Responses API 的客户端。

## 支持的接口

- `POST /v1/responses`
- `GET /v1/models`

## 必要请求头

对于 `POST /v1/responses`，请包含：

- `content-type: application/json`
- 如果 API Server 启动时配置了 `--api-keys`，请包含 `Authorization: Bearer <api_key>`

## 说明与当前限制

- `POST /v1/responses` 支持文本输入，以及视觉模型的 user 消息图片和工具结果图片输入。
- `input` 可以是字符串，也可以是 Responses input item 列表。
- user 的 content 数组可以混合 `input_text` 和 `input_image`。图片通过 `image_url` 指定 HTTP(S) URL 或 base64 图片 data URL；流式和非流式请求均支持多图，并保留图文顺序。
- 图片的 `detail` 和 `min_dynamic_patch` / `max_dynamic_patch` 扩展参数按 Chat 图片参数方式传递，实际效果取决于模型的图片处理器。
- `function_call_output.output` 也可以包含 `input_text` 和 `input_image` 数组。图片保留所属结果的 `call_id`，连续的 `function_call` 条目会转换为同一组并行工具调用。
- 暂不支持图片 `file_id` 引用，以及 system/developer/assistant 消息中的图片。
- `instructions`、`developer` 和 `system` 消息会合并为一个位于最前面的 system message，以兼容 chat template。
- function tools 会转换为 LMDeploy 的 OpenAI 兼容 tool 格式。工具调用需要在启动 API Server 时配置工具解析器（`--tool-call-parser ...`）。
- `parallel_tool_calls` 默认为 `true`。当设置为 `false` 时，LMDeploy 会保持与 vLLM 类似的兼容行为，仅返回解析到的第一个 function call。
- 非 function 类型的 hosted tools（例如 `web_search`）会被接受，但 LMDeploy 会忽略它们。
- Responses 专属的 logprob 序列化与流式混淆选项当前会被接受但忽略。
- 当前不支持 `background` 模式和 `previous_response_id`。

## 示例：`/v1/responses`

```bash
curl http://{server_ip}:{server_port}/v1/responses \
  -H "content-type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "input": "Reply exactly: pong",
    "max_output_tokens": 32
  }'
```

返回结果包含 `output` 列表和便捷字段 `output_text`：

```json
{
  "object": "response",
  "status": "completed",
  "output": [{
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "pong"}]
  }],
  "output_text": "pong"
}
```

## 示例：带图片的 `/v1/responses`

启动视觉模型时不要设置 `--language-model-only`，然后发送：

```bash
curl http://{server_ip}:{server_port}/v1/responses \
  -H "content-type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "input": [{
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this image."},
        {"type": "input_image", "image_url": "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg", "detail": "auto"}
      ]
    }],
    "max_output_tokens": 128
  }'
```

本地图片可将图片字节编码为 base64，再使用 `data:image/png;base64,<encoded bytes>` 作为 `image_url`。添加 `"stream": true` 可接收 SSE 事件。图片复用 Chat 的媒体加载器和 URL 限制；损坏的图片数据会在预处理阶段被拒绝。

## 示例：带 tools 的 `/v1/responses`

```bash
curl http://{server_ip}:{server_port}/v1/responses \
  -H "content-type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "input": "Call the search tool with query lmdeploy.",
    "max_output_tokens": 128,
    "tools": [{
      "type": "function",
      "name": "search",
      "description": "Search docs",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string"}
        },
        "required": ["query"]
      }
    }]
  }'
```

## 流式事件（SSE）

当 `stream=true` 时，接口会返回 `text/event-stream` 事件流，常见事件包括：

- `response.created`
- `response.in_progress`
- `response.output_item.added`
- `response.content_part.added`
- `response.output_text.delta`
- `response.output_text.done`
- `response.function_call_arguments.delta`
- `response.function_call_arguments.done`
- `response.output_item.done`
- `response.completed`

## Codex 集成说明

Codex 可以通过自定义 provider 连接到 LMDeploy。需要将 `wire_api` 设为 `"responses"`，并将 `base_url` 指向 LMDeploy 的 `/v1` 根路径：

```toml
model = "Qwen/Qwen3.5-35B-A3B"
model_provider = "lmdeploy"

[model_providers.lmdeploy]
name = "LMDeploy"
base_url = "http://{server_ip}:{server_port}/v1"
env_key = "LMDEPLOY_API_KEY"
wire_api = "responses"
requires_openai_auth = false
stream_idle_timeout_ms = 300000
```

`model` 的值必须与 LMDeploy 暴露的模型名完全一致。

## 示例：包含图片的工具结果

重放工具调用，并使用相同的 `call_id` 返回图片。并行调用时，将连续的 `function_call` 条目放在结果之前；结果可以按不同顺序返回，每个结果内部的图文顺序都会保留。

```json
{
  "model": "your-vision-model",
  "input": [
    {"role": "user", "content": "Describe the screenshot."},
    {"type": "function_call", "call_id": "call_screenshot", "name": "screenshot", "arguments": "{}"},
    {
      "type": "function_call_output",
      "call_id": "call_screenshot",
      "output": [
        {"type": "input_text", "text": "Screenshot result:"},
        {"type": "input_image", "image_url": "https://example.com/screenshot.png"}
      ]
    }
  ]
}
```
