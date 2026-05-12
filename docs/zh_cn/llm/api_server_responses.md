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

- `POST /v1/responses` 当前支持 text-first 子集。
- `input` 可以是字符串，也可以是 Responses input item 列表。
- `instructions`、`developer` 和 `system` 消息会合并为一个位于最前面的 system message，以兼容 chat template。
- function tools 会转换为 LMDeploy 的 OpenAI 兼容 tool 格式。工具调用需要在启动 API Server 时配置工具解析器（`--tool-call-parser ...`）。
- 非 function 类型的 hosted tools（例如 `web_search`）会被接受，但 LMDeploy 会忽略它们。
- 当前不支持 `background` 模式和 `previous_response_id`。

## 示例：`/v1/responses`

```bash
curl http://{server_ip}:{server_port}/v1/responses \
  -H "content-type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{
    "model": "internlm-chat-7b",
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

## 示例：带 tools 的 `/v1/responses`

```bash
curl http://{server_ip}:{server_port}/v1/responses \
  -H "content-type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{
    "model": "internlm-chat-7b",
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
model = "internlm-chat-7b"
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
