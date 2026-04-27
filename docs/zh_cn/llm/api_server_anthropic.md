# Anthropic 兼容接口

LMDeploy 提供了轻量级的 Anthropic 兼容接口，方便接入 Anthropic 风格的客户端和网关。

## 支持的接口

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `GET /anthropic/v1/models`

## 必要请求头

对于 Anthropic 的 `POST` 接口，请包含：

- `content-type: application/json`
- `anthropic-version: 2023-06-01`（或其他可接受的版本字符串）

## 说明与当前限制

- `POST /v1/messages` 支持文本输出、推理输出（`thinking` block）和工具调用输出（`tool_use` block）。
- 工具调用需要在启动 API Server 时配置工具解析器（`--tool-call-parser ...`）。
- 推理内容 block 的提取依赖解析器配置（与 OpenAI 兼容 chat 接口使用同一套解析器）。
- `count_tokens` 基于 tokenizer 和 chat template 计算，适合用于实用估算。

## 示例：`/v1/messages`

```bash
curl http://{server_ip}:{server_port}/v1/messages \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "internlm-chat-7b",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Hello from Anthropic client"}]
  }'
```

## 示例：带 tools 的 `/v1/messages`

```bash
curl http://{server_ip}:{server_port}/v1/messages \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "internlm-chat-7b",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Find lmdeploy docs"}],
    "tools": [{
      "name": "search",
      "description": "Search docs",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"}
        },
        "required": ["query"]
      }
    }],
    "tool_choice": {"type": "auto"}
  }'
```

## 流式事件（SSE）

当 `stream=true` 时，接口会返回 `text/event-stream` 事件流，常见事件包括：

- `message_start`
- `content_block_start`
- `content_block_delta`（`text_delta`、`thinking_delta`、`input_json_delta`）
- `content_block_stop`
- `message_delta`
- `message_stop`

## 示例：`/v1/messages/count_tokens`

```bash
curl http://{server_ip}:{server_port}/v1/messages/count_tokens \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "internlm-chat-7b",
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Count these tokens"}]
  }'
```
