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
- 模型支持工具结果图片时，`tool_result.content` 可以包含 text 和 image 块。这些块保留在同一工具结果中，并保留 `tool_use_id` 和内部顺序；工具结果前、后及结果之间的普通 user 内容也保留原有位置。
- 工具调用需要在启动 API Server 时配置工具解析器（`--tool-call-parser ...`）。
- 推理内容 block 的提取依赖解析器配置（与 OpenAI 兼容 chat 接口使用同一套解析器）。
- `thinking.type` 同时控制模板和推理解析器：`disabled` 关闭思考，`enabled` 和 `adaptive` 开启思考，要求模型模板支持 `enable_thinking`；对于始终输出 `<think>` 的早期 GLM-5.3 模板，关闭思考时 LMDeploy 会预填充 `</think>`。兼容接收 `budget_tokens`，但当前不单独限制思考预算；总输出仍由 `max_tokens` 限制。
- `usage.cache_read_input_tokens` 返回引擎前缀缓存命中数，`usage.input_tokens` 排除这些命中 token。这使用 LMDeploy 的前缀缓存，不实现 Anthropic 的缓存断点或 TTL，也不提供缓存写入统计。
- `usage.output_tokens_details.thinking_tokens` 是 LMDeploy 扩展，返回解析器统计的推理 token 数（统计不可用时为 0）。流式响应在 `message_delta.usage` 中返回最终统计。
- 配置的停止串触发引擎基于 token 的停止匹配时，流式和非流式响应均返回 `stop_reason="stop_sequence"` 和命中的 `stop_sequence`；自然 EOS 仍返回 `end_turn`。
- `count_tokens` 基于 tokenizer 和 chat template 计算，适合用于实用估算。

## 示例：`/v1/messages`

```bash
curl http://{server_ip}:{server_port}/v1/messages \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "intern-s2-preview",
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
    "model": "intern-s2-preview",
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
    "model": "intern-s2-preview",
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Count these tokens"}]
  }'
```
