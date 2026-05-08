# Anthropic-Compatible Endpoints

LMDeploy provides a lightweight Anthropic-compatible surface for easier integration with Anthropic-style clients and gateways.

## Supported Endpoints

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `GET /anthropic/v1/models`

## Required Headers

For Anthropic `POST` endpoints, include:

- `content-type: application/json`
- `anthropic-version: 2023-06-01` (or another accepted version string)

## Notes and Current Limits

- `POST /v1/messages` supports text output, reasoning output (`thinking` blocks), and tool-use output (`tool_use` blocks).
- Tool use requires launching API server with a configured tool parser (`--tool-call-parser ...`).
- Reasoning block extraction depends on parser configuration (same parser stack used by OpenAI-compatible chat endpoint).
- `count_tokens` is tokenizer/chat-template based and is intended for practical estimation.

## Example: `/v1/messages`

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

## Example: `/v1/messages` with tools

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

## Streaming Events (SSE)

When `stream=true`, the endpoint returns `text/event-stream` events such as:

- `message_start`
- `content_block_start`
- `content_block_delta` (`text_delta`, `thinking_delta`, `input_json_delta`)
- `content_block_stop`
- `message_delta`
- `message_stop`

## Example: `/v1/messages/count_tokens`

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
