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
- `tool_result.content` accepts text and image blocks when the model supports images in tool results. These blocks remain in the same tool result, retaining `tool_use_id` and their internal order. Ordinary user content before, between, or after tool results retains its position.
- Tool use requires launching API server with a configured tool parser (`--tool-call-parser ...`).
- Reasoning block extraction depends on parser configuration (same parser stack used by OpenAI-compatible chat endpoint).
- `thinking.type` controls the template and reasoning parser: `disabled` turns thinking off, while `enabled` and `adaptive` turn it on. The model's chat template must support `enable_thinking`; for early GLM-5.3 templates that always open `<think>`, LMDeploy prefills `</think>` when thinking is disabled. `budget_tokens` is accepted for compatibility but does not enforce a separate thinking budget; `max_tokens` limits total output.
- `usage.cache_read_input_tokens` reports engine prefix-cache hits; `usage.input_tokens` excludes those hits. This uses LMDeploy prefix caching, not Anthropic cache breakpoints or TTLs. Cache-write accounting is not exposed.
- `usage.output_tokens_details.thinking_tokens` is an LMDeploy extension containing the parser's reasoning token count (zero when unavailable). Streaming reports the final count in `message_delta.usage`.
- When a configured stop sequence triggers the engine's token-based stop matching, both streaming and non-streaming responses report `stop_reason="stop_sequence"` and the matched `stop_sequence`. Natural EOS still reports `end_turn`.
- `count_tokens` is tokenizer/chat-template based and is intended for practical estimation.

## Example: `/v1/messages`

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

## Example: `/v1/messages` with tools

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
    "model": "intern-s2-preview",
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Count these tokens"}]
  }'
```
