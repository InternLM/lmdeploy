# OpenAI Responses-Compatible Endpoint

LMDeploy provides a lightweight OpenAI Responses-compatible surface for easier integration with clients that use the newer Responses API.

## Supported Endpoints

- `POST /v1/responses`
- `GET /v1/models`

## Required Headers

For `POST /v1/responses`, include:

- `content-type: application/json`
- `Authorization: Bearer <api_key>` when the server is launched with `--api-keys`

## Notes and Current Limits

- `POST /v1/responses` currently supports a text-first subset of the Responses API.
- `input` may be a string or a list of Responses input items.
- `instructions`, `developer`, and `system` messages are merged into a single leading system message for chat-template compatibility.
- Function tools are converted to LMDeploy's OpenAI-compatible tool format. Tool calling requires launching API server with a configured tool parser (`--tool-call-parser ...`).
- Non-function hosted tools, such as `web_search`, are accepted but ignored by LMDeploy.
- `background` mode and `previous_response_id` are not supported.

## Example: `/v1/responses`

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

The response contains an `output` list and a convenience `output_text` field:

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

## Example: `/v1/responses` with tools

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

## Streaming Events (SSE)

When `stream=true`, the endpoint returns `text/event-stream` events such as:

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

## Codex Integration Note

Codex can connect to LMDeploy by configuring a custom provider with `wire_api = "responses"` and `base_url` pointing to the LMDeploy `/v1` root:

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

The `model` value must exactly match a model name exposed by LMDeploy.
