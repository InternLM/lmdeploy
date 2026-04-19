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

- Tool-call fields are **temporarily unsupported** in this phase (`tools`, `tool_choice`).
- If tool fields are provided, LMDeploy returns an Anthropic-style error response.
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
