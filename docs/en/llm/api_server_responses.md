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

- `POST /v1/responses` supports text, user-message images, and tool-result images with a vision-capable model.
- `input` may be a string or a list of Responses input items.
- With the default response parser, `text.format` values `json_object` and `json_schema` constrain the entire generated sequence. Thinking is disabled through the template and the JSON answer is returned in `output_text`, including streaming text deltas.
- User content arrays can mix `input_text` and `input_image` blocks. Images use `image_url` with an HTTP(S) URL or a base64 image data URL. Multiple images and their order are preserved in both streaming and non-streaming requests.
- Image `detail` and the `min_dynamic_patch` / `max_dynamic_patch` extensions are forwarded like Chat image parameters; their effect depends on the model's image processor. An omitted or `null` image `detail` uses the default behavior; no `detail` value is forwarded in either case.
- `function_call_output.output` can also contain an array of `input_text` and `input_image` blocks. Images stay associated with the result's `call_id`. Consecutive `function_call` items are treated as one parallel tool-call group.
- Image `file_id` references and images in system/developer/assistant messages are not supported.
- `instructions`, `developer`, and `system` messages are merged into a single leading system message for chat-template compatibility.
- Function tools are converted to LMDeploy's OpenAI-compatible tool format. Tool calling requires launching API server with a configured tool parser (`--tool-call-parser ...`).
- `parallel_tool_calls` defaults to `true`. When it is `false`, LMDeploy follows vLLM-style compatibility and returns only the first parsed function call.
- Non-function hosted tools, such as `web_search`, are accepted but ignored by LMDeploy.
- Responses-only logprob serialization and stream obfuscation options are accepted but currently ignored.
- `reasoning.effort="none"` disables thinking through the model's template and parser. `low`, `medium`, `high`, and `max` enable thinking and forward the effort to the template; model support determines its effect. Other effort values return HTTP 400. Reasoning summaries remain unsupported.
- `max_output_tokens` includes both reasoning and visible answer tokens. Reaching the limit returns `status="incomplete"` with `incomplete_details.reason="max_output_tokens"`, possibly before any visible text. Increase the budget, or use `reasoning: {"effort": "none"}` with a model that supports disabling thinking for a short direct answer. Accepting the request's fields does not guarantee `status="completed"`.
- `usage.input_tokens_details.cached_tokens` reports engine prefix-cache hits in both non-streaming and terminal streaming responses. `input_tokens` continues to include cached tokens.
- `background` mode and `previous_response_id` are not supported.

## Example: `/v1/responses`

```bash
curl http://{server_ip}:{server_port}/v1/responses \
  -H "content-type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "input": "Reply exactly: pong",
    "max_output_tokens": 32,
    "reasoning": {"effort": "none"}
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

## Example: `/v1/responses` with images

Start a vision-capable model without `--language-model-only`, then send:

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
    "max_output_tokens": 128,
    "reasoning": {"effort": "none"}
  }'
```

For a local image, base64-encode its bytes and use `data:image/png;base64,<encoded bytes>` as `image_url`. Add `"stream": true` to receive SSE events. Images use the same media loader and URL restrictions as Chat requests; malformed image data is rejected during preprocessing.

## Example: `/v1/responses` with tools

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

## Codex Integration

May refer to [codex](../integration/codex.md).

## Example: tool results with images

Replay the tool call and return its image using the same `call_id`. For parallel calls, place the consecutive `function_call` items before their results; results may arrive in a different order. Each result preserves its internal text/image order.

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
