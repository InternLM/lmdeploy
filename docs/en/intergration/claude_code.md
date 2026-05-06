# Integrate LMDeploy `/v1/messages` with Claude Code

Claude Code can connect to an Anthropic-compatible gateway by setting `ANTHROPIC_BASE_URL`.
LMDeploy exposes an Anthropic-compatible `POST /v1/messages` endpoint, so Claude Code can
route requests to a local or remote LMDeploy `api_server`.

The request path is:

```text
Claude Code -> http://<server>:<port>/v1/messages -> LMDeploy api_server
```

## 1. Start LMDeploy

Launch an LMDeploy API server with the model you want Claude Code to use:

```bash
lmdeploy serve api_server Qwen/Qwen3.5-35B-A3B --backend pytorch --server-port 23333
```

For tool calling, launch the server with a tool parser supported by your model:

```bash
lmdeploy serve api_server <model> \
  --server-port 23333 \
  --tool-call-parser <parser-name>
```

## 2. Verify the Messages Endpoint

Before configuring Claude Code, check that LMDeploy responds to Anthropic-style requests:

```bash
curl http://127.0.0.1:23333/v1/messages \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Say hello from LMDeploy"}
    ]
  }'
```

The `model` value must match a model name exposed by your LMDeploy server.

## 3. Configure Claude Code

Set `ANTHROPIC_BASE_URL` to the server root, not to `/v1`. Claude Code appends
`/v1/messages` itself. Add the LMDeploy gateway configuration to
`~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:23333",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_MODEL": "Qwen/Qwen3.5-35B-A3B",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "Qwen/Qwen3.5-35B-A3B",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "LMDeploy local model",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "Served by LMDeploy /v1/messages",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "Qwen/Qwen3.5-35B-A3B",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "Qwen/Qwen3.5-35B-A3B",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Qwen/Qwen3.5-35B-A3B",
    "CLAUDE_CODE_SUBAGENT_MODEL": "Qwen/Qwen3.5-35B-A3B",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

The model name must exactly match the name exposed by LMDeploy.

Then start Claude Code:

```bash
claude --model Qwen/Qwen3.5-35B-A3B
```

## 4. Streaming Behavior

Claude Code commonly uses streaming. LMDeploy's `/v1/messages` endpoint supports
`stream=true` and returns Anthropic-style server-sent events:

```text
message_start
content_block_start
content_block_delta
content_block_stop
message_delta
message_stop
```

Streaming supports text deltas, reasoning deltas when parser configuration enables them,
and tool input JSON deltas when tool calling is enabled.

## 5. Model Discovery Note

Claude Code may query `/v1/models` when `ANTHROPIC_BASE_URL` points to a gateway.
LMDeploy's Anthropic model list endpoint is currently documented as `GET /anthropic/v1/models`.
If Claude Code does not discover your model automatically, use `ANTHROPIC_MODEL` and
`ANTHROPIC_CUSTOM_MODEL_OPTION` as shown above.
