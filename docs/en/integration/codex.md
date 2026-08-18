# Integrate LMDeploy with Codex

Codex can connect to an OpenAI Responses-compatible gateway by configuring a
custom model provider. LMDeploy exposes `POST /v1/responses`, so Codex can route
requests to a local or remote LMDeploy `api_server`.

The request path is:

```text
Codex -> http://<server>:<port>/v1/responses -> LMDeploy api_server
```

## 1. Start LMDeploy

Launch an LMDeploy API server with the model you want Codex to use:

```bash
lmdeploy serve api_server Qwen/Qwen3.5-35B-A3B --backend pytorch --server-port 23333
```

For tool calling, launch the server with a tool parser supported by your model:

```bash
lmdeploy serve api_server <model> \
  --server-port 23333 \
  --tool-call-parser <parser-name>
```

## 2. Verify the Responses Endpoint

Before configuring Codex, check that LMDeploy responds to Responses-style requests:

```bash
curl http://127.0.0.1:23333/v1/responses \
  -H "content-type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "input": "Say hello from LMDeploy",
    "max_output_tokens": 32
  }'
```

The `model` value must match a model name exposed by your LMDeploy server. You can
check the model list with:

```bash
curl http://127.0.0.1:23333/v1/models
```

## 3. Configure Codex

Add the LMDeploy provider configuration to `~/.codex/config.toml`:

```toml
model = "Qwen/Qwen3.5-35B-A3B"
model_provider = "lmdeploy"

[model_providers.lmdeploy]
name = "LMDeploy"
base_url = "http://127.0.0.1:23333/v1"
env_key = "LMDEPLOY_API_KEY"
wire_api = "responses"
requires_openai_auth = false
stream_idle_timeout_ms = 300000
```

Set `base_url` to the `/v1` root, not to `/v1/responses`. Codex appends
`/responses` itself when `wire_api = "responses"`.

Set `LMDEPLOY_API_KEY` before starting Codex. If the LMDeploy server was not
launched with `--api-keys`, any non-empty value is sufficient. Otherwise, use the
matching key instead of `dummy`:

```bash
export LMDEPLOY_API_KEY=dummy
```

## 4. Run Codex

Start Codex:

```bash
codex
```

Or run a single non-interactive request:

```bash
codex exec "Say hello from LMDeploy"
```

## 5. Tool Calling

Codex commonly uses tool calls for shell commands and file inspection. LMDeploy's
`/v1/responses` endpoint supports function-call output when the API server is
launched with `--tool-call-parser`.

## 6. Endpoint Note

Codex uses the OpenAI Responses-compatible `/v1/responses` endpoint. The
Anthropic-compatible `/v1/messages` endpoint is for Anthropic-style clients such
as Claude Code.
