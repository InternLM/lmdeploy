# Intern-Decision

[Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) answers structured questions by
**scoring** instead of generating. A native request holds a `state`, up to 16 `questions` and optional images. The model
reads one prompt that ends in a JSON skeleton with a `<decision>` marker per question, and the logits right before
each marker, restricted to that question's answer symbols (`A`, `B`, ...), give the answer distribution.

LMDeploy serves it with the `intern-decision` task on the PyTorch engine. The checkpoint itself is an ordinary
Qwen3.5 vision-language model; the task adds the protocol on top. The decision protocol and preprocessing are
adapted from the Apache-2.0 [Intern-Decision project](https://github.com/InternLM/Intern-Decision); adapted files
keep their attribution.

Decision requests are currently supported through HTTP only; the Python Pipeline has no Decision entry point.

## HTTP

```shell
lmdeploy serve api_server /path/to/Intern-Decision-0.8B --task intern-decision --model-name Intern-Decision-0.8B
```

`--task intern-decision` enables decision scoring on the PyTorch backend. The old checkpoint-name task values are
no longer accepted. All PyTorch engine options (batch size, TP, prefix caching, ...) remain available. Decision
requests use input scoring and generate no tokens.

Startup reads a finite positive `DEFAULT_TEMPERATURE` constant from the local checkpoint's `inference.py` using
AST parsing, without executing that file. Missing, unreadable or invalid metadata produces a warning and falls
back to the 4B temperature `1.99241824`. This is a best-effort read: it does not download metadata for remote IDs
or verify that calibration belongs to the loaded weights. Warnings follow `--log-level` (CLI default: `WARNING`).

A read-only `DecisionContext` holds only the marker ID, answer-symbol IDs, image-token ID and temperature.
It is initialized once from existing engine resources; its presence enables the decision endpoints.

`POST /v1/decisions` accepts and returns only the OpenAI Decisions text/image protocol described below.
Its OpenAPI definition declares a single request model and a single success-response model.
`POST /v1/jev` accepts and returns only the native protocol. Existing native clients must use `/v1/jev`;
native requests sent to `/v1/decisions` are rejected with 400. Native requests contain `state`, a `questions` object,
and optional `images`. For example, upload a client-local image:

```shell
IMG=$(base64 -w0 crossing.png)
curl http://127.0.0.1:23333/v1/jev -H 'Content-Type: application/json' -d @- <<EOF
{"state": "Inspect the image.",
 "images": ["data:image/png;base64,$IMG"],
 "questions": {"red": {"type": "noul", "instructions": "Is a red light visible?"}}}
EOF
```

`images` uses the same `ImageDataFormat` as `/generate.image_data`: a string, a parameter dictionary such as
`{"url": "..."}`, or a list mixing both. Strings may be HTTP(S) URLs, complete base64 data URLs, or local paths.
Omit `images` or pass `[]` for text-only input. Dictionary contents are checked by shared multimodal preprocessing,
rather than a separate Decision image schema. Additional parameters are forwarded to that path; their effects
depend on the existing processor.
**Migration:** pass references such as `"https://..."` or `{"url": "https://..."}`, not complete
`{"type": "image_url", "image_url": {...}}` message blocks; the compiler wraps each reference once.
The upstream `{"type": "image/png", "data": "<raw base64>"}` upload format is not adapted: encode uploads as
complete data URLs instead.
Local paths refer to files on the machine running LMDeploy. Decision HTTP requests use the same multimodal
loader as chat, including its URL checks and `allowed_media_domains` configuration. Image formats, base64 decoding,
and animation handling follow that loader; there are no additional Decision-specific image count, upload-byte,
or original-pixel limits. The checkpoint processor controls image resizing and visual token expansion. Pixels use
their stored orientation; the loader does not apply EXIF rotation.

Errors use LMDeploy's existing flat error body: 400 for invalid input, including schema violations and images sent
to an engine without a vision encoder, and 503 when the task is not enabled. Image capability checks use the
shared multimodal processor.

See also [the HTTP client example](../../../examples/decision/decision.py).

## OpenAI Decisions input

For the supported subset of the [OpenAI Decisions protocol](https://developers.openai.com/api/docs/guides/decisions),
send `model`, a string or message-array `input`, and a `questions` array to `/v1/decisions`:

```json
{
  "model": "Intern-Decision-0.8B",
  "input": "The customer was charged twice.",
  "questions": [
    {"type": "predicate", "name": "duplicate", "instructions": "Was the customer charged twice?"},
    {"type": "choice", "name": "team", "instructions": "Which team should handle this?",
     "choices": [{"value": "billing", "description": "Payment issues"}, {"value": "support"}]},
    {"type": "score", "name": "severity", "instructions": "Rate the severity.",
     "levels": [{"label": "Low"}, {"label": "Medium"}, {"label": "High"}]}
  ]
}
```

`input` also accepts a nonempty array of `user` messages. Each message has string content or a nonempty list of
`input_text` / `input_image` blocks; `type: "message"` is optional. For example:

```json
{
  "model": "Intern-Decision-0.8B",
  "input": [{"role": "user", "content": [
    {"type": "input_text", "text": "Inspect this traffic light."},
    {"type": "input_image", "image_url": "https://example.com/light.png"}
  ]}],
  "questions": [{"type": "predicate", "name": "red", "instructions": "Is the light red?"}]
}
```

Image references accept HTTP(S) URLs and inline `data:image/...;base64,...` URLs, using the shared media loader's
URL restrictions. Server file paths remain a native-protocol feature. Image `detail` may be omitted, `null`, or
`"auto"`, which uses the local processor's default policy; other profiles are rejected.

The parser collects images in encounter order across all messages and writes one-based `image_index` references
into the internal state. Repeated references occupy separate positions. The compiler places actual images before
the state/schema text, so this preserves textual references, not interleaved image-token positions. A single message
with string content keeps the same prompt as a plain string input. Response mapping is unchanged.

Requests contain 1–16 questions, with string `instructions`. Choice questions have 2–62 unique string or boolean
values; `true` and `"true"` are distinct. Score questions have 2–10 ordered levels with a label and optional description.
Question names may be omitted, empty, or repeated: answers retain request order and the supplied name, or `null`
when unnamed. Internally generated field names prevent collisions; supplied names can appear in the model prompt.

The response contains `model`, an `answers` array and `usage`. Predicates return `probability = P(yes)`.
Choices return the original typed `choice`, `confidence`, and an array of `{value, probability}` entries.
Scores return the probability-weighted mean of zero-based level indices, `confidence`, and an array of
`{value, label, probability}` entries. Candidate probabilities are calibrated once using the loaded task's temperature.
For choices and scores, local `confidence` means the maximum calibrated candidate probability; it does not reproduce
OpenAI's confidence calibration. This scorer does not emit model refusals.

`usage.input_tokens` is the engine's input token count, including the scoring skeleton. `output_tokens` is the actual
generated token count (zero for this scoring path), and `total_tokens` is their sum. `input_tokens_details.cached_tokens`
comes from the engine. **`cache_write_tokens: 0` is an unmeasured compatibility placeholder required by the SDK,
not a measurement of cache writes.** `output_tokens_details.reasoning_tokens` is zero because no tokens are generated.
Native response usage retains its historical meaning, described below.

`model` is required in this protocol and must match the served model name returned by `/v1/models`.
It does not load or switch models. Both protocols share the same `DecisionContext` and resolved temperature.
Model identity is handled by the HTTP endpoint, outside that context. Responses identify the served model.

Other message roles, audio/video/file blocks, structured instructions, tool records, prompt cache controls, and
other unsupported fields are rejected with 400. Requests mixing `input` with native
`state`/`images` are rejected. `/v1/jev` does not accept OpenAI requests. These input limits, confidence and cache-write
semantics, and the flat error body mean this is a supported subset, not full API equivalence.

An OpenAI SDK exposing `client.decisions` can call this endpoint with `base_url="http://127.0.0.1:23333/v1"` and
the loaded model name. Older SDKs without that resource can use HTTP directly.

## Native response

Shape of a response (values are illustrative):

```json
{
  "request_id": "...", "model": "...",
  "answers": {
    "action": {"type": "choice", "probabilities": {"stop": 0.83, "go": 0.17}, "confidence": 0.83,
               "choice": "stop", "source": "local", "decision": "stop"},
    "risk":   {"type": "score", "probabilities": {"0": 0.12, "1": 0.30, "2": 0.58}, "confidence": 0.58,
               "score": 1.46, "legend": {"0": "Safe", "1": "Some risk", "2": "Dangerous"}, ...},
    "red":    {"type": "noul", "probabilities": {"no": 0.08, "yes": 0.92}, "confidence": 0.92, "noul": 0.92, ...}
  },
  "usage": {"input_tokens": 412, "output_tokens": 3, "decision_count": 3, "generated_tokens": 0},
  "timing": {"preprocess_ms": 4.1, "inference_ms": 38.0, "server_ms": 43.2},
  "calibration": {"method": "temperature-scaling", "temperature": 2.747760550703}
}
```

Probabilities use the same candidate softmax and temperature calibration as the checkpoint's `inference.py`
(`2.747760550703` when read from the official Intern-Decision-0.8B checkpoint). The native response's
`calibration` field records the temperature actually used, including any default fallback. Different inference kernels and floating-point dtypes can introduce
small numerical differences from the official `DecisionEngine`; this is not a claim of bitwise-equivalent model
outputs. `score` is the probability-weighted mean of the numeric option keys (or list indices); `noul` is the
probability of `yes`. `usage.output_tokens` is the number of answered questions for protocol compatibility;
`usage.generated_tokens` is always zero.

## How it runs

A decision runs in the decision endpoint, which drives the engine directly and owns the request session. `DecisionParser`
converts the decision protocol to engine input and interprets the scoring output; it performs no engine execution.
There is no separate `DecideRunner`, no shared-runner hooks, no decision-owned vision pipeline, and no independently
loaded `DecisionEncoder`:

1. `compile_row` compiles `state` and `questions` into messages with the trained assistant skeleton, wraps image
   references as `image_url` blocks as `/generate` does, and places them before the user text.
   `DecisionParser` passes the complete messages to preprocessing.
   Shared startup metadata validates marker/answer token IDs using the
   engine's existing tokenizer; it does not load a second tokenizer or processor.
2. The parser sets `do_preprocess=True`, `enable_thinking=False` and `add_vision_id=True`, reusing ordinary chat
   preprocessing with the service's configured template. Use the checkpoint's HF template; it trims the final
   assistant terminator. Images use the already loaded processor to tokenize and expand visual tokens.
3. After preprocessing, the parser locates `<decision>` in the final input IDs and sets `logprob_start_len` to the
   position immediately before the first marker. It saves each question's scoring row index and the expected
   row count. Image token counts need not be predicted beforehand.
4. `GenerationConfig(max_new_tokens=0, logprobs=0, logprob_extra_token_ids=...)` requests input scoring. The engine projects
   the LM head for the selected suffix and returns the requested 62 candidate scores on those rows, independent of
   top-k rank. This limits the returned scores, not the vocabulary dimension of the LM-head projection.
5. The parser validates the returned row count, selects rows using the saved indices and each question's candidate
   symbols, applies
   softmax and temperature calibration, and constructs the response. Raw logits and raw log probabilities both
   support this candidate normalization.

This is prefill-only execution, possibly split into prefill chunks; it never enters autoregressive decode. Scheduling,
session cleanup and cancellation reuse `ServerContext.create_session` and the shared request-cleanup helper.
Prefix reuse stops before the requested scoring positions so those scores are recomputed.

Decision image references are wrapped into message blocks without decoding them; shared preprocessing loads them.
The loader closes images that it creates after visual preprocessing, including partial failures, and waits for
outstanding worker threads before closing images on cancellation.

## Limitations

- PyTorch backend only; speculative decoding and non-hybrid (disaggregated) roles are rejected.
- After image expansion, input must be shorter than the engine's session length,
  which is the model's own context window unless overridden by `--session-len`.
  Longer inputs are rejected, never truncated.
- Native requests contain 1–16 questions. Each `choice` or `score` question has 1–62 options, assigned
  answer symbols in request order. The original option labels are restored in the response.
- In native requests, `model` may be omitted or match the served model name (`--model-name`, default: the model path).
  Checkpoint directory basenames are no longer additional aliases. Set `--model-name` explicitly for a short name.
- With `--dp > 1`, call each rank's server directly: the DP proxy does not route `/v1/decisions`.
