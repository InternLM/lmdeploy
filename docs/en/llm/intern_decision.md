# Intern-Decision

[Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) answers structured questions by
**scoring** instead of generating. A Systemone request holds a `state`, up to 16 `questions` and optional image `assets`. The model
reads one prompt that ends in a JSON skeleton with a `<decision>` marker per question, and the logits right before
each marker, restricted to that question's answer symbols (`A`, `B`, ...), give the answer distribution.

LMDeploy serves it with the `intern-decision` task on the PyTorch engine. The checkpoint itself is an ordinary
Qwen3.5 vision-language model; the task adds the protocol on top. The decision protocol and preprocessing are
adapted from the Apache-2.0 [Intern-Decision project](https://github.com/InternLM/Intern-Decision); adapted files
keep their attribution.

Decision requests are currently supported through HTTP only; the Python Pipeline has no Decision entry point.
Only `SystemoneRequest` / `SystemoneResponse` and `OpenAIDecisionRequest` / `OpenAIDecisionResponse`
are wire schemas. The parser adapts OpenAI input into a private `_DecisionRow` containing `state`, `questions`
and `images`; it does not accept the former native request format or emit its response envelope.

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
`POST /v1/systemone` accepts and returns the Systemone protocol described below, with `state`, a `questions`
object and optional `assets`. Native requests sent to `/v1/decisions` are rejected with 422.
Use `/v1/systemone` for text and image scoring; image uploads use `assets` with references in the request content.

See also [the HTTP client example](../../../examples/decision/decision.py).

## Systemone assets input

`POST /v1/systemone` reuses the image asset protocol from `intern-decision-inference` (a8fe089).
It runs on the same decision engine and accepts JSON or `multipart/form-data`:

```json
{
  "state": {"photo": "photo-ref"},
  "assets": {"photo-ref": {"format": "png", "data_base64": "<raw padded Base64>"}},
  "questions": {"red": {"type": "noul", "instructions": "Is photo-ref red?"}}
}
```

References are expanded in place in string values within `state`, `instructions` and `criteria`, including nested
JSON. Object keys remain unchanged. Every asset must be referenced; repeated references count separately toward
the maximum of eight image occurrences. Images are static PNG/JPEG files, at most 16 MiB each, with both original
dimensions between 280 and 1568 pixels. Base64 must be canonical padded standard Base64 without a data URL prefix.
The HTTP body is limited to 32 MiB. URLs and local paths are not accepted by this endpoint.

For multipart uploads, send the JSON `request` part first, with `Content-Type: application/json`, and reference
each image part exactly once through `assets.<ref>.part`. File format can be omitted and is inferred from its bytes:

```shell
curl http://localhost:23333/v1/systemone \
  -F 'request={"state":"photo-ref","assets":{"photo-ref":{"part":"photo"}},"questions":{"red":{"type":"noul","instructions":"Is it red?"}}};type=application/json' \
  -F 'photo=@image.png;type=image/png'
```

Images are converted to RGB without EXIF rotation and resized with the source service's 32-pixel alignment,
1024-visual-token per-image budget and 4096-token image-group budget. Invalid image content or references return
400; schema errors, including the removed `images` field, return 422; size limits return 413. These policies apply
to `/v1/systemone`; the `/v1/decisions` image contract is preserved.

Responses contain `model`, `answers` and `usage`. A `noul` answer contains `type` and `noul`; choice and score
answers include probabilities and confidence. Score legends preserve the original JSON values and image references.
`usage.input_tokens` includes expanded visual tokens; `usage.output_tokens` counts the tokenized assistant skeleton,
including its decision markers, rather than generated tokens. Images are input evidence and are not returned.

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
`input_text` / `input_image` / `input_json` blocks; `type: "message"` is optional. For example:

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
URL restrictions and configured domain allowlist. Server file paths are not accepted. Image `detail` may be omitted,
`null`, or `"auto"`; other profiles are rejected. Each request accepts at most eight image occurrences across `input` and all question `instructions`, including repeats.
Inline and downloaded images must be static PNG, JPEG, WEBP or GIF, at most 10 MiB of decoded file bytes,
20 million pixels, 20000 pixels on either side, and a 200:1 aspect ratio. Invalid image content returns 422 with
the source field location. HTTP downloads retain the shared loader's transport behavior; the byte limit is checked
after download. Images are converted to RGB and resized on a 32-pixel grid with a 1024-token per-image budget and
a 4096-token total budget, divided by the number of image occurrences, following a8fe089.

The parser collects images in encounter order across all messages and writes one-based `image_index` references
into the internal state. Repeated references occupy separate positions. The compiler places these input images before
the state/schema text, so this preserves textual references, not interleaved image-token positions. A single message
with string content keeps the same prompt as a plain string input. Response mapping is unchanged.

Question images expand at their positions within `instructions`, using the same request-owned cache as input images.
The input-image placement described above stays unchanged. Adjacent instruction text blocks concatenate in order;
`input_json.data` is rendered as compact JSON text and does not interpret strings as image references.
JSON evidence accepts an object or an array of objects, with at most 32 container levels, 10000 values (including
containers), and 256 KiB of compact UTF-8 JSON per block. Choice/level descriptions remain strings.
For example, a question can carry its own image and structured evidence:

```json
{
  "model": "Intern-Decision-4B",
  "input": "Inspect the delivery.",
  "questions": [{
    "type": "predicate", "name": "damaged",
    "instructions": [
      {"type": "input_text", "text": "Does this package look damaged? "},
      {"type": "input_image", "image_url": "https://example.com/package.png"},
      {"type": "input_json", "data": {"expected_condition": "intact"}}
    ]
  }]
}
```

These content blocks extend the a8fe089 deployment adapter; existing checkpoint limits and `detail=auto` policy
remain in effect. They do not add tool records, prompt-cache controls or other image detail profiles.

Requests contain 1–16 questions. Each `instructions` accepts a string or a nonempty array of
`input_text`, `input_image` and `input_json` blocks. Choice questions have 2–62 unique string or boolean
values; `true` and `"true"` are distinct. String choice values must not be blank.
Score questions have 2–62 ordered levels with a label and optional description. The 62-level limit is a local extension
matching the reference deployment; the OpenAI API currently documents a maximum of 10.
Question names may be omitted or `null`; supplied names must be nonblank and unique. Answers retain request order
and the supplied name, or `null` when unnamed. Internally generated field names prevent collisions;
supplied names can appear in the model prompt. A top-level string `input` must not be empty.
Validation does not trim names, choice values, or input text.

The response contains `model`, an `answers` array and `usage`. Predicates return `probability = P(yes)`.
Choices return the original typed `choice`, `confidence`, and an array of `{value, probability}` entries.
Scores return the probability-weighted mean of zero-based level indices, `confidence`, and an array of
`{value, label, probability}` entries. Candidate probabilities are calibrated once using the loaded task's temperature.
For choices and scores, local `confidence` means the maximum calibrated candidate probability; it does not reproduce
OpenAI's confidence calibration. This scorer does not emit model refusals.

`usage.input_tokens` is the engine's input token count, including the scoring skeleton. Like Systemone,
`output_tokens` counts the tokenized assistant skeleton, including decision markers but excluding chat wrappers.
No tokens are generated. `total_tokens` is the sum of the reported input and output counts; the skeleton is not
subtracted from `input_tokens`. This follows the reference deployment's usage convention. `input_tokens_details.cached_tokens`
comes from the engine. **`cache_write_tokens: 0` is an unmeasured compatibility placeholder required by the SDK,
not a measurement of cache writes.** `output_tokens_details.reasoning_tokens` is zero because no tokens are generated.

`model` is required in this protocol and accepts the served model name returned by `/v1/models`,
`InternLM/<served-name>`, or `internlm/<lowercase-served-name>`.
It does not load or switch models. Both protocols share the same `DecisionContext` and resolved temperature.
Model identity is handled by the HTTP endpoint, outside that context. Responses identify the served model.

`safety_identifier` is optional caller-supplied end-user metadata (a string of at most 64 characters, or `null`).
It is accepted for compatibility, is not an authentication credential, and does not enter the prompt or affect scoring.

Other message roles, audio/video/file blocks, tool records, prompt cache controls, and
other unsupported fields are rejected with 422. Requests mixing `input` with native
`state`/`images` are rejected. `/v1/systemone` does not accept OpenAI requests. These input limits, confidence and cache-write
semantics mean this is a supported subset, not full API equivalence. Both decision endpoints return schema and
model-name errors as 422 with `detail` and `error.details` entries identifying the request fields.
Asset and scoring errors retain their existing status codes; other API routes retain their existing error format.

An OpenAI SDK exposing `client.decisions` can call this endpoint with `base_url="http://127.0.0.1:23333/v1"` and
the loaded model name. Older SDKs without that resource can use HTTP directly.

## Systemone response and calibration

Shape of a response (values are illustrative):

```json
{
  "model": "Intern-Decision-0.8B",
  "answers": {
    "action": {"type": "choice", "probabilities": {"stop": 0.83, "go": 0.17},
               "confidence": 0.83, "choice": "stop"},
    "risk": {"type": "score", "probabilities": {"0": 0.12, "1": 0.30, "2": 0.58},
             "confidence": 0.58, "score": 1.46,
             "legend": {"0": "Safe", "1": "Some risk", "2": "Dangerous"}},
    "red": {"type": "noul", "noul": 0.92}
  },
  "usage": {"input_tokens": 412, "output_tokens": 37}
}
```

Probabilities use the same candidate softmax and temperature calibration as the checkpoint's `inference.py`
(`2.747760550703` when read from the official Intern-Decision-0.8B checkpoint), with the fallback described above.
Different inference kernels and floating-point dtypes can introduce small numerical differences from the official
`DecisionEngine`. `score` is the probability-weighted mean of numeric option keys (or list indices); `noul` is
P(yes). The response contains only `model`, `answers` and `usage`; temperature and timing metadata are not returned.
`usage.output_tokens` counts the tokenized assistant skeleton, including decision markers; no tokens are generated.

## How it runs

A decision runs in the decision endpoint, which drives the engine directly and owns the request session. `DecisionParser`
converts the decision protocol to engine input and interprets the scoring output; it performs no engine execution.
There is no separate `DecideRunner`, no shared-runner hooks, no decision-owned vision pipeline, and no independently
loaded `DecisionEncoder`:

1. `compile_row` compiles `state` and `questions` into messages with the trained assistant skeleton.
   Systemone asset references expand in place; the parser decodes and resizes their images and passes PIL-backed
   `image_data` blocks to shared preprocessing. OpenAI images use their own validation limits, the shared HTTP
   loader for remote URLs, and the same resize policy before becoming `image_data` blocks.
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

Both decision protocols prepare images in a worker thread. Prepared images belong to the request, and repeated
references share decoded and resized images. Partial preparation failures close the prepared images. Shared
preprocessing consumes PIL-backed blocks and expands visual tokens without downloading or decoding them again.

## Limitations

- PyTorch backend only; speculative decoding and non-hybrid (disaggregated) roles are rejected.
- After image expansion, input must be shorter than the engine's session length,
  which is the model's own context window unless overridden by `--session-len`.
  Longer inputs are rejected, never truncated.
- Native requests contain 1–16 questions. Each `choice` or `score` question has 1–62 options, assigned
  answer symbols in request order. The original option labels are restored in the response.
- Systemone `model` may be omitted or empty, or match the served model name (`--model-name`, default: the model path),
  `InternLM/<served-name>`, or `internlm/<lowercase-served-name>`. Set `--model-name` explicitly for a short name.
- With `--dp > 1`, call each rank's server directly: the DP proxy does not route `/v1/decisions`.
