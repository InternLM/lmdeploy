# Parser design

This document describes the internal parser architecture used by LMDeploy's
chat serving path. It is intended for contributors adding or reviewing a
reasoning or tool-call protocol. For user-facing setup and request examples,
see [Reasoning Outputs](../llm/api_server_reasoning.md) and
[Tools Calling](../llm/api_server_tools.md).

The parser layer follows one central rule:

> Recognize protocol boundaries, preserve the model's stable output, and emit
> normalized deltas as early as the output contract permits.

It is deliberately not a strict validator for model-generated JSON or XML.
Callers remain responsible for deciding whether duplicated parameters,
incomplete values, or other semantically questionable output is acceptable.

## Architecture

The serving path separates transport, response routing, and model-specific
syntax:

```text
engine output chunk
        |
        v
ChatRunner                         request/session and transport metadata
        |
        v
ResponseParser                    plain/reasoning/tool routing
        |                         and the unconsumed response suffix
        +--> ReasoningParser      reasoning tags and initial mode
        |
        +--> ToolParser           one outer tool block and normalized calls
                  |
                  +--> JSON, XML-like, DSML, or model-specific consumer
        |
        v
OpenAI / Anthropic response adapter
```

The ownership boundaries are:

| Component            | Owns                                                                                              | Does not own                           |
| -------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `ChatRunner`         | Engine iteration, request lifetime, finish status, token IDs, and log probabilities               | Model protocol syntax                  |
| `ResponseParser`     | Routing between plain content, reasoning, and tool blocks; the unconsumed response suffix         | Tool payload grammar                   |
| `ReasoningParser`    | Reasoning opening/closing tags and whether parsing starts in reasoning mode                       | Streaming buffers and response routing |
| `ToolParser`         | The outer tool block, common call identity/filtering rules, and normalized `DeltaToolCall` output | Plain or reasoning content             |
| Concrete tool parser | The inner payload grammar and its incremental state                                               | SSE framing and API transport metadata |
| `JsonValueScanner`   | The lexical end of one JSON value                                                                 | JSON grammar or schema validation      |

The corresponding implementation is organized as follows:

| Area                             | Main file                                                     |
| -------------------------------- | ------------------------------------------------------------- |
| Transport integration            | `lmdeploy/serve/core/chat_runner.py`                          |
| Response routing                 | `lmdeploy/serve/parsers/response_parser.py`                   |
| Reasoning protocol declarations  | `lmdeploy/serve/parsers/reasoning_parser/reasoning_parser.py` |
| Common tool contract             | `lmdeploy/serve/parsers/tool_parser/tool_parser.py`           |
| JSON envelope and value scanning | `json_tool_parser.py`, `json_value_scanner.py`                |
| Shared XML-like state machine    | `xml_tool_parser.py`                                          |

Most models use `BaseResponseParser`, configured with registered reasoning and
tool parser classes. `ResponseParserManager`, `ReasoningParserManager`, and
`ToolParserManager` resolve their configured names. A protocol whose
token-channel semantics do not fit this composition may register a specialized
`ResponseParser`; OpenAI Harmony is the current example.

Parser objects are stateful and belong to one response. Do not share one
instance between concurrent or sequential requests.

## The response parser

`BaseResponseParser` owns the top-level state machine:

```text
                       reasoning close
                 +---------------------------+
                 |                           v
plain -- reasoning open --> reasoning      plain
  |                              |
  +--------- tool open ----------+
                 |
                 v
                tool -- complete outer block --> plain
```

An engine chunk does not have to align with these boundaries. One chunk can,
for example, contain the end of reasoning, plain content, a tool opening tag,
and part of its payload. Therefore:

```python
ResponseParser.stream_chunk(...) -> list[tuple[DeltaMessage, bool]]
```

may return multiple messages for one engine chunk. It may also return an empty
list while a possible marker or payload fragment is buffered.

The boolean paired with each `DeltaMessage` reports whether that message emits
tool calls. `ChatRunner` uses it to track whether a terminal `stop` should be
reported to the client as `tool_calls`.

Although it is named a parser, `ReasoningParser` is deliberately a protocol
descriptor: it declares the reasoning tags and initial mode.
`BaseResponseParser` owns buffering, marker detection, and channel routing.

`BaseResponseParser` keeps only `self._pending`, the response suffix whose role
is not stable yet. In plain and reasoning modes this is normally a proper
prefix of an opening or closing marker. In tool mode, it is the suffix not
consumed by `ToolParser.feed_tool_block`.

`ChatRunner` attaches token IDs, log probabilities, and finish status only to
the last parser delta produced from an engine chunk. This prevents transport
metadata from being duplicated when one input chunk is split into several
visible response deltas.

### Complete and streaming responses

Complete parsing is not a separate grammar implementation. `parse_complete`
feeds the full text through `stream_chunk(..., final=True)`, then joins content
and reasoning fragments and builds complete tool calls from their deltas.
Consequently, fixes to boundary handling apply to both streaming and
non-streaming requests.

`final=True` means that no more decoded text will arrive. Parsers then release
otherwise ambiguous plain text and perform best-effort tool-call finalization.
This is not a promise that malformed model output will become valid. For
example, the XML-like parser closes the emitted OpenAI `function.arguments`
object, but it does not invent a missing function name or value.

### Separators between tool blocks

After a tool block closes, the response parser drops newlines only after it can
confirm that they immediately precede another tool opening tag. Thus the
separator in either of these sequences is not returned as assistant content:

```text
</tool_call>\n<tool_call>
</tool_call>\n\n<tool_call>
```

A newline followed by ordinary text, or a trailing newline at the end of the
response, remains content. This decision belongs to the response parser
because it spans two outer blocks; an individual tool parser must not consume
it.

## The consumed-prefix tool interface

Once the response parser has consumed a tool opening tag, it calls:

```python
consumed = tool_parser.feed_tool_block(text, deltas, final=final)
```

`text` starts immediately after the outer opening tag. It may contain retained
text from the previous call, newly decoded text, an outer closing tag, and
trailing assistant content. The method appends zero or more normalized tool
deltas and returns the number of leading characters that are safe to discard:

```python
pending = text[consumed:]
```

This interface is intentionally based on a consumed prefix instead of a
parser-owned copy of the response buffer. It gives each layer one buffer owner,
allows a tool parser to stop exactly before trailing content, and avoids
rescanning text that has already been emitted.

The concrete parser implements:

```python
_consume_stream_payload(text, deltas, final=final) -> int
```

It consumes only the inner dialect payload. It must not consume the outer
closing marker. When the inner grammar is complete it sets `_payload_closed`;
`ToolParser.feed_tool_block` then consumes the outer delimiter and exposes any
remaining text to the response parser.

Returning `0`, or consuming syntax without emitting a delta, is normal. It
means that more decoded text is required before a boundary or client-visible
fragment becomes stable.

### Two kinds of completion

`ToolParser` deliberately tracks two completion states:

| State             | Meaning                                                                                   | Owner                |
| ----------------- | ----------------------------------------------------------------------------------------- | -------------------- |
| `_payload_closed` | The model-specific inner payload is complete; the outer close marker may still be pending | Concrete tool parser |
| `block_closed`    | The complete outer block is consumed, or final input forced it closed                     | `ToolParser`         |

The distinction prevents an inner parser from knowing how the response parser
will handle outer delimiters or trailing content.

## Normalized tool-call output

All tool parsers use `_begin_call` and `_emit_delta` to enforce the same
client-visible contract.

### Call identity is emitted first

For each accepted call index, the first visible delta contains its ID, type,
and function name. Argument fragments follow it. For example, even if the
model emits:

```json
{"arguments":{"city":"Beijing"},"name":"weather"}
```

the normalized order is conceptually:

```text
DeltaToolCall(index=0, id=..., type="function", name="weather")
DeltaToolCall(index=0, arguments="{\"city\":\"Beijing\"}")
```

Arguments encountered before the name are retained as separate fragments and
released in their original order after the identity delta. This small delay is
required by downstream streaming APIs, which create a tool block from its
first event and cannot reliably repair a missing name later.

### Filtering and preservation

In the serving path, `adjust_request` records the tool names exposed by the
effective request. A generated call whose name is absent from that set is not
emitted. Accepted calls receive dense, zero-based output indices even when an
earlier generated call was rejected.

Beyond that runtime boundary, the parser preserves as much generated data as
possible. In particular, repeated argument keys or repeated XML parameters are
not deduplicated. Argument source fragments keep their order, and the caller
decides whether the resulting argument object is semantically acceptable.

Calls are sequential: a concrete parser finishes or abandons the active call
before starting the next one. `_begin_call` resets the common per-call state,
so concrete parsers do not need a map for interleaved call fragments.

## JSON tool payloads

`JsonToolParser` handles a JSON envelope with a function name and an argument
field. Qwen parsers use `arguments`; InternLM and Llama parsers override
`argument_field` with `parameters`. Subclasses normally only declare the outer
markers and the argument-field name.

The envelope parser incrementally recognizes keys, colons, names, values, and
the object end. The argument value is forwarded verbatim as stable source text
becomes available. Other values are scanned only far enough to find the next
envelope field. If the envelope has no argument field, `{}` is emitted exactly
once.

`JsonValueScanner` is a lexical scanner. It tracks container depth, quoted
strings, and escapes, but intentionally does not deserialize and reserialize
the argument value. This has three useful properties:

- Duplicate keys and original formatting are preserved.
- Large values do not require a second complete copy before streaming.
- Work remains approximately linear because already consumed input is not
  searched again.

For long strings and containers the scanner uses bulk searches for syntax;
very short suffixes use a direct loop. Do not replace this with repeated
searches from the beginning of the pending buffer.

## XML-like tool payloads

`XmlToolParser` is a shared driver for sequential, function-first protocols
whose argument values end at a dialect-specific closing marker. GLM-4.7 and
Qwen3-Coder are examples. Its state transitions are:

```text
function --> arg_start --> arg_name --> arg_value
                ^                         |
                +-------------------------+
                |
                +---- payload end ----> done
```

The base class alone owns `XmlParseState`. A dialect recognizes syntax through
three hooks and returns semantic results without mutating that shared state:

```python
_consume_function(payload, pos, final) -> (pos, name, "arg_start" | "done") | None
_consume_arg_start(payload, pos) -> (pos, "arg_name" | "done") | None
_consume_arg_name(payload, pos) -> (pos, argument_name) | None
```

`None` means that the unconsumed suffix is incomplete and must be retained.
The shared driver applies successful transitions. `XmlToolParser` itself owns
argument-value consumption, JSON-object emission, schema-aware scalar
coercion, and the transition back to `arg_start`.

This boundary is intentionally narrower than “all XML-shaped protocols.” A
format that can emit arguments before its function identity does not satisfy
the `XmlToolParser` contract because parameter-schema lookup requires the
function name. A format without close-delimited argument values should also
derive directly from `ToolParser` or use another suitable shared parser.

Plain unquoted string values can stream immediately after their type is known.
Quoted values and values declared as non-string types are buffered until the
closing marker, then normalized and serialized once. The parser holds at most
the syntax needed to make that decision and any value that inherently requires
whole-value conversion.

## Token-aligned marker prefixes

Decoded chunks end at token boundaries, not at arbitrary character positions.
A parser should therefore retain only proper marker prefixes that can really
occur at the end of a decoded token chunk.

For example, if `</parameter>` is tokenized as at most `</`, `parameter`, and
`>`, the only possible incomplete suffixes are:

```python
arg_value_close_prefixes = ("</parameter", "</")
```

There is no need to test every character prefix such as `<`, `</p`, or
`</para`. If a supported tokenizer represents the whole marker as one token,
the prefix tuple is empty. `_stable_prefix_end` uses these longest-first tuples
to return the safe consumable boundary without tokenizing in the hot path.

Always verify the tuple against every tokenizer supported by that parser.
Arbitrary character chunking is useful for general JSON scanner tests, but it
is not a valid simulation for atomic or multi-token protocol markers.

## Adding a parser

Choose the narrowest existing abstraction that matches the model protocol:

1. Subclass `JsonToolParser` for a JSON envelope that only changes its outer
   tags or argument-field name.
2. Subclass `XmlToolParser` only for a sequential, function-first,
   close-delimited argument protocol.
3. Subclass `ToolParser` for a materially different grammar such as DSML or a
   section/call protocol.
4. Add a specialized `ResponseParser` only when top-level response channels or
   token semantics cannot be expressed by `BaseResponseParser` composition.

A minimal JSON dialect looks like this:

```python
from .json_tool_parser import JsonToolParser
from .tool_parser import ToolParserManager


@ToolParserManager.register_module("example-json")
class ExampleJsonToolParser(JsonToolParser):
    argument_field = "arguments"

    @classmethod
    def get_tool_open_tag(cls) -> str:
        return "<tool_call>"

    @classmethod
    def get_tool_close_tag(cls) -> str:
        return "</tool_call>"
```

Import the module from `tool_parser/__init__.py` so registry discovery and CLI
validation can find it. If a closing marker can be split at supported token
boundaries, declare its proper prefixes in `tool_close_prefixes`. Do not add
speculative character prefixes.

`adjust_request` is the place for protocol-specific prompt settings. Always
call `super().adjust_request(request)` so tool normalization and allowed-name
filtering still run. A parser supports `tool_choice="required"` only when its
`structural_tag_model` (and, if needed, `reasoning_structural_tag_model`)
selects a matching XGrammar structural-tag format.

## Implementation invariants

Keep these invariants when implementing or reviewing a parser:

- Emit every stable fragment promptly, except when call identity or
  whole-value conversion requires buffering.
- Call `_begin_call` exactly once before emitting each logical call.
- Route all client-visible call fragments through `_emit_delta`.
- Consume only a stable prefix and never consume trailing assistant content.
- Leave the outer close marker to `ToolParser.feed_tool_block`.
- Set `_payload_closed` once the inner grammar is complete.
- Preserve fragment order and duplicate parameters; do not add semantic
  validation to the streaming hot path.
- Avoid decoding, tokenizing, reparsing, or rescanning previously consumed
  text.
- Treat `final=True` as end-of-input, not as proof that the payload is valid.

## Tests and benchmarks

Every parser change should compare reconstructed streaming output with
complete parsing and the expected normalized result. Cover at least:

- the whole response in one chunk;
- opening and closing markers at every valid token boundary;
- function names before and after arguments where the source format permits
  both;
- calls with no arguments, repeated parameters, multiple sequential calls,
  and unknown function names;
- incomplete or malformed final payloads and ordinary content after a tool
  block;
- reasoning, content, and tool segments sharing one engine chunk;
- token-ID and log-probability transport when one engine chunk produces
  multiple parser deltas.

For JSON scanner performance, include small and large chunks (for example 1,
4, 8, 16, and 256 characters) and inputs containing many short strings. A
benchmark fixture must use the actual grammar of the selected parser
(`arguments` versus `parameters`, XML tags, DSML, and so on), and must validate
the reconstructed result before reporting throughput.
