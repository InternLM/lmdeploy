# Copyright (c) OpenMMLab. All rights reserved.
"""Backend-neutral guided-decoding grammar compilation."""
from __future__ import annotations

import json
import weakref
from functools import lru_cache
from typing import Any


def get_interns1_encoded_vocab(tokenizer, vocab_size: int) -> list[bytes]:
    """Decode InternS1/S2's mixed vocabulary to bytes for grammar matching.

    The slow tokenizer combines byte-level BPE with domain-specific pieces.
    ``get_vocab`` uses synthetic names for those pieces, whereas ID lookup
    returns their actual text. Preserve incomplete UTF-8 bytes rather than
    decoding individual tokens to strings with replacement characters.
    """
    encoded_vocab = [b''] * vocab_size
    added_ids = set(tokenizer.get_added_vocab().values())
    for token_id in tokenizer.get_vocab().values():
        if token_id >= vocab_size:
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token_id in added_ids:
            encoded_vocab[token_id] = token.encode('utf-8')
        else:
            token = token.replace('▁', 'Ġ').replace('\n', 'Ċ')
            encoded_vocab[token_id] = bytes(tokenizer.byte_decoder[char] for char in token)
    return encoded_vocab


def _json_schema_from_response_format(response_format: dict[str, Any]) -> str:
    schema: Any = response_format['json_schema']
    if isinstance(schema, dict):
        for key in ('json_schema', 'schema'):
            if key in schema:
                schema = schema[key]
                break
    if isinstance(schema, (dict, bool)):
        return json.dumps(schema, ensure_ascii=False)
    if isinstance(schema, str):
        return schema
    raise ValueError(
        f'Cannot parse schema {schema}. The schema must be either a dictionary '
        'or a string that contains the JSON Schema specification')


def _grammar_source(response_format: dict[str, Any]) -> tuple[str, str]:
    """Return the XGrammar entry point and its serialized input."""
    schema_type = response_format.get('type', 'text')
    if schema_type == 'json_schema':
        return schema_type, _json_schema_from_response_format(response_format)
    if schema_type == 'regex_schema':
        return schema_type, str(response_format.get('regex_schema', ''))
    if schema_type == 'json_object':
        schema = json.dumps({'type': 'object', 'additionalProperties': True})
        return 'json_schema', schema
    if schema_type == 'structural_tag':
        # Internal compiler policy is separate from XGrammar's tag schema.
        tag = {key: value for key, value in response_format.items() if key != '_token_id_only'}
        return schema_type, json.dumps(tag, ensure_ascii=False)
    if schema_type == 'text':
        return schema_type, ''
    raise ValueError(f'unsupported format type: {schema_type}')


@lru_cache(maxsize=128)
def _check_response_format(serialized_format: str) -> None:
    """Check a response format before it reaches a shared engine loop."""
    import xgrammar as xgr

    response_format = json.loads(serialized_format)
    schema_type, source = _grammar_source(response_format)
    if schema_type == 'text':
        return
    if schema_type == 'json_schema':
        xgr.Grammar.from_json_schema(source)
    elif schema_type == 'regex_schema':
        xgr.Grammar.from_regex(source)
    elif schema_type == 'structural_tag':
        xgr.Grammar.from_structural_tag(source)


def ensure_response_format_compilable(response_format: dict[str, Any]) -> None:
    """Reject response formats that XGrammar cannot compile."""
    try:
        serialized_format = json.dumps(response_format, ensure_ascii=False, separators=(',', ':'))
        _check_response_format(serialized_format)
    except (KeyError, RuntimeError, TypeError) as err:
        raise ValueError(f'Unsupported response format: {err}') from err


class _CompilerIdentity:
    """Cache by compiler identity even for unhashable XGrammar wrappers."""

    def __init__(self, compiler, tokenizer_info, tokenizer=None):
        self.compiler = compiler
        self.tokenizer_info = tokenizer_info
        # The cached compiler must not keep a model's HF tokenizer alive.
        try:
            self.tokenizer_ref = weakref.ref(tokenizer) if tokenizer is not None else None
        except TypeError:
            self.tokenizer_ref = None

    def __hash__(self):
        return id(self.compiler)

    def __eq__(self, other):
        return isinstance(other, _CompilerIdentity) and self.compiler is other.compiler


@lru_cache(maxsize=8)
def _token_id_only_compiler(identity: _CompilerIdentity, token_ids: tuple[int, ...]):
    """Reserve protocol IDs for atomic Token/ExcludeToken grammar terminals.

    XGrammar also tries each token's decoded bytes against text rules. A GLM closing ID must not match a
    const/enum/regex inside an argument. Use a private compiler vocabulary whose reserved entries cannot match UTF-8
    text, while keeping their numeric IDs available to atomic terminals. The actual tokenizer, engine token IDs and
    other compilers are unchanged.
    """
    compiler, tokenizer_info = identity.compiler, identity.tokenizer_info
    if tokenizer_info is None:
        # Python XGrammar exposes tokenizer metadata on a compiled grammar.
        # TurboMind supplies its metadata explicitly through the same API.
        tokenizer_info = compiler.compile_regex('').tokenizer_info
    vocab_size = tokenizer_info.vocab_size
    for token_id in token_ids:
        if not isinstance(token_id, int) or not 0 <= token_id < vocab_size:
            raise ValueError(f'Invalid protocol token ID: {token_id!r}')
    clone = getattr(tokenizer_info, 'with_token_id_only', None)
    if clone is not None:
        # TurboMind can copy and patch the vocabulary entirely in C++.
        return type(compiler)(clone(token_ids))
    metadata = json.loads(tokenizer_info.dump_metadata())
    tokenizer = identity.tokenizer_ref() if identity.tokenizer_ref is not None else None
    if metadata['vocab_type'] == 2 and tokenizer is not None:  # BYTE_LEVEL
        # GLM's immutable engine tokenizer supplies the same encoded vocabulary
        # used to build tokenizer_info. Its Rust-backed bulk getter avoids one
        # Python/FFI call per decoded entry. Preserve model padding and metadata.
        vocab = [''] * vocab_size
        for token, token_id in tokenizer.get_vocab().items():
            if 0 <= token_id < vocab_size:
                vocab[token_id] = token
        # In byte-level encoding U+00FF represents the single byte 0xff.
        replacement = '\u00ff'
    else:
        vocab = list(tokenizer_info.decoded_vocab)
        metadata['vocab_type'] = 0  # RAW: vocab is already decoded to bytes.
        replacement = b'\xff'
    for token_id in token_ids:
        if token_id >= len(vocab):
            raise ValueError(f'Invalid protocol token ID: {token_id!r}')
        # 0xff cannot occur in UTF-8 text. An empty entry would instead mark
        # the ID as padding, which XGrammar rejects even in Token(id) rules.
        vocab[token_id] = replacement
    info = type(tokenizer_info).from_vocab_and_metadata(vocab, json.dumps(metadata))
    return type(compiler)(info)


def compile_response_format(compiler, response_format: dict[str, Any], *, tokenizer_info=None, tokenizer=None):
    """Compile one public or internal response format with XGrammar.

    Structural tags retain XGrammar's native shape. Internal token-ID policy is applied here before compilation; native
    backends supply tokenizer metadata, while Python XGrammar can expose it through a compiled grammar. If supplied,
    tokenizer must be the immutable tokenizer used to construct that metadata.
    """
    schema_type, source = _grammar_source(response_format)
    if schema_type == 'json_schema':
        return compiler.compile_json_schema(source)
    if schema_type == 'regex_schema':
        return compiler.compile_regex(source)
    if schema_type == 'structural_tag':
        token_ids = tuple(response_format.get('_token_id_only', ()))
        if token_ids:
            compiler = _token_id_only_compiler(_CompilerIdentity(compiler, tokenizer_info, tokenizer), token_ids)
        return compiler.compile_structural_tag(source)
    raise ValueError(f'unsupported format type: {schema_type}')
