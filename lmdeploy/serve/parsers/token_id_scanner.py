# Copyright (c) OpenMMLab. All rights reserved.
"""Locate declared control IDs in text produced by the existing detokenizer.

Like vLLM's TokenIDScanner, single-ID decodes are cached per request. Unlike its string-search fallback, recovery checks
decoded control-led blocks against the actual suffix. Repeated literal spellings therefore cannot choose a boundary.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from .token_id_buffer import ControlSpan, LocatedText

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TokenIDProfile:
    controls: Mapping[int, str]
    special_ids: frozenset[int]


def build_token_profile(tokenizer: PreTrainedTokenizerBase,
                        token_id_terminals: tuple[str, ...]) -> TokenIDProfile:
    """Resolve declared control tags and validate ByteLevel decoding.

    Only declared tags become boundaries. Token IDs and skipped special IDs come from the tokenizer; parser/model
    selection belongs to the caller.
    """
    get_added_vocab = getattr(tokenizer, 'get_added_vocab', None)
    if get_added_vocab is None:
        raise ValueError('Token-ID parsing requires added-token metadata')
    added = get_added_vocab()
    missing = sorted(set(token_id_terminals) - added.keys())
    if missing:
        raise ValueError(
            f'Missing control tokens for token-ID parsing: {missing}. '
            'Refusing to fall back to string parsing.')
    if not tokenizer.is_fast:
        raise ValueError('Token-ID parsing requires a fast ByteLevel tokenizer')
    backend = tokenizer.backend_tokenizer
    decoder = backend.decoder
    # Serialize only the decoder state, not the full vocabulary and BPE merges.
    decoder_config = json.loads(decoder.__getstate__()) if decoder is not None else None
    expected_decoder = dict(type='ByteLevel', add_prefix_space=True, trim_offsets=True, use_regex=True)
    if (decoder_config != expected_decoder or backend.normalizer is not None
            or tokenizer.clean_up_tokenization_spaces):
        raise ValueError('Unsupported decoder configuration for token-ID parsing')
    added_tokens = backend.get_added_tokens_decoder()
    special_ids = frozenset(tokenizer.all_special_ids)
    # Native incremental decoding filters through the HF wrapper, whereas
    # scanner.decode uses the backend. Both must skip the same IDs.
    if special_ids != frozenset(token_id for token_id, token in added_tokens.items() if token.special):
        raise ValueError('Tokenizer special-token metadata differs between tokenizer and backend')
    controls = {added[tag]: tag for tag in token_id_terminals}
    if len(controls) != len(token_id_terminals):
        raise ValueError('Declared control tokens must have distinct IDs')
    for token_id, tag in controls.items():
        token = added_tokens.get(token_id)
        if (token is None or token.content != tag or token.special
                or token.single_word or token.lstrip or token.rstrip or token.normalized):
            raise ValueError('Unsupported control-token configuration')
        if (tokenizer.encode(tag, add_special_tokens=False) != [token_id]
                or tokenizer.decode([token_id], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False) != tag):
            raise ValueError('Declared control token must encode and decode as a single ID')
    return TokenIDProfile(MappingProxyType(controls), special_ids)


class TokenIDScanner:
    """Request-local locator.

    It never replaces text from the serving engine.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, profile: TokenIDProfile):
        self.tokenizer = tokenizer
        self.profile = profile
        self.control_ids = frozenset(profile.controls)
        self.skip_special_tokens = True
        self._cache: dict[int, str] = {}
        self._waiting: list[int] = []

    def _decode(self, ids: list[int], skip: bool) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip, clean_up_tokenization_spaces=False)

    def scan(self, text: str, ids: list[int], *, engine_final: bool = False) -> LocatedText:
        # AsyncEngine's independent final result discards unrendered pending
        # bytes and decodes include_stop_str with skip_special_tokens=False.
        if engine_final:
            self._waiting.clear()
        skip = self.skip_special_tokens and not engine_final
        if not text:
            if not engine_final:
                self._waiting.extend(ids)
            return LocatedText('', ())
        if not ids and not self._waiting:
            raise ValueError('Token-ID parsing requires IDs for visible output')
        if not self._waiting and self.control_ids.isdisjoint(ids):
            return LocatedText(text, ())
        if self._waiting:
            ids = self._waiting + list(ids)
            self._waiting = []
        if self.control_ids.isdisjoint(ids):
            return LocatedText(text, ())

        decoded: list[str] = []
        control_indices: list[int] = []
        text_offsets: list[int] = []
        offset = 0
        value: str | None
        for i, token_id in enumerate(ids):
            if skip and token_id in self.profile.special_ids:
                value = ''
            else:
                value = self._cache.get(token_id)
                if value is None:
                    value = self._decode([token_id], False)
                    self._cache[token_id] = value
            if token_id in self.profile.controls:
                control_indices.append(i)
                text_offsets.append(offset)
            decoded.append(value)
            offset += len(value)
        reconstructed = ''.join(decoded)
        if reconstructed and text.endswith(reconstructed):
            prefix = len(text) - len(reconstructed)
            positions = [prefix + text_offset for text_offset in text_offsets]
        else:
            # A byte-split character can make per-ID decodes differ. Controls
            # are byte boundaries; recover each control-led block from right
            # to left, reusing cached strings before decoding a whole block.
            right = len(text)
            positions = []
            for i in range(len(control_indices) - 1, -1, -1):
                begin = control_indices[i]
                end = control_indices[i + 1] if i + 1 < len(control_indices) else len(ids)
                tag = self.profile.controls[ids[begin]]
                # Reuse the joined cache text, rather than join each block again.
                text_end = text_offsets[i + 1] if i + 1 < len(control_indices) else len(reconstructed)
                segment = reconstructed[text_offsets[i]:text_end]
                if not segment.startswith(tag) or not text.endswith(segment, 0, right):
                    segment = self._decode(ids[begin:end], skip)
                if not segment.startswith(tag) or not text.endswith(segment, 0, right):
                    raise ValueError('Cannot align control IDs with decoded output')
                right -= len(segment)
                positions.append(right)
            positions.reverse()
        spans = tuple(
            ControlSpan(pos, pos + len(self.profile.controls[ids[index]]), self.profile.controls[ids[index]])
            for pos, index in zip(positions, control_indices))
        return LocatedText(text, spans)
