# Copyright (c) OpenMMLab. All rights reserved.
"""CPU-only GLM control-token regressions with an in-memory ByteLevel
vocabulary."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParser, ReasoningParserManager
from lmdeploy.serve.parsers.response_parser import BaseResponseParser, _resolve_token_id_terminals
from lmdeploy.serve.parsers.token_id_buffer import BoundaryIndex, BoundaryView, ControlSpan, LocatedText, ParseBuffer
from lmdeploy.serve.parsers.token_id_scanner import TokenIDScanner, build_token_profile
from lmdeploy.serve.parsers.tool_parser import ToolParserManager
from lmdeploy.serve.parsers.tool_parser.glm47_tool_parser import Glm47ToolParser
from lmdeploy.tokenizer import DetokenizeState, HuggingFaceTokenizer

CONTROL_TAGS = ('<think>', '</think>', '<tool_call>', '</tool_call>',
                '<arg_key>', '</arg_key>', '<arg_value>', '</arg_value>')

TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {
    'type': 'object', 'properties': {'text': {'type': 'string'}}}}}]


# Tokenizer metadata is independent of the parser profile under test.
_SPECIAL_TOKENS = set(
    '<|endoftext|> [MASK] [gMASK] [sMASK] '
    '<sop> <eop> <|system|> <|user|> '
    '<|assistant|> <|observation|> <|begin_of_image|> <|end_of_image|> '
    '<|begin_of_video|> <|end_of_video|> <|begin_of_audio|> <|end_of_audio|> '
    '<|begin_of_transcription|> <|end_of_transcription|>'.split())
_ORDINARY_ADDED_TOKENS = set(
    '<|code_prefix|> <|code_middle|> <|code_suffix|> <think> '
    '</think> <tool_call> </tool_call> <tool_response> '
    '</tool_response> <arg_key> </arg_key> <arg_value> '
    '</arg_value> /nothink <|begin_of_box|> <|end_of_box|> '
    '<|image|> <|video|>'.split())


# Short original token fragments: B8's JSON suffix and E1/H2's final six
# output-ID groups. Full tag strings below are control IDs; split pieces
# such as ['tool', '_call', '>'] are ordinary IDs. These HTTP groups are
# used as decode batches for regression; they are not engine batch traces.
_JSON_PIECES = ['{"', 'open', '":"', '<', 'tool', '_call', '>', '","', 'close', '":"', '</', 'think', '>', '"}']
_ARGUMENT_PIECE_BATCHES = [
    ['Ġthat', '.', '</think>', '<tool_call>'],
    ['save', '_note', '<arg_key>', 'text'],
    ['</arg_key>', '<arg_value>', 'a', 'Ġ</'],
    ['tool', '_call', '>', 'Ġb'],
    ['Ġ</', 'arg', '_value', '>'],
    ['Ġc', '</arg_value>', '</tool_call>'],
]


@pytest.fixture(scope='module')
def vocabulary():
    return _make_vocabulary(_SPECIAL_TOKENS)


def _make_vocabulary(special_tokens, *, added_tokens=None):
    if added_tokens is None:
        added_tokens = _SPECIAL_TOKENS | _ORDINARY_ADDED_TOKENS
    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for piece in _JSON_PIECES + [p for batch in _ARGUMENT_PIECE_BATCHES for p in batch]:
        if piece not in _SPECIAL_TOKENS | _ORDINARY_ADDED_TOKENS:
            vocab.setdefault(piece, len(vocab))
    backend = Tokenizer(models.BPE(vocab, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    ordinary = Tokenizer.from_str(backend.to_str())
    for tag in sorted(added_tokens):
        token = AddedToken(tag, normalized=False, special=tag in special_tokens)
        if token.special:
            backend.add_special_tokens([token])
        else:
            backend.add_tokens([token])
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, clean_up_tokenization_spaces=False)
    # Transformers 4 and 5 both expose the backend's registered special IDs.
    hf.add_special_tokens({'additional_special_tokens': [
        AddedToken(t, normalized=False, special=True) for t in sorted(special_tokens)]})
    lm = object.__new__(HuggingFaceTokenizer)
    lm.model = hf
    return SimpleNamespace(hf=hf, lm=lm, enc=lambda s: ordinary.encode(s).ids,
                           ctl=lambda s: [hf.get_vocab()[s]])



def _configured_profile(tokenizer, reasoning_name, tool_name):
    rcls = ReasoningParserManager.get(reasoning_name) if reasoning_name else None
    tcls = ToolParserManager.get(tool_name) if tool_name else None
    terminals = _resolve_token_id_terminals(tokenizer, rcls, tcls)
    return build_token_profile(tokenizer, terminals) if terminals else None


def make_parser(vocabulary, *, thinking=True, reasoning=True, tools=True, choice='auto', request_tools=TOOLS):
    class Parser(BaseResponseParser):
        pass

    Parser.set_parsers('default' if reasoning else None, 'glm47' if tools else None, vocabulary.hf)
    request = ChatCompletionRequest(model='test', messages=[], tools=request_tools if tools else None,
                                    tool_choice=choice, chat_template_kwargs={'enable_thinking': thinking})
    return Parser(request)


def native_chunks(v, ids, batch, *, skip=True, spaces=True):
    batches = [ids[begin:begin + batch] for begin in range(0, len(ids), batch)]
    return native_id_batches(v, batches, skip=skip, spaces=spaces)


def native_id_batches(v, batches, *, skip=True, spaces=True):
    prompt = v.ctl('<think>')
    all_ids = list(prompt)
    state = DetokenizeState(len(prompt))
    rows = []
    for part in batches:
        all_ids.extend(part)
        text, state = v.lm.detokenize_incrementally(all_ids, state, skip_special_tokens=skip,
                                                   spaces_between_special_tokens=spaces)
        rows.append((text, part, False))
    rows.append(('', [], True))
    return rows


def parsed(parser, rows, complete):
    if complete:
        content, calls, reason = parser.parse_complete(
            ''.join(r[0] for r in rows), [i for r in rows for i in r[1]], output_chunks=rows)
    else:
        deltas = [d for text, ids, final in rows
                  for d, _ in parser.stream_chunk(text, ids, final=final, engine_final=final)]
        content = ''.join(d.content or '' for d in deltas)
        reason = ''.join(d.reasoning_content or '' for d in deltas)
        calls = parser.tool_parser.build_tool_calls([t for d in deltas for t in d.tool_calls or []]) \
            if parser.tool_parser else []
    return (content or '', reason or '',
            [(c.function.name, json.loads(c.function.arguments)) for c in calls or []])


@pytest.mark.parametrize('batch', [1, 4, 16, 10000])
@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('literal_first', [False, True])
def test_real_and_literal_reasoning_close(vocabulary, batch, complete, literal_first):
    v = vocabulary
    reason, content = ('🧑 A </think> B', 'answer') if literal_first else ('🧑', 'A </think> B 🧑')
    ids = v.enc(reason) + v.ctl('</think>') + v.enc(content)
    parser = make_parser(v)
    assert parsed(parser, native_chunks(v, ids, batch), complete) == (content, reason, [])
    assert parser.reasoning_tokens == len(v.enc(reason))


@pytest.mark.parametrize('tag', CONTROL_TAGS)
@pytest.mark.parametrize('batch', [1, 4, 10000])
@pytest.mark.parametrize('complete', [False, True])
def test_literal_inside_tool_argument(vocabulary, tag, batch, complete):
    v = vocabulary
    value = 'A 🧑 ' + tag + ' B'
    ids = (v.ctl('<tool_call>') + v.enc('get_weather') + v.ctl('<arg_key>') + v.enc('text')
           + v.ctl('</arg_key>') + v.ctl('<arg_value>') + v.enc(value)
           + v.ctl('</arg_value>') + v.ctl('</tool_call>'))
    parser = make_parser(v, thinking=False)
    assert parsed(parser, native_chunks(v, ids, batch), complete) == ('', '', [('get_weather', {'text': value})])


@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('batch', [1, 10000])
def test_literal_tool_call_is_content_even_with_declared_tool(vocabulary, complete, batch):
    text = '<tool_call>get_weather</tool_call>'
    parser = make_parser(vocabulary, thinking=False)
    assert parsed(parser, native_chunks(vocabulary, vocabulary.enc(text), batch), complete) == (text, '', [])


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('spaces', [False, True])
def test_all_added_tokens_follow_native_skip_policy(vocabulary, skip, spaces):
    v = vocabulary
    ids = [v.hf.get_vocab()[t] for t in sorted(_SPECIAL_TOKENS | _ORDINARY_ADDED_TOKENS)]
    text = v.hf.decode(ids, skip_special_tokens=skip, clean_up_tokenization_spaces=False)
    scanner = TokenIDScanner(v.hf, _configured_profile(v.hf, 'default', 'glm47'))
    scanner.skip_special_tokens = skip
    located = scanner.scan(text, ids)
    assert [s.tag for s in located.spans] == [t for t in sorted(CONTROL_TAGS)]
    assert located.text == text
    rows = native_chunks(v, ids, 1, skip=skip, spaces=spaces)
    scanner = TokenIDScanner(v.hf, scanner.profile)
    scanner.skip_special_tokens = skip
    assert [s.tag for text, part, final in rows for s in scanner.scan(text, part, engine_final=final).spans] \
        == sorted(CONTROL_TAGS)


def test_reasoning_only_and_tools_none(vocabulary):
    v = vocabulary
    ids = v.enc('A </think> B') + v.ctl('</think>') + v.enc('answer')
    for kwargs in ({'tools': False}, {'choice': 'none'}):
        parser = make_parser(v, **kwargs)
        assert parser.requires_output_chunks
        assert parsed(parser, native_chunks(v, ids, 1), True) == ('answer', 'A </think> B', [])


def test_wait_for_visible_text_then_discard_invisible_final_ids(vocabulary):
    v = vocabulary
    scanner = TokenIDScanner(v.hf, _configured_profile(v.hf, 'default', None))
    emoji = v.enc('🧑')
    ids = v.enc('X') + v.ctl('</think>') + emoji[:1]
    assert scanner.scan('', ids).spans == ()
    located = scanner.scan('X</think>🧑', emoji[1:])
    assert located.spans == (ControlSpan(1, 9, '</think>'),)
    scanner.scan('', ids)
    assert scanner.scan('', [], engine_final=True).spans == ()
    assert scanner._waiting == []
    # Included stop is decoded independently, with special tokens retained.
    located = scanner.scan('</think>', v.ctl('</think>'), engine_final=True)
    assert located.spans == (ControlSpan(0, 8, '</think>'),)


def test_scanner_fast_path_cache_and_block_repair(vocabulary, monkeypatch):
    v = vocabulary
    scanner = TokenIDScanner(v.hf, _configured_profile(v.hf, 'default', None))
    calls = []
    decode = scanner._decode

    def counted(ids, skip):
        calls.append(tuple(ids))
        return decode(ids, skip)

    monkeypatch.setattr(scanner, '_decode', counted)
    scanner.scan('ordinary', v.enc('ordinary'))
    assert calls == []
    ids = v.enc('🧑') + v.ctl('</think>') + v.enc('A </think> B')
    assert scanner.scan('🧑</think>A </think> B', ids).spans[0].start == 1
    assert all(len(c) == 1 for c in calls)  # Reused suffix resolves prefix emoji.
    calls.clear()
    scanner.scan('🧑</think>A </think> B', ids)
    assert calls == []
    ids = v.ctl('</think>') + v.enc('🧑 </think>')
    assert scanner.scan('</think>🧑 </think>', ids).spans[0].start == 0
    assert sum(len(c) > 1 for c in calls) == 1
    with pytest.raises(ValueError, match='align'):
        scanner.scan('wrong', v.ctl('</think>'))
    with pytest.raises(ValueError, match='requires IDs'):
        scanner.scan('</think>', [])


def test_explicit_buffer_keeps_relative_boundaries():
    text = 'A </think> B</think>answer'
    buffer = ParseBuffer(token_id_terminals=('</think>',))
    buffer.append(LocatedText(text, (ControlSpan(12, 20, '</think>'),)))
    view = buffer.boundaries
    assert buffer.find_structure('</think>') == 12
    assert buffer.take(2) == 'A '
    assert buffer.boundaries.index is view.index
    assert buffer.find_structure('</think>') == 10
    with pytest.raises(ValueError, match='part of a control'):
        buffer.take(11)
    buffer.append(LocatedText(' </think>', ()))
    assert buffer.find_structure('</think>') == 10
    assert buffer.take(18) == '</think> B</think>'
    assert buffer.find_structure('</think>') == -1
    legacy = ParseBuffer()
    legacy.append(LocatedText('</think>'))
    assert legacy.find_structure('</think>') == 0
    with pytest.raises(ValueError, match='mix'):
        buffer.append(LocatedText('no boundaries'))
    with pytest.raises(ValueError, match='Invalid'):
        BoundaryIndex('x', (ControlSpan(0, 8, '</think>'),))
    index = BoundaryIndex('</think>', (ControlSpan(0, 8, '</think>'),))
    assert BoundaryView(index, 0, 7, frozenset({'</think>'})).find('</think>') == -1


@pytest.mark.parametrize('consume_all', [False, True])
def test_strict_buffer_alternates_literal_and_control_tags(consume_all):
    buffer = ParseBuffer(token_id_terminals=('</think>',))
    literal = 'A </think> B'
    buffer.append(LocatedText(literal, ()))
    consumed = len(literal) if consume_all else 2
    assert buffer.take(consumed) == literal[:consumed]
    buffer.append(LocatedText(' </think>', ()))
    text = literal[consumed:] + ' </think>'
    assert buffer.text == text
    assert buffer.find_structure('</think>') == -1

    buffer.append(LocatedText('</think>tail', (ControlSpan(0, 8, '</think>'),)))
    assert buffer.find_structure('</think>') == len(text)
    assert buffer.take(len(text) + 8) == text + '</think>'
    buffer.append(LocatedText('</think>', ()))
    assert buffer.find_structure('</think>') == -1
    assert buffer.take(len(buffer)) == 'tail</think>'

    buffer.append(LocatedText('</think>', (ControlSpan(0, 8, '</think>'),)))
    assert buffer.find_structure('</think>') == 0
    with pytest.raises(ValueError, match='part of a control'):
        buffer.take(1)
    assert buffer.take(8) == '</think>'


@pytest.mark.parametrize('tools', [False, True], ids=['reasoning-only', 'reasoning-and-tools'])
@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('complete', [False, True], ids=['stream', 'nonstream'])
@pytest.mark.parametrize('batch', [1, 10000])
def test_legacy_glm_special_token_profile(tools, skip, complete, batch):
    # GLM-4.5/4.6/4.7 mark these four tokens special; Flash/GLM-5 do not.
    legacy_tags = ['<|code_prefix|>', '<|code_middle|>', '<|code_suffix|>', '/nothink']
    v = _make_vocabulary(_SPECIAL_TOKENS | set(legacy_tags))
    markers = [v.hf.get_vocab()[tag] for tag in legacy_tags]
    visible_markers = '' if skip else ''.join(legacy_tags)
    reason, content, value = 'R </think> ', 'A <tool_call> B', 'C </arg_value> '
    ids = v.enc(reason) + markers + v.ctl('</think>') + v.enc(content)
    calls = []
    if tools:
        ids += (v.ctl('<tool_call>') + v.enc('get_weather') + v.ctl('<arg_key>') + v.enc('text')
                + v.ctl('</arg_key>') + v.ctl('<arg_value>') + v.enc(value) + markers
                + v.ctl('</arg_value>') + v.ctl('</tool_call>'))
        calls = [('get_weather', {'text': value + visible_markers})]
    parser = make_parser(v, tools=tools)
    assert parser.requires_output_chunks
    parser.bind_generation_config(GenerationConfig(skip_special_tokens=skip))
    assert parsed(parser, native_chunks(v, ids, batch, skip=skip), complete) == (
        content, reason + visible_markers, calls)
    assert parser.reasoning_tokens == len(v.enc(reason)) + len(markers)


@pytest.mark.parametrize('tag', CONTROL_TAGS)
def test_special_control_tokens_are_rejected(tag):
    v = _make_vocabulary(_SPECIAL_TOKENS | {tag})
    with pytest.raises(ValueError, match='control-token configuration'):
        _configured_profile(v.hf, 'default', 'glm47')


@pytest.mark.parametrize('change', [
    'extra_ordinary', 'extra_special', 'missing_unused', 'reclassified', 'only_controls'])
@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('batch', [1, 10000])
def test_glm_profile_follows_tokenizer_metadata(change, skip, complete, batch):
    added = _SPECIAL_TOKENS | _ORDINARY_ADDED_TOKENS
    special = _SPECIAL_TOKENS.copy()
    markers = ['/nothink', '<|endoftext|>']
    if change in ('extra_ordinary', 'extra_special'):
        markers = ['!extra</think>!']
        added.update(markers)
        if change == 'extra_special':
            special.update(markers)
    elif change == 'missing_unused':
        removed = {'[gMASK]', '<|begin_of_image|>'}
        added -= removed
        special -= removed
    elif change == 'reclassified':
        special.add('/nothink')
        special.remove('<|endoftext|>')
    else:
        added, special, markers = set(CONTROL_TAGS), set(), []
    v = _make_vocabulary(special, added_tokens=added)
    if change == 'extra_ordinary':
        # Input matching flags on unrelated tokens do not change output decoding.
        v.hf.backend_tokenizer.add_tokens([
            AddedToken(markers[0], normalized=True, single_word=True, lstrip=True, rstrip=True)])
    marker_ids = [v.hf.convert_tokens_to_ids(tag) for tag in markers]
    visible = ''.join(tag for tag in markers if not skip or tag not in special)
    reason, content, value = 'R 🧑 </think> ', 'A <tool_call> B', 'C </arg_value> 🧑 '
    ids = (v.enc(reason) + marker_ids + v.ctl('</think>') + v.enc(content)
           + v.ctl('<tool_call>') + v.enc('get_weather') + v.ctl('<arg_key>') + v.enc('text')
           + v.ctl('</arg_key>') + v.ctl('<arg_value>') + v.enc(value) + marker_ids
           + v.ctl('</arg_value>') + v.ctl('</tool_call>'))
    parser = make_parser(v)
    parser.bind_generation_config(GenerationConfig(skip_special_tokens=skip))
    assert parsed(parser, native_chunks(v, ids, batch, skip=skip), complete) == (
        content, reason + visible, [('get_weather', {'text': value + visible})])
    assert parser.reasoning_tokens == len(v.enc(reason)) + len(marker_ids)


@pytest.mark.parametrize('mismatch', ['missing', 'extra'])
def test_inconsistent_special_ids_are_rejected(mismatch, monkeypatch):
    v = _make_vocabulary(_SPECIAL_TOKENS)
    special_ids = v.hf.all_special_ids
    if mismatch == 'missing':
        special_ids = special_ids[1:]
    else:
        special_ids = special_ids + v.ctl('</think>')
    monkeypatch.setattr(type(v.hf), 'all_special_ids', property(lambda self: special_ids))
    with pytest.raises(ValueError, match='special-token metadata'):
        _configured_profile(v.hf, 'default', 'glm47')


def test_profile_is_checked_once_and_cannot_leak_to_other_parsers(vocabulary):
    v = vocabulary
    parser = make_parser(v)
    assert parser.requires_output_chunks
    assert _configured_profile(None, 'default', 'glm47') is None
    assert _configured_profile(v.hf, 'default', 'qwen3') is None
    cls = type(parser)
    cls.set_parsers('default', None, None)
    assert not cls(ChatCompletionRequest(model='test', messages=[])).requires_output_chunks
    assert cls.tool_parser_cls is None
    old = v.hf.clean_up_tokenization_spaces
    try:
        v.hf.clean_up_tokenization_spaces = True
        with pytest.raises(ValueError, match='Unsupported decoder'):
            _configured_profile(v.hf, 'default', None)
    finally:
        v.hf.clean_up_tokenization_spaces = old


def test_standalone_complete_and_invalid_complete_chunks(vocabulary):
    v = vocabulary
    text = 'A </think> B</think>answer'
    ids = v.enc('A </think> B') + v.ctl('</think>') + v.enc('answer')
    parser = make_parser(v)
    assert parser.parse_complete(text, ids) == ('answer', None, 'A </think> B')
    with pytest.raises(ValueError, match='requires IDs'):
        make_parser(v).parse_complete(text)
    with pytest.raises(ValueError, match='do not match'):
        make_parser(v).parse_complete(text, ids, output_chunks=[('wrong', v.enc('wrong'), False)])
    with pytest.raises(ValueError, match='do not cover'):
        make_parser(v).parse_complete(text, ids, output_chunks=[])


@pytest.mark.parametrize('tag', CONTROL_TAGS)
@pytest.mark.parametrize('thinking', [False, True])
def test_partial_literal_tag_does_not_delay_plain_text(vocabulary, tag, thinking):
    v = vocabulary
    parser = make_parser(v, thinking=thinking)
    # A literal prefix has no structural ID: emit it in this very step.
    prefix = tag[:-1]
    messages = parser.stream_chunk(prefix, v.enc(prefix))
    assert len(messages) == 1
    message, _ = messages[0]
    assert (message.reasoning_content if thinking else message.content) == prefix


def test_profile_uses_only_tokenizer_metadata(vocabulary, monkeypatch):
    def unexpected_full_vocabulary(*args, **kwargs):
        pytest.fail('GLM setup must not export the full tokenizer or vocabulary')

    monkeypatch.setattr(Tokenizer, 'to_str', unexpected_full_vocabulary)
    monkeypatch.setattr(vocabulary.hf, 'get_vocab', unexpected_full_vocabulary)
    parser = make_parser(vocabulary)
    assert set(parser._scanner.profile.controls.values()) == set(CONTROL_TAGS)


@pytest.mark.parametrize('change', ['missing_decoder', 'other_decoder', 'normalizer'])
def test_unsupported_decoder_metadata_is_rejected(change):
    v = _make_vocabulary(_SPECIAL_TOKENS)
    backend = v.hf.backend_tokenizer
    if change == 'missing_decoder':
        backend.decoder = None
    elif change == 'other_decoder':
        backend.decoder = decoders.WordPiece()
    else:
        backend.normalizer = normalizers.Lowercase()
    with pytest.raises(ValueError, match='Unsupported decoder'):
        _configured_profile(v.hf, 'default', 'glm47')


@pytest.mark.parametrize('flag', ['single_word', 'lstrip', 'rstrip', 'normalized'])
def test_unsupported_added_token_attributes_are_rejected(flag):
    v = _make_vocabulary(_SPECIAL_TOKENS)
    options = {'normalized': False, flag: True}
    v.hf.backend_tokenizer.add_tokens([AddedToken('</arg_value>', **options)])
    with pytest.raises(ValueError, match='Unsupported control-token configuration'):
        _configured_profile(v.hf, 'default', 'glm47')


def test_request_counter_reuses_checked_profile(vocabulary, monkeypatch):
    parser = make_parser(vocabulary)
    controls = {tag: token_id for token_id, tag in parser._scanner.profile.controls.items()}

    def unexpected_vocab_copy():
        pytest.fail('A checked GLM request must not fetch the complete vocabulary')

    monkeypatch.setattr(vocabulary.hf, 'get_vocab', unexpected_vocab_copy)
    another = type(parser)(ChatCompletionRequest(model='test', messages=[], tools=TOOLS))
    another._update_reasoning_tokens(vocabulary.enc('reason') + [controls['</think>']])
    assert another.reasoning_tokens == len(vocabulary.enc('reason'))


@pytest.mark.parametrize('tool_name', [None, 'glm47'])
@pytest.mark.parametrize('tag', CONTROL_TAGS)
def test_missing_controls_do_not_silently_disable_fix(tool_name, tag):
    v = _make_vocabulary(_SPECIAL_TOKENS, added_tokens=(_SPECIAL_TOKENS | _ORDINARY_ADDED_TOKENS) - {tag})
    if tool_name is None and tag not in ('<think>', '</think>'):
        profile = _configured_profile(v.hf, 'default', None)
        assert set(profile.controls.values()) == {'<think>', '</think>'}
    else:
        with pytest.raises(ValueError, match='Missing control tokens'):
            _configured_profile(v.hf, 'default', tool_name)


def test_control_ids_must_be_distinct(vocabulary, monkeypatch):
    added = vocabulary.hf.get_added_vocab()
    added['</think>'] = added['<think>']
    monkeypatch.setattr(vocabulary.hf, 'get_added_vocab', lambda: added)
    with pytest.raises(ValueError, match='distinct IDs'):
        _configured_profile(vocabulary.hf, 'default', 'glm47')


def test_unrelated_reasoning_tokenizer_keeps_legacy_parsing():
    unrelated = SimpleNamespace(get_added_vocab=lambda: {'<think>': 1, '</think>': 2})
    assert _configured_profile(unrelated, 'default', None) is None
    with pytest.raises(ValueError, match='Missing control tokens'):
        _configured_profile(unrelated, 'default', 'glm47')
    with pytest.raises(ValueError, match='requires added-token metadata'):
        _configured_profile(SimpleNamespace(), 'default', 'glm47')


@pytest.mark.parametrize(('reasoning', 'tools', 'expected'), [
    ('default', None, CONTROL_TAGS[:2]),
    (None, 'glm47', CONTROL_TAGS[2:]),
    ('default', 'glm47', CONTROL_TAGS),
    (None, None, ()),
    ('default', 'qwen3', ()),
    ('default', 'qwen3coder', ()),
    ('deepseek-v3', None, ()),
    ('deepseek-v32', None, ()),
    ('deepseek-v4', None, ()),
    ('deepseek-v3', 'glm47', ()),
])
def test_terminal_selection_preserves_other_parser_paths(vocabulary, reasoning, tools, expected):
    class Parser(BaseResponseParser):
        pass

    Parser.set_parsers(reasoning, tools, vocabulary.hf)
    profile = Parser._token_profile
    assert set(profile.controls.values() if profile else ()) == set(expected)
    # Selecting GLM must not mutate the shared reasoning parser or subclasses.
    assert ReasoningParser.token_id_terminals == ()
    for name in ('deepseek-v3', 'deepseek-v32', 'deepseek-v4'):
        assert ReasoningParserManager.get(name).token_id_terminals == ()


@pytest.mark.parametrize('batch', [1, 10000])
@pytest.mark.parametrize('complete', [False, True])
def test_unconfigured_default_keeps_string_matching(batch, complete):
    # Think tags (even together with tool_call tags) do not identify GLM.
    v = _make_vocabulary(set(), added_tokens=set(CONTROL_TAGS[:4]))

    class Parser(BaseResponseParser):
        pass

    Parser.set_parsers('default', None, v.hf)
    parser = Parser(ChatCompletionRequest(model='test', messages=[]))
    assert not parser.requires_output_chunks
    assert parser._scanner is None
    # Preserve the old text-based boundary, including its literal behavior.
    ids = v.enc('reason</think>answer')
    assert parsed(parser, native_chunks(v, ids, batch), complete) == ('answer', 'reason', [])


@pytest.mark.parametrize('batch', [1, 10000])
@pytest.mark.parametrize('complete', [False, True])
def test_partial_reasoning_declaration_preserves_text_open_tag(vocabulary, monkeypatch, batch, complete):
    class PartialReasoning(ReasoningParser):
        token_id_terminals = ('</think>',)

    monkeypatch.setitem(ReasoningParserManager.module_dict, 'test-partial', PartialReasoning)

    class Parser(BaseResponseParser):
        pass

    v = vocabulary
    Parser.set_parsers('test-partial', None, v.hf)
    parser = Parser(ChatCompletionRequest(model='test', messages=[]))
    assert set(parser._scanner.profile.controls.values()) == {'</think>'}
    assert parser._reasoning_start_token_id == v.ctl('<think>')[0]
    ids = (v.enc('R </think> literal') + v.ctl('</think>') + v.enc('body<think>more')
           + v.ctl('</think>') + v.enc('end'))
    assert parsed(parser, native_chunks(v, ids, batch), complete) == (
        'bodyend', 'R </think> literalmore', [])


@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('tool_name', [None, 'glm47'])
def test_custom_protocol_uses_generic_scanner(monkeypatch, complete, tool_name):
    validated = []

    class CustomReasoning(ReasoningParser):
        token_id_terminals = ('[analysis]', '[/analysis]')

        @classmethod
        def validate_tokenizer(cls, tokenizer):
            validated.append(cls)
            super().validate_tokenizer(tokenizer)

        @classmethod
        def get_reasoning_open_tag(cls):
            return '[analysis]'

        @classmethod
        def get_reasoning_close_tag(cls):
            return '[/analysis]'

    monkeypatch.setitem(ReasoningParserManager.module_dict, 'test-custom', CustomReasoning)
    tags = set(CustomReasoning.token_id_terminals)
    if tool_name:
        tags.update(Glm47ToolParser.token_id_terminals)
    v = _make_vocabulary(set(), added_tokens=tags)

    class Parser(BaseResponseParser):
        pass

    Parser.set_parsers('test-custom', tool_name, v.hf)
    assert validated == [CustomReasoning]
    parser = Parser(ChatCompletionRequest(model='test', messages=[], tools=TOOLS if tool_name else None))
    assert set(parser._scanner.profile.controls.values()) == tags
    ids = v.enc('literal [/analysis]') + v.ctl('[/analysis]') + v.enc('answer')
    calls = []
    if tool_name:
        value = 'A </arg_value> B'
        ids += _tool_ids(v, 'get_weather', [('text', value)])
        calls = [('get_weather', {'text': value})]
    # Custom reasoning tags work alone and with an explicitly configured tool parser.
    rows = [(v.hf.decode(ids, clean_up_tokenization_spaces=False), ids, False), ('', [], True)]
    assert parsed(parser, rows, complete) == ('answer', 'literal [/analysis]', calls)


@pytest.mark.parametrize('terminal', ['<tool_call>', '</arg_value>'])
@pytest.mark.parametrize('batch', [1, 4, 10000])
@pytest.mark.parametrize('complete', [False, True])
def test_partial_tool_declaration_keeps_split_text_tags(vocabulary, monkeypatch, terminal, batch, complete):
    class PartialTool(Glm47ToolParser):
        token_id_terminals = (terminal,)

    monkeypatch.setitem(ToolParserManager.module_dict, 'test-partial', PartialTool)

    class Parser(BaseResponseParser):
        pass

    v = vocabulary
    Parser.set_parsers('default', 'test-partial', v.hf)
    parser = Parser(ChatCompletionRequest(model='test', messages=[], tools=TOOLS))
    assert set(parser._scanner.profile.controls.values()) == {terminal}
    assert parser._reasoning_end_token_id == v.ctl('</think>')[0]
    value = 'A ' + terminal + ' B'
    ids = v.enc('reason') + v.ctl('</think>')
    for index in range(2):
        if index:
            ids += v.enc('\n')
        parts = [
            '<tool_call>', 'get_weather', '<arg_key>', 'text', '</arg_key>',
            '<arg_value>', value, '</arg_value>', '</tool_call>',
        ]
        for part in parts:
            ids += v.ctl(part) if part == terminal else v.enc(part)
    assert parsed(parser, native_chunks(v, ids, batch), complete) == (
        '', 'reason', [('get_weather', {'text': value})] * 2)
    assert parser.reasoning_tokens == len(v.enc('reason'))


@pytest.mark.parametrize('complete', [False, True])
def test_reasoning_scanner_preserves_legacy_tool_signature(vocabulary, monkeypatch, complete):
    class ExplicitReasoning(ReasoningParser):
        token_id_terminals = CONTROL_TAGS[:2]

    class LegacyTool(Glm47ToolParser):
        token_id_terminals = ()

        def feed_tool_block(self, text, deltas, *, final):
            return super().feed_tool_block(text, deltas, final=final)

    monkeypatch.setitem(ReasoningParserManager.module_dict, 'test-explicit', ExplicitReasoning)
    monkeypatch.setitem(ToolParserManager.module_dict, 'test-legacy', LegacyTool)

    class Parser(BaseResponseParser):
        pass

    v = vocabulary
    Parser.set_parsers('test-explicit', 'test-legacy', v.hf)
    parser = Parser(ChatCompletionRequest(model='test', messages=[], tools=TOOLS))
    ids = v.enc('reason') + v.ctl('</think>') + _tool_ids(v, 'get_weather', [('text', 'value')])
    assert parsed(parser, native_chunks(v, ids, 1), complete) == (
        '', 'reason', [('get_weather', {'text': 'value'})])


# Minimized output-side regressions from the supplied GLM-5.3 recordings.
# Short text examples cover the reported behavior; they do not replay the
# original generations. enc() creates ordinary IDs; ctl() inserts control IDs.
# The two tests at the end retain original token pieces and selected boundaries.
_RECORDED_CALL_TEXT = (
    '<tool_call>get_weather<arg_key>city</arg_key>'
    '<arg_value>北京</arg_value></tool_call>'
)
_RECORDED_TOOLS = [
    {'type': 'function', 'function': {'name': 'save_note', 'parameters': {
        'type': 'object', 'properties': {'text': {'type': 'string'}}}}},
    {'type': 'function', 'function': {'name': 'get_weather', 'parameters': {
        'type': 'object', 'properties': {'city': {'type': 'string'}, 'days': {'type': 'integer'}}}}},
    {'type': 'function', 'function': {'name': 'add_tags', 'parameters': {
        'type': 'object', 'properties': {'tags': {'type': 'array'}, 'meta': {'type': 'object'}}}}},
]


@pytest.mark.parametrize(
    ('reason', 'content', 'declared_tools', 'choice', 'skip'),
    [
        pytest.param('S = "</think>"\n1. </think>\n2. </think>', '</think>\n好的',
                     False, 'auto', True, id='A1_think_end_in_reasoning'),
        pytest.param('S = "</think>"', '1: </think>\n2: </think>\n3: </think>',
                     False, 'auto', True, id='A2_think_end_in_content'),
        pytest.param('S = <think>\nS = <think>', '<think>\n</think>好的',
                     False, 'auto', True, id='A3_think_in_reasoning'),
        pytest.param('S = "<think>"', '1: <think>\n2: <think>',
                     False, 'auto', True, id='A4_think_in_content'),
        pytest.param('S = <think>推理</think>', '1: <think>推理</think>\n2: <think>推理</think>',
                     False, 'auto', True, id='A5_think_pair_both'),
        pytest.param('1. <tool_call>\n2. <tool_call>', '第一次：<tool_call>\n好的',
                     False, 'auto', True, id='B1_tool_call_in_reasoning'),
        pytest.param('S = <tool_call>', '1: <tool_call>\n2: <tool_call>',
                     False, 'auto', True, id='B2_tool_call_in_content'),
        pytest.param('S = </tool_call>', '1: </tool_call>\n2: </tool_call>',
                     False, 'auto', True, id='B3_tool_call_end_both'),
        pytest.param('S = <tool_call>hello</tool_call>', '1: <tool_call>hello</tool_call>',
                     False, 'auto', True, id='B4_tool_call_pair_both'),
        pytest.param(_RECORDED_CALL_TEXT, '1: ' + _RECORDED_CALL_TEXT,
                     False, 'auto', True, id='B5_fake_full_call_both'),
        pytest.param('<arg_key>city</arg_key><arg_value>北京</arg_value>',
                     '1: <arg_key>city</arg_key><arg_value>北京</arg_value>',
                     False, 'auto', True, id='B6_arg_tags_both'),
        pytest.param('<tool_response>ok</tool_response>', '1: <tool_response>ok</tool_response>',
                     False, 'auto', True, id='B7_tool_response_both'),
        pytest.param('A is <tool_call>, B is </think>', '{"open":"<tool_call>","close":"</think>"}',
                     False, 'auto', True, id='B8_json_context'),
        pytest.param('S = <tool_call> and </think>', '```python\ntag = "<tool_call>"\nprint("</think>")\n```',
                     False, 'auto', True, id='B9_code_block_context'),
        pytest.param(_RECORDED_CALL_TEXT, '1: ' + _RECORDED_CALL_TEXT,
                     True, 'none', True, id='C1_fake_call_tools_none'),
        pytest.param('x</think>y<tool_call>z', '1: x</think>y<tool_call>z',
                     True, 'none', True, id='C2_mixed_tools_none'),
        pytest.param(_RECORDED_CALL_TEXT, '1: ' + _RECORDED_CALL_TEXT,
                     True, 'auto', True, id='D1_fake_call_tools_auto_no_call'),
        pytest.param('S = "</think>"', '1: </think>\n2: </think>',
                     True, 'auto', True, id='D2_think_end_tools_auto_no_call'),
        pytest.param(_RECORDED_CALL_TEXT,
                     _RECORDED_CALL_TEXT + '\n<tool_call>get_weather<arg_key>city</arg_key>'
                     '<arg_value>杭州</arg_value></tool_call>',
                     True, 'auto', True, id='D3_text_mimic_call'),
        pytest.param('S = "<|endoftext|>"', '1: <|endoftext|>\n2: <|endoftext|>',
                     False, 'auto', True, id='F1_endoftext'),
        pytest.param('<|user|>和<|observation|>', '1: <|user|>和<|observation|>',
                     False, 'auto', True, id='F2_user_observation'),
        pytest.param('[gMASK]<sop><|system|><|assistant|>', '1: [gMASK]<sop><|system|><|assistant|>',
                     False, 'auto', True, id='F3_role_tokens'),
        pytest.param('<|begin_of_image|>x<|end_of_video|>', '1: <|begin_of_image|>x<|end_of_video|>',
                     False, 'auto', True, id='F4_media_tokens'),
        pytest.param('/nothink <|begin_of_box|>42<|end_of_box|> <|code_prefix|>',
                     '1: /nothink <|begin_of_box|>42<|end_of_box|> <|code_prefix|>',
                     False, 'auto', True, id='G1_misc_special_false'),
        pytest.param('S = </think>', '1: </think>\n2: </think>',
                     False, 'auto', False, id='H1_no_skip_think_end'),
        pytest.param('1+1 = 2.', '2', False, 'auto', True, id='J1_plain_chat'),
    ],
)
@pytest.mark.parametrize('batch', [1, 4, 10000], ids=['one-id', 'four-ids', 'all-ids'])
@pytest.mark.parametrize('complete', [False, True], ids=['stream', 'nonstream'])
def test_reported_literal_text(vocabulary, reason, content, declared_tools, choice, skip, batch, complete):
    v = vocabulary
    parser = make_parser(v, request_tools=_RECORDED_TOOLS if declared_tools else None, choice=choice)
    parser.bind_generation_config(GenerationConfig(skip_special_tokens=skip))
    ids = v.enc(reason) + v.ctl('</think>') + v.enc(content)
    assert parsed(parser, native_chunks(v, ids, batch, skip=skip), complete) == (content, reason, [])
    assert parser.reasoning_tokens == len(v.enc(reason))


def _tool_ids(v, name, arguments):
    ids = v.ctl('<tool_call>') + v.enc(name)
    for key, value in arguments:
        ids += (v.ctl('<arg_key>') + v.enc(key) + v.ctl('</arg_key>')
                + v.ctl('<arg_value>') + v.enc(value) + v.ctl('</arg_value>'))
    return ids + v.ctl('</tool_call>')


@pytest.mark.parametrize(
    ('value', 'skip'),
    [
        pytest.param('a </tool_call> b </arg_value> c', True, id='E1_args_close_tags'),
        pytest.param('k=<arg_key>x</arg_key> v=<arg_value>y', True, id='E2_args_key_tags'),
        pytest.param(_RECORDED_CALL_TEXT, True, id='E3_args_full_fake_call'),
        pytest.param('<think>草稿</think>结论', True, id='E4_args_think_tags'),
        pytest.param('end=<|endoftext|> user=<|user|>', True, id='E9_args_special_true_literals'),
        pytest.param('a </tool_call> b </arg_value> c', False, id='H2_no_skip_args'),
    ],
)
@pytest.mark.parametrize('batch', [1, 4, 10000], ids=['one-id', 'four-ids', 'all-ids'])
@pytest.mark.parametrize('complete', [False, True], ids=['stream', 'nonstream'])
def test_reported_literal_tool_arguments(vocabulary, value, skip, batch, complete):
    v = vocabulary
    reason = 'S = ' + value
    ids = v.enc(reason) + v.ctl('</think>') + _tool_ids(v, 'save_note', [('text', value)])
    parser = make_parser(v, request_tools=_RECORDED_TOOLS)
    parser.bind_generation_config(GenerationConfig(skip_special_tokens=skip))
    assert parsed(parser, native_chunks(v, ids, batch, skip=skip), complete) == (
        '', reason, [('save_note', {'text': value})])
    assert parser.reasoning_tokens == len(v.enc(reason))


@pytest.mark.parametrize(
    ('reason', 'content', 'payloads', 'expected'),
    [
        pytest.param(
            _RECORDED_CALL_TEXT, '',
            [('get_weather', [('city', '杭州'), ('days', '2')])],
            [('get_weather', {'city': '杭州', 'days': 2})],
            id='E5_fake_call_in_reasoning_then_real'),
        pytest.param(
            'The labels are <tool_call> and </think>.', '标签是 <tool_call> 和 </think>',
            [('get_weather', [('city', '杭州'), ('days', '2')])],
            [('get_weather', {'city': '杭州', 'days': 2})],
            id='E6_literal_in_content_before_call'),
        pytest.param(
            'Save two notes.', '',
            [('save_note', [('text', 'one </tool_call> two')]),
             ('save_note', [('text', 'three <tool_call> four </arg_value>')])],
            [('save_note', {'text': 'one </tool_call> two'}),
             ('save_note', {'text': 'three <tool_call> four </arg_value>'})],
            id='E7_parallel_calls_with_literals'),
        pytest.param(
            'Add tags and metadata.', '',
            [('add_tags', [('meta', '{"end": "</arg_value>"}'), ('tags', '["<tool_call>", "</think>"]')])],
            [('add_tags', {'meta': {'end': '</arg_value>'}, 'tags': ['<tool_call>', '</think>']})],
            id='E8_typed_args_with_literals'),
        pytest.param(
            'Check the weather.', '',
            [('get_weather', [('city', '上海'), ('days', '3')])],
            [('get_weather', {'city': '上海', 'days': 3})],
            id='J2_single_call'),
        pytest.param(
            'Check both cities.', '我来同时查询上海和北京未来 2 天的天气。',
            [('get_weather', [('city', '上海'), ('days', '2')]),
             ('get_weather', [('city', '北京'), ('days', '2')])],
            [('get_weather', {'city': '上海', 'days': 2}), ('get_weather', {'city': '北京', 'days': 2})],
            id='J3_parallel_calls'),
    ],
)
@pytest.mark.parametrize('batch', [1, 4, 10000], ids=['one-id', 'four-ids', 'all-ids'])
@pytest.mark.parametrize('complete', [False, True], ids=['stream', 'nonstream'])
def test_reported_tool_calls(vocabulary, reason, content, payloads, expected, batch, complete):
    v = vocabulary
    ids = v.enc(reason) + v.ctl('</think>') + v.enc(content)
    for name, arguments in payloads:
        ids += _tool_ids(v, name, arguments)
    parser = make_parser(v, request_tools=_RECORDED_TOOLS)
    assert parsed(parser, native_chunks(v, ids, batch), complete) == (content, reason, expected)
    assert parser.reasoning_tokens == len(v.enc(reason))


@pytest.mark.parametrize('batch', [1, 4, 10000])
@pytest.mark.parametrize('complete', [False, True])
def test_recorded_json_token_pieces(vocabulary, batch, complete):
    # B8: preserve the original JSON suffix's token pieces, including merged
    # punctuation and multi-character tokens. These are base-vocabulary IDs.
    v = vocabulary
    ids = v.enc('R') + v.ctl('</think>') + v.hf.convert_tokens_to_ids(_JSON_PIECES)
    parser = make_parser(v, request_tools=_RECORDED_TOOLS)
    assert parsed(parser, native_chunks(v, ids, batch), complete) == (
        '{"open":"<tool_call>","close":"</think>"}', 'R', [])


@pytest.mark.parametrize('skip', [True, False], ids=['E1', 'H2'])
@pytest.mark.parametrize('chunking', ['recorded', 'one-id', 'all-ids'])
@pytest.mark.parametrize('complete', [False, True], ids=['stream', 'nonstream'])
def test_recorded_argument_boundaries(vocabulary, skip, chunking, complete):
    # E1/H2: retain IDs [107:130] by their pieces, with the original six HTTP
    # groups. Removing leading reasoning leaves ' that.' then the entire call.
    # In particular, the literal closing tags remain split across batches.
    v = vocabulary
    batches = [v.hf.convert_tokens_to_ids(pieces) for pieces in _ARGUMENT_PIECE_BATCHES]
    ids = [token_id for batch in batches for token_id in batch]
    if chunking == 'one-id':
        batches = [[token_id] for token_id in ids]
    elif chunking == 'all-ids':
        batches = [ids]
    parser = make_parser(v, request_tools=_RECORDED_TOOLS)
    parser.bind_generation_config(GenerationConfig(skip_special_tokens=skip))
    assert parsed(parser, native_id_batches(v, batches, skip=skip), complete) == (
        '', ' that.', [('save_note', {'text': 'a </tool_call> b </arg_value> c'})])
    assert parser.reasoning_tokens == 2
