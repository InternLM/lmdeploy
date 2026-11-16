# Copyright (c) OpenMMLab. All rights reserved.
import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from lmdeploy.tokenizer import DetokenizeState, HuggingFaceTokenizer


@pytest.fixture(scope='module')
def tokenizer():
    backend = BackendTokenizer(models.BPE(unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ['Hello world! 中文🙂😆\n  spaces and byte boundaries.'],
        trainers.BpeTrainer(vocab_size=280, special_tokens=['<unk>', '<eos>'],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    wrapper = HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
    wrapper.model = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', eos_token='<eos>')
    return wrapper


@pytest.mark.parametrize('prompt_repeats', [0, 1, 4096])
@pytest.mark.parametrize('chunk_size', [1, 3, 17])
@pytest.mark.parametrize('skip_special_tokens', [False, True])
def test_incremental_decode_matches_complete_decode(tokenizer, prompt_repeats, chunk_size, skip_special_tokens):
    prompt_ids = tokenizer.model.encode('Hello world! 中文\n' * prompt_repeats, add_special_tokens=False)
    new_ids = tokenizer.model.encode('  🙂😆 中文<eos>\n spaces  ', add_special_tokens=False)
    # Invalid IDs are ignored by the incremental tokenizer.
    invalid_id = len(tokenizer.model) + 10
    new_ids.insert(2, invalid_id)
    state = DetokenizeState(len(prompt_ids))
    all_ids = list(prompt_ids)
    pieces = []
    for offset in range(0, len(new_ids), chunk_size):
        all_ids.extend(new_ids[offset:offset + chunk_size])
        piece, state = tokenizer.detokenize_incrementally(all_ids, state, skip_special_tokens)
        pieces.append(piece)
        # A backend can yield an empty chunk without advancing the request.
        empty, state = tokenizer.detokenize_incrementally(all_ids, state, skip_special_tokens)
        assert empty == ''
        assert state.ids_offset == len(all_ids)
    expected = tokenizer.model.decode([i for i in new_ids if i != invalid_id],
                                      skip_special_tokens=skip_special_tokens)
    assert ''.join(pieces) == expected


def test_interleaved_requests_keep_separate_histories(tokenizer):
    texts = ['Hello 中文🙂 world', '  😆 spaces\n中文']
    ids = [tokenizer.model.encode(text, add_special_tokens=False) for text in texts]
    states = [DetokenizeState(), DetokenizeState()]
    outputs = ['', '']
    for offset in range(1, max(map(len, ids)) + 1):
        for i in range(2):
            piece, states[i] = tokenizer.detokenize_incrementally(ids[i][:offset], states[i])
            outputs[i] += piece
        assert states[0].prev_tokens is not states[1].prev_tokens
    assert outputs == texts
