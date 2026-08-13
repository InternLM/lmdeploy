from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from lmdeploy import GenerationConfig
from lmdeploy.pytorch.messages import SamplingParam
from lmdeploy.pytorch.strategies.ar.sampling import ARSamplingStrategy
from lmdeploy.serve.core.generation_config import build_generation_config
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest
from lmdeploy.serve.openai.responses.protocol import ResponsesRequest


@pytest.mark.parametrize('penalty', [-2.0, 0.0, 2.0])
def test_generation_config_accepts_frequency_penalty_range(penalty):
    config = GenerationConfig(frequency_penalty=penalty)

    assert config.frequency_penalty == penalty
    assert SamplingParam.from_gen_config(config).frequency_penalty == penalty


@pytest.mark.parametrize('penalty', [-2.01, 2.01])
def test_generation_config_rejects_frequency_penalty_out_of_range(penalty):
    with pytest.raises(AssertionError, match='frequency_penalty'):
        GenerationConfig(frequency_penalty=penalty)


def test_ar_sampling_marks_only_generated_tokens_for_frequency_penalty():
    def make_seq(seq_id, prompt_ids, generated_ids, frequency_penalty):
        valid_ids = prompt_ids + generated_ids
        return SimpleNamespace(
            sampling_param=SamplingParam(frequency_penalty=frequency_penalty),
            num_valid_ids=len(valid_ids),
            num_new_tokens=len(generated_ids),
            valid_ids=np.array(valid_ids, dtype=np.int64),
            generated_ids=np.array(generated_ids, dtype=np.int64),
            session=SimpleNamespace(session_id=seq_id),
            seq_id=seq_id,
        )

    strategy = ARSamplingStrategy(pad_token_id=0)
    sampling_inputs = strategy.make_sampling_inputs([
        make_seq(0, [9], [4, 1, 4], 0.5),
        make_seq(1, [2], [], 0.0),
    ])

    assert sampling_inputs.frequency_penalty.tolist() == [0.5, 0.0]
    assert sampling_inputs.all_ids.tolist() == [
        [9, 4, 1, 4],
        [0, 0, 0, 2],
    ]
    assert sampling_inputs.all_ids_mask.tolist() == [
        [False, True, True, True],
        [False, False, False, False],
    ]
    assert sampling_inputs.generated_ids is None
    assert sampling_inputs.generated_ids_cpu is None


def test_sampling_delta_preserves_frequency_state_across_decode_merge_and_reindex():
    from lmdeploy.pytorch.engine.logits_process import SamplingInputsDelta
    from lmdeploy.pytorch.strategies.ar.step_inputs import (
        merge_sampling_delta,
        reindex_sampling_delta,
        step_sampling_delta,
    )

    active = SamplingInputsDelta(
        num_ignore_eos=torch.tensor([4]),
        random_offsets=torch.tensor([3]),
        all_ids=torch.tensor([[4, 1, 4]]),
        all_ids_mask=torch.tensor([[False, True, True]]),
        frequency_penalty=torch.tensor([0.5]),
    )
    active = step_sampling_delta(active, torch.tensor([7]))

    assert active.all_ids.tolist() == [[4, 1, 4, 7]]
    assert active.all_ids_mask.tolist() == [[False, True, True, True]]

    inactive = SamplingInputsDelta(
        num_ignore_eos=torch.tensor([2]),
        random_offsets=torch.tensor([1]),
    )
    merged = merge_sampling_delta(active, inactive, pad_token_id=0)

    assert merged.all_ids.tolist() == [
        [4, 1, 4, 7],
        [0, 0, 0, 0],
    ]
    assert merged.all_ids_mask.tolist() == [
        [False, True, True, True],
        [False, False, False, False],
    ]
    assert merged.frequency_penalty.tolist() == [0.5, 0.0]

    reindexed = reindex_sampling_delta(
        merged,
        SimpleNamespace(indices=torch.tensor([1, 0])),
    )
    assert reindexed.all_ids.tolist() == [
        [0, 0, 0, 0],
        [4, 1, 4, 7],
    ]
    assert reindexed.all_ids_mask.tolist() == [
        [False, False, False, False],
        [False, True, True, True],
    ]
    assert reindexed.frequency_penalty.tolist() == [0.0, 0.5]


def test_dllm_rejects_frequency_penalty():
    from lmdeploy.pytorch.strategies.dllm.sampling import DLLMSamplingStrategy

    seq = SimpleNamespace(
        sampling_param=SamplingParam(frequency_penalty=0.5),
        num_valid_ids=1,
        num_new_tokens=0,
        valid_ids=np.array([4], dtype=np.int64),
        generated_ids=np.array([], dtype=np.int64),
        session=SimpleNamespace(session_id=0),
        seq_id=0,
    )
    strategy = DLLMSamplingStrategy(pad_token_id=0, dllm_block_length=4)

    with pytest.raises(ValueError, match='frequency_penalty'):
        strategy.make_sampling_inputs([seq])


@pytest.mark.parametrize(
    'request_obj',
    [
        ChatCompletionRequest(
            model='m',
            messages=[{'role': 'user', 'content': 'repeat repeat'}],
            frequency_penalty=0.75,
        ),
        CompletionRequest(
            model='m',
            prompt='repeat repeat',
            frequency_penalty=0.75,
        ),
        ResponsesRequest(
            model='m',
            input='repeat repeat',
            frequency_penalty=0.75,
        ),
    ],
)
def test_openai_frequency_penalty_is_forwarded(request_obj):
    config = build_generation_config(request_obj, {}, max_new_tokens=8)

    assert config.frequency_penalty == 0.75


@pytest.mark.parametrize(
    ('request_cls', 'request_args'),
    [
        (
            ChatCompletionRequest,
            dict(model='m', messages=[{'role': 'user', 'content': 'hi'}]),
        ),
        (
            CompletionRequest,
            dict(model='m', prompt='hi'),
        ),
        (
            ResponsesRequest,
            dict(model='m', input='hi'),
        ),
    ],
)
@pytest.mark.parametrize('penalty', [-2.01, 2.01])
def test_openai_frequency_penalty_range_validation(request_cls, request_args, penalty):
    with pytest.raises(ValidationError):
        request_cls(**request_args, frequency_penalty=penalty)
