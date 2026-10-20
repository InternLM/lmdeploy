import sys
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.flash_kda import check_flash_kda, flash_kda_prefill


@pytest.mark.parametrize('arguments', [
    ['checkpoint_state', 'checkpoint_offsets'],
    ['checkpoint_state'],
    ['checkpoint_offsets'],
])
def test_native_flash_kda_capability_check(monkeypatch, arguments):
    module = SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'flash_kda', module)
    schema = SimpleNamespace(arguments=[SimpleNamespace(name=name) for name in arguments])
    monkeypatch.setattr(torch.ops, 'flash_kda',
                        SimpleNamespace(fwd=SimpleNamespace(default=SimpleNamespace(_schema=schema))))
    if len(arguments) == 2:
        assert check_flash_kda() is module
    else:
        with pytest.raises(RuntimeError, match='checkpoint_state and checkpoint_offsets'):
            check_flash_kda()


def make_inputs():
    query = torch.zeros((1, 64, 2, 128), dtype=torch.bfloat16)
    return dict(query=query, key=query.clone(), value=query.clone(), gate=query.clone(),
                beta=torch.zeros((1, 64, 2), dtype=torch.bfloat16),
                a_log=torch.zeros(2), dt_bias=torch.zeros(256),
                initial_state=torch.zeros((1, 2, 128, 128)),
                cu_seqlens=torch.tensor([0, 64], dtype=torch.int64), lower_bound=-5.0)


@pytest.mark.parametrize('capture', [False, True])
def test_native_flash_kda_forwards_optional_checkpoint(monkeypatch, capture):
    recorded = {}

    def forward(*args, **kwargs):
        recorded.update(kwargs)
        recorded['bias_shape'] = args[9].shape
        args[6].zero_()
        kwargs['final_state'].fill_(2)

    monkeypatch.setitem(sys.modules, 'flash_kda', SimpleNamespace(get_workspace_size=lambda *args: 64))
    monkeypatch.setattr(torch.ops, 'flash_kda', SimpleNamespace(fwd=forward))
    inputs = make_inputs()
    checkpoint = (SimpleNamespace(recurrent_state_out=torch.empty_like(inputs['initial_state']),
                                  offsets=torch.tensor([32], dtype=torch.int64)) if capture else None)
    output, final_state = flash_kda_prefill(**inputs, checkpoint=checkpoint)
    assert output.shape == inputs['query'].shape
    assert torch.count_nonzero(output) == 0
    assert torch.all(final_state == 2)
    assert recorded['bias_shape'] == (2, 128)
    assert recorded['checkpoint_state'] is (checkpoint.recurrent_state_out if capture else None)
    assert recorded['checkpoint_offsets'] is (checkpoint.offsets if capture else None)
    assert recorded['cu_seqlens'] is inputs['cu_seqlens']


@pytest.mark.parametrize('field', ['query', 'initial_state'])
def test_native_flash_kda_rejects_incompatible_dtype(monkeypatch, field):
    monkeypatch.setitem(sys.modules, 'flash_kda', SimpleNamespace())
    inputs = make_inputs()
    inputs[field] = inputs[field].to(torch.float16)
    with pytest.raises(ValueError, match='BF16 inputs'):
        flash_kda_prefill(**inputs)


def test_native_flash_kda_requires_gate_bound(monkeypatch):
    monkeypatch.setitem(sys.modules, 'flash_kda', SimpleNamespace())
    inputs = make_inputs()
    inputs['lower_bound'] = None
    with pytest.raises(ValueError, match='gate lower bound'):
        flash_kda_prefill(**inputs)
