import pytest
from peft import LoraConfig

from lmdeploy.pytorch.adapter.adapter import get_ranks_and_scalings


@pytest.mark.parametrize(('use_rslora', 'expected'), [(False, 1.0), (True, 4.0)])
def test_get_ranks_and_scalings_matches_peft(use_rslora, expected):
    cfg = LoraConfig(r=16, lora_alpha=16, target_modules=['q_proj'], use_rslora=use_rslora)
    ranks, scalings = get_ranks_and_scalings('q_proj', [cfg])
    assert ranks.tolist() == [16]
    assert scalings.tolist() == [expected]
