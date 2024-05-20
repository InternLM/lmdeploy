import pytest
import torch

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.tools.make_inputs import make_step_context


class TestMakeInputs:

    @pytest.fixture
    def seq_length(self):
        yield torch.tensor([2, 4, 3])

    @pytest.fixture
    def history_length(self):
        yield torch.tensor([10, 12, 6])

    @pytest.fixture
    def input_ids(self, seq_length):
        batch_size = len(seq_length)
        max_seq_len = max(seq_length)
        yield torch.randint(0, 100, (batch_size, max_seq_len))

    @pytest.fixture
    def block_size(self):
        yield 4

    @pytest.fixture
    def num_key_value_heads(self):
        yield 1

    @pytest.fixture
    def head_size(self):
        yield 4

    @pytest.fixture
    def kv_cache_dtype(self):
        yield torch.float16

    @pytest.fixture
    def model_config(self, head_size, kv_cache_dtype, num_key_value_heads):
        hidden_size = head_size * num_key_value_heads
        yield ModelConfig(
            hidden_size=hidden_size,
            num_layers=1,
            num_attention_heads=num_key_value_heads,
            num_key_value_heads=num_key_value_heads,
            bos_token_id=0,
            eos_token_id=0,
            head_dim=head_size,
            dtype=kv_cache_dtype,
            k_head_dim=head_size,
            v_head_dim=head_size,
        )

    @pytest.fixture
    def past_key_values(self, history_length, num_key_value_heads, head_size):
        max_len = max(history_length)
        batch_size = len(history_length)
        k_cache = torch.rand(batch_size, num_key_value_heads, max_len,
                             head_size)
        v_cache = k_cache + 1
        yield [(k_cache, v_cache)]

    def test_make_step_context(self, input_ids, seq_length, history_length,
                               past_key_values, block_size, model_config):
        step_ctx = make_step_context(
            input_ids,
            seq_length=seq_length,
            history_length=history_length,
            past_key_values=past_key_values,
            world_size=1,
            block_size=block_size,
            model_config=model_config,
        )
        block_offsets = step_ctx.block_offsets
        assert block_offsets[0][3] == 0
        assert block_offsets[1][3] != 0
        assert block_offsets[2][3] == 0

        kv_caches = step_ctx.kv_caches
        assert len(kv_caches) == len(past_key_values)
