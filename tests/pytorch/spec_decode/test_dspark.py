import asyncio
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.attention.v4 import (
    CudaV4AttentionMetadata,
    TritonV4AttentionImpl,
    _V4DecodeExecutor,
)
from lmdeploy.pytorch.config import CacheConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.engine.cache_engine import StateCacheEngine
from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import (
    build_prefill_sparse_indices,
)
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.models.deepseek_v4 import DeepseekV4ForCausalLM
from lmdeploy.pytorch.models.deepseek_v4_dspark import (
    DeepseekV4ForCausalLMDSpark,
)
from lmdeploy.pytorch.models.deepseek_v32 import DeepseekV32Model
from lmdeploy.pytorch.models.dspark_heads import (
    GatedMarkovHead,
    RNNMarkovHead,
    VanillaMarkovHead,
    compute_dspark_proposal_ids,
)
from lmdeploy.pytorch.spec_decode.dspark_utils import (
    parse_dspark_config,
    prepare_dspark_hf_config,
    validate_dspark_runtime_config,
    validate_dspark_target_config,
)
from lmdeploy.pytorch.spec_decode.proposers.dspark import DSpark


def _raw_config(**kwargs):
    values = dict(
        architectures=['DSparkDraftModel'],
        speculators_model_type='dspark',
        transformer_layer_config=SimpleNamespace(
            model_type='qwen3',
            hidden_size=16,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
        ),
        vocab_size=100,
        block_size=8,
        mask_token_id=99,
        draft_vocab_size=100,
        markov_rank=4,
        markov_head_type='vanilla',
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        aux_hidden_state_layer_ids=[2, 20, 39, 58, 75],
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def _bundled_config(**kwargs):
    values = dict(
        architectures=['DeepseekV4ForCausalLM'],
        model_type='deepseek_v4',
        vocab_size=129280,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_markov_rank=256,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_parse_current_speculators_dspark_layout():
    cfg = _raw_config(sample_from_anchor=True)
    prepare_dspark_hf_config(cfg)
    resolved = parse_dspark_config(cfg, 7, target_num_layers=80)

    assert cfg.architectures == ['Qwen3DSparkModel']
    assert cfg.target_layer_ids == [1, 19, 38, 57, 74]
    assert resolved.target_layer_ids == (1, 19, 38, 57, 74)
    assert resolved.sample_from_anchor is True
    assert resolved.draft_query_len == 7
    assert resolved.verify_block_len == 8
    assert resolved.checkpoint_block_capacity == 8
    assert resolved.dynamic_verify_policy == 'fixed'


def test_parse_preview_speculators_dspark_defaults_to_bonus_anchor():
    cfg = _raw_config(aux_hidden_state_layer_ids=[8, 23, 39, 55, 70])
    prepare_dspark_hf_config(cfg)
    resolved = parse_dspark_config(cfg, 7, target_num_layers=80)

    assert resolved.sample_from_anchor is False
    assert resolved.draft_query_len == 8
    assert resolved.verify_block_len == 8
    assert resolved.target_layer_ids == (7, 22, 38, 54, 69)


def test_parse_bundled_deepseek_v4_dspark_uses_gamma_capacity():
    cfg = _bundled_config()
    prepare_dspark_hf_config(cfg)
    resolved = parse_dspark_config(cfg, 5, target_num_layers=43)

    assert cfg.architectures == ['DeepseekV4ForCausalLMDSpark']
    assert resolved.bundled_draft is True
    assert resolved.sample_from_anchor is True
    assert resolved.draft_query_len == 5
    assert resolved.verify_block_len == 6
    assert resolved.checkpoint_block_capacity == 5
    assert resolved.mask_token_id == 128799
    assert resolved.markov_head_type == 'vanilla'


def test_parse_native_dense_qwen_dspark_uses_gamma_capacity():
    cfg = SimpleNamespace(
        architectures=['Qwen3DSparkModel'],
        model_type='qwen3',
        vocab_size=100,
        block_size=5,
        mask_token_id=99,
        target_layer_ids=[1, 3],
        markov_rank=4,
        sample_from_anchor=True,
    )
    prepare_dspark_hf_config(cfg)
    resolved = parse_dspark_config(cfg, 5, target_num_layers=4)

    assert cfg.dflash_config == {
        'mask_token_id': 99,
        'target_layer_ids': [1, 3],
        'causal': False,
    }
    assert resolved.bundled_draft is False
    assert resolved.checkpoint_block_capacity == 5
    assert resolved.draft_query_len == 5
    assert resolved.verify_block_len == 6


def test_dspark_v1_accepts_eager_and_cudagraph_runtime():
    validate_dspark_runtime_config(
        cache_config=SimpleNamespace(device_type='cuda'),
        backend_config=SimpleNamespace(device_type='cuda', eager_mode=True))
    validate_dspark_runtime_config(
        cache_config=SimpleNamespace(device_type='cuda'),
        backend_config=SimpleNamespace(device_type='cuda', eager_mode=False))


def test_parse_dspark_rejects_capacity_and_bad_layers():
    with pytest.raises(ValueError, match='required=8, capacity=7'):
        parse_dspark_config(_raw_config(block_size=7), 7, target_num_layers=80)
    with pytest.raises(ValueError, match='strictly increasing'):
        parse_dspark_config(_bundled_config(dspark_target_layer_ids=[41, 40]), 5,
                            target_num_layers=43)
    with pytest.raises(ValueError, match='outside target depth'):
        parse_dspark_config(_bundled_config(dspark_target_layer_ids=[43]), 5,
                            target_num_layers=43)


def test_specdecode_config_stores_resolved_dspark_fields(monkeypatch):
    draft = SimpleNamespace(hf_config=_raw_config(sample_from_anchor=True))
    target = SimpleNamespace(num_layers=4,
                             llm_config=SimpleNamespace(num_hidden_layers=80))

    def from_pretrained(model, **kwargs):
        return draft if kwargs['is_draft_model'] else target

    monkeypatch.setattr(ModelConfig, 'from_pretrained', from_pretrained)
    cfg = SpecDecodeConfig.from_config(
        method='dspark',
        num_speculative_tokens=7,
        model='draft',
        target_model='target',
        target_cache_cfg=CacheConfig(max_batches=1,
                                     block_size=64,
                                     num_cpu_blocks=0,
                                     num_gpu_blocks=1),
    )

    assert cfg.target_layer_ids == (1, 19, 38, 57, 74)
    assert cfg.mask_token_id == 99
    assert cfg.dspark_sample_from_anchor is True
    assert cfg.dspark_draft_query_len == 7
    assert cfg.dspark_verify_block_len == 8
    assert cfg.markov_rank == 4
    assert cfg.dynamic_verify_policy == 'fixed'
    assert draft.hf_config.dspark_sample_from_anchor is True
    assert draft.hf_config.dspark_draft_query_len == 7
    assert draft.hf_config.dspark_num_speculative_tokens == 7


def test_specdecode_config_rejects_native_draft_target_hidden_mismatch(
        monkeypatch):
    draft_hf_config = SimpleNamespace(
        architectures=['Qwen3DSparkModel'],
        model_type='qwen3',
        hidden_size=16,
        vocab_size=100,
        num_hidden_layers=3,
        num_target_layers=4,
        block_size=3,
        mask_token_id=99,
        target_layer_ids=[1, 3],
        markov_rank=4,
        sample_from_anchor=True,
    )
    prepare_dspark_hf_config(draft_hf_config)
    draft = SimpleNamespace(hf_config=draft_hf_config)
    target = SimpleNamespace(
        num_layers=4,
        llm_config=SimpleNamespace(
            num_hidden_layers=4,
            hidden_size=32,
            vocab_size=100,
        ),
    )

    def from_pretrained(model, **kwargs):
        return draft if kwargs['is_draft_model'] else target

    monkeypatch.setattr(ModelConfig, 'from_pretrained', from_pretrained)
    with pytest.raises(ValueError, match='target hidden-size mismatch'):
        SpecDecodeConfig.from_config(
            method='dspark',
            num_speculative_tokens=3,
            model='draft',
            target_model='target',
            target_cache_cfg=CacheConfig(
                max_batches=1,
                block_size=64,
                num_cpu_blocks=0,
                num_gpu_blocks=1,
            ),
        )


@pytest.mark.parametrize(
    ('target_config', 'match'),
    [
        (SimpleNamespace(num_hidden_layers=5, hidden_size=16,
                         vocab_size=100), 'target depth mismatch'),
        (SimpleNamespace(num_hidden_layers=4, hidden_size=16,
                         vocab_size=101), 'target vocabulary mismatch'),
        (SimpleNamespace(num_hidden_layers=4, hidden_size=16,
                         vocab_size=99), 'mask token is outside'),
    ],
)
def test_validate_native_dspark_target_contract(target_config, match):
    draft = SimpleNamespace(
        architectures=['Qwen3DSparkModel'],
        hidden_size=16,
        vocab_size=100 if target_config.vocab_size != 99 else None,
        num_target_layers=4,
    )
    resolved = SimpleNamespace(mask_token_id=99)
    with pytest.raises(ValueError, match=match):
        validate_dspark_target_config(draft, target_config, resolved)


def test_dspark_query_layouts():
    inputs = ModelInputs(
        input_ids=torch.tensor([[10, 11, 20, 21]]),
        seq_length=torch.tensor([2, 2]),
        history_lengths=torch.tensor([0, 5]),
        block_offsets=torch.zeros((2, 2), dtype=torch.int32),
        is_decoding=False,
        num_ignored_history=torch.zeros(2, dtype=torch.long),
        max_q_seqlen=2,
        max_kv_seqlen=7,
        sum_kv_seqlen=9,
    )
    for sample_from_anchor, query_len in ((True, 3), (False, 4)):
        proposer = DSpark(SimpleNamespace(
            mask_token_id=99,
            target_layer_ids=(1, 5),
            num_speculative_tokens=3,
            dspark_draft_query_len=query_len,
            dspark_sample_from_anchor=sample_from_anchor,
            model_config=None,
        ), device='cpu')
        query = proposer._build_query_inputs(inputs, torch.tensor([2, 2]),
                                             torch.tensor([7, 8]))
        assert query.max_q_seqlen == query_len
        assert query.input_ids.reshape(2, query_len)[:, 0].tolist() == [7, 8]
        assert torch.all(query.input_ids.reshape(2, query_len)[:, 1:] == 99)
        assert query.target_position_ids.tolist() == [
            list(range(2, 2 + query_len)) + list(range(7, 7 + query_len))
        ]


def test_dspark_compacts_rejected_v4_context_rows():
    proposer = DSpark(SimpleNamespace(
        mask_token_id=99,
        target_layer_ids=(1, 5),
        num_speculative_tokens=5,
        dspark_draft_query_len=5,
        dspark_sample_from_anchor=True,
        model_config=None,
    ), device='cpu')
    inputs = ModelInputs(
        input_ids=torch.arange(12).reshape(1, 12),
        seq_length=torch.tensor([6, 6]),
        history_lengths=torch.tensor([10, 20]),
        block_offsets=torch.zeros((2, 2), dtype=torch.int32),
        is_decoding=True,
        num_ignored_history=torch.zeros(2, dtype=torch.long),
        max_q_seqlen=6,
        max_kv_seqlen=26,
        sum_kv_seqlen=42,
    )
    target_hidden = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2)
    extra = SimpleNamespace(
        target_hidden_states=target_hidden,
        num_rejected_tokens=torch.tensor([4, 1]),
    )

    context, hidden, lengths, starts = (
        proposer._prepare_context_materialization(inputs, extra))

    assert lengths.tolist() == [2, 5]
    assert starts.tolist() == [12, 25]
    assert context.seq_length.tolist() == [2, 5]
    assert context.input_ids.tolist() == [[0, 1, 6, 7, 8, 9, 10]]
    torch.testing.assert_close(
        hidden,
        target_hidden.reshape(2, 6, 2)[
            torch.tensor([[True, True, False, False, False, False],
                          [True, True, True, True, True, False]])])


def test_dspark_query_does_not_commit_v4_window_rows():
    proposer = DSpark(SimpleNamespace(
        mask_token_id=99,
        target_layer_ids=(1, 5),
        num_speculative_tokens=5,
        dspark_draft_query_len=5,
        dspark_sample_from_anchor=True,
        model_config=None,
    ), device='cpu')
    query = ModelInputs(
        input_ids=torch.arange(5).reshape(1, 5),
        seq_length=torch.tensor([5]),
        history_lengths=torch.tensor([126]),
        block_offsets=torch.zeros((1, 2), dtype=torch.int32),
        is_decoding=True,
        num_ignored_history=torch.zeros(1, dtype=torch.long),
        max_q_seqlen=5,
        max_kv_seqlen=131,
        sum_kv_seqlen=131,
        state_offsets=torch.tensor([0]),
    )
    state_engine = object.__new__(StateCacheEngine)
    state_engine._named_state_caches = {
        'v4_window_kv_fp8':
        torch.arange(134, dtype=torch.float32).reshape(1, 1, 134, 1),
    }
    cache_engine = SimpleNamespace(state_cache_engine=state_engine)
    before = state_engine.named_state_caches['v4_window_kv_fp8'].clone()
    output = object()

    def fake_forward(model_inputs, cache_engine):
        rows = torch.tensor([126, 127, 128, 129, 130])
        window_cache = cache_engine.state_cache_engine.named_state_caches[
            'v4_window_kv_fp8']
        window_cache[:, 0, rows] = -1
        return output

    proposer._forward = fake_forward

    assert proposer._forward_query(query, cache_engine) is output
    torch.testing.assert_close(
        state_engine.named_state_caches['v4_window_kv_fp8'], before)


def _fill_markov(head):
    with torch.no_grad():
        head.markov_w1.weight.copy_(torch.arange(
            head.markov_w1.weight.numel(), dtype=torch.float32).reshape_as(
                head.markov_w1.weight) / 10)
        head.markov_w2.weight.copy_(torch.arange(
            head.markov_w2.weight.numel(), dtype=torch.float32).reshape_as(
                head.markov_w2.weight) / 20)


def test_vanilla_markov_matches_embedding_then_projection():
    head = VanillaMarkovHead(7, 5, 3, dtype=torch.float32, device='cpu')
    _fill_markov(head)
    ids = torch.tensor([1, 4])
    bias, state = head.step(ids, None, None)
    expected = torch.nn.functional.linear(head.markov_w1.weight[ids],
                                          head.markov_w2.weight)
    torch.testing.assert_close(bias, expected)
    assert state is None


def test_gated_and_rnn_heads_use_hidden_and_block_local_state():
    hidden = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    ids = torch.tensor([1, 2])
    gated = GatedMarkovHead(7, 7, 3, hidden_size=2,
                            dtype=torch.float32, device='cpu')
    _fill_markov(gated)
    with torch.no_grad():
        gated.gate_proj.weight.fill_(0.1)
        gated.gate_proj.bias.zero_()
    bias1, _ = gated.step(ids, hidden, None)
    bias2, _ = gated.step(ids, hidden + 1, None)
    assert not torch.equal(bias1, bias2)

    rnn = RNNMarkovHead(7, 7, 3, hidden_size=2,
                        dtype=torch.float32, device='cpu')
    _fill_markov(rnn)
    with torch.no_grad():
        rnn.joint_proj.weight.fill_(0.1)
        rnn.joint_proj.bias.zero_()
    state0 = rnn.init_state(2, torch.float32, torch.device('cpu'))
    _, state1 = rnn.step(ids, hidden, state0)
    _, state2 = rnn.step(ids, hidden, state1)
    assert torch.count_nonzero(state0) == 0
    assert not torch.equal(state1, state2)
    torch.testing.assert_close(rnn.init_state(2, torch.float32,
                                               torch.device('cpu')), state0)


def test_deepseek_v4_cudagraph_preserves_aux_hidden_states():
    output_buffers = {
        'hidden_states': torch.arange(48).view(1, 3, 2, 8),
        'aux_hidden_states': torch.arange(60).view(1, 3, 20),
    }
    model = object.__new__(DeepseekV4ForCausalLM)
    outputs = model.get_outputs_cudagraph(output_buffers,
                                           torch.zeros((1, 2), dtype=torch.long))
    assert outputs['hidden_states'].shape == (1, 2, 2, 8)
    assert outputs['aux_hidden_states'].shape == (1, 2, 20)


def test_deepseek_v4_dspark_cudagraph_discards_padded_proposals():
    output_buffers = {
        'hidden_states': torch.arange(80).view(1, 20, 4),
        'draft_token_ids': torch.arange(20).view(4, 5),
    }
    model = object.__new__(DeepseekV4ForCausalLMDSpark)
    model.dspark_draft_query_len = 5
    outputs = model.get_outputs_cudagraph(
        output_buffers, torch.zeros((1, 15), dtype=torch.long))

    assert outputs['hidden_states'].shape == (1, 15, 4)
    assert outputs['draft_token_ids'].shape == (3, 5)
    assert outputs['draft_token_ids'].tolist() == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ]


def test_v4_rectangular_spec_block_is_marked_for_packed_decode():
    attn = SimpleNamespace(
        is_decoding=True,
        block_offsets=torch.zeros((2, 4), dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 5, 10], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 15, 35], dtype=torch.int32),
        kv_seqlens=torch.tensor([15, 20], dtype=torch.int32),
        q_seqlens=torch.tensor([5, 5], dtype=torch.int32),
    )
    step = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, num_gpu_blocks=4),
        max_q_seqlen=5,
        max_kv_seqlen=20,
        sum_kv_seqlen=35,
    )
    meta = CudaV4AttentionMetadata.from_step_context(attn, step)

    assert meta.is_decoding is False
    assert meta.is_rectangular_decode is True
    assert meta.max_kv_seqlen == 20
    assert meta.causal is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_v4_rectangular_cudagraph_builds_packed_decode_metadata(monkeypatch):
    device = torch.device('cuda')
    attn = SimpleNamespace(
        is_decoding=True,
        block_offsets=torch.zeros((4, 8), dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor([0, 5, 10, 15, 20], dtype=torch.int32,
                                 device=device),
        cu_seqlens_k=torch.tensor([0, 5, 10, 15, 20], dtype=torch.int32,
                                 device=device),
        kv_seqlens=torch.full((4, ), 5, dtype=torch.int32, device=device),
        q_seqlens=torch.full((4, ), 5, dtype=torch.int32, device=device),
        is_cuda_graph=True,
        graph_max_kv_seqlen=512,
        graph_sum_kv_seqlen=2048,
    )
    step = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, num_gpu_blocks=8,
                                     max_session_len=512),
        max_q_seqlen=5,
        max_kv_seqlen=5,
        sum_kv_seqlen=20,
    )
    # Index-score scheduling is unrelated to this metadata assertion.
    monkeypatch.setattr(
        CudaV4AttentionMetadata,
        '_build_index_score_meta',
        staticmethod(lambda *args, **kwargs: None),
    )
    slot = torch.tensor([0, 1, 2, -1], dtype=torch.long, device=device)
    meta = CudaV4AttentionMetadata.from_step_context(
        attn, step, window_size=128, ring_storage_capacity=132, slot=slot)

    assert meta.is_decoding is False
    assert meta.is_rectangular_decode is True
    assert meta.max_kv_seqlen == 512
    assert meta.sum_kv_seqlen == 20
    # Every token belonging to the padded graph row retains the negative
    # sentinel, so the FP8 pack kernel suppresses its persistent-state writes.
    assert meta.rectangular_decode.write_slot.tolist() == [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        -1, -1, -1, -1, -1,
    ]
    assert meta.rectangular_decode.write_pos[:5].tolist() == [0, 1, 2, 3, 4]
    # Causal target rows expose only their position-specific candidate prefix.
    assert meta.rectangular_decode.topk_length[:5].tolist() == [1, 2, 3, 4, 5]
    assert meta.rectangular_decode.topk_length[-5:].tolist() == [1] * 5


def test_v4_decode_window_write_preserves_padded_slot_sentinel():
    seen = {}

    class FakeImpl:

        @staticmethod
        def _pack_window_fp8(kv, cache, slot, position):
            seen['slot'] = slot.clone()

    executor = _V4DecodeExecutor(FakeImpl())
    kv = torch.zeros(2, 1, 4)
    cache = torch.zeros(2, 8, 4)
    slot = torch.tensor([0, -1])
    meta = SimpleNamespace(
        decode_window=SimpleNamespace(window_pos=torch.tensor([3, 3])))

    selected = executor._write_window(
        kv, cache, slot, meta)

    assert seen['slot'].tolist() == [0, -1]
    assert selected.shape == (2, 8, 4)


@pytest.mark.parametrize(('causal', 'expected'), [
    (True, 'rectangular'),
    (False, 'prefill'),
])
def test_v4_rectangular_dispatch_preserves_causal_mode(causal, expected):
    """Draft blocks cannot enter the causal target-only executor."""

    class FakeExecutor:

        def __init__(self, name):
            self.name = name

        def forward(self, *args, **kwargs):
            return self.name

    impl = object.__new__(TritonV4AttentionImpl)
    impl._decode_executor = FakeExecutor('decode')
    impl._rectangular_decode_executor = FakeExecutor('rectangular')
    impl._prefill_executor = FakeExecutor('prefill')
    # Supply rectangular metadata in both cases so the dispatch itself, not
    # merely metadata construction, enforces the causal-only contract.
    meta = SimpleNamespace(
        is_decoding=False,
        is_rectangular_decode=True,
        causal=causal,
        rectangular_decode=object(),
    )

    actual = impl.forward(None, None, None, meta, None, None, None)

    assert actual == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_v4_dspark_draft_metadata_builds_block_noncausal_indices(monkeypatch):
    """Exercise the metadata/index path used by the bundled draft."""
    device = torch.device('cuda')
    attn = SimpleNamespace(
        is_decoding=True,
        block_offsets=torch.zeros((1, 1), dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, 13], dtype=torch.int32, device=device),
        kv_seqlens=torch.tensor([13], dtype=torch.int32, device=device),
        q_seqlens=torch.tensor([3], dtype=torch.int32, device=device),
    )
    step = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, num_gpu_blocks=1,
                                     max_session_len=32),
        max_q_seqlen=3,
        max_kv_seqlen=13,
        sum_kv_seqlen=13,
    )
    monkeypatch.setattr(
        CudaV4AttentionMetadata,
        '_build_prefill_index_score_meta',
        staticmethod(lambda *args, **kwargs: None),
    )
    meta = CudaV4AttentionMetadata.from_step_context(
        attn,
        step,
        window_size=4,
        ring_storage_capacity=6,
        slot=torch.tensor([0], dtype=torch.long, device=device),
        causal=False,
    )

    # Rectangular describes the scheduler shape, but the draft deliberately
    # receives sparse-prefill metadata rather than causal target metadata.
    assert meta.is_rectangular_decode is True
    assert meta.causal is False
    assert meta.rectangular_decode is None
    assert meta.prefill_shared is not None
    prefill = meta.get_ratio_meta(0).prefill
    topk, topk_len = build_prefill_sparse_indices(
        start_pos=meta.start_pos,
        total_lens=meta.kv_seqlens,
        token_seq=meta.prefill_shared.token_seq,
        token_pos=meta.prefill_shared.token_pos,
        cu_seqlens_k=prefill.cu_seqlens_k,
        uncompressed_kv_lens=meta.prefill_shared.uncompressed_kv_lens,
        window_size=4,
        causal=False,
        max_q_seqlen=3,
    )

    expected = torch.arange(7, dtype=torch.int32, device=device)
    torch.testing.assert_close(topk[:, 0, :7], expected.expand(3, -1))
    torch.testing.assert_close(topk_len, torch.full((3,), 7,
                                                    dtype=torch.int32,
                                                    device=device))


def test_deepseek_v32_aux_capture_is_selected_and_target_output_is_stable():

    class FakeLayer(torch.nn.Module):

        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, hidden, **kwargs):
            residual = hidden if kwargs['residual'] is None else kwargs['residual']
            return hidden.new_full(hidden.shape, self.value), residual + hidden

    class FakeNorm(torch.nn.Module):

        def forward(self, hidden, residual):
            return hidden + residual, residual

    model = object.__new__(DeepseekV32Model)
    torch.nn.Module.__init__(model)
    model.embed_tokens = torch.nn.Embedding(8, 2)
    with torch.no_grad():
        model.embed_tokens.weight.zero_()
    model.rotary_emb = lambda hidden, positions: (
        hidden.new_zeros(1, hidden.size(1), 1),
        hidden.new_zeros(1, hidden.size(1), 1))
    model.layers = torch.nn.ModuleList([FakeLayer(1), FakeLayer(2), FakeLayer(3)])
    model.norm = FakeNorm()
    inputs = dict(input_ids=torch.tensor([[1, 2]]),
                  position_ids=torch.tensor([[0, 1]]),
                  past_key_values=[None, None, None])

    model._aux_hidden_state_layers_set = frozenset()
    plain = model.forward(**inputs)
    model._aux_hidden_state_layers_set = frozenset({0, 2})
    captured = model.forward(**inputs)

    torch.testing.assert_close(captured['hidden_states'], plain)
    assert captured['aux_hidden_states'].shape == (1, 2, 4)
    torch.testing.assert_close(captured['aux_hidden_states'][..., :2],
                               torch.ones(1, 2, 2))
    torch.testing.assert_close(captured['aux_hidden_states'][..., 2:],
                               torch.full((1, 2, 2), 6.0))


def test_dspark_sequential_sampling_feeds_each_token_to_next_step():

    class FakeDraft:

        dspark_sample_from_anchor = True
        dspark_draft_query_len = 3
        dspark_num_speculative_tokens = 3

        @staticmethod
        def compute_base_logits(hidden):
            return hidden.new_zeros(*hidden.shape[:2], 5)

        @staticmethod
        def init_sequential_state(batch_size, dtype, device):
            return None

        @staticmethod
        def sequential_head_step(prev, hidden, state):
            del hidden
            bias = torch.zeros(prev.size(0), 5, device=prev.device)
            bias.scatter_(1, ((prev + 1) % 5)[:, None], 100)
            return bias, state

        @staticmethod
        def map_draft_to_target(ids):
            return ids

    hidden = torch.zeros(1, 6, 2)
    query_ids = torch.tensor([[1, 99, 99, 3, 99, 99]])
    captured = compute_dspark_proposal_ids(FakeDraft(), hidden, query_ids)
    assert captured.tolist() == [[2, 3, 4], [4, 0, 1]]

    proposer = DSpark(SimpleNamespace(
        mask_token_id=99,
        target_layer_ids=(1, 5),
        num_speculative_tokens=3,
        dspark_draft_query_len=3,
        dspark_sample_from_anchor=True,
        model_config=None,
    ), device='cpu')
    inputs = ModelInputs(
        input_ids=torch.tensor([[10, 11, 20, 21]]),
        seq_length=torch.tensor([2, 2]),
        history_lengths=torch.tensor([0, 5]),
        block_offsets=torch.zeros((2, 2), dtype=torch.int32),
        is_decoding=False,
        num_ignored_history=torch.zeros(2, dtype=torch.long),
        max_q_seqlen=2,
        max_kv_seqlen=7,
        sum_kv_seqlen=9,
    )
    proposer._prepare_context_materialization = lambda *args: (
        inputs, torch.zeros(4, 2), torch.tensor([2, 2]), None)
    proposer._materialize_context = lambda *args: None
    proposer._forward = lambda *args, **kwargs: {
        'hidden_states': hidden,
        'draft_token_ids': captured,
    }
    proposer._draft_model = lambda: (_ for _ in ()).throw(
        AssertionError('proposal epilogue must not run in proposer Python'))
    result = asyncio.run(
        proposer.propose_block(
            inputs,
            SimpleNamespace(next_token_ids=torch.tensor([1, 3])),
            cache_engine=object()))
    assert result.tolist() == [[2, 3, 4], [4, 0, 1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_dspark_proposal_epilogue_cuda_graph_replays_all_steps():

    class TinyDraft(torch.nn.Module):

        dspark_sample_from_anchor = True
        dspark_draft_query_len = 3
        dspark_num_speculative_tokens = 3

        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(2, 5, bias=False, device='cuda')
            self.markov_head = VanillaMarkovHead(
                vocab_size=5, draft_vocab_size=5, markov_rank=2,
                dtype=torch.float32, device='cuda')

        def compute_base_logits(self, hidden):
            return self.base(hidden)

        def init_sequential_state(self, batch_size, dtype, device):
            return None

        def sequential_head_step(self, prev, hidden, state):
            return self.markov_head.step(prev, hidden, state)

        @staticmethod
        def map_draft_to_target(ids):
            return ids

    torch.manual_seed(7)
    model = TinyDraft()
    static_hidden = torch.randn(1, 6, 2, device='cuda')
    static_ids = torch.tensor([[0, 4, 4, 1, 4, 4]], device='cuda')

    # Compile all participating kernels before stream capture.
    compute_dspark_proposal_ids(model, static_hidden, static_ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = compute_dspark_proposal_ids(
            model, static_hidden, static_ids)

    replay_hidden = torch.randn_like(static_hidden)
    replay_ids = torch.tensor([[2, 4, 4, 3, 4, 4]], device='cuda')
    expected = compute_dspark_proposal_ids(
        model, replay_hidden, replay_ids)
    static_hidden.copy_(replay_hidden)
    static_ids.copy_(replay_ids)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_output, expected)


def test_dspark_v1_rejects_nongreedy_sampling_before_forward():
    proposer = DSpark(SimpleNamespace(
        mask_token_id=99,
        target_layer_ids=(1,),
        num_speculative_tokens=3,
        dspark_draft_query_len=3,
        dspark_sample_from_anchor=True,
        model_config=None,
    ), device='cpu')
    sampling = SamplingInputs(max_top_k=8, min_top_p=1.0)
    with pytest.raises(NotImplementedError, match='greedy sampling only'):
        asyncio.run(proposer.propose(None, None, sampling))
