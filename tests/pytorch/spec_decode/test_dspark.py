import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

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
from lmdeploy.pytorch.models.qwen3_dspark import Qwen3DSparkModel
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


@pytest.mark.parametrize('token_ids', [None, (1, 2)])
def test_raw_speculators_build_config_without_generation_tokens(token_ids):
    from lmdeploy.pytorch.configurations.default import DefaultModelConfigBuilder

    # Standalone draft checkpoints need no generation BOS/EOS: the target
    # owns stopping, but the common model-config builder still reads these.
    cfg = _raw_config(target_hidden_size=None)
    if token_ids is not None:
        cfg.bos_token_id, cfg.eos_token_id = token_ids
    prepare_dspark_hf_config(cfg)
    model_config = DefaultModelConfigBuilder.build(cfg)

    expected = (None, None) if token_ids is None else token_ids
    assert (model_config.bos_token_id, model_config.eos_token_id) == expected
    assert model_config.hidden_size == cfg.transformer_layer_config.hidden_size
    assert cfg.sample_from_anchor is False


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


@pytest.mark.parametrize('cache_kind', [
    'pageable', 'v4', 'small_ring', 'small_window', 'compressed',
    'extra_state', 'unknown', 'missing_ring', 'mismatched_ring',
])
@pytest.mark.parametrize('dp,ep', [(1, 1), (2, 1), (1, 2)])
def test_dspark_context_materialization_layout(cache_kind, dp, ep):
    proposer = DSpark(SimpleNamespace(
        mask_token_id=99,
        target_layer_ids=(1, 5),
        num_speculative_tokens=5,
        dspark_draft_query_len=5,
        dspark_sample_from_anchor=True,
        model_config=None,
        dist_config=SimpleNamespace(dp=dp, ep=ep),
    ), device='cpu')
    inputs = ModelInputs(
        input_ids=torch.arange(18).reshape(1, 18),
        seq_length=torch.tensor([6, 6, 6]),
        history_lengths=torch.tensor([10, 20, 30]),
        block_offsets=torch.zeros((3, 2), dtype=torch.int32),
        is_decoding=True,
        num_ignored_history=torch.zeros(3, dtype=torch.long),
        max_q_seqlen=6,
        max_kv_seqlen=36,
        sum_kv_seqlen=78,
    )
    target_hidden = torch.arange(36, dtype=torch.float32).reshape(1, 18, 2)
    inputs.target_hidden_states = target_hidden
    named_caches = {}
    if cache_kind == 'pageable':
        model = Qwen3DSparkModel.__new__(Qwen3DSparkModel)
        torch.nn.Module.__init__(model)
    elif cache_kind == 'unknown':
        model = SimpleNamespace()
    else:
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        torch.nn.Module.__init__(model)
        model.dspark_num_speculative_tokens = 5
        model.args = SimpleNamespace(
            window_size=4 if cache_kind == 'small_window' else 128,
            ring_storage_capacity=128 if cache_kind == 'small_ring' else 134,
            compress_ratios=(4,) if cache_kind == 'compressed' else (0,),
        )
        capacity = model.args.ring_storage_capacity
        if cache_kind == 'mismatched_ring':
            capacity -= 2
        if cache_kind != 'missing_ring':
            named_caches['v4_window_kv_fp8'] = torch.empty(1, 3, capacity, 1)
        if cache_kind == 'extra_state':
            named_caches['unvalidated_state'] = torch.empty(1)
    proposer.model = model
    cache = SimpleNamespace(
        state_cache_engine=SimpleNamespace(named_state_caches=named_caches),
        cache_config=SimpleNamespace(block_size=64),
    )
    proposer._materialize_context = lambda *args: None
    if cache_kind == 'mismatched_ring':
        with pytest.raises(RuntimeError, match='ring capacity'):
            proposer.prepare_warmup_forward(inputs, cache)
        return
    # Exercise actual warmup routing, including wrapped graph-runner models.
    proposer.model = SimpleNamespace(get_model=lambda: model)
    if (dp > 1 or ep > 1) and cache_kind not in ('pageable', 'v4'):
        def unexpected_materialization(*args):
            raise AssertionError('Unsupported geometry must fail before warmup writes any cache.')

        proposer._materialize_context = unexpected_materialization
        with pytest.raises(ValueError, match='validated full-context cache geometry'):
            proposer.prepare_warmup_forward(inputs, cache)
        return
    proposer.prepare_warmup_forward(inputs, cache)
    full_context = cache_kind in ('pageable', 'v4')
    assert proposer._full_context_materialization == full_context
    extra = SimpleNamespace(
        target_hidden_states=target_hidden,
        num_rejected_tokens=torch.tensor([5, 3, 0]),
    )

    class NoNonzero(TorchDispatchMode):

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func != torch.ops.aten.nonzero.default
            return func(*args, **(kwargs or {}))

    with NoNonzero() if full_context else nullcontext():
        context, hidden, lengths, starts = (
            proposer._prepare_context_materialization(inputs, extra))

    assert lengths.tolist() == [1, 3, 6]
    assert starts.tolist() == [11, 23, 36]
    assert context.is_decoding is False
    if full_context:
        assert context.seq_length.tolist() == [6, 6, 6]
        assert context.input_ids.data_ptr() == inputs.input_ids.data_ptr()
        assert hidden.data_ptr() == target_hidden.data_ptr()
        assert hidden.shape == (18, 2)
    else:
        assert context.seq_length.tolist() == [1, 3, 6]
        indices = torch.tensor([0, 6, 7, 8, 12, 13, 14, 15, 16, 17])
        torch.testing.assert_close(context.input_ids, inputs.input_ids[:, indices])
        torch.testing.assert_close(hidden, target_hidden[0, indices])
    query = proposer._build_query_inputs(
        inputs, lengths, torch.tensor([7, 8, 9]), query_start_positions=starts)
    assert query.history_lengths.tolist() == [11, 23, 36]
    assert query.target_position_ids.reshape(3, 5)[:, 0].tolist() == [11, 23, 36]


@pytest.mark.parametrize('query_len', [5, 6])
def test_dspark_query_does_not_commit_v4_window_rows(query_len):
    """Full/compact context have identical visible state across repeated
    wraps."""
    proposer = DSpark(SimpleNamespace(
        mask_token_id=99,
        target_layer_ids=(1, 5),
        num_speculative_tokens=5,
        dspark_draft_query_len=query_len,
        dspark_sample_from_anchor=query_len == 5,
        model_config=None,
    ), device='cpu')
    window, capacity, width = 128, 134, 6
    slots = torch.tensor([2, 0, 3])
    history = torch.tensor([126, 132, 400])
    engines = []
    for _ in range(2):
        state = object.__new__(StateCacheEngine)
        state._named_state_caches = {
            'v4_window_kv_fp8': torch.zeros(1, 4, capacity, 1),
        }
        for slot, start in zip(slots.tolist(), history.tolist(), strict=True):
            positions = torch.arange(max(0, start - window), start)
            state.named_state_caches['v4_window_kv_fp8'][0, slot, positions % capacity, 0] = (
                positions.float() + slot * 10000)
        engines.append(SimpleNamespace(state_cache_engine=state))

    def visible(cache, starts):
        tensor = cache.state_cache_engine.named_state_caches['v4_window_kv_fp8']
        return [tensor[0, slot, torch.arange(max(0, start - window), start) % capacity].clone()
                for slot, start in zip(slots.tolist(), starts.tolist(), strict=True)]

    def fake_forward(inputs, cache_engine):
        # Observe exactly the historical range used by the raw-KV flatten path,
        # then simulate query KV writes that must be rolled back.
        result = visible(cache_engine, inputs.history_lengths)
        tensor = cache_engine.state_cache_engine.named_state_caches['v4_window_kv_fp8']
        rows = (inputs.history_lengths[:, None] + torch.arange(query_len)[None]) % capacity
        tensor[:, slots[:, None], rows] = -1
        return result

    proposer._forward = fake_forward
    for step in range(96):
        lengths = torch.tensor([1, 3, 6]).roll(step % 3)
        for full, cache in enumerate(engines):
            tensor = cache.state_cache_engine.named_state_caches['v4_window_kv_fp8']
            for slot, start, length in zip(slots.tolist(), history.tolist(), lengths.tolist(), strict=True):
                positions = torch.arange(start, start + (width if full else length))
                tensor[0, slot, positions % capacity, 0] = positions.float() + slot * 10000
        starts = history + lengths
        query = ModelInputs(
            input_ids=torch.zeros((1, 3 * query_len), dtype=torch.long),
            seq_length=torch.full((3,), query_len),
            history_lengths=starts,
            block_offsets=torch.zeros((3, 2), dtype=torch.int32),
            is_decoding=True,
            num_ignored_history=torch.zeros(3, dtype=torch.long),
            max_q_seqlen=query_len,
            max_kv_seqlen=int(starts.max()) + query_len,
            sum_kv_seqlen=int(starts.sum()) + 3 * query_len,
            state_offsets=slots,
        )
        expected = visible(engines[0], starts)
        for cache in engines:
            before = cache.state_cache_engine.named_state_caches['v4_window_kv_fp8'].clone()
            actual = proposer._forward_query(query, cache)
            for lhs, rhs in zip(actual, expected, strict=True):
                torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
            torch.testing.assert_close(
                cache.state_cache_engine.named_state_caches['v4_window_kv_fp8'], before, rtol=0, atol=0)
        history = starts


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
        # CUDA graphs pad this table to the global pool. Keep it deliberately
        # wider than the logical per-request session bound below.
        block_offsets=torch.zeros((4, 64), dtype=torch.int32, device=device),
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
    # R4 prefix indices are capped by max_session_len=512 rather than the
    # 64-page graph table (4096 tokens): 512 / 4 = 128 entries.
    assert meta.get_ratio_meta(4).decode.indices.shape == (20, 1, 128)


def test_v4_rectangular_workspace_caps_global_graph_pool():
    # Regression for TP8/B128 graph buffers with a ~73k-block global pool.
    # A meta tensor keeps this CPU test allocation-free while preserving the
    # real page-table shape used by the bound calculation.
    meta = SimpleNamespace(
        is_decoding=False,
        max_kv_seqlen=1_048_576,
        block_offsets=torch.empty((128, 73_108), device='meta'),
        block_size=256,
    )
    assert CudaV4AttentionMetadata._get_index_score_max_len(meta, 4) == 262_144
    assert CudaV4AttentionMetadata._get_index_score_max_len(meta, 128) == 8192

    # The physical page table remains the fallback and upper bound when it is
    # smaller than the logical session capacity.
    meta.block_offsets = torch.empty((128, 128), device='meta')
    assert CudaV4AttentionMetadata._get_index_score_max_len(meta, 4) == 8192


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
    assert result.data_ptr() != captured.data_ptr()
    captured.fill_(-1)
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


def test_dspark_delegates_nongreedy_sampling_to_dflash(monkeypatch):
    proposer = DSpark(SimpleNamespace(
        mask_token_id=99,
        target_layer_ids=(1,),
        num_speculative_tokens=3,
        dspark_draft_query_len=3,
        dspark_sample_from_anchor=True,
        model_config=None,
    ), device='cpu')
    sampling = SamplingInputs(max_top_k=8, min_top_p=0.9, max_num_logprobs=5)
    draft_ids = torch.tensor([[1, 2, 3]])
    cache_engine = object()

    async def fake_propose_block(model_inputs, extra_inputs, cache):
        assert cache is cache_engine
        return draft_ids

    monkeypatch.setattr(proposer, 'propose_block', fake_propose_block)
    extra_inputs = SimpleNamespace(
        next_token_ids=torch.tensor([0]),
        num_rejected_tokens=torch.tensor([0]),
        output_token_ids=torch.tensor([[0]]),
        logprobs=None,
    )
    output = asyncio.run(
        proposer.propose(
            SimpleNamespace(is_chunk=False, dp_meta=None),
            extra_inputs,
            sampling,
            proposal_ctx=SimpleNamespace(cache_engine=cache_engine),
        ))

    assert output.output_draft_token_ids is draft_ids
    assert output.next_token_ids is extra_inputs.next_token_ids
    assert output.num_rejected_tokens is extra_inputs.num_rejected_tokens


@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
def test_v4_kv_only_metadata_matches_window_writes_without_scalar_read(device_name):
    """The KV-only path preserves cutoff, wrap, empty requests and padded
    slots."""
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    device = torch.device(device_name)
    impl = TritonV4AttentionImpl.__new__(TritonV4AttentionImpl)
    impl.compress_ratio = 0
    impl.window_size = 128
    impl.ring_storage_capacity = 134
    lengths = [2, 0, 137, 1]
    starts = [133, 0, 30, 260]
    slots = [2, -1, 0, -1]
    q = torch.tensor(lengths, dtype=torch.int32, device=device)
    cu = torch.tensor([0, 2, 2, 139, 140], dtype=torch.int32, device=device)
    kv = torch.tensor([s + n for s, n in zip(starts, lengths)], dtype=torch.int32, device=device)
    positions = torch.tensor([s + i for s, n in zip(starts, lengths) for i in range(n)], device=device)
    state_ids = torch.tensor(slots, device=device)
    attn = SimpleNamespace(is_decoding=False, q_seqlens=q, cu_seqlens_q=cu, kv_seqlens=kv)

    class NoScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func != torch.ops.aten._local_scalar_dense.default
            return func(*args, **(kwargs or {}))

    def build():
        return impl.build_cache_write_metadata(attn, positions, state_ids, positions.numel())

    with NoScalarRead():
        meta = build()
    expected_slots = torch.tensor([slot for slot, n in zip(slots, lengths) for _ in range(n)], device=device)
    expected_pos = torch.tensor([
        (s + i) % 134 if s + i >= max(0, s + n - 128) else -1
        for s, n in zip(starts, lengths) for i in range(n)
    ], device=device)
    torch.testing.assert_close(meta.slot, expected_slots)
    torch.testing.assert_close(meta.ring_pos, expected_pos)
    if device_name == 'cuda':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = build()
        for shift in (1, 134, 129):
            positions.add_(shift)
            kv.add_(shift)
            graph.replay()
            reference = build()
            torch.testing.assert_close(replayed.slot, reference.slot)
            torch.testing.assert_close(replayed.ring_pos, reference.ring_pos)
    impl.compress_ratio = 4
    assert build() is None
    impl.compress_ratio = 0
    attn.is_decoding = True
    assert build() is None
