import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from lmdeploy.pytorch.config import CacheConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.model_inputs import ModelInputs
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
    assert resolved.checkpoint_block_capacity == 5
    assert resolved.draft_query_len == 5
    assert resolved.verify_block_len == 6


def test_parse_dspark_rejects_capacity_and_bad_layers():
    with pytest.raises(ValueError, match='required=8, capacity=7'):
        parse_dspark_config(_raw_config(block_size=7), 7, target_num_layers=80)
    with pytest.raises(ValueError, match='strictly increasing'):
        cfg = prepare_dspark_hf_config(_raw_config(aux_hidden_state_layer_ids=[42, 41]))
        parse_dspark_config(cfg, 5, target_num_layers=43)
    with pytest.raises(ValueError, match='outside target depth'):
        cfg = prepare_dspark_hf_config(_raw_config(aux_hidden_state_layer_ids=[44]))
        parse_dspark_config(cfg, 5, target_num_layers=43)


def test_dspark_rejects_empty_init_before_resolving_checkpoint():
    from lmdeploy.messages import PytorchEngineConfig, SpeculativeConfig
    from lmdeploy.pytorch.engine.config_builder import ConfigBuilder

    with pytest.raises(ValueError, match='empty_init'):
        ConfigBuilder.build_specdecode_config('unused', SpeculativeConfig(method='dspark', model='unused'),
                                              PytorchEngineConfig(empty_init=True), None, None)


@pytest.mark.parametrize('nested,override', [(False, True), (True, True), (True, False)])
def test_flat_alias_and_effective_aux_ids(monkeypatch, nested, override):
    from transformers import PretrainedConfig

    backbone = {'target_layer_ids': [1, 3], 'aux_hidden_state_layer_ids': [2, 4]} if nested else None
    draft = PretrainedConfig(**vars(_raw_config(transformer_layer_config=backbone, speculators_model_type=None)))
    if not override:
        del draft.aux_hidden_state_layer_ids
        del backbone['target_layer_ids']
        backbone['aux_hidden_state_layer_ids'] = [1, 3]
    target = PretrainedConfig(num_hidden_layers=80)
    monkeypatch.setattr('lmdeploy.pytorch.transformers.config_from_pretrained',
                        lambda model, **_: draft if model == 'draft' else target)
    monkeypatch.setattr(ModelConfig, 'from_hf_config', lambda config, *args, **kwargs:
                        SimpleNamespace(hf_config=config, llm_config=config, num_layers=4))
    overrides = {'aux_hidden_state_layer_ids': [3, 21, 40, 59, 76]}
    cfg = SpecDecodeConfig.from_config('dspark', 7, 'draft',
                                       CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=2),
                                       target_model='target', hf_overrides=overrides if override else None)
    assert draft.architectures == (['Qwen3DSparkModel'] if nested else ['DSparkDraftModel'])
    assert cfg.target_layer_ids == ((2, 20, 39, 58, 75) if override else (0, 2))
    assert tuple(draft.dflash_config['target_layer_ids']) == cfg.target_layer_ids
    assert overrides == {'aux_hidden_state_layer_ids': [3, 21, 40, 59, 76]}


def test_specdecode_config_stores_resolved_dspark_fields(monkeypatch):
    draft = SimpleNamespace(hf_config=prepare_dspark_hf_config(_raw_config(sample_from_anchor=True)))
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
    assert cfg.dspark.sample_from_anchor is True
    assert cfg.dspark.draft_query_len == 7
    assert cfg.dspark.verify_block_len == 8
    assert cfg.dspark.markov_rank == 4
    assert cfg.dspark.dynamic_verify_policy == 'fixed'
    assert draft.hf_config.dspark_sample_from_anchor is True
    assert draft.hf_config.dspark_draft_query_len == 7
    assert draft.hf_config.dspark_num_speculative_tokens == 7


@pytest.mark.parametrize('target_hidden,target_vocab,declared,expected_error', [
    (32, 32, {}, None),
    (64, 32, {'target_hidden_size': None}, 'hidden-size mismatch'),
    (64, 32, {'target_hidden_size': 64}, None),
    (32, 64, {}, 'Markov input vocabulary'),
    (32, 64, {'target_vocab_size': 64}, 'Markov input vocabulary'),
])
def test_raw_dspark_target_dimensions(tmp_path, target_hidden, target_vocab, declared, expected_error):
    backbone = dict(model_type='qwen3', hidden_size=32, vocab_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    raw = dict(model_type='qwen3', architectures=['DSparkDraftModel'], transformer_layer_config=backbone,
               block_size=4, aux_hidden_state_layer_ids=[1, 2], mask_token_id=3, draft_vocab_size=16,
               markov_rank=4, **declared)
    target = dict(backbone, architectures=['Qwen3ForCausalLM'], hidden_size=target_hidden, vocab_size=target_vocab)
    for name, cfg in [('draft', raw), ('target', target)]:
        (tmp_path / name).mkdir()
        (tmp_path / name / 'config.json').write_text(json.dumps(cfg))
    args = dict(method='dspark', num_speculative_tokens=3, model=str(tmp_path / 'draft'),
                target_model=str(tmp_path / 'target'), dtype='float16',
                target_cache_cfg=CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=2))
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            SpecDecodeConfig.from_config(**args)
    else:
        assert SpecDecodeConfig.from_config(**args).dspark.draft_vocab_size == 16


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
            dspark=SimpleNamespace(draft_query_len=query_len, sample_from_anchor=sample_from_anchor),
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


@pytest.mark.parametrize('cache_kind', ['pageable', 'unknown'])
@pytest.mark.parametrize('dp,ep', [(1, 1), (2, 1), (1, 2)])
def test_dspark_context_materialization_layout(cache_kind, dp, ep):
    proposer = DSpark(SimpleNamespace(
        mask_token_id=99,
        target_layer_ids=(1, 5),
        num_speculative_tokens=5,
        dspark=SimpleNamespace(draft_query_len=5, sample_from_anchor=True),
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
    proposer.model = SimpleNamespace(supports_full_context_materialization=cache_kind == 'pageable')
    if cache_kind == 'unknown':
        with pytest.raises(ValueError, match='validated full-context cache geometry'):
            proposer._validate_context_materialization()
        return
    proposer._validate_context_materialization()
    extra = SimpleNamespace(
        target_hidden_states=target_hidden,
        num_rejected_tokens=torch.tensor([5, 3, 0]),
    )

    class NoNonzero(TorchDispatchMode):

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func != torch.ops.aten.nonzero.default
            return func(*args, **(kwargs or {}))

    with NoNonzero():
        context, hidden, lengths, starts = (
            proposer._prepare_context_materialization(inputs, extra))

    assert lengths.tolist() == [1, 3, 6]
    assert starts.tolist() == [11, 23, 36]
    assert context.is_decoding is False
    assert context.seq_length.tolist() == [6, 6, 6]
    assert context.input_ids.data_ptr() == inputs.input_ids.data_ptr()
    assert hidden.data_ptr() == target_hidden.data_ptr()
    assert hidden.shape == (18, 2)
    query = proposer._build_query_inputs(
        inputs, lengths, torch.tensor([7, 8, 9]), query_start_positions=starts)
    assert query.history_lengths.tolist() == [11, 23, 36]
    assert query.target_position_ids.reshape(3, 5)[:, 0].tolist() == [11, 23, 36]


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
        dspark=SimpleNamespace(draft_query_len=3, sample_from_anchor=True),
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
        dspark=SimpleNamespace(draft_query_len=3, sample_from_anchor=True),
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


# Plain TP, attention DP, and combined attention DP/TP are distinct paths.


@pytest.mark.parametrize('model', [None, ''])
def test_dspark_without_external_draft_fails_before_model_config(monkeypatch, model):
    def unexpected_model_config(*args, **kwargs):
        pytest.fail('DSpark must reject a missing draft before resolving model configs.')

    monkeypatch.setattr(ModelConfig, 'from_pretrained', unexpected_model_config)
    with pytest.raises(ValueError, match='explicit external draft model'):
        SpecDecodeConfig.from_config('dspark', 3, model, None, target_model='target')


@pytest.mark.parametrize('schema', [
    dict(model_type='deepseek_v4', dspark_block_size=8),
    dict(model_type='deepseek_v4', dspark_config={'block_size': 8}),
])
def test_dspark_rejects_v4_draft_and_target(schema):
    v4 = SimpleNamespace(**schema)
    with pytest.raises(ValueError, match='does not support DeepSeek-V4 or bundled drafts'):
        prepare_dspark_hf_config(v4)
    with pytest.raises(ValueError, match='does not support DeepSeek-V4 or bundled drafts'):
        validate_dspark_target_config(_raw_config(), v4, None)


@pytest.mark.parametrize('field,value', [
    ('sample_from_anchor', 'false'),
    ('draft_vocab_size', 0),
    ('aux_hidden_state_layer_ids', [1.9]),
    ('mask_token_id', True),
    ('markov_rank', 2.5),
])
def test_dspark_rejects_raw_hf_values_before_coercion(field, value):
    with pytest.raises(ValueError, match=field):
        cfg = prepare_dspark_hf_config(_raw_config(**{field: value}))
        parse_dspark_config(cfg, 3, target_num_layers=80)


@pytest.mark.parametrize('method,arch', [('dflash', 'Qwen3DSparkModel'), ('dspark', 'DFlashDraftModel')])
def test_block_draft_config_checks_device_and_method_architecture(monkeypatch, method, arch):
    cfg = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=2, device_type='ascend')

    def no_load(*args, **kwargs):
        pytest.fail('CUDA guard must run before checkpoint config loading')

    monkeypatch.setattr(ModelConfig, 'from_pretrained', no_load)
    with pytest.raises(ValueError, match='requires CUDA'):
        SpecDecodeConfig.from_config(method, 3, 'draft', cfg)
    cfg.device_type = 'cuda'
    with pytest.raises(ValueError, match='KV-cache quantization'):
        cfg.quant_policy = 8
        SpecDecodeConfig.from_config(method, 3, 'draft', cfg)
    cfg.quant_policy = 0
    monkeypatch.setattr(ModelConfig, 'from_pretrained',
                        lambda *args, **kwargs: SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch])))
    with pytest.raises(ValueError, match='requires draft architecture'):
        SpecDecodeConfig.from_config(method, 3, 'draft', cfg)


@pytest.mark.parametrize('kind,projection', [('gated', 'gate_proj'), ('rnn', 'joint_proj')])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_dspark_weight_validation_runs_after_all_shards(kind, projection, device):
    from lmdeploy.pytorch.weight_loader.model_weight_loader import process_weights_after_loading

    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    # Exercise the real per-shard DSpark loader without allocating an unrelated backbone.
    model = Qwen3DSparkModel.__new__(Qwen3DSparkModel)
    torch.nn.Module.__init__(model)
    model.markov_head = {'gated': GatedMarkovHead, 'rnn': RNNMarkovHead}[kind](8, 4, 2, 4, device=device)
    model.output_head = torch.nn.Linear(4, 4, bias=False, device=device)
    model.draft_id_to_target_id = torch.nn.Parameter(
        torch.zeros(4, dtype=torch.long, device=device), requires_grad=False)
    model.has_own_output_head = False
    model._loaded_dspark_weights = set()
    weights = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    model.load_weights(item for item in weights.items() if item[0].startswith('markov_head.markov_'))
    with pytest.raises(ValueError, match=rf'd2t.*{projection}'):
        process_weights_after_loading(model)
    model.load_weights([(n, w) for n, w in weights.items() if f'.{projection}.' in n])
    model.load_weights([('d2t', weights['draft_id_to_target_id']), ('lm_head.weight', weights['output_head.weight'])])
    process_weights_after_loading(model)
    model.set_output_head(None)
    assert model.map_draft_to_target(torch.tensor([0, 2], device=device)).tolist() == [1, 3]


@pytest.mark.parametrize('interval,expected', [(1, 12), (16, 96)])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_dspark_decode_scheduler_covers_query_crossing_page(interval, expected, device):
    from lmdeploy.pytorch.config import SchedulerConfig
    from lmdeploy.pytorch.messages import SequenceMeta
    from lmdeploy.pytorch.paging.scheduler import Scheduler
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    from lmdeploy.pytorch.strategies.ar_spec.engine import ARSpecEngineStrategy

    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    cfg = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=8)
    scfg = SchedulerConfig(max_batches=1, max_session_len=256, max_request_output_len=128,
                           prefill_interval=interval, eviction_type='recompute')
    strategy = ARSpecEngineStrategy(scfg, cfg, num_spec_tokens=5)
    scheduler = Scheduler(scfg, cfg, SequenceMeta(64, strategy=ARSequenceStrategy()))
    seq = scheduler.add_session(0).add_sequence(list(range(53)))
    assert scheduler.schedule(is_prefill=True).running == [seq]
    scheduler.activate_seqs([seq])
    assert seq.num_blocks == 1
    required = strategy.get_num_required_tokens()
    prealloc = strategy.get_prealloc_size(True)
    assert (required, prealloc) == (12, expected)
    assert scheduler.schedule_running([seq], required, prealloc) == [True]
    assert seq.num_blocks >= 2
    blocks = torch.as_tensor(scheduler.get_block_tables([seq])[0], device=device).unsqueeze(0)
    lengths = torch.tensor([6], device=device)
    inputs = ModelInputs(
        input_ids=torch.zeros((1, 6), dtype=torch.long, device=device),
        seq_length=lengths, history_lengths=torch.tensor([53], device=device),
        block_offsets=blocks, is_decoding=True, num_ignored_history=torch.zeros_like(lengths),
        max_q_seqlen=6, max_kv_seqlen=59, sum_kv_seqlen=59)
    proposer = DSpark(SimpleNamespace(mask_token_id=99, target_layer_ids=(1,),
                                     num_speculative_tokens=5, model_config=None,
                                     dspark=SimpleNamespace(draft_query_len=6)), device=device)
    query = proposer._build_query_inputs(inputs, lengths, torch.tensor([1], device=device))
    assert query.target_position_ids.tolist() == [list(range(59, 65))]
    if device == 'cuda':
        from lmdeploy.pytorch.kernels.cuda import fill_kv_cache

        keys = torch.arange(1, 7, dtype=torch.float16, device=device)[:, None, None].expand(-1, 1, 32).contiguous()
        target = torch.full((8, 64, 1, 32), -1, dtype=keys.dtype, device=device)
        draft = torch.full_like(target, -2)
        start = torch.zeros_like(lengths)
        fill_kv_cache(keys, None, target, None, start, lengths, inputs.history_lengths + lengths, 6, blocks)
        fill_kv_cache(keys, None, draft, None, start, lengths, query.history_lengths + lengths, 6, query.block_offsets)
        p0, p1 = blocks[0, :2].tolist()
        assert target[p0, 53:59, 0, 0].tolist() == list(range(1, 7))
        assert target[p1, 0, 0, 0].item() == -1
        assert draft[p0, 58:64, 0, 0].tolist() == [-2, 1, 2, 3, 4, 5]
        assert draft[p1, :2, 0, 0].tolist() == [6, -2]
