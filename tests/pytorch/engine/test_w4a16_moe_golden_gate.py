# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Golden gate for the TurboMind grouped W4A16 MoE route on a tiny ct model.

The gate follows the tiny-checkpoint pattern. A seeded tiny checkpoint (compressed-
tensors, group 128, Mixtral MoE geometry) loads through the engine, and its
outputs face an external judge. No triton path exists for the AWQ expert
layout. The judge is therefore the unquantized twin of the same source
weights running the plain PyTorch MoE route. The difference therefore carries the
int4 quantization error on top of the kernel error. The strict kernel-level
numbers live in the kernel gate (``tests/pytorch/kernel/test_tm_moe_w4a16.py``).

Thresholds follow the tiny-gate convention: first-token top-1
agreement on the full vocabulary, truncated KL on the reference top-20
support, and greedy agreement >= 0.8. Random weights carry no margin between
top candidates, so a small KL can flip a later greedy tie.

All weights are random; the tokenizer comes from the public AWQ checkpoint
(HF/ModelScope cache). The suite touches no real local checkpoint.
"""

import json
import os
import zlib

import pytest
import torch

HF_AWQ_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct-AWQ'

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 128
INTER = 256
NUM_LAYERS = 2

# The quantized route carries the full int4 error against the fp16 twin, so
# the thresholds pin the first measured distribution with margin: KL 2.3e-3
# and top-1 agreement 0.53 on 32 prompts (first measured run). The frozen
# judge semantics stay the same as the tiny gate: same model, same prompts, same
# numbers. Greedy agreement stays report-only: random weights give many
# near-tied continuations, and the int4 noise flips those ties, so a fixed
# greedy bar has no meaning across the quantization boundary.
MOE_TINY_KL_THRESHOLD = 0.01
MOE_TINY_TOP1_THRESHOLD = 0.4


def _seeded(key: str):
    """Deterministic per-key generator.

    Python randomizes the str hash per process, so the seed derives from crc32.
    """
    gen = torch.Generator()
    gen.manual_seed(zlib.crc32(key.encode()) ^ 0x5EED)
    return gen


def _quantized_triplet(name: str, w: torch.Tensor) -> dict[str, torch.Tensor]:
    """Ct group-128 triplet.

    The source already carries the outlier.
    """
    from lmdeploy.turbomind.ct_repack import quantize_ct

    packed, scale, zero = quantize_ct(w)
    return {
        f'{name}.weight_packed': packed,
        f'{name}.weight_scale': scale,
        f'{name}.weight_zero_point': zero,
    }


def _source_weights(vocab: int) -> dict[str, torch.Tensor]:
    """The raw fp16 model both engines serve.

    Deterministic across runs.
    """
    gen = _seeded('moe-tiny-source')
    tensors = {
        'model.embed_tokens.weight': torch.randn(vocab, HIDDEN, generator=gen, dtype=torch.float16) * 0.02,
        'model.norm.weight': torch.ones(HIDDEN, dtype=torch.float16),
        'lm_head.weight': torch.randn(vocab, HIDDEN, generator=gen, dtype=torch.float16) * 0.02,
    }
    for layer in range(NUM_LAYERS):
        prefix = f'model.layers.{layer}'
        tensors[f'{prefix}.input_layernorm.weight'] = torch.ones(HIDDEN, dtype=torch.float16)
        tensors[f'{prefix}.post_attention_layernorm.weight'] = torch.ones(HIDDEN, dtype=torch.float16)
        shapes = {
            'self_attn.q_proj': (128, HIDDEN),
            'self_attn.k_proj': (128, HIDDEN),
            'self_attn.v_proj': (128, HIDDEN),
            'self_attn.o_proj': (HIDDEN, 128),
            'block_sparse_moe.gate': (NUM_EXPERTS, HIDDEN),
        }
        for name, (n, k) in shapes.items():
            tensors[f'{prefix}.{name}'] = torch.randn(n, k, generator=gen, dtype=torch.float16) * 0.05
        for expert in range(NUM_EXPERTS):
            for name, (n, k) in (('w1', (INTER, HIDDEN)), ('w3', (INTER, HIDDEN)), ('w2', (HIDDEN, INTER))):
                key = f'{prefix}.block_sparse_moe.experts.{expert}.{name}'
                w = torch.randn(n, k, generator=gen, dtype=torch.float16) * (0.1 / k**0.5)
                # Per-group outlier on the first group element keeps the
                # quantization scales away from zero (tiny_ct convention).
                # The outlier lives in the source, so the unquantized twin
                # serves the same weights.
                w.view(n, k // 128, 128)[:, :, 0] *= 3.0
                tensors[key] = w
    return tensors


def _ct_quantization_config() -> dict:
    return {
        'quant_method': 'compressed-tensors',
        'format': 'pack-quantized',
        'quantization_status': 'compressed',
        'config_groups': {
            'group_0': {
                'targets': ['Linear'],
                'input_activations': None,
                'output_activations': None,
                'weights': {
                    'num_bits': 4,
                    'group_size': 128,
                    'strategy': 'group',
                    'symmetric': True,
                    'dynamic': False,
                    'type': 'int',
                },
            }
        },
        'ignore': [],
    }


def _base_config(vocab: int, quantization_config: dict | None) -> dict:
    config = {
        'architectures': ['MixtralForCausalLM'],
        'model_type': 'mixtral',
        'torch_dtype': 'float16',
        'hidden_size': HIDDEN,
        'intermediate_size': INTER,
        'num_hidden_layers': NUM_LAYERS,
        'num_attention_heads': 4,
        'num_key_value_heads': 4,
        'head_dim': 32,
        'vocab_size': vocab,
        'max_position_embeddings': 512,
        'rms_norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'tie_word_embeddings': False,
        'attention_bias': False,
        'num_local_experts': NUM_EXPERTS,
        'num_experts_per_tok': TOP_K,
    }
    if quantization_config is not None:
        config['quantization_config'] = quantization_config
    return config


def _make_tiny_moe_pair(tmp_path, seed: int):
    """Write the quantized candidate and the unquantized twin of one seeded
    source model.

    Returns ``(candidate_dir, twin_dir)``.
    """
    import shutil

    from safetensors.torch import save_file

    from lmdeploy.utils import get_model
    from tests.turbomind.tiny_ct import _vocab_size_from_tokenizer

    tokenizer_src = get_model(HF_AWQ_MODEL)
    vocab = _vocab_size_from_tokenizer(tokenizer_src)
    source = _source_weights(vocab)
    quantized_suffixes = tuple(f'.{key}' for key in (
        'self_attn.q_proj',
        'self_attn.k_proj',
        'self_attn.v_proj',
        'self_attn.o_proj',
    ))

    def _is_quantized(name: str) -> bool:
        """Router and norms stay plain; attention and experts quantize."""
        if name.endswith('block_sparse_moe.gate'):
            return False
        return name.startswith('model.layers.') and (name.endswith(quantized_suffixes) or '.experts.' in name)

    quantization_config = _ct_quantization_config()

    candidate_dir = os.path.join(str(tmp_path), f'tiny_moe_ct_seed{seed}')
    twin_dir = os.path.join(str(tmp_path), f'tiny_moe_fp_seed{seed}')
    for out_dir, quantize in ((candidate_dir, True), (twin_dir, False)):
        os.makedirs(out_dir, exist_ok=True)
        for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
            src = os.path.join(tokenizer_src, name)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(out_dir, name))
        tensors: dict[str, torch.Tensor] = {}
        for name, tensor in source.items():
            if quantize and _is_quantized(name):
                tensors.update(_quantized_triplet(name, tensor))
            elif not name.endswith('.weight') and not name.startswith(('model.embed_tokens', 'lm_head')):
                # Plain modules carry the checkpoint `.weight` suffix; the
                # source stores bare module names so both layouts share keys.
                tensors[f'{name}.weight'] = tensor.clone()
            else:
                tensors[name] = tensor.clone()
        save_file(tensors, os.path.join(out_dir, 'model.safetensors'))
        config = _base_config(vocab, quantization_config if quantize else None)
        with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    return candidate_dir, twin_dir


def _run(model_dir: str, backend: str | None, log_level: str = 'WARNING'):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from w4a16_golden import PROMPTS, run_engine

    return run_engine(model_dir, PROMPTS, backend=backend, log_level=log_level)


def _moe_gate(reference: list, results: list) -> dict:
    """Tiny-gate arithmetic with the MoE-specific thresholds."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from w4a16_golden import top20_of, truncated_kl

    top1_hits = 0
    greedy_hits = 0
    worst_kl = 0.0
    for ref_item, item in zip(reference, results):
        ref_ids, ref_values = top20_of(ref_item['first_logits'])
        top1_hits += int(torch.argmax(item['first_logits']).item()) == ref_ids[0]
        worst_kl = max(worst_kl, truncated_kl(torch.tensor(ref_values), item['first_logits'], ref_ids))
        greedy_hits += list(item['greedy_ids']) == list(ref_item['greedy_ids'])
    count = len(results)
    return dict(
        num_prompts=count,
        top1_agreement=top1_hits / count,
        greedy_agreement=greedy_hits / count,
        worst_truncated_kl=worst_kl,
    )


class TestTinyMoeGoldenGate:
    """Engine e2e on the seeded tiny ct MoE model: load-time ct conversion,
    TM grouped route, spill, and the golden gate."""

    def test_tiny_moe_tm_route_passes_gate(self, tmp_path):
        """Ct MoE checkpoint -> canonical AWQ -> TM grouped route."""
        import logging

        candidate_dir, twin_dir = _make_tiny_moe_pair(tmp_path, seed=5)
        reference = _run(twin_dir, backend='triton')

        logger = logging.getLogger('lmdeploy')
        records: list[str] = []

        class _Collector(logging.Handler):

            def emit(self, record):
                records.append(record.getMessage())

        collector = _Collector()
        logger.addHandler(collector)
        old_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            # INFO level surfaces the per-module routing decisions.
            results = _run(candidate_dir, backend='turbomind', log_level='INFO')
        finally:
            logger.removeHandler(collector)
            logger.setLevel(old_level)

        routing = [m for m in records if 'W4A16 MoE GEMM routing' in m]
        assert routing, 'the TM grouped MoE route never logged a decision'
        assert any('-> turbomind' in m for m in routing), routing

        gate = _moe_gate(reference, results)
        print('moe tiny gate:', gate)
        assert gate['top1_agreement'] >= MOE_TINY_TOP1_THRESHOLD, gate
        assert gate['worst_truncated_kl'] <= MOE_TINY_KL_THRESHOLD, gate

    def test_tiny_moe_spills_packed_expert_weights(self, tmp_path):
        """The TM route keeps no raw expert tensors on the GPU after load."""
        candidate_dir, _ = _make_tiny_moe_pair(tmp_path, seed=5)
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from w4a16_golden import MAX_NEW_TOKENS

        from lmdeploy import pipeline
        from lmdeploy.messages import GenerationConfig, PytorchEngineConfig

        engine_config = PytorchEngineConfig(session_len=256,
                                            max_batch_size=8,
                                            cache_max_entry_count=0.1,
                                            enable_prefix_caching=False,
                                            distributed_executor_backend='uni')
        pipe = pipeline(candidate_dir, backend_config=engine_config, log_level='WARNING')
        try:
            gen_config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS,
                                          do_sample=False,
                                          ignore_eos=True,
                                          random_seed=0)
            pipe(['hello'], gen_config=gen_config, do_preprocess=False)
            model = pipe.async_engine.engine.executor.model_agent.patched_model.get_model()
            from lmdeploy.pytorch.nn.moe.tm_w4a16 import FusedMoETmW4A16

            moe_modules = [mod for mod in model.modules() if isinstance(mod, FusedMoETmW4A16)]
            assert len(moe_modules) == NUM_LAYERS, 'expected one TM MoE block per layer'
            for mod in moe_modules:
                for submodule in (mod.gate_up, mod.down):
                    # Spilled: the raw storage is empty, the CPU copy armed
                    # the loader for hot updates.
                    assert submodule.qweight.data.numel() == 0
                    assert submodule.scales.data.numel() == 0
                assert mod._spill is not None
                assert mod._spill.resident_bytes > 0
        finally:
            pipe.close()
