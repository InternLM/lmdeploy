# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Tiny compressed-tensors W4A16 test checkpoint generator.

The generator synthesizes a tiny ct checkpoint directory from config-only
geometry. Nothing downloads. All tensors come from ``torch.randn``, pushed
through the real quantization in ``lmdeploy.turbomind.ct_repack``. The same
seed reproduces byte-identical shards.

Geometry presets mirror the entries in ``tests/turbomind/linear/models.yaml``.
Keep the two lists in sync. Every preset line must keep the input dim a
multiple of 128. The output dim must stay a multiple of 8.

A generated directory holds one ``config.json`` plus one ``.pt`` shard per
linear, keyed ``model.layers.<i>.<name>``. The keys are placeholders with no
loader behind them.
"""

from __future__ import annotations

import json
import os

import torch

from lmdeploy.turbomind.ct_repack import GROUP_SIZE, WEIGHT_DTYPE, quantize_ct, validate_ct_geometry

# (name, out_features, in_features) per model, copied from models.yaml.
PRESETS = {
    # Llama-3-8B.
    'llama3_8b': (
        ('gate_up', 28672, 4096),
        ('down', 4096, 14336),
        ('qkv', 6144, 4096),
        ('o', 4096, 4096),
    ),
    # GLM-4-9B.
    'glm4_9b': (
        ('gate_up', 27392, 4096),
        ('down', 4096, 13696),
        ('qkv', 4608, 4096),
        ('o', 4096, 4096),
    ),
    # GLM-5.2. Dense blocks.
    'glm52': (
        ('dense_gate_up', 24576, 6144),
        ('dense_down', 6144, 12288),
        # MoE blocks.
        ('gate_up', 4096, 6144),
        ('down', 6144, 2048),
        ('shared_gate_up', 4096, 6144),
        ('shared_down', 6144, 2048),
        # MLA blocks.
        ('mla_q_a', 2048, 6144),
        ('mla_q_b', 36864, 2048),
        ('mla_kv_a', 576, 6144),
        ('mla_o', 6144, 36864),
        # Indexer blocks.
        ('index_q', 4096, 2048),
        ('index_k', 128, 6144),
        ('index_weight', 32, 6144),
    ),
}


def shard_key(layer: int, name: str) -> str:
    return f'model.layers.{layer}.{name}'


def select_lines(preset: str, only=None):
    """Resolve a preset to its linear list, optionally filtered by name."""

    lines = PRESETS[preset]
    if only is None:
        return lines
    wanted = set(only)
    kept = tuple(item for item in lines if item[0] in wanted)
    missing = wanted - {item[0] for item in kept}
    if missing:
        raise ValueError(f'unknown linear names for preset {preset}: {sorted(missing)}')
    return kept


def gen_tiny_ct(out_dir: str,
                preset: str = 'glm52',
                num_layers: int = 2,
                seed: int = 9,
                only=None) -> dict:
    """Write a tiny ct checkpoint directory. Returns the written config.

    ``only`` restricts the preset to the given linear names. Random data
    carries a per-group outlier on the first group element. This keeps the
    quantization scales away from zero.
    """
    lines = select_lines(preset, only)
    validate_ct_geometry(lines)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    layers = {}
    for layer in range(num_layers):
        for name, n, k in lines:
            w = torch.randn(n, k, dtype=WEIGHT_DTYPE)
            w.view(n, k // GROUP_SIZE, GROUP_SIZE)[:, :, 0] *= 3.0
            packed, scale, zero = quantize_ct(w)
            bias = torch.randn(n, dtype=WEIGHT_DTYPE) * 0.01
            key = shard_key(layer, name)
            torch.save(
                {
                    'weight_packed': packed,
                    'weight_scale': scale,
                    'weight_zero_point': zero,
                    'bias': bias,
                },
                os.path.join(out_dir, f'{key}.pt'),
            )
            layers[key] = {'out_features': n, 'in_features': k}
    config = {
        'format': 'compressed-tensors',
        'quant_config': {
            'scheme': 'W4A16',
            'bits': 4,
            'group_size': GROUP_SIZE,
            'symmetric': True,
            'pack_format': 'pack-quantized',
        },
        'activation_dtype': 'float16',
        'preset': preset,
        'num_layers': num_layers,
        'seed': seed,
        'layers': layers,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    return config


def load_shard(model_dir: str, key: str) -> dict:
    """Load one shard of a generated (or converted) directory by key."""
    return torch.load(os.path.join(model_dir, f'{key}.pt'), map_location='cpu', weights_only=True)


# Dense llama geometry scaled down to engine-test size. Every input dim is a
# multiple of GROUP_SIZE and every output dim a multiple of 128, which the
# AWQ weight loaders require even at TP=1 (their chunk alignment asserts on
# the checkpoint shard dims). head_dim stays a legal attention width.
TINY_HF_GEOMETRY = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=32,
)

_TINY_TOKENIZER_REQUIRED_FILES = ('tokenizer.json', 'tokenizer_config.json')
_TINY_TOKENIZER_FILES = _TINY_TOKENIZER_REQUIRED_FILES + ('special_tokens_map.json',)


def _vocab_size_from_tokenizer(tokenizer_src: str) -> int:
    """Read the largest vocab id plus one from the tokenizer files."""
    with open(os.path.join(tokenizer_src, 'tokenizer.json'), encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    size = len(tokenizer_json['model']['vocab'])
    for added in tokenizer_json.get('added_tokens', ()):
        size = max(size, added['id'] + 1)
    return size


def _token_id(tokenizer_src: str, token) -> int | None:
    """Resolve a tokenizer special token entry to its id.

    ``tokenizer_config.json`` may carry the token as a string or as a dict
    with the id; plain ints pass through. Unknown shapes return None so the
    model config omits the field.
    """
    if isinstance(token, int):
        return token
    if isinstance(token, dict):
        token = token.get('id')
        return token if isinstance(token, int) else None
    if isinstance(token, str):
        with open(os.path.join(tokenizer_src, 'tokenizer.json'), encoding='utf-8') as f:
            tokenizer_json = json.load(f)
        vocab = tokenizer_json['model']['vocab']
        if token in vocab:
            return vocab[token]
        for added in tokenizer_json.get('added_tokens', ()):
            if added.get('content') == token:
                return added['id']
    return None


def gen_tiny_ct_hf(out_dir: str, tokenizer_src: str, seed: int = 9, num_layers: int = 2) -> dict:
    """Write a tiny compressed-tensors W4A16 checkpoint in HF layout.

    The directory loads through the pytorch engine like any ct checkpoint:
    ``config.json`` carries the compressed-tensors quantization config, the
    safetensors file carries ct keys (``weight_packed``/``weight_scale``/
    ``weight_zero_point``) on every linear, and embedding, norms, and lm_head
    stay unquantized. Tokenizer files are copied from ``tokenizer_src``, so
    prompts tokenize with real text. The same seed reproduces byte-identical
    weights.
    """
    import shutil

    from safetensors.torch import save_file

    # tokenizer.json and tokenizer_config.json must exist; optional files
    # (special_tokens_map.json) copy only when the source carries them.
    for name in _TINY_TOKENIZER_REQUIRED_FILES:
        src = os.path.join(tokenizer_src, name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f'tokenizer file missing: {src}')
    os.makedirs(out_dir, exist_ok=True)
    for name in _TINY_TOKENIZER_FILES:
        src = os.path.join(tokenizer_src, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(out_dir, name))

    geometry = dict(TINY_HF_GEOMETRY)
    geometry['num_hidden_layers'] = num_layers
    hidden = geometry['hidden_size']
    heads = geometry['num_attention_heads']
    kv_heads = geometry['num_key_value_heads']
    head_dim = geometry['head_dim']
    inter = geometry['intermediate_size']
    vocab_size = _vocab_size_from_tokenizer(tokenizer_src)

    torch.manual_seed(seed)

    def quantized(name: str, n: int, k: int) -> dict:
        # Per-group outlier on the first group element keeps scales away
        # from zero, matching the placeholder generator.
        w = torch.randn(n, k, dtype=WEIGHT_DTYPE)
        w.view(n, k // GROUP_SIZE, GROUP_SIZE)[:, :, 0] *= 3.0
        packed, scale, zero = quantize_ct(w)
        return {
            f'{name}.weight_packed': packed,
            f'{name}.weight_scale': scale,
            f'{name}.weight_zero_point': zero,
        }

    tensors: dict[str, torch.Tensor] = {
        'model.embed_tokens.weight': torch.randn(vocab_size, hidden, dtype=WEIGHT_DTYPE) * 0.02,
    }
    for layer in range(num_layers):
        prefix = f'model.layers.{layer}'
        tensors.update(quantized(f'{prefix}.self_attn.q_proj', heads * head_dim, hidden))
        tensors.update(quantized(f'{prefix}.self_attn.k_proj', kv_heads * head_dim, hidden))
        tensors.update(quantized(f'{prefix}.self_attn.v_proj', kv_heads * head_dim, hidden))
        tensors.update(quantized(f'{prefix}.self_attn.o_proj', hidden, heads * head_dim))
        tensors.update(quantized(f'{prefix}.mlp.gate_proj', inter, hidden))
        tensors.update(quantized(f'{prefix}.mlp.up_proj', inter, hidden))
        tensors.update(quantized(f'{prefix}.mlp.down_proj', hidden, inter))
        tensors[f'{prefix}.input_layernorm.weight'] = torch.ones(hidden, dtype=WEIGHT_DTYPE)
        tensors[f'{prefix}.post_attention_layernorm.weight'] = torch.ones(hidden, dtype=WEIGHT_DTYPE)
    tensors['model.norm.weight'] = torch.ones(hidden, dtype=WEIGHT_DTYPE)
    tensors['lm_head.weight'] = torch.randn(vocab_size, hidden, dtype=WEIGHT_DTYPE) * 0.02
    save_file(tensors, os.path.join(out_dir, 'model.safetensors'))

    tokenizer_config = {}
    tok_cfg_path = os.path.join(tokenizer_src, 'tokenizer_config.json')
    if os.path.isfile(tok_cfg_path):
        with open(tok_cfg_path, encoding='utf-8') as f:
            tokenizer_config = json.load(f)
    config = {
        'architectures': ['LlamaForCausalLM'],
        'model_type': 'llama',
        'torch_dtype': 'float16',
        'hidden_size': hidden,
        'intermediate_size': inter,
        'num_hidden_layers': num_layers,
        'num_attention_heads': heads,
        'num_key_value_heads': kv_heads,
        'head_dim': head_dim,
        'vocab_size': vocab_size,
        'max_position_embeddings': 512,
        'rms_norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'tie_word_embeddings': False,
        'attention_bias': False,
        'bos_token_id': _token_id(tokenizer_src, tokenizer_config.get('bos_token')),
        'eos_token_id': _token_id(tokenizer_src, tokenizer_config.get('eos_token')),
        'quantization_config': {
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
                        'group_size': GROUP_SIZE,
                        'strategy': 'group',
                        'symmetric': True,
                        'dynamic': False,
                        'type': 'int',
                    },
                }
            },
            'ignore': [],
        },
        'seed': seed,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    return config
