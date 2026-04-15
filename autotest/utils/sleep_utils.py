from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from lmdeploy.utils import serialize_state_dict

UPDATE_WEIGHTS_CUDA_DEVICE_ENV = 'LMDEPLOY_UPDATE_WEIGHTS_CUDA_DEVICE'

# Same layer id regex as ``Qwen3_5ReaderMixin.attn_layer_pattern`` (turbomind deploy).
QWEN_TURBOMIND_LAYER_PATTERN = r'(?:model\.language_model\.|model\.)layers\.([0-9]+)\.'

LEVEL2_GREEDY_MESSAGES = [{'role': 'user', 'content': '424242'}]
LEVEL2_MAX_TOKENS = 64
LEVEL2_BASELINE_RUNS = 3
MAX_SINGLE_CHAR_FRACTION = 0.75


def resolve_update_weights_cuda_device_index() -> int:
    raw = os.environ.get(UPDATE_WEIGHTS_CUDA_DEVICE_ENV, '').strip()
    if not raw:
        return torch.cuda.current_device()
    try:
        idx = int(raw)
    except ValueError as e:
        raise AssertionError(
            f'{UPDATE_WEIGHTS_CUDA_DEVICE_ENV} must be an int, got {raw!r}') from e
    n = torch.cuda.device_count()
    assert 0 <= idx < n, (
        f'{UPDATE_WEIGHTS_CUDA_DEVICE_ENV}={idx} out of range for cuda.device_count()={n}')
    return idx


def resolve_hf_checkpoint_dir(config: dict, model_case: str) -> Path:
    if os.environ.get('LMDEPLOY_USE_MODELSCOPE', 'False') == 'True':
        return Path(model_case)
    return Path(config['model_path']) / model_case


def shard_paths(model_dir: Path) -> tuple[str, list[Path]]:
    if (model_dir / SAFE_WEIGHTS_NAME).is_file():
        return 'safetensors', [model_dir / SAFE_WEIGHTS_NAME]
    if (model_dir / SAFE_WEIGHTS_INDEX_NAME).is_file():
        with open(model_dir / SAFE_WEIGHTS_INDEX_NAME, encoding='utf-8') as f:
            index = json.load(f)
        paths = sorted(set(index['weight_map'].values()))
        return 'safetensors', [model_dir / p for p in paths]
    if (model_dir / WEIGHTS_NAME).is_file():
        return 'pytorch', [model_dir / WEIGHTS_NAME]
    if (model_dir / WEIGHTS_INDEX_NAME).is_file():
        with open(model_dir / WEIGHTS_INDEX_NAME, encoding='utf-8') as f:
            index = json.load(f)
        paths = sorted(set(index['weight_map'].values()))
        return 'pytorch', [model_dir / p for p in paths]
    raise FileNotFoundError(f'No HF weights under {model_dir}')


def _safetensors_weight_index(model_dir: Path, shards: list[Path]) -> dict[str, str]:
    """``weight_name -> shard_basename`` like ``SafetensorsLoader``."""
    if (model_dir / SAFE_WEIGHTS_INDEX_NAME).is_file():
        with open(model_dir / SAFE_WEIGHTS_INDEX_NAME, encoding='utf-8') as f:
            return dict(json.load(f)['weight_map'])
    index: dict[str, str] = {}
    for shard in shards:
        fn = shard.name
        with safe_open(str(shard), framework='pt') as f:
            for k in f.keys():
                index[k] = fn
    return index


def _qwen35_moe_map_packed_key(name: str) -> str:
    return re.sub(r'(mlp\.experts\.(?:gate_up|down)_proj)$', r'\1.weight', name)


def _qwen35_moe_name_mapper(index: dict[str, str]) -> Callable[[str], str] | None:
    keys = index.keys()
    if any('mlp.experts.gate_up_proj' in x for x in keys) and any('mlp.experts.down_proj' in x for x in keys):
        return _qwen35_moe_map_packed_key
    return None


def _turbomind_layer_param_counts(index: dict[str, str], pattern: str) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for k in index.keys():
        m = re.findall(pattern, k)
        if m:
            counts[int(m[0])] += 1
    return dict(counts)


def _count_safetensors_turbomind_chunks(
    shards: list[Path],
    index: dict[str, str],
    pattern: str,
    item_count: dict[int, int],
) -> int:
    n = 0
    params: dict[int, int] = defaultdict(int)
    for shard in shards:
        filename = shard.name
        with safe_open(str(shard), framework='pt') as f:
            misc_keys: list[str] = []
            for k in f.keys():
                if k not in index or index[k] != filename:
                    continue
                m = re.findall(pattern, k)
                if not m:
                    misc_keys.append(k)
                else:
                    idx = int(m[0])
                    params[idx] += 1
                    if params[idx] == item_count[idx]:
                        n += 1
                        del params[idx]
            if misc_keys:
                n += 1
    if params:
        raise RuntimeError(
            f'turbomind weight packing (dry-run): incomplete layers {sorted(params.keys())}')
    return n


def _iterate_safetensors_turbomind_cpu_chunks(
    shards: list[Path],
    index: dict[str, str],
    pattern: str,
    item_count: dict[int, int],
    map_key: Callable[[str], str],
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield CPU state dicts in the same order as ``SafetensorsLoader.items`` (one decoder layer or misc)."""
    params: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    for shard in shards:
        filename = shard.name
        with safe_open(str(shard), framework='pt') as f:
            misc_keys: list[str] = []
            for k in f.keys():
                if k not in index or index[k] != filename:
                    continue
                m = re.findall(pattern, k)
                if not m:
                    misc_keys.append(k)
                else:
                    idx = int(m[0])
                    bucket = params[idx]
                    bucket[map_key(k)] = f.get_tensor(k)
                    if len(bucket) == item_count[idx]:
                        yield dict(params.pop(idx))
            if misc_keys:
                yield {k: f.get_tensor(k) for k in misc_keys}
    if params:
        raise RuntimeError(f'turbomind weight packing: incomplete layers {sorted(params.keys())}')


def load_shard_tensors(kind: str, path: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if kind == 'safetensors':
        with safe_open(str(path), framework='pt') as f:
            for key in f.keys():
                out[key] = f.get_tensor(key)
    else:
        state = torch.load(str(path), weights_only=True, map_location='cpu')
        try:
            out.update(state)
        finally:
            del state
    return out


def assistant_content_from_openai_completion_dict(output: dict) -> str:
    choices = output.get('choices') or []
    assert len(choices) == 1, f'expected 1 choice, got {len(choices)}'
    msg = choices[0].get('message') or {}
    return (msg.get('content') or '').strip()


def assert_assistant_not_degenerate(content: str, *, label: str) -> None:
    assert content, f'{label}: empty assistant content'
    compact = content.replace('\n', ' ').strip()
    assert len(set(compact)) >= 4, (
        f'{label}: degenerate assistant text (too few distinct chars): {content!r}')
    top_cnt = Counter(compact).most_common(1)[0][1]
    assert top_cnt / len(compact) <= MAX_SINGLE_CHAR_FRACTION, (
        f'{label}: one token/char dominates assistant text: {content!r}')


def level2_update_weights_request_dict(serialized_data: object, finished: bool) -> dict[str, Any]:
    return {
        'serialized_named_tensors': serialized_data,
        'finished': finished,
    }


def assert_chat_decode_unchanged(ref: dict, cur: dict, *, label: str) -> None:
    a, b = assistant_content_from_openai_completion_dict(ref), assistant_content_from_openai_completion_dict(cur)
    assert a == b, f'{label}: assistant content changed\n before={a!r}\n after={b!r}'
    rt = ref.get('usage', {}).get('completion_tokens')
    ct = cur.get('usage', {}).get('completion_tokens')
    assert rt == ct, f'{label}: completion_tokens changed {rt} -> {ct}'
    rfr = ref['choices'][0].get('finish_reason')
    cfr = cur['choices'][0].get('finish_reason')
    if rfr is not None and cfr is not None:
        assert rfr == cfr, f'{label}: finish_reason changed {rfr!r} -> {cfr!r}'


def apply_serialized_hf_segments_for_level2_weights(
    model_dir: Path,
    emit_segment: Callable[[Any, bool], None],
) -> None:
    kind, shards = shard_paths(model_dir)
    num_segment = len(shards)
    dev_idx = resolve_update_weights_cuda_device_index()
    device = torch.device('cuda', dev_idx)
    with torch.cuda.device(dev_idx):
        for seg_idx in range(num_segment):
            cpu_dict = load_shard_tensors(kind, shards[seg_idx])
            seg_gpu = {k: v.to(device, non_blocking=True) for k, v in cpu_dict.items()}
            del cpu_dict
            serialized_data = serialize_state_dict(seg_gpu)
            del seg_gpu
            torch.cuda.empty_cache()
            emit_segment(serialized_data, seg_idx == num_segment - 1)


def apply_serialized_hf_segments_for_turbomind_level2_weights(
    model_dir: Path,
    emit_segment: Callable[[Any, bool], None],
) -> None:
    """Upload HF weights in **layer-sized** chunks (TurboMind ``update_params`` / ``StateDictLoader``)."""
    kind, shards = shard_paths(model_dir)
    if kind != 'safetensors':
        raise RuntimeError(
            f'turbomind update_weights packing is only implemented for safetensors checkpoints (got {kind!r})')
    index = _safetensors_weight_index(model_dir, shards)
    mapper = _qwen35_moe_name_mapper(index)
    map_key: Callable[[str], str] = mapper if mapper is not None else (lambda x: x)
    item_count = _turbomind_layer_param_counts(index, QWEN_TURBOMIND_LAYER_PATTERN)
    n = _count_safetensors_turbomind_chunks(shards, index, QWEN_TURBOMIND_LAYER_PATTERN, item_count)
    if n <= 0:
        raise RuntimeError(f'no turbomind weight chunks to emit under {model_dir}')
    dev_idx = resolve_update_weights_cuda_device_index()
    device = torch.device('cuda', dev_idx)
    seg_i = 0
    with torch.cuda.device(dev_idx):
        for cpu_dict in _iterate_safetensors_turbomind_cpu_chunks(
                shards, index, QWEN_TURBOMIND_LAYER_PATTERN, item_count, map_key):
            seg_gpu = {k: v.to(device, non_blocking=True) for k, v in cpu_dict.items()}
            del cpu_dict
            serialized_data = serialize_state_dict(seg_gpu)
            del seg_gpu
            torch.cuda.empty_cache()
            emit_segment(serialized_data, seg_i == n - 1)
            seg_i += 1
    assert seg_i == n, f'chunk count mismatch: emitted={seg_i} expected={n}'
