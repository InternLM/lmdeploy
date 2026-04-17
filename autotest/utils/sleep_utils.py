from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from lmdeploy.utils import serialize_state_dict
from safetensors.torch import safe_open
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

UPDATE_WEIGHTS_CUDA_DEVICE_ENV = 'LMDEPLOY_UPDATE_WEIGHTS_CUDA_DEVICE'

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
    from lmdeploy.turbomind.deploy.converter import get_input_model_registered_name
    from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS

    root = str(model_dir.resolve())
    try:
        input_model_name = get_input_model_registered_name(root, 'hf')
        if input_model_name == 'qwen3_5-moe':
            raise RuntimeError(
                'turbomind update_weights is unsupported for qwen3_5-moe in the current server build: '
                'server-side StateDictLoader has no `index`, but Qwen3_5MoeModel.readers() accesses loader.index')
        input_model_cls = INPUT_MODELS.get(input_model_name)
        input_model = input_model_cls(model_path=root, tokenizer_path=root)
    except Exception as e:
        raise RuntimeError(
            f'turbomind update_weights: failed to build input_model readers for {model_dir}: {e}') from e

    dev_idx = resolve_update_weights_cuda_device_index()
    device = torch.device('cuda', dev_idx)
    with torch.cuda.device(dev_idx):
        it = iter(dict(reader.params) for _, reader in input_model.readers())
        try:
            chunk = next(it)
        except StopIteration:
            raise RuntimeError(f'no turbomind weight chunks to emit under {model_dir}') from None

        for cpu_dict_next in it:
            seg_gpu = {k: v.to(device, non_blocking=True) for k, v in chunk.items()}
            try:
                emit_segment(serialize_state_dict(seg_gpu), False)
            finally:
                del seg_gpu
                torch.cuda.empty_cache()
            chunk = cpu_dict_next

        seg_gpu = {k: v.to(device, non_blocking=True) for k, v in chunk.items()}
        try:
            emit_segment(serialize_state_dict(seg_gpu), True)
        finally:
            del seg_gpu
            torch.cuda.empty_cache()
