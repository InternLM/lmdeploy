# Copyright (c) OpenMMLab. All rights reserved.
"""Execution identity and physical layouts for PD peers."""
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING

from .conn.protocol import DistServeCachePoolInfo, DistServeKVTransferEndpointInfo

if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.cache_engine.layout import CachePool


def execution_schema(model_config, cache_config, attn_tp, block_plans, state_specs, spec_config=None) -> str:
    """Hash execution semantics, excluding capacities and DP/EP deployment
    size."""
    target = model_config.hf_config.to_dict()
    # A local path is not a weight revision. Operators use the same model_name
    # (and revision in HF config) for replicas of the same deployment.
    target.pop('_name_or_path', None)
    contract = dict(model=target, dtype=str(model_config.dtype), block_size=cache_config.block_size,
                    kernel_block_size=cache_config.kernel_block_size, quant_policy=cache_config.quant_policy,
                    attn_tp=attn_tp, blocks=[[asdict(s) for s in p.tensor_specs] for p in block_plans],
                    state=[asdict(s) for s in state_specs])
    if spec_config is not None:
        draft = spec_config.model_config.hf_config.to_dict()
        draft.pop('_name_or_path', None)
        contract['mtp'] = dict(model=draft, method=spec_config.method, width=spec_config.num_speculative_tokens,
                               dtype=str(spec_config.model_config.dtype))
    encoded = json.dumps(contract, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def describe_pd_pools(groups: Mapping[str, Sequence['CachePool']]) -> tuple[DistServeCachePoolInfo, ...]:
    """Describe every owning allocation, independent of Store transfer
    policy."""
    result = []
    for group, pools in groups.items():
        for pool in pools:
            tensor = pool.tensor
            if not tensor.is_contiguous():
                raise ValueError(f'PD {group} pool must be contiguous.')
            result.append(DistServeCachePoolInfo(shape=tuple(tensor.shape), dtype=str(tensor.dtype),
                                                 element_size=tensor.element_size(), entry_axis=pool.entry_axis,
                                                 group=group))
    return tuple(result)


def validate_pd_peer(local: DistServeKVTransferEndpointInfo, remote: DistServeKVTransferEndpointInfo) -> None:
    """Reject incompatible peers before registering any receive operation."""
    if remote.version != 2 or local.schema != remote.schema:
        raise ValueError('PD requires version 2 peers with matching model, MTP and attention TP schemas.')
    if remote.cache_pools is None or len(local.cache_pools) != len(remote.cache_pools):
        raise ValueError('PD peers must expose the same ordered cache groups.')
    for lhs, rhs in zip(local.cache_pools, remote.cache_pools):
        axis = lhs.entry_axis
        if (lhs.group != rhs.group or axis != rhs.entry_axis or lhs.dtype != rhs.dtype
                or lhs.element_size != rhs.element_size
                or lhs.shape[:axis] + lhs.shape[axis + 1:] != rhs.shape[:axis] + rhs.shape[axis + 1:]):
            raise ValueError(f'PD {lhs.group} cache layout differs between peers.')
        if lhs.shape[axis] < 0 or rhs.shape[axis] < 0:
            raise ValueError('PD cache capacities must be nonnegative.')
