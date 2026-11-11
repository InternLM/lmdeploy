# Copyright (c) OpenMMLab. All rights reserved.
"""PD cache-pool description and byte-transfer planning."""

from collections.abc import Iterator, Mapping, Sequence
from math import prod

from lmdeploy.pytorch.disagg.conn.protocol import DistServeCachePoolInfo
from lmdeploy.pytorch.disagg.messages import AssignmentInstruct

from .layout import CachePool


def describe_cache_pools(pools: Sequence[CachePool], num_blocks: int) -> tuple[DistServeCachePoolInfo, ...]:
    """Validate owning pools and describe their transferable entry layout."""
    descriptions = []
    for pool_id, pool in enumerate(pools):
        tensor = pool.tensor
        if not tensor.is_contiguous():
            raise RuntimeError(f'PD migration requires contiguous cache pool {pool_id}.')
        if tensor.shape[pool.entry_axis] != num_blocks:
            raise RuntimeError(
                f'PD migration requires cache pool {pool_id} entry count {num_blocks}, '
                f'but got {tensor.shape[pool.entry_axis]}.')
        descriptions.append(
            DistServeCachePoolInfo(shape=tuple(tensor.shape),
                                   dtype=str(tensor.dtype),
                                   element_size=tensor.element_size(),
                                   entry_axis=pool.entry_axis))
    return tuple(descriptions)


def infer_remote_pool_without_metadata(local_pools: Sequence[DistServeCachePoolInfo],
                                       remote_num_blocks: int) -> tuple[DistServeCachePoolInfo, ...]:
    """Infer a single-pool endpoint shape for a peer without metadata."""
    if len(local_pools) != 1 or local_pools[0].entry_axis != 1:
        raise RuntimeError('The remote PD endpoint must provide cache-pool metadata for this layout.')

    local_pool = local_pools[0]
    remote_shape = list(local_pool.shape)
    remote_shape[local_pool.entry_axis] = remote_num_blocks
    return (DistServeCachePoolInfo(shape=tuple(remote_shape),
                                   dtype=local_pool.dtype,
                                   element_size=local_pool.element_size,
                                   entry_axis=local_pool.entry_axis), )


def validate_cache_pool_layouts(local_pools: Sequence[DistServeCachePoolInfo],
                                remote_pools: Sequence[DistServeCachePoolInfo], local_num_blocks: int,
                                remote_num_blocks: int) -> None:
    """Validate one-to-one logical payload compatibility across endpoints."""
    if len(local_pools) != len(remote_pools):
        raise RuntimeError('PD endpoints must expose the same number of cache pools, '
                           f'but got {len(local_pools)} local and {len(remote_pools)} remote pools.')

    for pool_id, (local, remote) in enumerate(zip(local_pools, remote_pools)):
        _validate_cache_pool(local, local_num_blocks, pool_id, 'local')
        _validate_cache_pool(remote, remote_num_blocks, pool_id, 'remote')
        if _payload_shape(local) != _payload_shape(remote):
            raise RuntimeError(f'PD cache pool {pool_id} payload shapes differ: '
                               f'{_payload_shape(local)} local and {_payload_shape(remote)} remote.')
        if local.dtype != remote.dtype or local.element_size != remote.element_size:
            raise RuntimeError(f'PD cache pool {pool_id} dtypes differ: '
                               f'{local.dtype} local and {remote.dtype} remote.')


def build_cache_pool_assignments(
        local_pools: Sequence[DistServeCachePoolInfo], remote_pools: Sequence[DistServeCachePoolInfo],
        block_pairs: Sequence[tuple[int, int]], *,
        group_pairs: Mapping[str, Sequence[tuple[int, int]]] | None = None) -> list[AssignmentInstruct]:
    """Map source and target block entries into contiguous byte assignments."""
    if len(local_pools) != len(remote_pools):
        raise RuntimeError('PD endpoints must expose the same number of cache pools.')

    pairs_by_group = group_pairs if group_pairs is not None else dict(target=block_pairs)
    runs_by_group = {group: _paired_runs(pairs) for group, pairs in pairs_by_group.items()}
    assignments = []
    for pool_id, (local, remote) in enumerate(zip(local_pools, remote_pools)):
        for remote_start, local_start, count in runs_by_group.get(local.group, ()):
            if local.entry_axis != remote.entry_axis:
                # Legacy layouts may split one entry differently on each side.
                for offset in range(count):
                    assignments.extend(_pair_segments(
                        pool_id, _entry_segments(local, local_start + offset, pool_id, 'source'),
                        _entry_segments(remote, remote_start + offset, pool_id, 'target')))
                continue
            local_count = local.shape[local.entry_axis]
            remote_count = remote.shape[remote.entry_axis]
            if (local_start < 0 or local_start + count > local_count
                    or remote_start < 0 or remote_start + count > remote_count):
                raise RuntimeError(f'PD entry run is outside cache pool {pool_id}.')
            if _payload_shape(local) != _payload_shape(remote) or local.element_size != remote.element_size:
                raise RuntimeError(f'PD cache pool {pool_id} payload sizes differ.')
            entry_bytes = prod(local.shape[local.entry_axis + 1:]) * local.element_size
            if not entry_bytes:
                continue
            for row in range(prod(local.shape[:local.entry_axis])):
                assignments.append(AssignmentInstruct(
                    mr_key=pool_id, source_offset=(row * local_count + local_start) * entry_bytes,
                    target_offset=(row * remote_count + remote_start) * entry_bytes, length=count * entry_bytes))
    return assignments


def _paired_runs(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Coalesce indices before expanding layers or allocating descriptors."""
    runs = []
    for remote, local in pairs:
        if runs and remote == runs[-1][0] + runs[-1][2] and local == runs[-1][1] + runs[-1][2]:
            start_remote, start_local, count = runs[-1]
            runs[-1] = (start_remote, start_local, count + 1)
        else:
            runs.append((int(remote), int(local), 1))
    return runs


def _validate_cache_pool(pool: DistServeCachePoolInfo, num_blocks: int, pool_id: int, endpoint: str) -> None:
    if pool.entry_axis < 0 or pool.entry_axis >= len(pool.shape):
        raise RuntimeError(f'PD {endpoint} cache pool {pool_id} has invalid entry_axis={pool.entry_axis}.')
    if pool.element_size <= 0:
        raise RuntimeError(f'PD {endpoint} cache pool {pool_id} has invalid element_size={pool.element_size}.')
    if any(dim < 0 for dim in pool.shape):
        raise RuntimeError(f'PD {endpoint} cache pool {pool_id} has invalid shape {pool.shape}.')
    if pool.shape[pool.entry_axis] != num_blocks:
        raise RuntimeError(f'PD {endpoint} cache pool {pool_id} must contain {num_blocks} block entries, '
                           f'but got {pool.shape[pool.entry_axis]}.')


def _payload_shape(pool: DistServeCachePoolInfo) -> tuple[int, ...]:
    return pool.shape[:pool.entry_axis] + pool.shape[pool.entry_axis + 1:]


def _entry_segments(pool: DistServeCachePoolInfo, entry: int, pool_id: int,
                    endpoint: str) -> tuple[tuple[int, int], ...]:
    num_entries = pool.shape[pool.entry_axis]
    if entry < 0 or entry >= num_entries:
        raise RuntimeError(
            f'PD {endpoint} block {entry} is outside cache pool {pool_id} with {num_entries} entries.')

    prefix_count = prod(pool.shape[:pool.entry_axis])
    suffix_nbytes = prod(pool.shape[pool.entry_axis + 1:]) * pool.element_size
    prefix_stride = num_entries * suffix_nbytes
    if suffix_nbytes == 0:
        return ()
    return tuple((prefix * prefix_stride + entry * suffix_nbytes, suffix_nbytes)
                 for prefix in range(prefix_count))


def _pair_segments(pool_id: int, source_segments: Sequence[tuple[int, int]],
                   target_segments: Sequence[tuple[int, int]]) -> Iterator[AssignmentInstruct]:
    source_id = target_id = 0
    source_used = target_used = 0
    while source_id < len(source_segments) and target_id < len(target_segments):
        source_offset, source_length = source_segments[source_id]
        target_offset, target_length = target_segments[target_id]
        length = min(source_length - source_used, target_length - target_used)
        yield AssignmentInstruct(mr_key=pool_id,
                                 target_offset=target_offset + target_used,
                                 source_offset=source_offset + source_used,
                                 length=length)

        source_used += length
        target_used += length
        if source_used == source_length:
            source_id += 1
            source_used = 0
        if target_used == target_length:
            target_id += 1
            target_used = 0

    if source_id != len(source_segments) or target_id != len(target_segments):
        raise RuntimeError(f'PD cache pool {pool_id} source and target payload sizes differ.')
