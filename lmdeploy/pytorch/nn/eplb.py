# Copyright (c) OpenMMLab. All rights reserved.
import json
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lmdeploy.pytorch.envs import (
    eplb_experts_statistic_file,
    eplb_num_groups,
    eplb_num_redundant_experts,
    eplb_ranks_per_node,
)


def balanced_packing(weight: torch.Tensor, num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack expert groups with approximately balanced weights."""
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs
    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1), dtype=torch.int64, device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64)
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min((idx for idx in range(num_packs) if pack_items[idx] < groups_per_pack),
                       key=pack_weights.__getitem__)
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(weight: torch.Tensor, num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create redundant physical experts for the heaviest logical experts."""
    num_layers, num_log = weight.shape
    assert num_phy >= num_log
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(num_layers, 1)
    rank = torch.zeros(num_layers, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(num_layers, num_log, dtype=torch.int64, device=device)
    arange_layers = torch.arange(num_layers, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arange_layers, redundant_indices]
        logcnt[arange_layers, redundant_indices] += 1
    return phy2log, rank, logcnt


def _inverse(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(perm.shape))
    return inv


def rebalance_experts_hierarchical(weight: torch.Tensor, num_physical_experts: int, num_groups: int, num_nodes: int,
                                   num_gpus: int):
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)).flatten(-2)
    mlog2log = _inverse(log2mlog)
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = _inverse(phy2pphy)
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    node_offsets = torch.arange(0,
                                num_logical_experts,
                                num_logical_experts // num_nodes,
                                dtype=torch.int64,
                                device=weight.device)
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + node_offsets.view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(weight: torch.Tensor, num_replicas: int, num_groups: int, num_nodes: int,
                      num_gpus: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_layers, num_logical_experts = weight.shape
    weight = weight.float()
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)
    maxlogcnt = logcnt.max().item()
    log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt),
                         -1,
                         dtype=torch.int64,
                         device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(
        -1, phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1))
    return phy2log, log2phy, logcnt


def logical_to_all_physical_raw(logical_to_all_physical_map: torch.Tensor, layer_id: int,
                                logical_expert_id: int) -> list[int]:
    return [
        physical_expert_id for physical_expert_id in logical_to_all_physical_map[layer_id,
                                                                                 logical_expert_id].tolist()
        if physical_expert_id != -1
    ]


def _compute_gpu_id_of_physical_expert(physical_expert_id: int, num_local_physical_experts: int) -> int:
    return physical_expert_id // num_local_physical_experts


def _fair_choices(arr: list[int], k: int, r: random.Random) -> list[int]:
    quotient, remainder = divmod(k, len(arr))
    res = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(res)
    return res


def compute_logical_to_rank_dispatch_physical_map(
    logical_to_all_physical_map: torch.Tensor,
    num_gpus: int,
    num_physical_experts: int,
    seed: int = 42,
):
    r = random.Random(seed)
    num_local_physical_experts = num_physical_experts // num_gpus
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype
    logical_to_rank_dispatch_physical_map = torch.full(
        size=(num_gpus, num_layers, num_logical_experts),
        fill_value=-1,
        dtype=dtype,
        device=logical_to_all_physical_map.device,
    )

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            candidates = logical_to_all_physical_raw(logical_to_all_physical_map, layer_id, logical_expert_id)
            output_partial = logical_to_rank_dispatch_physical_map[:, layer_id, logical_expert_id]
            for gpu_id in range(num_gpus):
                same_gpu = [
                    physical_expert_id for physical_expert_id in candidates
                    if _compute_gpu_id_of_physical_expert(physical_expert_id, num_local_physical_experts) == gpu_id
                ]
                if same_gpu:
                    output_partial[gpu_id] = same_gpu[0]

            num_remain = torch.sum(output_partial == -1).item()
            output_partial[output_partial == -1] = torch.tensor(_fair_choices(candidates, k=num_remain, r=r),
                                                                dtype=dtype,
                                                                device=logical_to_all_physical_map.device)

    assert torch.all(logical_to_rank_dispatch_physical_map != -1)
    return logical_to_rank_dispatch_physical_map


@dataclass
class EPLBMetadata:
    physical_to_logical_map: torch.Tensor
    logical_to_all_physical_map: torch.Tensor
    logical_to_all_physical_map_num_valid: torch.Tensor
    logical_to_rank_dispatch_physical_map: torch.Tensor

    def num_physical_experts(self) -> int:
        return self.physical_to_logical_map.shape[1]

    def __post_init__(self):
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = self.logical_to_all_physical_map.shape
        num_layers_2, num_logical_experts_1 = self.logical_to_all_physical_map_num_valid.shape
        _, num_layers_3, num_logical_experts_2 = self.logical_to_rank_dispatch_physical_map.shape
        assert num_layers_0 == num_layers_1 == num_layers_2 == num_layers_3
        assert num_logical_experts_0 == num_logical_experts_1 == num_logical_experts_2
        assert num_physical_experts_0 == num_physical_experts_1

    @staticmethod
    def _init_raw(ep_size: int, physical_to_logical_map: torch.Tensor, logical_to_all_physical_map: torch.Tensor):
        _, num_physical_experts = physical_to_logical_map.shape
        logical_to_all_physical_map_padded = F.pad(
            logical_to_all_physical_map,
            (0, num_physical_experts - logical_to_all_physical_map.shape[-1]),
            value=-1,
        )
        logical_to_all_physical_map_num_valid = torch.count_nonzero(logical_to_all_physical_map != -1, dim=-1)
        return EPLBMetadata(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map_padded,
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            logical_to_rank_dispatch_physical_map=compute_logical_to_rank_dispatch_physical_map(
                logical_to_all_physical_map,
                num_gpus=ep_size,
                num_physical_experts=num_physical_experts,
            ),
        )

    @staticmethod
    def init(ep_size: int, num_routed_experts: int, num_hidden_layers: int):
        num_groups = eplb_num_groups
        weight_path = eplb_experts_statistic_file
        ranks_per_node = eplb_ranks_per_node
        num_redundant_experts = eplb_num_redundant_experts
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if weight_path is None:
            experts_statistic = torch.arange(num_routed_experts, dtype=torch.int32,
                                             device=device).flip(dims=(0, )).expand(num_hidden_layers, -1)
        else:
            try:
                with open(weight_path) as f:
                    experts_statistic = torch.tensor(json.load(f), dtype=torch.float32, device=device)
            except Exception as exc:
                raise RuntimeError(f'Load eplb experts statistic data failed, path: {weight_path}') from exc
            target_shape = torch.Size([num_hidden_layers, num_routed_experts])
            assert experts_statistic.shape == target_shape, f'Shape of {weight_path} must be {target_shape}'

        num_nodes = 1 if ep_size < ranks_per_node else ep_size // ranks_per_node
        num_physical_experts = num_routed_experts + num_redundant_experts
        physical_to_logical_map, logical_to_all_physical_map, _ = rebalance_experts(
            weight=experts_statistic,
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=ep_size,
        )
        return EPLBMetadata._init_raw(
            ep_size=ep_size,
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )


_global_eplb_metadata: EPLBMetadata | None = None


def init_global_eplb_metadata(ep_size: int, num_routed_experts: int, num_hidden_layers: int):
    global _global_eplb_metadata
    if _global_eplb_metadata is not None:
        raise RuntimeError('Global EPLB metadata has already been initialized.')
    _global_eplb_metadata = EPLBMetadata.init(ep_size=ep_size,
                                             num_routed_experts=num_routed_experts,
                                             num_hidden_layers=num_hidden_layers)


def get_global_eplb_metadata():
    global _global_eplb_metadata
    if _global_eplb_metadata is None:
        raise RuntimeError('Global EPLB metadata has not been initialized.')
    return _global_eplb_metadata


def get_eplb_phy2log_metadata_by_layer(layer_idx: int):
    return get_global_eplb_metadata().physical_to_logical_map[layer_idx]


@dataclass
class _EPLBDispatchInfo:
    partial_logical_to_rank_dispatch_physical_map: torch.Tensor
    partial_logical_to_all_physical_map: torch.Tensor
    partial_logical_to_all_physical_map_num_valid: torch.Tensor

    @classmethod
    def init_new(cls, ep_rank: int, layer_idx: int):
        eplb_metadata = get_global_eplb_metadata()
        return cls(
            partial_logical_to_rank_dispatch_physical_map=eplb_metadata.logical_to_rank_dispatch_physical_map[
                ep_rank, layer_idx, :],
            partial_logical_to_all_physical_map=eplb_metadata.logical_to_all_physical_map[layer_idx, :],
            partial_logical_to_all_physical_map_num_valid=eplb_metadata.logical_to_all_physical_map_num_valid[
                layer_idx, :],
        )


def topk_ids_logical_to_physical(topk_ids: torch.Tensor, info: _EPLBDispatchInfo | None) -> torch.Tensor:
    if info is None:
        return topk_ids
    original_shape = topk_ids.shape
    topk_ids = topk_ids.flatten()
    chosen_dispatch_index = (torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=topk_ids.device) %
                             info.partial_logical_to_all_physical_map_num_valid[topk_ids])
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]
    return topk_ids.view(original_shape)


class EPLBDispatchInfo:

    def __init__(self, info) -> None:
        self.info = info


class EPLBManager:

    @classmethod
    def init_global_eplb_metadata(cls, ep_size: int, num_routed_experts: int, num_hidden_layers: int):
        assert ep_size > 1, 'eplb requires ep_size > 1'
        init_global_eplb_metadata(ep_size=ep_size,
                                  num_routed_experts=num_routed_experts,
                                  num_hidden_layers=num_hidden_layers)

    @classmethod
    def num_physical_experts(cls) -> int:
        return get_global_eplb_metadata().num_physical_experts()

    @classmethod
    def topk_ids_logical_to_physical(cls, topk_ids: torch.Tensor, eplb_dispatch_info: EPLBDispatchInfo):
        return topk_ids_logical_to_physical(topk_ids=topk_ids, info=eplb_dispatch_info.info)

    @classmethod
    def get_dispatch_info(cls, ep_rank, layer_idx) -> EPLBDispatchInfo:
        return EPLBDispatchInfo(_EPLBDispatchInfo.init_new(ep_rank=ep_rank, layer_idx=layer_idx))
