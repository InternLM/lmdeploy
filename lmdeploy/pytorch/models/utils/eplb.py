# Copyright (c) OpenMMLab. All rights reserved.
import random
from dataclasses import dataclass
from os import getenv
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from lmdeploy.pytorch.distributed import get_ep_world_rank
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
            pack = min((i for i in range(num_packs) if pack_items[i] < groups_per_pack), key=pack_weights.__getitem__)
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


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

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(perm.shape))
        return inv

    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1) +
        torch.arange(0, num_logical_experts, num_logical_experts // num_nodes).cuda().view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(weight: torch.Tensor, num_replicas: int, num_groups: int, num_nodes: int,
                      num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_layers, num_logical_experts = weight.shape
    weight = weight.float()
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full((num_layers, num_logical_experts, maxlogcnt),
                                       -1,
                                       dtype=torch.int64,
                                       device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(
        -1, phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1))
    return phy2log, log2phy, logcnt


def logical_to_all_physical_raw(logical_to_all_physical_map, layer_id: int, logical_expert_id: int) -> List[int]:
    return [
        physical_expert_id for physical_expert_id in logical_to_all_physical_map[layer_id, logical_expert_id].tolist()
        if physical_expert_id != -1
    ]


def _compute_gpu_id_of_physical_expert(physical_expert_id: int, num_local_physical_experts: int) -> int:
    return physical_expert_id // num_local_physical_experts


def _fair_choices(arr: List, k: int, r: random.Random) -> List:
    quotient, remainder = divmod(k, len(arr))
    res = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(res)
    return res


# This is rarely called, so we use for loops for maximum clarity
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
    )

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            candidate_physical_expert_ids = (logical_to_all_physical_raw(logical_to_all_physical_map, layer_id,
                                                                         logical_expert_id))
            output_partial = logical_to_rank_dispatch_physical_map[:, layer_id, logical_expert_id]

            for gpu_id in range(num_gpus):
                same_gpu_physical_expert_ids = [
                    physical_expert_id for physical_expert_id in candidate_physical_expert_ids
                    if _compute_gpu_id_of_physical_expert(physical_expert_id, num_local_physical_experts) == gpu_id
                ]
                if len(same_gpu_physical_expert_ids) > 0:
                    output_partial[gpu_id] = same_gpu_physical_expert_ids[0]

            num_remain = torch.sum(output_partial == -1).item()
            output_partial[output_partial == -1] = torch.tensor(
                _fair_choices(candidate_physical_expert_ids, k=num_remain, r=r),
                dtype=dtype,
            )

    assert torch.all(logical_to_rank_dispatch_physical_map != -1)
    return logical_to_rank_dispatch_physical_map


@dataclass
class EPLBMetadata:
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)
    # (num_gpus, layers, num_logical_experts)
    logical_to_rank_dispatch_physical_map: torch.Tensor

    def num_physical_experts(self) -> int:
        return self.physical_to_logical_map.shape[1]

    def __post_init__(self):
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = (self.logical_to_all_physical_map.shape)
        num_layers_2, num_logical_experts_1 = (self.logical_to_all_physical_map_num_valid.shape)
        _, num_layers_3, num_logical_experts_2 = (self.logical_to_rank_dispatch_physical_map.shape)
        assert num_layers_0 == num_layers_1 == num_layers_2 == num_layers_3
        assert num_logical_experts_0 == num_logical_experts_1 == num_logical_experts_2
        assert num_physical_experts_0 == num_physical_experts_1

    @staticmethod
    def _init_raw(
        ep_size: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
    ):
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
    def init(num_routed_experts: int, num_hidden_layers: int):
        num_groups: int = int(getenv('EPLB_NUM_GROUPS', 4))
        weight_path: str = getenv('EPLB_EXPERTS_STATISTIC_FILE', None)
        ranks_per_node: int = int(getenv('RANKS_PER_NODES', 8))
        num_redundant_experts: int = int(getenv('EPLB_NUM_REDUNDANT_EXPERTS', 32))
        if weight_path is None:
            experts_statistic = torch.arange(num_routed_experts, dtype=torch.int32,
                                             device='cuda').flip(dims=(0, )).expand(num_hidden_layers, -1)
        else:
            try:
                import json
                with open(weight_path, 'r') as f:
                    experts_statistic = torch.tensor(json.load(f), dtype=torch.float32, device='cuda')
            except Exception:
                raise RuntimeError(f'Load eplb experts statistic data failed, path: {weight_path}')
        ep_size, _ = get_ep_world_rank()
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


_global_eplb_metadata: Optional[EPLBMetadata] = None


def init_global_eplb_metadata(num_routed_experts: int, num_hidden_layers: int):
    global _global_eplb_metadata
    assert _global_eplb_metadata is None
    _global_eplb_metadata = EPLBMetadata.init(num_routed_experts=num_routed_experts,
                                              num_hidden_layers=num_hidden_layers)


def get_global_eplb_metadata():
    global _global_eplb_metadata
    assert _global_eplb_metadata is not None
    return _global_eplb_metadata


def get_eplb_phy2log_metadata_by_layer(layer_idx: int):
    global _global_eplb_metadata
    assert _global_eplb_metadata is not None
    return _global_eplb_metadata.physical_to_logical_map[layer_idx]


@dataclass
class EPLBDispatchInfo:
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


def _topk_ids_logical_to_physical_random(topk_ids: torch.Tensor, info: Optional[EPLBDispatchInfo]) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()
    chosen_dispatch_index = (torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device) %
                             info.partial_logical_to_all_physical_map_num_valid[topk_ids])
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]
    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids


def topk_ids_logical_to_physical(topk_ids: torch.Tensor, info: Optional[EPLBDispatchInfo]) -> torch.Tensor:
    if info is None:
        return topk_ids
    return _topk_ids_logical_to_physical_random(topk_ids, info)
