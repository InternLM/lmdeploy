# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from os import getenv
from typing import Optional, Tuple

import torch

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


@dataclass
class EPLBMetadata:
    phy2log: torch.Tensor  # (layers, num_physical_experts)
    log2phy: torch.Tensor  # (layers, num_logical_experts, X)
    expert_count: torch.Tensor
    num_phy_experts: int

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

        phy2log, log2phy, expert_count = rebalance_experts(
            weight=experts_statistic,
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=ep_size,
        )
        return EPLBMetadata(
            phy2log=phy2log,
            log2phy=log2phy,
            expert_count=expert_count,
            num_phy_experts=num_physical_experts,
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


def get_eplb_metadata_by_layer(layer_idx: int):
    global _global_eplb_metadata
    assert _global_eplb_metadata is not None
    return _global_eplb_metadata.log2phy[layer_idx], _global_eplb_metadata.phy2log[
        layer_idx], _global_eplb_metadata.expert_count[layer_idx]
