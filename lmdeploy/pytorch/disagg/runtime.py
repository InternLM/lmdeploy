# Copyright (c) OpenMMLab. All rights reserved.
"""Worker ownership of PD registrations, peer layouts and native transfers."""
import json
from time import perf_counter

from lmdeploy.pytorch.disagg.backend.mooncake import MooncakeBackend
from lmdeploy.pytorch.disagg.conn.protocol import DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment, PDTransferResult
from lmdeploy.pytorch.disagg.schema import describe_pd_pools, execution_schema, validate_pd_peer
from lmdeploy.pytorch.engine.cache_engine.migration import build_cache_pool_assignments
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class PDWorkerRuntime:
    """Register complete cache allocations once; keep Store registration
    separate."""

    def __init__(self, target, draft, state, model_config, dist_config, spec_config):
        self.rank, self.tp_rank = target.rank, target.tp_rank
        self.cache_config = target.cache_config
        self.pages_per_block = target.block_cache_plan.kernel_blocks_per_logical_block
        groups = dict(target=target.gpu_allocation.pools)
        plans = [target.block_cache_plan]
        if draft is not None:
            groups['draft'] = draft.gpu_allocation.pools
            plans.append(draft.block_cache_plan)
        if state.compact_plan is not None:
            groups['state'] = state.unit_allocation.pools
            groups['auxiliary'] = state.auxiliary_allocation.pools
        else:
            groups['legacy_state'] = state.allocation.pools
        self.pools = tuple(pool for pools in groups.values() for pool in pools)
        self.pool_infos = describe_pd_pools(groups)
        self.schema = execution_schema(model_config, self.cache_config, dist_config.attn_tp, plans,
                                        model_config.state_cache_specs, spec_config)
        self.backend = MooncakeBackend()
        self.local_endpoint = None
        self.remote_endpoints = {}

    def p2p_initialize(self, request):
        """Register local cache pools and return the worker's transfer
        endpoint."""
        request = request.model_copy(update=dict(rank=self.rank))
        self.backend.p2p_initialize(request)
        for key, pool in enumerate(self.pools):
            if pool.nbytes:
                self.backend.register_memory_region(DistServeRegisterMRMessage(
                    protocol=request.protocol, remote_engine_id=request.remote_engine_id, mr_key=key,
                    addr=pool.tensor.data_ptr(), offset=0, length=pool.nbytes))
        self.local_endpoint = DistServeKVTransferEndpointInfo(
            protocol=request.protocol, endpoint_info=json.dumps(
                self.backend.endpoint_info(request.remote_engine_id, request.protocol)),
            version=2, cache_schema=self.schema, cache_pools=self.pool_infos)
        return self.local_endpoint

    def p2p_connect(self, remote_engine_id, endpoints):
        """Validate and connect to the peer endpoint for this attention TP
        rank."""
        remote = endpoints[self.tp_rank]
        validate_pd_peer(self.local_endpoint, remote)
        self.backend.p2p_connect(remote_engine_id, remote)
        self.remote_endpoints[remote_engine_id] = remote

    async def migrate(self, batch):
        """Transfer cache pools into destinations assigned by the scheduler.

        Return one transfer result per receive, preserving failed or unknown completion statuses for the coordinator.
        Legacy requests support only target cache pools and raise if a transfer does not complete successfully.
        """
        results = []
        for receive in batch.receives:
            handoff = receive.handoff
            remote = self.remote_endpoints[receive.remote_engine_id]
            if handoff.cache_schema != self.schema or handoff.block_size != self.cache_config.block_size:
                raise ValueError('PD handoff does not match the connected cache schema.')
            pairs = [(src * self.pages_per_block + page, dst * self.pages_per_block + page)
                     for src, dst in receive.block_pairs for page in range(self.pages_per_block)]
            groups = dict(target=pairs, draft=pairs)
            if receive.state_id is not None:
                groups['state'] = [(handoff.state_ids[self.tp_rank], receive.state_id)]
            if receive.auxiliary_id is not None:
                groups['auxiliary'] = [(handoff.auxiliary_id, receive.auxiliary_id)]
            if receive.legacy_state_id is not None:
                groups['legacy_state'] = [(handoff.legacy_state_id, receive.legacy_state_id)]
            for pool in self.pool_infos:
                if pool.group not in groups:
                    raise ValueError(f'PD handoff is missing {pool.group} destinations.')
            begin = perf_counter()
            plan = build_cache_pool_assignments(self.pool_infos, remote.cache_pools, (), group_pairs=groups)
            prepared = perf_counter()
            result = await self.backend.p2p_migrate(MigrationAssignment(
                protocol=batch.protocol, remote_engine_id=receive.remote_engine_id, batch=plan,
                transfer_id=handoff.transfer_id, deadline=receive.deadline))
            logger.info('PD %s: %d segments, %d bytes, plan %.3f ms, transfer %.3f ms, %s',
                         handoff.transfer_id, len(plan), sum(t.length for t in plan),
                         (prepared - begin) * 1000, (perf_counter() - prepared) * 1000, result.status)
            results.append(result)
        if batch.requests:
            # Existing callers only describe target blocks. They cannot safely
            # initialize draft or recurrent caches through this legacy path.
            if any(pool.group != 'target' for pool in self.pool_infos):
                raise ValueError('Hybrid and MTP PD require a full-prompt handoff.')
            for peer, pairs in batch.requests:
                remote = self.remote_endpoints[peer]
                plan = build_cache_pool_assignments(self.pool_infos, remote.cache_pools, pairs)
                result = await self.backend.p2p_migrate(MigrationAssignment(
                    protocol=batch.protocol, remote_engine_id=peer, batch=plan))
                if result.status != 'completed':
                    raise RuntimeError(result.error or 'Legacy PD transfer failed; caches remain quarantined.')
                results.append(result)
        return results

    def close(self):
        """Release transport resources after native transfers have drained.

        Peer layouts are cleared only after backend cleanup succeeds.
        """
        self.backend.close()
        self.remote_endpoints.clear()


def aggregate_transfer_results(rank_results, transfer_ids):
    """An RPC failure cannot prove that a worker stopped using its
    allocations."""
    merged = []
    for index, transfer_id in enumerate(transfer_ids):
        results = [rank[index] for rank in rank_results
                   if not isinstance(rank, BaseException) and len(rank) == len(transfer_ids)]
        if (len(results) != len(rank_results)
                or any(r.transfer_id != transfer_id or r.status == 'completion_unknown' for r in results)):
            status = 'completion_unknown'
        elif any(r.status == 'failed_drained' for r in results):
            status = 'failed_drained'
        else:
            status = 'completed'
        errors = [str(rank) for rank in rank_results if isinstance(rank, BaseException)]
        errors.extend(r.error for r in results if r.error)
        merged.append(PDTransferResult(transfer_id=transfer_id, status=status, error='; '.join(errors)))
    return merged
