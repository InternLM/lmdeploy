# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise real GPU RDMA bytes, fragmented plans and peer failure recovery.

With scatter tickets, --fail-peer checks failed-drained completion and reads from a replacement peer using the same
receiver engine. Older APIs cannot prove drain and exit without unregistering their allocations.
"""
import argparse
import asyncio
import json
import multiprocessing as mp
import os
import time

import torch

from lmdeploy.pytorch.disagg.backend.mooncake import MooncakeBackend
from lmdeploy.pytorch.disagg.config import MooncakeEngineConfig
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeInitRequest,
    DistServeKVTransferEndpointInfo,
    MigrationProtocol,
)
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment
from lmdeploy.pytorch.disagg.schema import describe_pd_pools
from lmdeploy.pytorch.engine.cache_engine.layout import CachePool
from lmdeploy.pytorch.engine.cache_engine.migration import build_cache_pool_assignments


def allocation(source, device):
    torch.cuda.set_device(device)
    capacity = 1024 if source else 1027
    shapes = dict(target=(60, capacity, 4096), draft=(2, capacity, 4096),
                  state=(2, 8, 1024 * 1024), auxiliary=(4, 8, 4096))
    groups = {}
    for name, shape in shapes.items():
        tensor = torch.full(shape, 255, dtype=torch.uint8, device=device)
        if source:
            pattern = (torch.arange(shape[0], device=device)[:, None] * 17
                       + torch.arange(shape[1], device=device)[None, :]) % 251
            tensor.copy_(pattern[..., None])
        groups[name] = [CachePool(tensor, 1, transfer_ranges=())]
    torch.cuda.synchronize()
    return groups


def register(groups, source, device):
    local, peer = ('source', 'receiver') if source else ('receiver', 'source')
    config = MooncakeEngineConfig(tp_size=1, ep_size=1, dp_size=1, pp_size=1, dp_rank=0,
                                  block_size=64, num_cpu_blocks=0, num_gpu_blocks=1024)
    backend = MooncakeBackend()
    backend.p2p_initialize(DistServeInitRequest(local_engine_id=local, remote_engine_id=peer,
                                               local_engine_config=config, remote_engine_config=config,
                                               protocol=MigrationProtocol.RDMA, rank=device))
    for key, pools in enumerate(groups.values()):
        tensor = pools[0].tensor
        backend.register_memory_region(DistServeRegisterMRMessage(
            protocol=MigrationProtocol.RDMA, remote_engine_id=peer, mr_key=key,
            addr=tensor.data_ptr(), offset=0, length=tensor.nbytes))
    endpoint = DistServeKVTransferEndpointInfo(
        protocol=MigrationProtocol.RDMA, version=2,
        endpoint_info=json.dumps(backend.endpoint_info(peer, MigrationProtocol.RDMA)),
        cache_pools=describe_pd_pools(groups))
    return backend, endpoint


def source_process(pipe, device):
    groups = allocation(True, device)
    backend, endpoint = register(groups, True, device)
    pipe.send(endpoint)
    assert pipe.recv() == 'stop'
    backend.close()


async def receive(args, pipe, source):
    groups = allocation(False, args.destination_gpu)
    backend, endpoint = register(groups, False, args.destination_gpu)
    if not pipe.poll(60):
        raise TimeoutError('Source did not publish its RDMA endpoint.')
    remote = pipe.recv()
    backend.p2p_connect('source', remote)
    for blocks in (128, 512):
        for layout in ('continuous', 'unmerged', 'fragmented'):
            fragmented = layout == 'fragmented'
            pairs = [(i, 2 * i + 1 if fragmented else i + 1) for i in range(blocks)]
            if layout == 'unmerged':
                # Same contiguous destinations, one descriptor per entry/row.
                pairs = pairs[::2] + pairs[1::2]
            group_pairs = dict(target=pairs, draft=pairs, state=[(2, 3)], auxiliary=[(2, 3)])
            for pools in groups.values():
                pools[0].tensor.fill_(255)
            torch.cuda.synchronize()
            plan_ms = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                plan = build_cache_pool_assignments(endpoint.cache_pools, remote.cache_pools, (),
                                                     group_pairs=group_pairs)
                plan_ms.append((time.perf_counter() - start) * 1000)
            assignment = MigrationAssignment(protocol=MigrationProtocol.RDMA, remote_engine_id='source',
                                              transfer_id=f'{blocks}-{layout}', batch=plan)
            durations = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                result = await backend.p2p_migrate(assignment)
                durations.append((time.perf_counter() - start) * 1000)
                assert result.status == 'completed', result
            for name, pools in groups.items():
                actual = pools[0].tensor.cpu()
                expected = torch.full_like(actual, 255)
                for src, dst in group_pairs[name]:
                    pattern = (torch.arange(actual.shape[0]) * 17 + src) % 251
                    expected[:, dst].copy_(pattern[:, None])
                assert torch.equal(actual, expected), name
            print(json.dumps(dict(blocks=blocks, layout=layout, segments=len(plan),
                                  bytes=sum(t.length for t in plan), plan_ms=plan_ms,
                                  transfer_ms=durations, bytes_equal=True)), flush=True)
    pipe.send('stop')
    source.join(30)
    assert source.exitcode == 0
    if args.fail_peer:
        start = time.perf_counter()
        result = await backend.p2p_migrate(assignment.model_copy(update=dict(batch=assignment.batch[:1])))
        failure_ms = (time.perf_counter() - start) * 1000
        expected_status = 'failed_drained' if backend.shared.has_drain_ticket else 'completion_unknown'
        assert result.status == expected_status, result
        if result.status == 'completion_unknown':
            try:
                backend.close()
            except RuntimeError as error:
                assert 'unknown' in str(error)
            else:
                raise AssertionError('Undrained allocations were unregistered.')
            print(json.dumps(dict(peer_failure=result.model_dump(), failure_ms=failure_ms,
                                  allocations_retained=True)), flush=True)
            os._exit(0)

        engine = backend.shared.engine
        ctx = mp.get_context('spawn')
        parent, child = ctx.Pipe()
        replacement = ctx.Process(target=source_process, args=(child, args.source_gpu))
        replacement.start()
        try:
            assert parent.poll(60), 'Replacement peer did not publish its endpoint.'
            backend.p2p_connect('source', parent.recv())
            expected = {name: pools[0].tensor.cpu() for name, pools in groups.items()}
            for pools in groups.values():
                pools[0].tensor.fill_(255)
            torch.cuda.synchronize()
            recovered = await backend.p2p_migrate(assignment.model_copy(update=dict(transfer_id='replacement')))
            assert recovered.status == 'completed', recovered
            assert backend.shared.engine is engine
            for name, pools in groups.items():
                assert torch.equal(pools[0].tensor.cpu(), expected[name]), name
            parent.send('stop')
            replacement.join(30)
            assert replacement.exitcode == 0
        finally:
            if replacement.is_alive():
                replacement.terminate()
            replacement.join(10)
        print(json.dumps(dict(peer_failure=result.model_dump(), failure_ms=failure_ms,
                              recovered_without_engine_restart=True, bytes_equal=True)), flush=True)
    backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-gpu', type=int, default=0)
    parser.add_argument('--destination-gpu', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--fail-peer', action='store_true')
    args = parser.parse_args()
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    source = ctx.Process(target=source_process, args=(child, args.source_gpu))
    source.start()
    try:
        asyncio.run(receive(args, parent, source))
    finally:
        if source.is_alive():
            source.terminate()
        source.join(10)


if __name__ == '__main__':
    main()
