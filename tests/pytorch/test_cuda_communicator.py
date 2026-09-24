# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends import communicator as base_communicator_module
from lmdeploy.pytorch.backends.cuda.comm import communicator as communicator_module


def _run_dcp_query_gather(rank, rendezvous, enabled):
    from datetime import timedelta

    from torch import distributed as dist

    from lmdeploy.pytorch.backends.cuda.attention.cp import get_dcp_manager
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.config import DistConfig
    from lmdeploy.pytorch.distributed import DistContext, get_dist_manager

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    backend = 'auto' if enabled[rank] else 'nccl'
    if enabled[0] != enabled[1]:
        try:
            with pytest.raises(ValueError, match='agree across ranks'):
                DistContext.build(rank, DistConfig(tp=2, dcp=2, communication_backend=backend),
                                  communicator_builder=CudaOpsBackend.build_communicator)
        finally:
            dist.destroy_process_group()
        return
    ctx = DistContext.build(rank, DistConfig(tp=2, dcp=2, communication_backend=backend),
                            communicator_builder=CudaOpsBackend.build_communicator)
    try:
        with get_dist_manager().context(ctx):
            manager = get_dcp_manager()
            manager.prepare_query_gather(32, 576)
            workspace = manager._query_workspace
            if all(enabled):
                assert workspace.is_available()
            else:
                assert workspace is None
            manager = get_dcp_manager()
            manager.prepare_query_gather(32, 576)
            assert manager._query_workspace is workspace
            query = torch.empty(384, 32, 576, device='cuda', dtype=torch.bfloat16)
            # Ineligible inputs still return correct results through NCCL.
            for fallback in (query[:1], query[..., ::2], query.float(),
                             torch.empty(1024, 32, 576, device='cuda', dtype=query.dtype)):
                fallback.fill_(rank)
                output = manager.gather_query(fallback)
                expected = torch.arange(2, device='cuda', dtype=fallback.dtype)
                expected = expected.repeat_interleave(32)[None, :, None].expand_as(output)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
            rows = (2, 16, 96, 384)

            def merge(query):
                lse = torch.zeros(query.shape[:2], device='cuda')
                counts = torch.ones(query.size(0), device='cuda', dtype=torch.int32)
                return manager.combine(query, lse, counts)

            query.fill_(rank)
            for count in rows:
                torch.testing.assert_close(merge(manager.gather_query(query[:count])), query[:count], rtol=0, atol=0)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            outputs = []
            with torch.cuda.graph(graph):
                for step in range(16):
                    query.fill_(rank + step * 8)
                    gathered = manager.gather_query(query[:rows[step % len(rows)]])
                    # A slow reader must finish before any rank reuses the arena.
                    if rank == step % 2:
                        torch.cuda._sleep(20000)
                    outputs.append(gathered.clone())
                    merge(gathered)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            graph.reset()
            for step, output in enumerate(outputs):
                expected = (torch.arange(2, device='cuda', dtype=query.dtype) + step * 8)
                expected = expected.repeat_interleave(32)[None, :, None].expand_as(output)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
            _check_lm_head_lifecycle(rank, all(enabled))
            _check_dcp_candidate_gather(rank, ctx.dcp_group, all(enabled))
    finally:
        ctx.close()
        dist.destroy_process_group()


def _check_dcp_candidate_gather(rank, group, direct):
    from lmdeploy.pytorch.backends.cuda.attention.cp import get_dcp_manager
    from lmdeploy.pytorch.backends.cuda.nsa import TritonNSAIndexFP8Impl
    from lmdeploy.pytorch.kernels.cuda.sparse_index_dcp_topk import pack_dcp_topk_candidates, sparse_dcp_global_topk

    k = 512
    # Candidate merging consumes existing scores and does not require DeepGEMM.
    impl = object.__new__(TritonNSAIndexFP8Impl)
    impl.topk, impl.fill = k, -1
    impl.dcp_world_size, impl.dcp_rank = 2, rank
    impl.dcp_manager = get_dcp_manager()
    impl.dcp_manager.prepare_candidate_gather(k)
    workspace = impl.dcp_manager._candidate_workspace
    assert (workspace is not None and workspace.is_available()) == direct
    get_dcp_manager().prepare_candidate_gather(k)
    assert impl.dcp_manager._candidate_workspace is workspace
    native = base_communicator_module.DeviceCommunicator(group.gpu_group)
    # IDs are bitcast, not converted to FP32; include large IDs and NaN encodings.
    bits = (torch.arange(16 * k * 2, device='cuda', dtype=torch.int32) * 1234567 + rank).view(16, k * 2)
    payload = bits.view(torch.float32)
    for value in (payload, payload[:1], payload[:, ::2], payload.double()):
        actual = group.communicator.all_gather(value, dim=0, workspace=workspace)
        expected = native.all_gather(value, dim=0)
        torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
    if direct:
        # Capacity rejection must use NCCL consistently, including a captured
        # shape that was not warmed before capture.
        oversized = payload[:1].expand(workspace._state.max_token_num // 2 + 1, -1).contiguous()
        torch.testing.assert_close(
            group.communicator.all_gather(oversized, dim=0, workspace=workspace).view(torch.int32),
            native.all_gather(oversized, dim=0).view(torch.int32), rtol=0, atol=0)
    scores = torch.randn(16, k * 2, device='cuda')
    indices = torch.arange(k, device='cuda', dtype=torch.int32).expand(16, -1).clone()

    def reference():
        packed = pack_dcp_topk_candidates(scores, indices, dcp_world_rank=(2, rank))
        gathered = native.all_gather(packed.flatten(1), dim=0).view(2, 16, k, 2)
        return sparse_dcp_global_topk(gathered, k=k)

    impl._merge_dcp_topk(scores, indices)
    graph = torch.cuda.CUDAGraph()
    outputs = []
    with torch.cuda.graph(graph):
        for step in range(4):
            if rank == step % 2:
                torch.cuda._sleep(20000)
            outputs.append(impl._merge_dcp_topk(scores, indices))
        unwarmed = group.communicator.all_gather(payload[:3], dim=0, workspace=workspace)
    for case in range(3):
        scores.normal_()
        if case == 1:
            scores.zero_()
        if case == 2:
            indices[:, k // 2:] = -1
            indices[0] = -1
        expected = reference()
        torch.testing.assert_close(impl._merge_dcp_topk(scores, indices), expected, rtol=0, atol=0)
        graph.replay()
        torch.cuda.synchronize()
        for output in outputs:
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(unwarmed.view(torch.int32),
                                   native.all_gather(payload[:3], dim=0).view(torch.int32), rtol=0, atol=0)
    graph.reset()


def _run_auto_all_reduce(rank, rendezvous):
    from datetime import timedelta

    from torch import distributed as dist

    from lmdeploy.pytorch.config import DistConfig

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    cpu_group = dist.new_group(backend='gloo')
    communicator = communicator_module.build_cuda_communicator(
        cpu_group, dist.group.WORLD, DistConfig(tp=2, communication_backend='auto'))
    try:
        provider = communicator._all_reduce_provider
        assert provider is not None
        provider.all_reduce_ = Mock(wraps=provider.all_reduce_)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            tensor = torch.empty(7, 4096, device='cuda', dtype=dtype)
            tensor.fill_(rank + 1)
            communicator.all_reduce_(tensor)
            provider.all_reduce_.assert_called_with(tensor)
            torch.testing.assert_close(tensor, torch.full_like(tensor, 3), rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                tensor.fill_(rank + 1)
                communicator.all_reduce_(tensor)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(tensor, torch.full_like(tensor, 3), rtol=0, atol=0)
            graph.reset()
        # Strided input retains the native NCCL error.
        strided = torch.empty(7, 8192, device='cuda', dtype=torch.bfloat16)[:, ::2]
        assert not provider.all_reduce_(strided)
        with pytest.raises(ValueError, match='contiguous'):
            communicator.all_reduce_(strided)
    finally:
        communicator.close()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two CUDA GPUs')
def test_auto_all_reduce_eager_and_graph(tmp_path):
    from torch.distributed._symmetric_memory import DeviceType, _SymmetricMemory

    if any(torch.cuda.get_device_capability(i) != (9, 0)
           or not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, i) for i in range(2)):
        pytest.skip('requires SM90 with multicast support')
    torch.multiprocessing.spawn(_run_auto_all_reduce,
                                args=(f'file://{tmp_path}/auto_all_reduce',), nprocs=2, join=True)


def _check_lm_head_lifecycle(rank, direct):
    from lmdeploy.pytorch.nn import ParallelEmbedding, ParallelLMHead

    head = ParallelLMHead(250, 128, dtype=torch.bfloat16, device='cuda')
    if direct:
        assert head._logits_gather_workspace.is_available()
    head.weight.data.fill_(rank + 1)
    hidden = torch.ones(2, 128, dtype=torch.bfloat16, device='cuda')
    expected = torch.arange(1, 3, device='cuda', dtype=hidden.dtype).repeat_interleave(128)[:250] * 128
    logits = head(hidden)
    head(hidden * 2)
    torch.testing.assert_close(logits, expected.expand_as(logits), atol=0, rtol=0)
    if direct:
        assert head._logits_gather_workspace.is_available()
    # A coordinated dtype move retires the BF16 arena and uses native gathering.
    head.to(dtype=torch.float16)
    if direct:
        assert not head._logits_gather_workspace.is_available()
    logits = head(hidden)
    torch.testing.assert_close(logits, expected.to(logits.dtype).expand_as(logits), atol=0, rtol=0)
    # Tying BF16 weights must re-admit the arena before the next forward.
    embedding = ParallelEmbedding(250, 128, None, dtype=torch.bfloat16, device='cuda', is_tp=True)
    embedding.weight.data.fill_(rank + 1)
    head.tie_weights(embedding)
    if direct:
        assert head._logits_gather_workspace.is_available()
    logits = head(hidden)
    torch.testing.assert_close(logits, expected.expand_as(logits), atol=0, rtol=0)
    if direct:
        head._logits_gather_workspace.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two CUDA GPUs')
@pytest.mark.parametrize('enabled', [(False, False), (True, True), (True, False)],
                         ids=['nccl', 'auto', 'mixed_config'])
def test_dcp_query_gather_graph_reuses_arena(tmp_path, enabled):
    if all(enabled):
        from torch.distributed._symmetric_memory import DeviceType, _SymmetricMemory
        if any(torch.cuda.get_device_capability(i)[0] < 9
               or not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, i) for i in range(2)):
            pytest.skip('requires SM90 or newer with multicast support')
        if not torch.cuda.can_device_access_peer(0, 1):
            pytest.skip('requires peer access')
    torch.multiprocessing.spawn(_run_dcp_query_gather,
                                args=((tmp_path / 'rendezvous').as_uri(), enabled), nprocs=2)


@pytest.fixture
def comm_env(monkeypatch):
    """Two ranks with available providers; no CUDA or distributed runtime."""
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'GPU')
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(communicator_module.dist, 'get_world_size', lambda group: 2)
    monkeypatch.setattr(communicator_module.dist, 'get_rank', lambda group: 0)
    monkeypatch.setattr(communicator_module.dist, 'all_gather_object',
                        lambda output, value, group: output.__setitem__(slice(None), [value, value]))
    provider = Mock()
    provider.is_available.return_value = True
    monkeypatch.setattr(communicator_module, 'SymmetricMemoryAllReduce', Mock(return_value=provider))
    return provider


def test_gather_registration_and_native_fallback(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.attention.cp import get_dcp_manager
    from lmdeploy.pytorch.distributed import DistContext, DistGroup, get_dist_manager

    module = base_communicator_module
    monkeypatch.setattr(module.dist, 'get_world_size', lambda group: 2)

    def gather(output, input, group):
        output.copy_(torch.cat((input, input + 10), dim=0))

    monkeypatch.setattr(module.dist, 'all_gather_into_tensor', gather)
    communicator = module.DeviceCommunicator('group')
    group = DistGroup(gpu_group='group', communicator=communicator)
    context = DistContext(dcp_group=group)
    with get_dist_manager().context(context):
        manager = get_dcp_manager()
        manager.prepare_query_gather(2, 4)
    assert manager._query_workspace is None
    logits = communicator.create_all_gather_workspace(6, device=torch.device('cpu'), dtype=torch.float32)
    assert logits is None
    query = torch.arange(48).view(3, 2, 8)[..., ::2]
    with get_dist_manager().context(context):
        torch.testing.assert_close(manager.gather_query(query), torch.cat((query, query + 10), dim=1))
    input = torch.ones(3, 3)
    torch.testing.assert_close(communicator.all_gather(input, workspace=logits), torch.cat((input, input + 10), dim=-1))
    torch.testing.assert_close(communicator.all_gather(input, dim=0), torch.cat((input, input + 10), dim=0))


def test_gather_preparation_and_weight_transition(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.comm.symm_mem_allgather import SymmetricMemoryAllGather

    workspace = SymmetricMemoryAllGather('group', 0, 16, device='cpu', dtype=torch.bfloat16, capacity_bytes=1024)
    prepare = Mock(return_value=False)
    monkeypatch.setattr(workspace, '_prepare', prepare)
    workspace.reset('cpu', torch.float16)
    prepare.assert_not_called()
    workspace.prepare()
    workspace.prepare()
    prepare.assert_called_once_with(torch.device('cpu'), torch.float16)
    workspace.reset('cpu', torch.float16)
    prepare.assert_called_once()
    workspace.reset('cpu', torch.bfloat16)
    assert prepare.call_count == 2
    prepare.assert_called_with(torch.device('cpu'), torch.bfloat16)
    assert not workspace.is_available()
    workspace.close()
    workspace.prepare()
    assert prepare.call_count == 3


def test_cuda_gather_dispatch_and_workspace_ownership(monkeypatch, comm_env):
    from lmdeploy.pytorch.backends.cuda.attention.cp import get_dcp_manager
    from lmdeploy.pytorch.backends.cuda.comm import symm_mem_allgather
    from lmdeploy.pytorch.distributed import DistContext, DistGroup, get_dist_manager

    # The context shares a DCP manager; logits workspaces remain caller-owned.
    factory = Mock(side_effect=lambda *args, **kwargs: Mock())
    monkeypatch.setattr(symm_mem_allgather, 'SymmetricMemoryAllGather', factory)
    communicator = communicator_module.CudaCommunicator('cpu', 'gpu', group_name='dcp')
    group = DistGroup(gpu_group='gpu', communicator=communicator)
    context = DistContext(dcp_group=group)
    with get_dist_manager().context(context):
        manager = get_dcp_manager()
        manager.prepare_query_gather(2, 4)
        manager = get_dcp_manager()
        manager.prepare_query_gather(2, 4)
    query_workspace = manager._query_workspace
    logits_workspace = communicator.create_all_gather_workspace(6, device=torch.device('cpu'), dtype=torch.float32)
    assert factory.call_count == 2
    query_workspace.prepare.assert_called_once()
    logits_workspace.prepare.assert_called_once()

    query = torch.arange(24).view(3, 2, 4)
    gathered_query = torch.cat((query, query + 10), dim=1)
    query_workspace.all_gather.return_value = gathered_query.flatten(1)
    with get_dist_manager().context(context):
        torch.testing.assert_close(manager.gather_query(query), gathered_query)
    assert query_workspace.all_gather.call_args.kwargs == {'dim': -1, 'copy_output': False}

    native = Mock(side_effect=lambda output, input, group: output.copy_(torch.cat((input, input + 10))))
    monkeypatch.setattr(base_communicator_module.dist, 'all_gather_into_tensor', native)
    input = torch.ones(3, 3)
    logits_workspace.all_gather.return_value = torch.ones(3, 6)
    torch.testing.assert_close(communicator.all_gather(input, workspace=logits_workspace), torch.ones(3, 6))
    logits_workspace.all_gather.assert_called_once_with(input, dim=-1, copy_output=True)
    native.assert_not_called()
    logits_workspace.all_gather.return_value = None
    torch.testing.assert_close(communicator.all_gather(input, workspace=logits_workspace),
                               torch.cat((input, input + 10), dim=-1))
    native.assert_called_once()

    manager.prepare_candidate_gather(512)
    candidate_workspace = manager._candidate_workspace
    manager.prepare_candidate_gather(512)
    assert manager._candidate_workspace is candidate_workspace
    assert context.dcp_manager is manager
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    group_close = Mock(wraps=group.close)
    monkeypatch.setattr(group, 'close', group_close)
    query_workspace.close.side_effect = lambda: group_close.assert_not_called()
    candidate_workspace.close.side_effect = lambda: group_close.assert_not_called()
    context.close()
    context.close()
    query_workspace.close.assert_called_once()
    candidate_workspace.close.assert_called_once()
    assert context.dcp_manager is None
    assert manager._query_workspace is manager._candidate_workspace is None
    logits_workspace.close.assert_not_called()


@pytest.mark.parametrize('native', ['none', 'async', 'max', 'cpu', 'other_group', 'no_communicator'])
def test_public_all_reduce_routes_only_synchronous_cuda_tp_sum(monkeypatch, native):
    import lmdeploy.pytorch.distributed as distributed

    communicator = Mock()
    group = object()
    context = distributed.DistContext(
        attn_tp_group=distributed.DistGroup(gpu_group=group, communicator=communicator))
    if native == 'no_communicator':
        context.attn_tp_group.communicator = None
    tensor = SimpleNamespace(is_cuda=native != 'cpu')
    op = distributed.ReduceOp.MAX if native == 'max' else distributed.ReduceOp.SUM
    async_op = native == 'async'
    selected_group = object() if native == 'other_group' else group
    native_reduce = Mock()
    monkeypatch.setattr(distributed.dist, 'all_reduce', native_reduce)
    with distributed.get_dist_manager().context(context):
        result = distributed.all_reduce(tensor, op=op, group=selected_group, async_op=async_op)
    if native == 'none':
        communicator.all_reduce_.assert_called_once_with(tensor)
        native_reduce.assert_not_called()
    else:
        native_reduce.assert_called_once_with(tensor, op, selected_group, async_op)
        communicator.all_reduce_.assert_not_called()
        assert result is native_reduce.return_value


def test_all_reduce_workspace_failure_agrees_across_ranks(monkeypatch, comm_env):
    def peer_failure(flags, ready, group):
        flags[:] = [ready, not comm_env.prepare.called]

    monkeypatch.setattr(communicator_module.dist, 'all_gather_object', peer_failure)
    communicator = communicator_module.CudaCommunicator('cpu', 'gpu', group_name='tp')
    comm_env.close.assert_called_once()
    comm_env.prepare.assert_called_once()
    assert communicator._all_reduce_provider is None
    native = Mock()
    monkeypatch.setattr(communicator_module.dist, 'all_reduce', native)
    input = torch.ones(2, 4)
    communicator.all_reduce_(input)
    native.assert_called_once_with(input, group='gpu')


@pytest.mark.parametrize('handled', [True, False])
def test_auto_all_reduce_falls_back_for_unsupported_inputs(monkeypatch, comm_env, handled):
    communicator = communicator_module.CudaCommunicator('cpu', 'gpu', group_name='tp')
    comm_env.all_reduce_.return_value = handled
    native = Mock()
    monkeypatch.setattr(communicator_module.dist, 'all_reduce', native)
    comm_env.prepare.assert_called_once()
    input = torch.ones(7, 128, dtype=torch.bfloat16)
    communicator.all_reduce_(input)
    comm_env.all_reduce_.assert_called_once_with(input)
    if handled:
        native.assert_not_called()
    else:
        native.assert_called_once_with(input, group='gpu')


@pytest.mark.parametrize('group_name,available', [('tp', True), ('tp', False), ('dcp', True)])
def test_auto_initializes_all_reduce_by_availability(comm_env, group_name, available):
    comm_env.is_available.return_value = available
    communicator = communicator_module.CudaCommunicator('cpu', 'gpu', group_name=group_name)
    assert communicator._all_reduce_provider is (comm_env if available and group_name == 'tp' else None)
    assert communicator_module.SymmetricMemoryAllReduce.call_count == (group_name == 'tp')
    assert comm_env.close.call_count == (group_name == 'tp' and not available)
    assert comm_env.prepare.call_count == (group_name == 'tp' and available)


def test_symm_mem_allreduce_selects_group_algorithm(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.comm.symm_mem_allreduce import SymmetricMemoryAllReduce

    multimem = Mock()
    two_shot = Mock()
    monkeypatch.setattr(
        torch.ops,
        'symm_mem',
        SimpleNamespace(
            multimem_all_reduce_=multimem,
            two_shot_all_reduce_=two_shot,
        ),
    )

    communicator = SymmetricMemoryAllReduce.__new__(SymmetricMemoryAllReduce)
    communicator.group = SimpleNamespace(group_name='group')
    communicator._buffer = torch.empty(8, dtype=torch.bfloat16)
    communicator._max_size = communicator._buffer.nbytes
    input = torch.ones(4, dtype=torch.bfloat16)

    communicator._use_multimem = True
    assert communicator.all_reduce_(input)
    multimem.assert_called_once()
    two_shot.assert_not_called()

    communicator._use_multimem = False
    assert communicator.all_reduce_(input)
    two_shot.assert_called_once()


def test_symm_mem_allreduce_peer_allocation_failure(monkeypatch):
    import torch.distributed._symmetric_memory as symm_mem

    from lmdeploy.pytorch.backends.cuda.comm import symm_mem_allreduce as module

    monkeypatch.setattr(module.dist, 'get_world_size', lambda group: 2)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.ops, 'symm_mem', SimpleNamespace(two_shot_all_reduce_=Mock()))
    monkeypatch.setattr(symm_mem, 'empty', Mock(return_value=object()))
    rendezvous = Mock()
    monkeypatch.setattr(symm_mem, 'rendezvous', rendezvous)

    def peer_failure(flags, ready, group):
        flags[:] = [ready, False]

    monkeypatch.setattr(module.dist, 'all_gather_object', peer_failure)
    provider = module.SymmetricMemoryAllReduce(SimpleNamespace(group_name='group'))
    symm_mem.empty.assert_not_called()
    provider.prepare()
    assert not provider.is_available()
    assert provider._buffer is None
    rendezvous.assert_not_called()


def test_communicator_rejects_rank_backend_mismatch(monkeypatch, comm_env):
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.config import DistConfig

    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (9, 0))

    def disagree(output, local, group):
        output[:] = [local, ('auto', *local[1:])]

    monkeypatch.setattr(communicator_module.dist, 'all_gather_object', disagree)
    optimized = Mock()
    monkeypatch.setattr(communicator_module, 'CudaCommunicator', optimized)
    with pytest.raises(ValueError, match='agree across ranks'):
        CudaOpsBackend.build_communicator('cpu', 'gpu', DistConfig(tp=2, communication_backend='nccl'))
    optimized.assert_not_called()


@pytest.mark.parametrize('shared_tp_group', [False, True])
def test_communicator_group_names_and_tp_deduplication(shared_tp_group):
    from lmdeploy.pytorch.distributed import DistContext, DistGroup, _build_communicators

    tp_group = DistGroup(cpu_group='tp_cpu', gpu_group='tp_gpu')
    mlp_group = tp_group if shared_tp_group else DistGroup(cpu_group='mlp_cpu', gpu_group='mlp_gpu')
    dcp_group = DistGroup(cpu_group='dcp_cpu', gpu_group='dcp_gpu')
    builder = Mock(side_effect=lambda **kwargs: Mock())
    context = DistContext(attn_tp_group=tp_group, mlp_tp_group=mlp_group, moe_tp_group=mlp_group,
                          dcp_group=dcp_group, communicator_builder=builder)
    _build_communicators(context)

    names = [call.kwargs['group_name'] for call in builder.call_args_list]
    assert names == (['tp', 'dcp'] if shared_tp_group else ['tp', 'tp', 'dcp'])
    assert (tp_group.communicator is mlp_group.communicator) == shared_tp_group
    assert tp_group.communicator is not dcp_group.communicator
