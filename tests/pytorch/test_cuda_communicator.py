# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends import communicator as base_communicator_module
from lmdeploy.pytorch.backends.cuda.comm import communicator as communicator_module
from tests.pytorch.distributed_test_utils import spawn_with_timeout


@pytest.fixture(autouse=True)
def communication_environment(monkeypatch):
    for name in ('enable_flashinfer_allreduce', 'enable_symm_mem_allreduce', 'enable_symm_mem_lmhead'):
        monkeypatch.setattr(communicator_module._envs, name, False)


def _run_auto_all_gather(rank, rendezvous):
    from datetime import timedelta

    from torch import distributed as dist

    from lmdeploy.pytorch.config import DistConfig

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    cpu_group = dist.new_group(backend='gloo')
    communicator = communicator_module.build_cuda_communicator(
        cpu_group,
        dist.group.WORLD,
        DistConfig(tp=2, dcp=2, communication_backend='auto'),
        group_name='dcp')
    try:
        workspace = communicator.create_all_gather_workspace(
            256, device=torch.device('cuda'), dtype=torch.bfloat16)
        assert workspace.is_available()

        rows = (2, 16, 96, 384)
        inputs = [torch.full((row, 128), rank, device='cuda', dtype=torch.bfloat16) for row in rows]
        optimized_gather = Mock(wraps=workspace._kernels.all_gather_inner)
        workspace._kernels.all_gather_inner = optimized_gather
        # Admission is per shape. Warm outside capture on a side stream so the
        # graph records the optimized path, not a cold-shape NCCL fallback.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for input in inputs:
                output = communicator.all_gather(input, workspace=workspace, copy_output=False).clone()
                expected = torch.cat((torch.zeros_like(input), torch.ones_like(input)), dim=-1)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        assert optimized_gather.call_count == len(rows)
        optimized_gather.reset_mock()
        graph = torch.cuda.CUDAGraph()
        outputs = []
        with torch.cuda.graph(graph):
            for input in inputs:
                # The next gather reuses the arena, even for a different shape.
                outputs.append(communicator.all_gather(input, workspace=workspace, copy_output=False).clone())
        assert optimized_gather.call_count == len(rows)

        for step in range(3):
            for input in inputs:
                input.fill_(rank + step)
            graph.replay()
            torch.cuda.synchronize()
            for input, output in zip(inputs, outputs):
                expected = torch.cat((torch.full_like(input, step), torch.full_like(input, step + 1)), dim=-1)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
        graph.reset()
        # The single-token policy intentionally uses native gathering.
        optimized_gather.reset_mock()
        input = torch.full((1, 128), rank, device='cuda', dtype=torch.bfloat16)
        output = communicator.all_gather(input, workspace=workspace)
        expected = torch.cat((torch.zeros_like(input), torch.ones_like(input)), dim=-1)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        optimized_gather.assert_not_called()
        workspace.close()
    finally:
        communicator.close()
        dist.destroy_process_group()


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
    spawn_with_timeout(_run_auto_all_reduce, args=(f'file://{tmp_path}/auto_all_reduce',))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two CUDA GPUs')
def test_auto_all_gather_reuses_workspace(tmp_path):
    from torch.distributed._symmetric_memory import DeviceType, _SymmetricMemory

    if any(torch.cuda.get_device_capability(i)[0] < 9
            or not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, i) for i in range(2)):
        pytest.skip('requires SM90 or newer with multicast support')
    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip('requires peer access')
    spawn_with_timeout(_run_auto_all_gather, args=(f'file://{tmp_path}/auto_all_gather',))


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
        manager.prepare_attention(2, 4)
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
        manager.prepare_attention(2, 4)
        manager = get_dcp_manager()
        manager.prepare_attention(2, 4)
    query_workspace = manager._query_workspace
    logits_workspace = communicator.create_all_gather_workspace(6, device=torch.device('cpu'), dtype=torch.float32)
    assert factory.call_count == 3
    lse_workspace = manager._lse_workspace
    lse_workspace.prepare.assert_called_once()
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
    manager.prepare_prefix_gather(576, 128, 576)
    prefix_workspaces = list(manager._prefix_workspaces.values())
    assert len(prefix_workspaces) == 2
    assert context.dcp_manager is manager
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    group_close = Mock(wraps=group.close)
    monkeypatch.setattr(group, 'close', group_close)
    query_workspace.close.side_effect = lambda: group_close.assert_not_called()
    lse_workspace.close.side_effect = lambda: group_close.assert_not_called()
    candidate_workspace.close.side_effect = lambda: group_close.assert_not_called()
    for workspace in prefix_workspaces:
        workspace.close.side_effect = lambda: group_close.assert_not_called()
    context.close()
    context.close()
    query_workspace.close.assert_called_once()
    lse_workspace.close.assert_called_once()
    candidate_workspace.close.assert_called_once()
    for workspace in prefix_workspaces:
        workspace.close.assert_called_once()
    assert not manager._prefix_workspaces
    assert context.dcp_manager is None
    assert manager._query_workspace is manager._lse_workspace is manager._candidate_workspace is None
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


@pytest.mark.parametrize('backend,group,flashinfer,symm,lmhead,expected', [
    ('nccl', 'tp', False, False, False, None),
    ('nccl', 'tp', True, False, False, ('flashinfer', False)),
    ('nccl', 'tp', False, True, False, ('symm_mem', False)),
    ('nccl', 'tp', False, False, True, (None, True)),
    ('auto', 'tp', False, False, False, ('symm_mem', True)),
    ('auto', 'tp', True, False, False, ('flashinfer', True)),
    ('nccl', 'dcp', True, False, True, None),
    ('auto', 'dcp', True, False, False, ('symm_mem', True)),
])
def test_communication_policy_preserves_legacy_options(monkeypatch, comm_env, backend, group,
                                                        flashinfer, symm, lmhead, expected):
    from lmdeploy.pytorch.config import DistConfig

    module = communicator_module
    monkeypatch.setattr(module._envs, 'enable_flashinfer_allreduce', flashinfer)
    monkeypatch.setattr(module._envs, 'enable_symm_mem_allreduce', symm)
    monkeypatch.setattr(module._envs, 'enable_symm_mem_lmhead', lmhead)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *args: (9, 0))
    monkeypatch.setattr(torch.cuda, 'can_device_access_peer', lambda *args: True)

    def peers(output, local, group):
        other = (local[0], local[1], 1, *local[3:]) if isinstance(local, tuple) else local
        output[:] = [local, other]

    monkeypatch.setattr(module.dist, 'all_gather_object', peers)
    factory = Mock()
    monkeypatch.setattr(module, 'CudaCommunicator', factory)
    result = module.build_cuda_communicator('cpu', 'gpu', DistConfig(tp=2, communication_backend=backend),
                                            group_name=group)
    if expected is None:
        assert result is None
        factory.assert_not_called()
    else:
        factory.assert_called_once_with(cpu_group='cpu', device_group='gpu', group_name=group,
                                         all_reduce_backend=expected[0], enable_all_gather=expected[1])


@pytest.mark.parametrize('mismatch', [False, True])
def test_communication_policy_rejects_conflicting_providers(monkeypatch, comm_env, mismatch):
    from lmdeploy.pytorch.config import DistConfig

    module = communicator_module
    monkeypatch.setattr(module._envs, 'enable_flashinfer_allreduce', True)
    monkeypatch.setattr(module._envs, 'enable_symm_mem_allreduce', not mismatch)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *args: (9, 0))
    if mismatch:
        def peers(output, local, group):
            policy = ('nccl', 'tp', False, False, False)
            output[:] = [local, (policy, *local[1:])]
        monkeypatch.setattr(module.dist, 'all_gather_object', peers)
    error = 'must agree across ranks' if mismatch else 'cannot be enabled together'
    with pytest.raises(ValueError, match=error):
        module.build_cuda_communicator('cpu', 'gpu', DistConfig(tp=2))
    # No provider or symmetric workspace may be created before consensus.
    communicator_module.SymmetricMemoryAllReduce.assert_not_called()


@pytest.mark.parametrize('fused', [False, True])
def test_flashinfer_norm_dispatch_and_fallback(monkeypatch, comm_env, fused):
    from lmdeploy.pytorch.distributed import DistContext, DistGroup, get_dist_manager
    from lmdeploy.pytorch.nn import norm as norm_module

    provider = Mock()
    provider.is_available.return_value = True
    monkeypatch.setattr(communicator_module, 'FlashInferAllReduce', Mock(return_value=provider))
    communicator = communicator_module.CudaCommunicator('cpu', 'gpu', group_name='tp',
                                                        all_reduce_backend='flashinfer', enable_all_gather=False)
    communicator_module.SymmetricMemoryAllReduce.assert_not_called()
    assert communicator.create_all_gather_workspace(16, torch.device('cpu'), torch.bfloat16) is None
    group = DistGroup(gpu_group='gpu', communicator=communicator)
    context = DistContext(attn_tp_group=group)
    x, residual = torch.ones(2, 8), torch.ones(2, 8)
    expected = (x * 3, residual * 4)
    provider.fused_all_reduce_residual_rms_norm.return_value = expected if fused else None
    provider.all_reduce_.return_value = False
    native = Mock(side_effect=lambda tensor, group: tensor.mul_(2))
    monkeypatch.setattr(communicator_module.dist, 'all_reduce', native)
    impl = Mock()
    impl.forward.return_value = expected
    monkeypatch.setattr(norm_module, 'get_backend', lambda: SimpleNamespace(build_op=lambda *a, **k: impl))
    monkeypatch.setattr(norm_module, 'get_build_model_context', lambda: SimpleNamespace(enable_deterministic=False))
    with get_dist_manager().context(context):
        norm = norm_module.RMSNorm(8, device='cpu', dtype=torch.float32, all_reduce_group='attn')
        result = norm(x, residual)
    assert result is expected
    provider.fused_all_reduce_residual_rms_norm.assert_called_once_with(
        input=x, residual=residual, weight=norm.weight, eps=norm.eps)
    if fused:
        native.assert_not_called()
        provider.all_reduce_.assert_not_called()
        impl.forward.assert_not_called()
    else:
        native.assert_called_once()
        provider.all_reduce_.assert_called_once_with(x)
        impl.forward.assert_called_once_with(x, norm.weight, residual)
        torch.testing.assert_close(x, torch.full_like(x, 2))
    communicator.close()
    provider.close.assert_called_once()


@pytest.mark.parametrize('budget_mb', [8, 64])
def test_lmhead_preserves_gather_budget(monkeypatch, budget_mb):
    from lmdeploy.pytorch.config import DistConfig
    from lmdeploy.pytorch.distributed import DistContext, DistGroup, get_dist_manager
    from lmdeploy.pytorch.nn import embedding as module

    communicator = Mock()
    group = DistGroup(gpu_group='gpu', communicator=communicator)
    context = DistContext(dist_config=DistConfig(tp=2), attn_tp_group=group)
    monkeypatch.setattr(module._envs, 'symm_mem_lmhead_max_mb', budget_mb)
    monkeypatch.setattr(module, 'get_backend', lambda: SimpleNamespace(build_op=Mock()))
    monkeypatch.setattr(module, 'get_build_model_context',
                        lambda: SimpleNamespace(fp32_lm_head=False, tie_word_embeddings=False,
                                                enable_deterministic=False))
    with get_dist_manager().context(context):
        head = module.ParallelLMHead(128, 16, dtype=torch.bfloat16, device='cpu')
    communicator.create_all_gather_workspace.assert_called_once_with(
        128, device=head.weight.device, dtype=torch.bfloat16, capacity_bytes=budget_mb * 1024 * 1024)


@pytest.mark.parametrize('layer_idx', [0, 1, 2])
@pytest.mark.parametrize('optimized', [False, True])
def test_deepseek_defers_only_nonterminal_reductions(monkeypatch, layer_idx, optimized):
    from lmdeploy.pytorch.models import deepseek_v32 as module

    norm = Mock()
    norm.can_handle_all_reduce.return_value = optimized
    attn, mlp = Mock(), Mock()
    monkeypatch.setattr(module, 'RMSNorm', norm)
    monkeypatch.setattr(module.DeepseekV32DecoderLayer, 'attention_cls', attn)
    monkeypatch.setattr(module, 'DeepseekV2MLP', mlp)
    config = SimpleNamespace(num_hidden_layers=3, hidden_size=8, rms_norm_eps=1e-6, n_routed_experts=None)
    module.DeepseekV32DecoderLayer(config, layer_idx, device='cpu')
    assert attn.call_args.kwargs['all_reduce'] is not optimized
    assert mlp.call_args.kwargs['all_reduce'] is not (optimized and layer_idx < 2)
    assert [call.kwargs['all_reduce_group'] for call in norm.call_args_list] == ['mlp', 'attn']
