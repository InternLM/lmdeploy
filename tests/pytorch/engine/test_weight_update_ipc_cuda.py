# Copyright (c) OpenMMLab. All rights reserved.
"""Real colocated CUDA IPC through UniExecutor, without HTTP or a full LLM.

Only model construction/routing is small: deserialization, event wait/ack, loading, finalization and producer-buffer
reuse use the production path.
"""
import multiprocessing as mp
import pickle
from contextlib import nullcontext
from types import SimpleNamespace

import pybase64
import pytest
import torch


class _Model(torch.nn.Module):
    def __init__(self, tag, failure):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(128, device='cuda'), requires_grad=False)
        self.tag, self.failure, self.finalized = tag, failure, 0

    def rename_weight(self, name):
        return name.split('.')[-1]

    def load_weights(self, weights):
        # A late consumer makes premature producer reuse observable.
        torch.cuda._sleep(20_000_000)
        for name, tensor in weights:
            getattr(self, name).copy_(tensor)
        if self.failure == self.tag:
            raise RuntimeError('injected ' + self.tag)

    def update_weights(self):
        self.finalized += 1
        if self.failure == 'finalize':
            raise RuntimeError('injected finalize')

    def forward(self):
        return self.weight.sum()


class _Spec:
    def __init__(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def split_updated_weights(self, weights):
        return ([x for x in weights if x[0].startswith('main.')],
                [x for x in weights if x[0].startswith('draft.')])


def _receiver(pipe, draft, failure):
    from lmdeploy.pytorch.engine.executor.uni_executor import UniExecutor
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
    from lmdeploy.serve.openai.protocol import UpdateParamsRequest
    torch.cuda.set_device(0)
    main = _Model('main', failure)
    extra = _Model('draft', failure) if draft else None
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.cache_config = SimpleNamespace(compact_state_capacity=object() if failure == 'reject' else None,
                                        use_mooncake_store=failure == 'reject')
    agent.patched_model = SimpleNamespace(get_model=lambda: main)
    agent.spec_agent = _Spec(extra)
    agent.all_context = nullcontext
    agent._update_params_ipc_tensor = None
    agent._update_params_ipc_event = None
    executor = UniExecutor.__new__(UniExecutor)
    executor.model_agent = agent
    pipe.send('ready')
    while True:
        command, data = pipe.recv()
        if command == 'stop':
            break
        if command == 'snapshot':
            pipe.send([(m.forward().item(), m.finalized) for m in (main, extra) if m is not None])
            continue
        error = None
        try:
            executor.update_params(UpdateParamsRequest(**data))
        except RuntimeError as exc:
            error = str(exc)
        # Do not synchronize here: producer must rely on the real IPC ack.
        pipe.send(dict(error=error, released=agent._update_params_ipc_tensor is None
                       and agent._update_params_ipc_event is None))
    torch.cuda.synchronize()
    pipe.close()


def _receive(pipe):
    assert pipe.poll(90), 'IPC receiver timed out'
    return pipe.recv()


@pytest.mark.parametrize('draft', [False, True])
@pytest.mark.parametrize('clone', [False, True])
def test_real_ipc_multiple_buckets_and_empty_finalize(draft, clone):
    _exercise(draft, clone, None)


@pytest.mark.parametrize('failure', ['main', 'draft', 'finalize', 'reject'])
def test_real_ipc_failure_releases_after_consumption(failure):
    _exercise(True, False, failure)


def _exercise(draft, clone, failure):
    from torch.multiprocessing.reductions import reduce_tensor

    from lmdeploy.utils import FlattenedTensorBucket
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_receiver, args=(child, draft, failure))
    process.start()
    child.close()
    try:
        assert _receive(parent) == 'ready'
        tensors = [('main.weight', torch.zeros(128, device='cuda'))]
        if draft:
            tensors.append(('draft.weight', torch.zeros(128, device='cuda')))
        bucket = FlattenedTensorBucket(named_tensors=tensors)
        flat = bucket.get_flattened_tensor()
        event = torch.cuda.Event(interprocess=True)
        event.record()
        event.synchronize()
        handle = event.ipc_handle()
        producer = torch.cuda.Stream()
        for step in range(1, 2 if failure else 4):
            if step == 1 or clone:
                descriptor = reduce_tensor(flat)
            with torch.cuda.stream(producer):
                event.wait()
                torch.cuda._sleep(20_000_000)
                flat.fill_(step)
                event.record()
            payload = dict(metadata=bucket.metadata, require_clone=clone)
            # Zero-copy mode reuses the IPC mapping; cloned buckets resend it.
            if step == 1 or clone:
                payload.update(flattened_tensor=descriptor, event_ipc_handle=handle)
            encoded = pybase64.b64encode(pickle.dumps(payload)).decode()
            parent.send(('update', dict(serialized_named_tensors=encoded, load_format='flattened_bucket',
                                       finished=failure == 'finalize')))
            reply = _receive(parent)
            if failure:
                assert reply['error'] and reply['released'], reply
                assert ('immutable namespace' if failure == 'reject' else 'injected') in reply['error']
            else:
                assert reply == dict(error=None, released=False)
            # Overwrite immediately after ack, before asking for a snapshot.
            with torch.cuda.stream(producer):
                event.wait()
                flat.fill_(-999)
            producer.synchronize()
            parent.send(('snapshot', None))
            actual = _receive(parent)
            for i, (value, finalized) in enumerate(actual):
                loaded = failure != 'reject' and not (failure == 'main' and i == 1)
                assert value == (128 * step if loaded else 0), (failure, actual)
                assert finalized == (1 if failure == 'finalize' and i == 0 else 0)
        if not failure:
            encoded = pybase64.b64encode(pickle.dumps(dict(metadata=[]))).decode()
            parent.send(('update', dict(serialized_named_tensors=encoded,
                                        load_format='flattened_bucket', finished=True)))
            assert _receive(parent) == dict(error=None, released=True)
            parent.send(('snapshot', None))
            assert _receive(parent) == [(384., 1)] * (2 if draft else 1)
        parent.send(('stop', None))
        process.join(30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()
