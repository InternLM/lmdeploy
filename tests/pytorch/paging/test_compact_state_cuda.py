# Copyright (c) OpenMMLab. All rights reserved.
"""Real event-delayed consumption of compact units (not engine integration)."""
import pytest
import torch

from lmdeploy.pytorch.config import CompactStateCapacity
from lmdeploy.pytorch.paging.compact_state import CompactStateManager
from tests.pytorch.compact_state_reference import StateAccessRow, submit


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA event lifetime test')
@pytest.mark.parametrize('bank_pin', [False, True])
def test_delayed_producer_two_stream_readers_eviction_and_cancel(bank_pin):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

    torch.manual_seed(113)
    owner = CompactStateManager(CompactStateCapacity(4, 1, 2, 0))
    pool = torch.zeros(owner.capacity.num_units, 1, 128, 128, device='cuda')
    q, k, v = [torch.randn(1, 1, 1, 128, device='cuda', dtype=torch.bfloat16) / 8 for _ in range(3)]
    g = -torch.ones(1, 1, 1, device='cuda')
    beta = torch.ones_like(g)

    def forward(src, dst):
        return fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=pool, output_final_state=True,
                                                state_indices=src, output_state_indices=dst)

    zero = torch.tensor([0], device='cuda')
    warm_destination = torch.tensor([[1]], device='cuda')
    forward(zero, warm_destination)  # compile outside the delayed experiment
    pool.zero_()
    init = torch.cuda.Event()
    init.record()
    streams = [torch.cuda.Stream() for _ in range(3)]
    banks = [owner.allocate_bank(stream=s.cuda_stream) for s in streams]
    checkpoint = owner.reserve_checkpoint(banks[0], 64)
    checkpoint_id = checkpoint.unit_id
    source = torch.tensor([checkpoint_id], device='cuda')
    destinations = [torch.tensor([[checkpoint_id]], device='cuda')] + [
        torch.tensor([[bank.unit_ids[0]]], device='cuda') for bank in banks[1:]]
    metadata_ready = torch.cuda.Event()
    metadata_ready.record()
    producer = submit(owner, (StateAccessRow(banks[0], 0, (checkpoint_id,), checkpoint),),
                            stream=streams[0].cuda_stream)
    ready = torch.cuda.Event()
    with torch.cuda.stream(streams[0]):
        streams[0].wait_event(init)
        streams[0].wait_event(metadata_ready)
        torch.cuda._sleep(150_000_000)
        forward(zero, destinations[0])
        checkpoint_bytes = pool[checkpoint_id].clone()  # test-only immutability oracle
        ready.record()
    trie = owner.retain(checkpoint, 'trie')
    reads, done = [], []
    for bank, stream, dst in zip(banks[1:], streams[1:], destinations[1:]):
        owner.bind_input(bank, checkpoint)
        if bank_pin:
            use = owner.submit_decode((bank,), stream=stream.cuda_stream, wait_for=(producer,))
            assert checkpoint_id not in use.bank_ids[0]
        else:
            use = submit(owner, (StateAccessRow(bank, checkpoint_id, (bank.unit_ids[0],)),),
                               stream=stream.cuda_stream, wait_for=(producer,))
        event = torch.cuda.Event()
        with torch.cuda.stream(stream):
            stream.wait_event(ready)  # actual CUDA dependency, not only a host record
            forward(source, dst)
            event.record()
        reads.append(use)
        done.append(event)
    owner.release_reference(trie)
    owner.release_reference(checkpoint)
    for bank in banks:
        owner.release_bank(bank)
    assert not ready.query(), 'Delay must cover host cancellation to exercise in-flight retention'
    assert checkpoint_id in owner._units
    live_units = 1 + 2 * owner.capacity.bank_width if bank_pin else 3
    assert owner.get_num_free() == owner.capacity.num_units - 1 - live_units
    with pytest.raises(RuntimeError, match='complete state bank'):
        owner.allocate_bank()  # auxiliary slots also remain occupied
    ready.synchronize()
    owner.complete(producer)
    for event, use in zip(done, reads):
        event.synchronize()
        torch.testing.assert_close(pool[checkpoint_id], checkpoint_bytes, rtol=0, atol=0)
        owner.complete(use)
    assert owner.get_num_free() == owner.capacity.num_units - 1
    assert not owner._units and not owner._banks
    next_bank = owner.allocate_bank()
    owner.complete(producer)
    for use in reads:
        owner.complete(use)
    assert owner._banks[next_bank.auxiliary_id] is next_bank
    owner.release_bank(next_bank)


def test_failed_producer_and_queued_cross_stream_reader_keep_physical_units_until_events():
    from lmdeploy.pytorch.config import CompactStateCapacity
    from lmdeploy.pytorch.paging.compact_state import CompactStateManager

    owner = CompactStateManager(CompactStateCapacity(3, 1, 1))
    producer_bank = owner.allocate_bank(stream=0)
    reader_bank = owner.allocate_bank(stream=1)
    checkpoint = owner.reserve_checkpoint(producer_bank, 64)
    producer = owner.submit_decode((producer_bank,), checkpoints=(checkpoint,))
    owner.bind_input(reader_bank, checkpoint)
    reader = owner.submit_decode((reader_bank,), stream=1, wait_for=(producer,))
    pool = torch.zeros((owner.capacity.num_units, 1024), device='cuda')
    producing, reading = torch.cuda.Stream(), torch.cuda.Stream()
    produced, consumed = torch.cuda.Event(), torch.cuda.Event()
    torch.cuda.synchronize()  # Fixture initialization only, not production failure handling.
    with torch.cuda.stream(producing):
        torch.cuda._sleep(100_000_000)
        pool[checkpoint.unit_id].fill_(7)
        produced.record()
    with torch.cuda.stream(reading):
        reading.wait_event(produced)
        pool[reader_bank.unit_ids[0]].copy_(pool[checkpoint.unit_id])
        consumed.record()
    owner.invalidate(producer.access_id)
    owner.release_bank(producer_bank)
    owner.release_bank(reader_bank)
    assert owner.get_num_free() == 0 and not producer.finished and not reader.finished
    with pytest.raises(RuntimeError, match='Failed compact work'):
        owner.acknowledge(reader.access_id)
    produced.synchronize()
    owner.complete(producer, success=False)
    assert checkpoint.unit_id in owner._units  # Reader remains live regardless of producer outcome.
    consumed.synchronize()
    assert torch.all(pool[reader_bank.unit_ids[0]] == 7)
    owner.complete(reader, success=False)
    assert not owner._units and not owner._banks and not owner._accesses
