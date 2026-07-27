"""Unit tests for the flatten_v4_kv Triton kernel and pack_window_tokens_fp8
kernel."""

import pytest
import torch

DEVICE = 'cuda'
DTYPE = torch.bfloat16


def _flat_kv_bounds(kv_seqlens, window_size, compress_ratio, start_pos=None,
                    q_seqlens=None):
    """Compute safe upper bounds for flatten_v4_kv (no GPU sync)."""
    kv = kv_seqlens.cpu() if kv_seqlens.is_cuda else kv_seqlens
    cr = compress_ratio if compress_ratio > 0 else 1

    if start_pos is not None and q_seqlens is not None:
        sp = start_pos.cpu() if start_pos.is_cuda else start_pos
        qs = q_seqlens.cpu() if q_seqlens.is_cuda else q_seqlens
        prev_window_lens = sp.clamp(max=window_size)
        raw_kv_lens = qs
    else:
        prev_window_lens = kv.clamp(max=window_size)
        raw_kv_lens = torch.zeros_like(kv)

    if compress_ratio > 0:
        num_compressed = torch.div(kv, cr, rounding_mode='floor').long()
    else:
        num_compressed = kv.new_zeros(kv.shape, dtype=torch.long)
    flat_kv_lens = prev_window_lens + raw_kv_lens + num_compressed
    return int(flat_kv_lens.sum().item()), int(flat_kv_lens.max().item())


def _reference_flatten_v4_kv(window_kv_cache, compressed_kv_cache, block_offsets,
                              kv_seqlens, window_size, compress_ratio,
                              raw_kv=None, raw_kv_lens=None, start_pos=None,
                              slot=None):
    """Python reference for flatten_v4_kv.

    Produces the same output as the Triton kernel: per-sequence flat KV with
    layout [prev_window (ring buffer) | raw_kv (current chunk) | compressed_kv].
    """
    bsz = kv_seqlens.size(0)
    head_dim = window_kv_cache.size(-1)
    entries_per_block = compressed_kv_cache.size(1) if compressed_kv_cache is not None else 1

    has_raw_kv = raw_kv is not None

    all_flat = []
    cu_seqlens = [0]

    for b in range(bsz):
        total_len = kv_seqlens[b].item()
        flat = []

        if has_raw_kv:
            sp = start_pos[b].item()
            prev_window_len = min(sp, window_size)
            rkv_len = raw_kv_lens[b].item()

            # Previous window region from ring buffer
            window_start = max(0, sp - window_size)
            for t in range(prev_window_len):
                actual_pos = window_start + t
                ring_pos = actual_pos % window_size
                if slot is not None:
                    s = slot[b].item()
                    if s < 0:
                        flat.append(torch.zeros(head_dim, dtype=window_kv_cache.dtype,
                                                 device=window_kv_cache.device))
                    else:
                        flat.append(window_kv_cache[s, ring_pos])
                else:
                    flat.append(window_kv_cache[b, ring_pos])

            # Raw KV region (current chunk)
            cu_raw = 0 if b == 0 else sum(raw_kv_lens[:b].tolist())
            for t in range(rkv_len):
                flat.append(raw_kv[cu_raw + t])
        else:
            # Legacy path: window from ring buffer (min(total_len, window_size))
            window_kv_len = min(total_len, window_size)
            window_start = max(0, total_len - window_size)
            for t in range(window_kv_len):
                actual_pos = window_start + t
                ring_pos = actual_pos % window_size
                flat.append(window_kv_cache[b, ring_pos])

        # Compressed region
        num_compressed = total_len // compress_ratio if compress_ratio > 0 else 0
        for c in range(num_compressed):
            if compressed_kv_cache is not None:
                page_id = c // entries_per_block
                page_off = c % entries_per_block
                phys_block = block_offsets[b, page_id].item()
                flat.append(compressed_kv_cache[phys_block, page_off])

        if flat:
            flat_tensor = torch.stack(flat)
        else:
            flat_tensor = torch.empty(0, head_dim, dtype=window_kv_cache.dtype,
                                      device=window_kv_cache.device)
        all_flat.append(flat_tensor)
        cu_seqlens.append(cu_seqlens[-1] + len(flat))

    flat_kv = torch.cat(all_flat, dim=0).unsqueeze(1)
    cu_seqlens_k = torch.tensor(cu_seqlens, dtype=torch.int32, device=kv_seqlens.device)

    return flat_kv, cu_seqlens_k


def _pack_bf16_window_to_fp8(window_kv, bsz, window_size, device):
    """Pack a BF16 window cache [bsz, window_size, 512] to FP8."""
    from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
    from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8

    head_dim = V4_FLASHMLA_HEAD_DIM
    packed_dim = 584
    fp8_window = torch.zeros(bsz, window_size, packed_dim,
                              dtype=torch.float8_e4m3fn, device=device)
    slot_all = torch.arange(bsz, device=device).repeat_interleave(window_size).long()
    pos_all = torch.arange(window_size, device=device).repeat(bsz).long()
    pack_window_tokens_fp8(window_kv.reshape(-1, head_dim), fp8_window, slot_all, pos_all)
    return fp8_window


def _pack_global_bf16_window_to_fp8(global_cache, num_slots, window_size, device):
    """Pack a global BF16 window cache [num_slots, window_size, 512] to FP8,
    per slot."""
    from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
    from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8

    head_dim = V4_FLASHMLA_HEAD_DIM
    packed_dim = 584
    fp8_window = torch.zeros(num_slots, window_size, packed_dim,
                              dtype=torch.float8_e4m3fn, device=device)
    for s in range(num_slots):
        slot_arr = torch.full((window_size,), s, dtype=torch.long, device=device)
        pos_arr = torch.arange(window_size, device=device, dtype=torch.long)
        pack_window_tokens_fp8(global_cache[s].reshape(-1, head_dim), fp8_window,
                               slot_arr, pos_arr)
    return fp8_window


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestFlattenV4KV:

    @pytest.fixture
    def device(self):
        yield DEVICE

    @pytest.fixture
    def dtype(self):
        yield DTYPE

    # ------------------------------------------------------------------
    # Window-only (compress_ratio=0), FP8 window
    # ------------------------------------------------------------------

    def test_fp8_window_short(self, device, dtype):
        """FP8 window: total_len < window_size, no ring wrap."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size = 1, 8
        head_dim = V4_FLASHMLA_HEAD_DIM
        total_len = 5
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = torch.randn(bsz, window_size, head_dim, dtype=dtype, device=device)
        window_kv[0, total_len:].zero_()

        fp8_window = _pack_bf16_window_to_fp8(window_kv, bsz, window_size, device)

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, None, block_offsets, total_lens, window_size, 0)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, 0, tot, mxf)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    def test_fp8_window_ring_wrap(self, device, dtype):
        """FP8 window: total_len > window_size, ring wraps."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size = 1, 4
        head_dim = V4_FLASHMLA_HEAD_DIM
        total_len = 20
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for p in range(total_len):
            ring_pos = p % window_size
            window_kv[0, ring_pos] = torch.randn(head_dim, dtype=dtype, device=device)

        fp8_window = _pack_bf16_window_to_fp8(window_kv, bsz, window_size, device)

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, None, block_offsets, total_lens, window_size, 0)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, 0, tot, mxf)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == [0, 4]

    def test_fp8_window_multi_batch(self, device, dtype):
        """FP8 window: multiple sequences with different lengths."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size = 2, 4
        head_dim = V4_FLASHMLA_HEAD_DIM
        total_lens = torch.tensor([6, 10], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for b in range(bsz):
            for p in range(total_lens[b].item()):
                ring_pos = p % window_size
                window_kv[b, ring_pos] = torch.randn(head_dim, dtype=dtype, device=device)

        fp8_window = _pack_bf16_window_to_fp8(window_kv, bsz, window_size, device)

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, None, block_offsets, total_lens, window_size, 0)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, 0, tot, mxf)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    # ------------------------------------------------------------------
    # FP8 window + slot indexing
    # ------------------------------------------------------------------

    def test_fp8_window_slot_indexing(self, device, dtype):
        """FP8 window with slot-based global cache indexing."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        num_slots, window_size = 4, 8
        head_dim = V4_FLASHMLA_HEAD_DIM
        total_lens = torch.tensor([3, 5], dtype=torch.long, device=device)
        bsz = total_lens.numel()

        global_cache = torch.zeros(num_slots, window_size, head_dim, dtype=dtype, device=device)
        for p in range(3):
            global_cache[2, p] = torch.randn(head_dim, dtype=dtype, device=device)
        for p in range(5):
            global_cache[0, p] = torch.randn(head_dim, dtype=dtype, device=device)

        fp8_window = _pack_global_bf16_window_to_fp8(global_cache, num_slots, window_size, device)

        slot = torch.tensor([2, 0], dtype=torch.long, device=device)
        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)

        batch_cache = global_cache[slot.long()]
        ref_kv, ref_cu = _reference_flatten_v4_kv(
            batch_cache, None, block_offsets, total_lens, window_size, 0)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, 0, tot, mxf,
            slot=slot)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    def test_fp8_window_negative_slot(self, device, dtype):
        """FP8 window: negative slot values produce all-zero window region."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        num_slots, window_size = 4, 8
        head_dim = V4_FLASHMLA_HEAD_DIM
        total_lens = torch.tensor([3, 0, 5], dtype=torch.long, device=device)
        bsz = total_lens.numel()

        global_cache = torch.zeros(num_slots, window_size, head_dim, dtype=dtype, device=device)
        for p in range(3):
            global_cache[1, p] = torch.randn(head_dim, dtype=dtype, device=device)
        for p in range(5):
            global_cache[3, p] = torch.randn(head_dim, dtype=dtype, device=device)

        fp8_window = _pack_global_bf16_window_to_fp8(global_cache, num_slots, window_size, device)

        slot = torch.tensor([1, -1, 3], dtype=torch.long, device=device)
        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)

        batch_cache = global_cache[slot.clamp(min=0).long()]
        ref_kv, ref_cu = _reference_flatten_v4_kv(
            batch_cache, None, block_offsets, total_lens, window_size, 0, slot=slot)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, 0, tot, mxf,
            slot=slot)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        if cu[1].item() < cu[2].item():
            seq1 = flat_kv[cu[1]:cu[2], 0, :].cpu()
            assert seq1.abs().max().item() < 0.15, 'Negative slot window should be near-zero'

    # ------------------------------------------------------------------
    # FP8 window + FP8 compressed
    # ------------------------------------------------------------------

    def test_fp8_window_fp8_compress(self, device, dtype):
        """FP8 window + FP8 compressed KV cache combined path."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        from .dsv4_utils import quantize_v4_flashmla_sparse

        bsz, window_size = 1, 4
        compress_ratio = 4
        head_dim = V4_FLASHMLA_HEAD_DIM
        block_size = 4
        total_len = 10
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for p in range(total_len):
            ring_pos = p % window_size
            window_kv[0, ring_pos] = torch.randn(head_dim, dtype=dtype, device=device)

        fp8_window = _pack_bf16_window_to_fp8(window_kv, bsz, window_size, device)

        num_comp = total_len // compress_ratio
        block_offsets = torch.tensor([[0]], dtype=torch.long, device=device)
        compressed_kv_bf16 = torch.randn(1, block_size, head_dim, dtype=dtype, device=device)
        for e in range(num_comp, block_size):
            compressed_kv_bf16[0, e].zero_()

        fp8_compress = quantize_v4_flashmla_sparse(
            compressed_kv_bf16.unsqueeze(2)).squeeze(2)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, compressed_kv_bf16, block_offsets, total_lens,
            window_size, compress_ratio)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, compress_ratio,
            tot, mxf,
            fp8_compressed_kv_cache=fp8_compress)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    # ------------------------------------------------------------------
    # FP8 window + raw_kv (chunked prefill)
    # ------------------------------------------------------------------

    def test_fp8_window_raw_kv_chunked(self, device, dtype):
        """FP8 window: chunked prefill with prev_window from FP8 + raw_kv."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size = 1, 4
        head_dim = V4_FLASHMLA_HEAD_DIM
        start_pos_val = 8
        q_seqlens_val = 3
        total_len = start_pos_val + q_seqlens_val
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)
        start_pos = torch.tensor([start_pos_val], dtype=torch.long, device=device)
        q_seqlens = torch.tensor([q_seqlens_val], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for p in range(start_pos_val):
            ring_pos = p % window_size
            window_kv[0, ring_pos] = torch.randn(head_dim, dtype=dtype, device=device)

        fp8_window = _pack_bf16_window_to_fp8(window_kv, bsz, window_size, device)

        raw_kv = torch.randn(q_seqlens_val, head_dim, dtype=dtype, device=device)

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, None, block_offsets, total_lens, window_size, 0,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens, start_pos=start_pos)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0, start_pos, q_seqlens)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, 0, tot, mxf,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens, start_pos=start_pos)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    def test_fp8_window_raw_kv_first_time(self, device, dtype):
        """FP8 window: first-time prefill (start_pos=0), all tokens from raw_kv."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size = 1, 4
        head_dim = V4_FLASHMLA_HEAD_DIM
        total_len = 10
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)
        start_pos = torch.tensor([0], dtype=torch.long, device=device)
        q_seqlens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        fp8_window = _pack_bf16_window_to_fp8(window_kv, bsz, window_size, device)

        raw_kv = torch.randn(total_len, head_dim, dtype=dtype, device=device)

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, None, block_offsets, total_lens, window_size, 0,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens, start_pos=start_pos)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0, start_pos, q_seqlens)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, 0, tot, mxf,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens, start_pos=start_pos)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0, rtol=0)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    def test_fp8_window_raw_kv_fp8_compress(self, device, dtype):
        """Full path: FP8 window + raw_kv + FP8 compressed KV."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        from .dsv4_utils import quantize_v4_flashmla_sparse

        bsz, window_size = 1, 4
        compress_ratio = 4
        head_dim = V4_FLASHMLA_HEAD_DIM
        block_size = 4
        start_pos_val = 8
        q_seqlens_val = 4
        total_len = start_pos_val + q_seqlens_val
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)
        start_pos = torch.tensor([start_pos_val], dtype=torch.long, device=device)
        q_seqlens = torch.tensor([q_seqlens_val], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for p in range(start_pos_val):
            ring_pos = p % window_size
            window_kv[0, ring_pos] = torch.randn(head_dim, dtype=dtype, device=device)

        fp8_window = _pack_bf16_window_to_fp8(window_kv, bsz, window_size, device)

        raw_kv = torch.randn(q_seqlens_val, head_dim, dtype=dtype, device=device)

        block_offsets = torch.tensor([[0]], dtype=torch.long, device=device)
        compressed_kv_bf16 = torch.randn(1, block_size, head_dim, dtype=dtype, device=device)

        fp8_compress = quantize_v4_flashmla_sparse(
            compressed_kv_bf16.unsqueeze(2)).squeeze(2)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, compressed_kv_bf16, block_offsets, total_lens,
            window_size, compress_ratio,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens, start_pos=start_pos)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio, start_pos, q_seqlens)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, compress_ratio, tot, mxf,
            fp8_compressed_kv_cache=fp8_compress,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens, start_pos=start_pos)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    def test_fp8_window_raw_kv_mixed_batch(self, device, dtype):
        """FP8 window: mixed batch — first-time prefill + chunked prefill."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size = 2, 4
        head_dim = V4_FLASHMLA_HEAD_DIM
        total_lens = torch.tensor([6, 11], dtype=torch.long, device=device)
        start_pos = torch.tensor([0, 8], dtype=torch.long, device=device)
        q_seqlens = torch.tensor([6, 3], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for p in range(8):
            ring_pos = p % window_size
            window_kv[1, ring_pos] = torch.randn(head_dim, dtype=dtype, device=device)

        fp8_window = _pack_bf16_window_to_fp8(window_kv, bsz, window_size, device)

        raw_kv = torch.randn(9, head_dim, dtype=dtype, device=device)

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, None, block_offsets, total_lens, window_size, 0,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens, start_pos=start_pos)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0, start_pos, q_seqlens)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, 0, tot, mxf,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens, start_pos=start_pos)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    def test_fp8_window_slot_raw_kv_compress(self, device, dtype):
        """Full integration: FP8 window + slot + raw_kv + FP8 compressed."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        from .dsv4_utils import quantize_v4_flashmla_sparse

        num_slots, window_size = 4, 4
        compress_ratio = 4
        head_dim = V4_FLASHMLA_HEAD_DIM
        block_size = 4
        total_lens = torch.tensor([10, 12], dtype=torch.long, device=device)
        start_pos = torch.tensor([6, 8], dtype=torch.long, device=device)
        q_seqlens = torch.tensor([4, 4], dtype=torch.long, device=device)
        bsz = total_lens.numel()

        global_cache = torch.zeros(num_slots, window_size, head_dim, dtype=dtype, device=device)
        for p in range(6):
            global_cache[1, p % window_size] = torch.randn(head_dim, dtype=dtype, device=device)
        for p in range(8):
            global_cache[3, p % window_size] = torch.randn(head_dim, dtype=dtype, device=device)

        fp8_window = _pack_global_bf16_window_to_fp8(global_cache, num_slots, window_size, device)

        slot = torch.tensor([1, 3], dtype=torch.long, device=device)
        raw_kv = torch.randn(8, head_dim, dtype=dtype, device=device)

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        block_offsets[0, 0] = 0
        block_offsets[1, 0] = 1
        compressed_kv_bf16 = torch.randn(2, block_size, head_dim, dtype=dtype, device=device)

        fp8_compress = quantize_v4_flashmla_sparse(
            compressed_kv_bf16.unsqueeze(2)).squeeze(2)

        batch_cache = global_cache[slot.long()]
        ref_kv, ref_cu = _reference_flatten_v4_kv(
            batch_cache, compressed_kv_bf16, block_offsets, total_lens,
            window_size, compress_ratio,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens, start_pos=start_pos)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio, start_pos, q_seqlens)
        flat_kv, cu = flatten_v4_kv(
            fp8_window, block_offsets, total_lens, window_size, compress_ratio, tot, mxf,
            fp8_compressed_kv_cache=fp8_compress,
            slot=slot,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens, start_pos=start_pos)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestPackWindowTokensFP8:
    """Tests for the pack_window_tokens_fp8 Triton kernel."""

    @pytest.fixture
    def device(self):
        yield DEVICE

    @pytest.fixture
    def dtype(self):
        yield DTYPE

    def test_round_trip_single_token(self, device, dtype):
        """Pack one token, dequantize, compare vs original."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8

        from .dsv4_utils import dequantize_v4_flashmla_sparse

        window_size = 4
        packed_dim = 584
        kv = torch.randn(1, V4_FLASHMLA_HEAD_DIM, dtype=dtype, device=device)
        fp8_cache = torch.zeros(1, window_size, packed_dim, dtype=torch.float8_e4m3fn, device=device)

        slot = torch.tensor([0], dtype=torch.long, device=device)
        positions = torch.tensor([2], dtype=torch.long, device=device)

        pack_window_tokens_fp8(kv, fp8_cache, slot, positions)

        deq = dequantize_v4_flashmla_sparse(fp8_cache.unsqueeze(2)).squeeze(2)
        err = (deq[0, 2] - kv[0]).abs().max().item()
        assert err < 0.15, f'Max error {err}'

    def test_round_trip_multiple_tokens(self, device, dtype):
        """Pack multiple tokens, dequantize, compare."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8

        from .dsv4_utils import dequantize_v4_flashmla_sparse

        num_tokens = 3
        window_size = 4
        packed_dim = 584
        kv = torch.randn(num_tokens, V4_FLASHMLA_HEAD_DIM, dtype=dtype, device=device)
        fp8_cache = torch.zeros(1, window_size, packed_dim, dtype=torch.float8_e4m3fn, device=device)

        slot = torch.tensor([0, 0, 0], dtype=torch.long, device=device)
        positions = torch.tensor([0, 1, 2], dtype=torch.long, device=device)

        pack_window_tokens_fp8(kv, fp8_cache, slot, positions)

        deq = dequantize_v4_flashmla_sparse(fp8_cache.unsqueeze(2)).squeeze(2)
        for i in range(num_tokens):
            err = (deq[0, i] - kv[i]).abs().max().item()
            assert err < 0.15, f'Token {i} max error {err}'

    def test_match_block_quantize(self, device, dtype):
        """Pack all tokens individually via kernel, compare vs block-level
        quantize."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8

        from .dsv4_utils import dequantize_v4_flashmla_sparse, quantize_v4_flashmla_sparse

        num_slots = 2
        window_size = 4
        packed_dim = 584
        head_dim = V4_FLASHMLA_HEAD_DIM

        block_kv = torch.randn(num_slots, window_size, 1, head_dim, dtype=dtype, device=device)
        packed_ref = quantize_v4_flashmla_sparse(block_kv)
        deq_ref = dequantize_v4_flashmla_sparse(packed_ref).squeeze(2)

        fp8_kernel = torch.zeros(num_slots, window_size, packed_dim, dtype=torch.float8_e4m3fn, device=device)
        slot_all = torch.arange(num_slots, device=device).repeat_interleave(window_size).long()
        pos_all = torch.arange(window_size, device=device).repeat(num_slots).long()
        kv_all = block_kv.squeeze(2).reshape(-1, head_dim)

        pack_window_tokens_fp8(kv_all, fp8_kernel, slot_all, pos_all)

        deq_kernel = dequantize_v4_flashmla_sparse(fp8_kernel.unsqueeze(2)).squeeze(2)

        max_diff = (deq_kernel - deq_ref).abs().max().item()
        assert max_diff == 0.0, f'Kernel vs block pack differ by {max_diff}'

    def test_multi_slot(self, device, dtype):
        """Pack tokens into different slots."""
        from lmdeploy.pytorch.consts import V4_FLASHMLA_HEAD_DIM
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8

        from .dsv4_utils import dequantize_v4_flashmla_sparse

        window_size = 4
        packed_dim = 584
        kv = torch.randn(3, V4_FLASHMLA_HEAD_DIM, dtype=dtype, device=device)
        fp8_cache = torch.zeros(2, window_size, packed_dim, dtype=torch.float8_e4m3fn, device=device)

        slot = torch.tensor([0, 1, 1], dtype=torch.long, device=device)
        positions = torch.tensor([0, 1, 3], dtype=torch.long, device=device)

        pack_window_tokens_fp8(kv, fp8_cache, slot, positions)

        deq = dequantize_v4_flashmla_sparse(fp8_cache.unsqueeze(2)).squeeze(2)
        for i, (s, p) in enumerate(zip(slot.tolist(), positions.tolist())):
            err = (deq[s, p] - kv[i]).abs().max().item()
            assert err < 0.15, f'Token {i} (slot={s}, pos={p}) error {err}'
