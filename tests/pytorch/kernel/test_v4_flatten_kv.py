"""Unit tests for the flatten_v4_kv Triton kernel and pack_window_tokens_fp8 kernel."""

import pytest
import torch

DEVICE = 'cuda'
DTYPE = torch.bfloat16


def _flat_kv_bounds(kv_seqlens, window_size, compress_ratio):
    """Compute safe upper bounds for flatten_v4_kv (no GPU sync)."""
    kv = kv_seqlens.cpu() if kv_seqlens.is_cuda else kv_seqlens
    cr = compress_ratio if compress_ratio > 0 else 1
    window_kv_lens = kv.clamp(max=window_size)
    num_compressed = torch.div(kv, cr, rounding_mode='floor').long() if compress_ratio > 0 else kv.new_zeros(kv.shape, dtype=torch.long)
    flat_kv_lens = window_kv_lens + num_compressed
    return int(flat_kv_lens.sum().item()), int(flat_kv_lens.max().item())


def _reference_flatten_v4_kv(window_kv_cache, compressed_kv_cache, block_offsets,
                              kv_seqlens, window_size, compress_ratio):
    """Python reference for flatten_v4_kv.

    Produces the same output as the Triton kernel: per-sequence flat KV with
    window region (chronological from ring buffer) followed by compressed region
    (sequential from paged cache).
    """
    bsz = kv_seqlens.size(0)
    head_dim = window_kv_cache.size(-1)
    entries_per_block = compressed_kv_cache.size(1) if compressed_kv_cache is not None else 1

    all_flat = []
    cu_seqlens = [0]

    for b in range(bsz):
        total_len = kv_seqlens[b].item()
        window_kv_len = min(total_len, window_size)
        window_start = max(0, total_len - window_size)

        flat = []

        # Window region: chronological from ring buffer
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


def _make_ring_buffer_window(bsz, window_size, head_dim, total_lens, token_fn):
    """Populate a ring-buffer window cache.

    token_fn(batch_idx, position) -> scalar value to store (broadcast over head_dim).
    Writes each token at position % window_size, mimicking the real ring-buffer write.
    """
    window_kv = torch.zeros(bsz, window_size, head_dim, dtype=DTYPE, device=DEVICE)
    for b in range(bsz):
        tl = total_lens[b].item()
        for p in range(tl):
            ring_pos = p % window_size
            window_kv[b, ring_pos] = token_fn(b, p)
    return window_kv


def _make_paged_cache(num_blocks, entries_per_block, head_dim, num_entries,
                       block_offsets, entry_fn):
    """Populate a paged compressed KV cache.

    entry_fn(batch_idx, entry_idx) -> scalar value (broadcast over head_dim).
    Scatters entries into the correct physical blocks.
    """
    compressed_kv = torch.zeros(num_blocks, entries_per_block, head_dim,
                                dtype=DTYPE, device=DEVICE)
    for b in range(block_offsets.size(0)):
        for e in range(num_entries[b].item()):
            page_id = e // entries_per_block
            page_off = e % entries_per_block
            phys_block = block_offsets[b, page_id].item()
            compressed_kv[phys_block, page_off] = entry_fn(b, e)
    return compressed_kv


class TestFlattenV4KV:

    @pytest.fixture
    def device(self):
        yield DEVICE

    @pytest.fixture
    def dtype(self):
        yield DTYPE

    # ------------------------------------------------------------------
    # Window-only (compress_ratio=0)
    # ------------------------------------------------------------------

    def test_window_only_short(self, device, dtype):
        """total_len < window_size: no ring wrap, no compressed entries."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 2, 8, 16
        total_lens = torch.tensor([3, 5], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for b in range(bsz):
            for p in range(total_lens[b].item()):
                window_kv[b, p] = b * 100 + p

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(window_kv, None, block_offsets,
                                    total_lens, window_size, 0, tot, mxf)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, None, block_offsets, total_lens, window_size, 0)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0, rtol=0)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    def test_window_only_equal_window(self, device, dtype):
        """total_len == window_size: exactly fills the ring, no wrap."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 6, 8
        total_lens = torch.tensor([6], dtype=torch.long, device=device)

        window_kv = _make_ring_buffer_window(
            bsz, window_size, head_dim, total_lens,
            token_fn=lambda b, p: p)

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(window_kv, None, block_offsets,
                                    total_lens, window_size, 0, tot, mxf)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, None, block_offsets, total_lens, window_size, 0)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0, rtol=0)
        assert cu.cpu().tolist() == [0, 6]

    def test_window_only_long(self, device, dtype):
        """total_len >> window_size: ring wraps multiple times."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 4, 8
        total_lens = torch.tensor([20], dtype=torch.long, device=device)

        window_kv = _make_ring_buffer_window(
            bsz, window_size, head_dim, total_lens,
            token_fn=lambda b, p: p)

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(window_kv, None, block_offsets,
                                    total_lens, window_size, 0, tot, mxf)

        # Expected: last 4 tokens = 16, 17, 18, 19 in chronological order
        vals = flat_kv[:, 0, 0].cpu().tolist()
        assert vals == [16.0, 17.0, 18.0, 19.0]
        assert cu.cpu().tolist() == [0, 4]

    # ------------------------------------------------------------------
    # Window + compressed (ratio=4)
    # ------------------------------------------------------------------

    def test_ratio4_short(self, device, dtype):
        """total_len < window_size with ratio=4: window + 1 compressed entry."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 8, 16
        compress_ratio = 4
        block_size = 4
        total_len = 5
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for p in range(total_len):
            window_kv[0, p] = p

        num_comp = total_len // compress_ratio  # 1
        block_offsets = torch.tensor([[0]], dtype=torch.long, device=device)
        compressed_kv = torch.zeros(1, block_size, head_dim, dtype=dtype, device=device)
        compressed_kv[0, 0] = 100

        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio)
        flat_kv, cu = flatten_v4_kv(window_kv, compressed_kv, block_offsets,
                                    total_lens, window_size, compress_ratio, tot, mxf)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, compressed_kv, block_offsets, total_lens,
            window_size, compress_ratio)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0, rtol=0)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    def test_ratio4_ring_wrap(self, device, dtype):
        """total_len > window_size with ratio=4: ring wraps, multiple compressed."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 4, 8
        compress_ratio = 4
        block_size = 4
        total_len = 10
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = _make_ring_buffer_window(
            bsz, window_size, head_dim, total_lens,
            token_fn=lambda b, p: p)

        num_comp = total_len // compress_ratio  # 2
        block_offsets = torch.tensor([[0]], dtype=torch.long, device=device)
        compressed_kv = torch.zeros(1, block_size, head_dim, dtype=dtype, device=device)
        compressed_kv[0, 0] = 100
        compressed_kv[0, 1] = 101

        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio)
        flat_kv, cu = flatten_v4_kv(window_kv, compressed_kv, block_offsets,
                                    total_lens, window_size, compress_ratio, tot, mxf)

        # Window chronological: tokens 6,7,8,9
        vals = flat_kv[:, 0, 0].cpu().tolist()
        assert vals[:4] == [6.0, 7.0, 8.0, 9.0], f"Window: {vals[:4]}"
        assert vals[4:6] == [100.0, 101.0], f"Compressed: {vals[4:6]}"
        assert cu.cpu().tolist() == [0, 6]

    def test_ratio4_multi_block(self, device, dtype):
        """Compressed entries span multiple pages in the paged cache."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 4, 8
        compress_ratio = 4
        block_size = 2  # small block so 2 entries = 2 blocks
        total_len = 12
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = _make_ring_buffer_window(
            bsz, window_size, head_dim, total_lens,
            token_fn=lambda b, p: p)

        num_comp = total_len // compress_ratio  # 3
        # 3 entries with block_size=2: needs 2 physical blocks
        block_offsets = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        compressed_kv = torch.zeros(2, block_size, head_dim, dtype=dtype, device=device)
        compressed_kv[0, 0] = 200
        compressed_kv[0, 1] = 201
        compressed_kv[1, 0] = 202

        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio)
        flat_kv, cu = flatten_v4_kv(window_kv, compressed_kv, block_offsets,
                                    total_lens, window_size, compress_ratio, tot, mxf)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, compressed_kv, block_offsets, total_lens,
            window_size, compress_ratio)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0, rtol=0)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    # ------------------------------------------------------------------
    # Window + compressed (ratio=128)
    # ------------------------------------------------------------------

    def test_ratio128(self, device, dtype):
        """ratio=128 with large total_len, many compressed entries."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 4, 8
        compress_ratio = 128
        block_size = 4
        total_len = 512
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = _make_ring_buffer_window(
            bsz, window_size, head_dim, total_lens,
            token_fn=lambda b, p: p % 100)

        num_comp = total_len // compress_ratio  # 4
        num_blocks = (num_comp + block_size - 1) // block_size  # 1
        block_offsets = torch.zeros(1, num_blocks, dtype=torch.long, device=device)
        compressed_kv = torch.zeros(num_blocks, block_size, head_dim,
                                    dtype=dtype, device=device)
        for e in range(num_comp):
            compressed_kv[e // block_size, e % block_size] = 500 + e

        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio)
        flat_kv, cu = flatten_v4_kv(window_kv, compressed_kv, block_offsets,
                                    total_lens, window_size, compress_ratio, tot, mxf)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, compressed_kv, block_offsets, total_lens,
            window_size, compress_ratio)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0, rtol=0)

    # ------------------------------------------------------------------
    # Multi-batch
    # ------------------------------------------------------------------

    def test_batched_different_lengths(self, device, dtype):
        """Two sequences with different total_len and different compressed counts."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 2, 4, 8
        compress_ratio = 4
        block_size = 4
        total_lens = torch.tensor([6, 10], dtype=torch.long, device=device)

        window_kv = _make_ring_buffer_window(
            bsz, window_size, head_dim, total_lens,
            token_fn=lambda b, p: b * 100 + p)

        # seq0: 6//4=1 compressed, seq1: 10//4=2 compressed
        num_comp = total_lens // compress_ratio  # [1, 2]
        max_blocks = (num_comp.max() + block_size - 1) // block_size
        block_offsets = torch.zeros(bsz, max_blocks, dtype=torch.long, device=device)
        for b in range(bsz):
            block_offsets[b, 0] = b

        compressed_kv = torch.zeros(bsz, block_size, head_dim,
                                    dtype=dtype, device=device)
        compressed_kv[0, 0] = 200
        compressed_kv[1, 0] = 300
        compressed_kv[1, 1] = 301

        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio)
        flat_kv, cu = flatten_v4_kv(window_kv, compressed_kv, block_offsets,
                                    total_lens, window_size, compress_ratio, tot, mxf)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, compressed_kv, block_offsets, total_lens,
            window_size, compress_ratio)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0, rtol=0)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    def test_batched_one_seq_no_compress(self, device, dtype):
        """Mixed: one sequence with compression, one without (ratio=0)."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        # ratio=0 means no compressed entries for any sequence
        bsz, window_size, head_dim = 2, 8, 16
        total_lens = torch.tensor([5, 3], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for b in range(bsz):
            for p in range(total_lens[b].item()):
                window_kv[b, p] = b * 50 + p

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(window_kv, None, block_offsets,
                                    total_lens, window_size, 0, tot, mxf)

        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, None, block_offsets, total_lens, window_size, 0)

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0, rtol=0)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_single_token(self, device, dtype):
        """total_len=1: single window token, no compressed."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 4, 8
        total_lens = torch.tensor([1], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        window_kv[0, 0] = 42

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(window_kv, None, block_offsets,
                                    total_lens, window_size, 0, tot, mxf)

        assert flat_kv.shape == (1, 1, head_dim)
        assert flat_kv[0, 0, 0].item() == 42.0
        assert cu.cpu().tolist() == [0, 1]

    def test_exactly_one_compressed_entry(self, device, dtype):
        """total_len == compress_ratio: exactly 1 compressed entry."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 4, 8
        compress_ratio = 4
        block_size = 4
        total_len = 4
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = _make_ring_buffer_window(
            bsz, window_size, head_dim, total_lens,
            token_fn=lambda b, p: p)

        block_offsets = torch.tensor([[0]], dtype=torch.long, device=device)
        compressed_kv = torch.zeros(1, block_size, head_dim, dtype=dtype, device=device)
        compressed_kv[0, 0] = 99

        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio)
        flat_kv, cu = flatten_v4_kv(window_kv, compressed_kv, block_offsets,
                                    total_lens, window_size, compress_ratio, tot, mxf)

        # window_kv_len = min(4, 4) = 4, num_compressed = 4//4 = 1
        vals = flat_kv[:, 0, 0].cpu().tolist()
        assert vals[:4] == [0.0, 1.0, 2.0, 3.0]
        assert vals[4] == 99.0
        assert cu.cpu().tolist() == [0, 5]

    def test_total_len_below_compress_ratio(self, device, dtype):
        """total_len < compress_ratio: 0 compressed entries."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 8, 16
        compress_ratio = 4
        total_len = 3
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for p in range(total_len):
            window_kv[0, p] = p

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        compressed_kv = torch.zeros(1, 1, head_dim, dtype=dtype, device=device)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio)
        flat_kv, cu = flatten_v4_kv(window_kv, compressed_kv, block_offsets,
                                    total_lens, window_size, compress_ratio, tot, mxf)

        # 3 < 4 → 0 compressed, flat_kv_len = 3
        assert flat_kv.shape[0] == 3
        vals = flat_kv[:, 0, 0].cpu().tolist()
        assert vals == [0.0, 1.0, 2.0]
        assert cu.cpu().tolist() == [0, 3]

    # ------------------------------------------------------------------
    # FP8 compressed cache path
    # ------------------------------------------------------------------

    def test_fp8_compressed_path(self, device, dtype):
        """Flatten with FP8 compressed cache: dequantize → compare vs BF16 reference."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv
        from lmdeploy.pytorch.backends.cuda.attention.flashmla_utils import (
            MODEL1_D, quantize_model1_fp8_sparse)

        bsz, window_size = 1, 4
        compress_ratio = 4
        head_dim = MODEL1_D  # must be 512 for FP8 path
        block_size = 4
        total_len = 10
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        # BF16 window cache (ring buffer)
        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for p in range(total_len):
            ring_pos = p % window_size
            window_kv[0, ring_pos] = torch.randn(head_dim, dtype=dtype, device=device)

        # BF16 compressed cache (paged)
        num_comp = total_len // compress_ratio  # 2
        block_offsets = torch.tensor([[0]], dtype=torch.long, device=device)
        compressed_kv_bf16 = torch.randn(1, block_size, head_dim,
                                         dtype=dtype, device=device)
        # Zero out unused entries
        for e in range(num_comp, block_size):
            compressed_kv_bf16[0, e].zero_()

        # BF16 reference
        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, compressed_kv_bf16, block_offsets, total_lens,
            window_size, compress_ratio)

        # Quantize compressed cache to FP8 MODEL1 format
        fp8_cache = quantize_model1_fp8_sparse(
            compressed_kv_bf16.unsqueeze(2))  # [1, block_size, 1, packed_dim]

        # Flatten using FP8 path
        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio)
        flat_kv, cu = flatten_v4_kv(
            window_kv, None, block_offsets, total_lens,
            window_size, compress_ratio, tot, mxf,
            fp8_compressed_kv_cache=fp8_cache.squeeze(2))  # [1, block_size, packed_dim]

        # FP8 quantization has limited precision — allow small tolerance
        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    def test_fp8_multi_batch(self, device, dtype):
        """FP8 path with multiple sequences of different lengths."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv
        from lmdeploy.pytorch.backends.cuda.attention.flashmla_utils import (
            MODEL1_D, quantize_model1_fp8_sparse)

        bsz, window_size = 2, 4
        compress_ratio = 4
        head_dim = MODEL1_D
        block_size = 4
        total_lens = torch.tensor([8, 12], dtype=torch.long, device=device)

        # BF16 window cache
        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for b in range(bsz):
            for p in range(total_lens[b].item()):
                ring_pos = p % window_size
                window_kv[b, ring_pos] = torch.randn(head_dim, dtype=dtype, device=device)

        # BF16 compressed cache
        num_comp = total_lens // compress_ratio  # [2, 3]
        max_blocks = (num_comp.max() + block_size - 1) // block_size
        block_offsets = torch.zeros(bsz, max_blocks, dtype=torch.long, device=device)
        for b in range(bsz):
            block_offsets[b, 0] = b

        compressed_kv_bf16 = torch.zeros(bsz * max_blocks, block_size, head_dim,
                                         dtype=dtype, device=device)
        for b in range(bsz):
            for e in range(num_comp[b].item()):
                phys_block = block_offsets[b, e // block_size].item()
                off = e % block_size
                compressed_kv_bf16[phys_block, off] = torch.randn(
                    head_dim, dtype=dtype, device=device)

        # BF16 reference
        ref_kv, ref_cu = _reference_flatten_v4_kv(
            window_kv, compressed_kv_bf16, block_offsets, total_lens,
            window_size, compress_ratio)

        # Quantize to FP8
        fp8_input = compressed_kv_bf16.unsqueeze(2)  # [num_blocks, block_size, 1, head_dim]
        fp8_cache = quantize_model1_fp8_sparse(fp8_input)

        # Flatten using FP8 path
        tot, mxf = _flat_kv_bounds(total_lens, window_size, compress_ratio)
        flat_kv, cu = flatten_v4_kv(
            window_kv, None, block_offsets, total_lens,
            window_size, compress_ratio, tot, mxf,
            fp8_compressed_kv_cache=fp8_cache.squeeze(2))

        torch.testing.assert_close(flat_kv.cpu(), ref_kv.cpu(), atol=0.1, rtol=0.05)
        assert cu.cpu().tolist() == ref_cu.cpu().tolist()

    # ------------------------------------------------------------------
    # cu_seqlens_k passthrough
    # ------------------------------------------------------------------

    def test_provided_cu_seqlens_k(self, device, dtype):
        """When cu_seqlens_k is provided, it should be used as-is."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 8, 8
        total_len = 5
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = torch.zeros(bsz, window_size, head_dim, dtype=dtype, device=device)
        for p in range(total_len):
            window_kv[0, p] = p

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)

        # Provide cu_seqlens_k explicitly
        cu_seqlens_k = torch.tensor([0, 5], dtype=torch.int32, device=device)

        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu_out = flatten_v4_kv(
            window_kv, None, block_offsets, total_lens, window_size, 0, tot, mxf,
            cu_seqlens_k=cu_seqlens_k)

        assert cu_out.data_ptr() == cu_seqlens_k.data_ptr() or cu_out.cpu().tolist() == [0, 5]
        vals = flat_kv[:, 0, 0].cpu().tolist()
        assert vals == [0.0, 1.0, 2.0, 3.0, 4.0]

    # ------------------------------------------------------------------
    # Ring buffer wrapping stress
    # ------------------------------------------------------------------

    def test_ring_wrap_many_cycles(self, device, dtype):
        """Ring buffer wraps many times — verify only the last window_size tokens."""
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv

        bsz, window_size, head_dim = 1, 4, 8
        total_len = 100
        total_lens = torch.tensor([total_len], dtype=torch.long, device=device)

        window_kv = _make_ring_buffer_window(
            bsz, window_size, head_dim, total_lens,
            token_fn=lambda b, p: p % 256)

        block_offsets = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        tot, mxf = _flat_kv_bounds(total_lens, window_size, 0)
        flat_kv, cu = flatten_v4_kv(window_kv, None, block_offsets,
                                    total_lens, window_size, 0, tot, mxf)

        # Expected: tokens 96, 97, 98, 99 (mod 256)
        vals = flat_kv[:, 0, 0].cpu().tolist()
        assert vals == [96.0, 97.0, 98.0, 99.0]


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
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8
        from lmdeploy.pytorch.backends.cuda.attention.flashmla_utils import (
            MODEL1_D, dequantize_model1_fp8_sparse)

        window_size = 4
        packed_dim = 584
        kv = torch.randn(1, MODEL1_D, dtype=dtype, device=device)
        fp8_cache = torch.zeros(1, window_size, packed_dim, dtype=torch.float8_e4m3fn, device=device)

        slot = torch.tensor([0], dtype=torch.long, device=device)
        positions = torch.tensor([2], dtype=torch.long, device=device)

        pack_window_tokens_fp8(kv, fp8_cache, slot, positions)

        deq = dequantize_model1_fp8_sparse(fp8_cache.unsqueeze(2)).squeeze(2)
        err = (deq[0, 2] - kv[0]).abs().max().item()
        assert err < 0.15, f'Max error {err}'

    def test_round_trip_multiple_tokens(self, device, dtype):
        """Pack multiple tokens, dequantize, compare."""
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8
        from lmdeploy.pytorch.backends.cuda.attention.flashmla_utils import (
            MODEL1_D, dequantize_model1_fp8_sparse)

        num_tokens = 3
        window_size = 4
        packed_dim = 584
        kv = torch.randn(num_tokens, MODEL1_D, dtype=dtype, device=device)
        fp8_cache = torch.zeros(1, window_size, packed_dim, dtype=torch.float8_e4m3fn, device=device)

        slot = torch.tensor([0, 0, 0], dtype=torch.long, device=device)
        positions = torch.tensor([0, 1, 2], dtype=torch.long, device=device)

        pack_window_tokens_fp8(kv, fp8_cache, slot, positions)

        deq = dequantize_model1_fp8_sparse(fp8_cache.unsqueeze(2)).squeeze(2)
        for i in range(num_tokens):
            err = (deq[0, i] - kv[i]).abs().max().item()
            assert err < 0.15, f'Token {i} max error {err}'

    def test_match_block_quantize(self, device, dtype):
        """Pack all tokens individually via kernel, compare vs block-level quantize."""
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8
        from lmdeploy.pytorch.backends.cuda.attention.flashmla_utils import (
            MODEL1_D, quantize_model1_fp8_sparse, dequantize_model1_fp8_sparse)

        num_slots = 2
        window_size = 4
        packed_dim = 584
        head_dim = MODEL1_D

        # Block-level reference
        block_kv = torch.randn(num_slots, window_size, 1, head_dim, dtype=dtype, device=device)
        packed_ref = quantize_model1_fp8_sparse(block_kv)
        fp8_ref = packed_ref.squeeze(2).clone()
        deq_ref = dequantize_model1_fp8_sparse(packed_ref).squeeze(2)

        # Per-token kernel pack
        fp8_kernel = torch.zeros(num_slots, window_size, packed_dim, dtype=torch.float8_e4m3fn, device=device)
        slot_all = torch.arange(num_slots, device=device).repeat_interleave(window_size).long()
        pos_all = torch.arange(window_size, device=device).repeat(num_slots).long()
        kv_all = block_kv.squeeze(2).reshape(-1, head_dim)

        pack_window_tokens_fp8(kv_all, fp8_kernel, slot_all, pos_all)

        deq_kernel = dequantize_model1_fp8_sparse(fp8_kernel.unsqueeze(2)).squeeze(2)

        # Should be exactly identical (same quantization logic)
        max_diff = (deq_kernel - deq_ref).abs().max().item()
        assert max_diff == 0.0, f'Kernel vs block pack differ by {max_diff}'

    def test_multi_slot(self, device, dtype):
        """Pack tokens into different slots."""
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8
        from lmdeploy.pytorch.backends.cuda.attention.flashmla_utils import (
            MODEL1_D, dequantize_model1_fp8_sparse)

        window_size = 4
        packed_dim = 584
        kv = torch.randn(3, MODEL1_D, dtype=dtype, device=device)
        fp8_cache = torch.zeros(2, window_size, packed_dim, dtype=torch.float8_e4m3fn, device=device)

        slot = torch.tensor([0, 1, 1], dtype=torch.long, device=device)
        positions = torch.tensor([0, 1, 3], dtype=torch.long, device=device)

        pack_window_tokens_fp8(kv, fp8_cache, slot, positions)

        deq = dequantize_model1_fp8_sparse(fp8_cache.unsqueeze(2)).squeeze(2)
        for i, (s, p) in enumerate(zip(slot.tolist(), positions.tolist())):
            err = (deq[s, p] - kv[i]).abs().max().item()
            assert err < 0.15, f'Token {i} (slot={s}, pos={p}) error {err}'
