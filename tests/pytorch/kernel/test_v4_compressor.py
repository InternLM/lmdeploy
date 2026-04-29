import pytest
import torch


def _reference_fill_compress_state(kv, score, ape, kv_state, score_state, state_ids,
                                   cu_q_seqlens, kv_seqlens, overlap):
    """Python reference for fill_compress_state, matching the official
    DeepSeek-V4 Compressor state-fill logic.

    kv/score are flat [S, D] tensors indexed via cu_q_seqlens. overlap=True  (ratio=4): ring-buffer state layout
    [ratio*2, D]. overlap=False (ratio=128): state layout [ratio, D].
    """
    ratio = ape.size(0)
    coff = 1 + overlap
    max_write = ratio * coff
    B = state_ids.size(0)
    kv_state = kv_state.clone()
    score_state = score_state.clone()

    for b in range(B):
        seq_start = cu_q_seqlens[b].item()
        seq_end = cu_q_seqlens[b + 1].item()
        seqlen = seq_end - seq_start
        kvlen = kv_seqlens[b].item()
        sid = state_ids[b].item()

        t_size = min(seqlen, max_write)
        if t_size == 0:
            continue

        for t in range(t_size):
            kv_pos = seq_start + seqlen - t_size + t
            abs_pos = kvlen - t_size + t

            k = kv[kv_pos]
            s = score[kv_pos]
            a = ape[abs_pos % ratio]

            if overlap:
                head = (abs_pos // ratio % 2) * ratio
                write_pos = (head + ratio + abs_pos % ratio) % max_write
            else:
                write_pos = abs_pos % max_write

            kv_state[sid, write_pos] = k
            score_state[sid, write_pos] = s + a

    return kv_state, score_state


class TestFillCompressState:

    @pytest.fixture
    def head_dim(self):
        yield 128

    @pytest.fixture
    def num_states(self):
        yield 8

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def dtype(self):
        yield torch.bfloat16

    @pytest.fixture
    def overlap(self, request):
        yield request.param

    @pytest.fixture
    def ratio(self, request):
        yield request.param

    def _run_test(self, kv_seqlens_list, q_seqlens_list, ratio, head_dim, num_states,
                  overlap, device, dtype):
        B = len(kv_seqlens_list)
        coff = 1 + overlap
        max_write = ratio * coff
        D = coff * head_dim
        total_q = sum(q_seqlens_list)

        kv_seqlens = torch.tensor(kv_seqlens_list, dtype=torch.int32, device=device)
        cu_q_seqlens = torch.tensor([0] + list(q_seqlens_list), dtype=torch.int32,
                                    device=device).cumsum(0)

        # kv/score: [S, D] flat across batches
        kv = torch.randn(total_q, D, dtype=dtype, device=device)
        score = torch.randn(total_q, D, dtype=dtype, device=device)
        # ape: [ratio, D]
        ape = torch.randn(ratio, D, dtype=torch.float32, device=device)

        state_ids = torch.arange(B, dtype=torch.int32, device=device)
        # kv_state: [N, max_write, D]
        kv_state = torch.zeros(num_states, max_write, D, dtype=dtype, device=device)
        score_state = torch.full((num_states, max_write, D), float('-inf'),
                                 dtype=torch.float32, device=device)

        # reference
        ref_kv_state, ref_score_state = _reference_fill_compress_state(
            kv, score, ape, kv_state, score_state, state_ids,
            cu_q_seqlens, kv_seqlens, overlap)

        # kernel
        from lmdeploy.pytorch.kernels.cuda.v4_compressor import fill_compress_state
        fill_compress_state(kv, score, ape, kv_state, score_state, state_ids,
                            cu_q_seqlens, kv_seqlens)

        torch.testing.assert_close(kv_state.float(), ref_kv_state.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(score_state, ref_score_state, atol=1e-2, rtol=1e-2)

    # ---- overlap=True, ratio=4 ----

    @pytest.mark.parametrize('overlap', [True], indirect=True)
    @pytest.mark.parametrize('ratio', [4], indirect=True)
    @pytest.mark.parametrize('kv_seqlens_list, q_seqlens_list', [
        # decode: history + 1
        ([13, 17, 9], [1, 1, 1]),
        # decode: various history lengths
        ([5, 21, 33], [1, 1, 1]),
        # prefill: kv_seqlens == q_seqlens (no history)
        ([8, 16], [8, 16]),
        # prefill: long (more than max_write)
        ([32, 64], [32, 64]),
        # prefill: with history (multi-turn, start_pos > 0)
        ([20, 48], [8, 16]),
        ([128, 256], [16, 32]),
    ])
    def test_overlap(self, kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                     num_states, overlap, device, dtype):
        self._run_test(kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                       num_states, overlap, device, dtype)

    # ---- overlap=False, ratio=4 ----

    @pytest.mark.parametrize('overlap', [False], indirect=True)
    @pytest.mark.parametrize('ratio', [4], indirect=True)
    @pytest.mark.parametrize('kv_seqlens_list, q_seqlens_list', [
        # decode
        ([13, 9, 128], [1, 1, 1]),
        # prefill
        ([4, 8], [4, 8]),
        ([16], [16]),
        # prefill: with history
        ([20, 40], [8, 16]),
    ])
    def test_no_overlap(self, kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                        num_states, overlap, device, dtype):
        self._run_test(kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                       num_states, overlap, device, dtype)

    # ---- overlap=False, ratio=128 (r128 compress path) ----

    @pytest.mark.parametrize('overlap', [False], indirect=True)
    @pytest.mark.parametrize('ratio', [128], indirect=True)
    @pytest.mark.parametrize('kv_seqlens_list, q_seqlens_list', [
        # decode
        ([257, 513], [1, 1]),
        # prefill
        ([256, 512], [256, 512]),
        # prefill: with history
        ([512, 1024], [256, 128]),
    ])
    def test_ratio_128(self, kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                       num_states, overlap, device, dtype):
        self._run_test(kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                       num_states, overlap, device, dtype)


def _reference_score_kv(kv, score, ape, kv_state, score_state, state_ids,
                        cu_q_seqlens, kv_seqlens, overlap):
    """Python reference for score_kv.

    Compression points are at abs_pos = n*ratio - 1 (0-indexed),
    i.e. kv positions 3, 7, 11, ... for ratio=4.

    This reference does NOT modify state (matching kernel behavior).
    Returns: compressed_kv [S, head_dim] tensor.
    """
    ratio = ape.size(0)
    coff = 1 + overlap
    head_dim = ape.size(1) // coff
    B = state_ids.size(0)
    S = kv.size(0)
    compressed_kv = torch.zeros(S, head_dim, dtype=kv.dtype, device=kv.device)

    for b in range(B):
        seq_start = cu_q_seqlens[b].item()
        seq_end = cu_q_seqlens[b + 1].item()
        seqlen = seq_end - seq_start
        kvlen = kv_seqlens[b].item()
        sid = state_ids[b].item()
        start_pos = kvlen - seqlen

        if seqlen == 1:
            # ======== DECODE PATH ========
            emit = (start_pos + 1) % ratio == 0
            pos_in_ratio = start_pos % ratio

            if overlap:
                cur_kv = kv[seq_start, head_dim:].float()
                cur_score = score[seq_start, head_dim:].float() + ape[pos_in_ratio, head_dim:]
            else:
                cur_kv = kv[seq_start, :head_dim].float()
                cur_score = score[seq_start, :head_dim].float() + ape[pos_in_ratio, :head_dim]

            if overlap:
                cap = 2 * ratio
                head = (start_pos // ratio % 2) * ratio
                # prev window: ring buffer rows, LEFT half
                prev_rows = [(head + i) % cap for i in range(ratio)]
                prev_kv = kv_state[sid, prev_rows, :head_dim].float()
                prev_score = score_state[sid, prev_rows, :head_dim].float()
                # curr window: ring buffer rows, RIGHT half
                curr_rows = [(head + ratio + i) % cap for i in range(ratio)]
                curr_kv = kv_state[sid, curr_rows, head_dim:].float()
                curr_score = score_state[sid, curr_rows, head_dim:].float()
                # replace current token's slot in curr window
                curr_kv[pos_in_ratio] = cur_kv
                curr_score[pos_in_ratio] = cur_score
                # merge
                merged_kv = torch.cat([prev_kv, curr_kv], dim=0)
                merged_score = torch.cat([prev_score, curr_score], dim=0)
            else:
                merged_kv = kv_state[sid, :, :head_dim].float()
                merged_score = score_state[sid, :, :head_dim].float()
                merged_kv[pos_in_ratio] = cur_kv
                merged_score[pos_in_ratio] = cur_score

            if emit:
                compressed = (merged_kv * merged_score.softmax(dim=0)).sum(dim=0)
                compressed_kv[seq_start] = compressed.to(compressed_kv.dtype)

        else:
            # ======== PREFILL PATH ========
            # Compression points at abs_pos = n*ratio - 1
            # First one in the new token range:
            first_compress = ((start_pos + ratio) // ratio) * ratio - 1
            last_pos = start_pos + seqlen - 1
            if first_compress > last_pos:
                continue

            compress_points = list(range(first_compress, last_pos + 1, ratio))

            for compress_abs in compress_points:
                if overlap:
                    # prev window: abs_pos in [compress_abs - 2*ratio + 1, compress_abs - ratio]
                    prev_abs_base = compress_abs - 2 * ratio + 1
                    # curr window: abs_pos in [compress_abs - ratio + 1, compress_abs]
                    curr_abs_base = compress_abs - ratio + 1

                    # prev window
                    if prev_abs_base < 0:
                        prev_kv = torch.zeros(ratio, head_dim, dtype=torch.float32, device=kv.device)
                        prev_score = torch.full((ratio, head_dim), -1e30, dtype=torch.float32, device=kv.device)
                    elif prev_abs_base < start_pos:
                        # read from state
                        cap = 2 * ratio
                        _prev_head = (prev_abs_base // ratio % 2) * ratio
                        prev_rows = [(_prev_head + i) % cap for i in range(ratio)]
                        prev_kv = kv_state[sid, prev_rows, :head_dim].float()
                        prev_score = score_state[sid, prev_rows, :head_dim].float()
                    else:
                        prev_kv = torch.zeros(ratio, head_dim, dtype=torch.float32, device=kv.device)
                        prev_score = torch.zeros(ratio, head_dim, dtype=torch.float32, device=kv.device)
                        for i in range(ratio):
                            abs_pos_i = prev_abs_base + i
                            kv_pos = seq_start + (abs_pos_i - start_pos)
                            prev_kv[i] = kv[kv_pos, :head_dim].float()
                            prev_score[i] = score[kv_pos, :head_dim].float() + ape[abs_pos_i % ratio, :head_dim]

                    # curr window
                    curr_kv = torch.zeros(ratio, head_dim, dtype=torch.float32, device=kv.device)
                    curr_score = torch.zeros(ratio, head_dim, dtype=torch.float32, device=kv.device)
                    for i in range(ratio):
                        abs_pos_i = curr_abs_base + i
                        kv_pos = seq_start + (abs_pos_i - start_pos)
                        curr_kv[i] = kv[kv_pos, head_dim:].float()
                        curr_score[i] = score[kv_pos, head_dim:].float() + ape[abs_pos_i % ratio, head_dim:]

                    merged_kv = torch.cat([prev_kv, curr_kv], dim=0)
                    merged_score = torch.cat([prev_score, curr_score], dim=0)
                else:
                    merged_kv = torch.zeros(ratio, head_dim, dtype=torch.float32, device=kv.device)
                    merged_score = torch.zeros(ratio, head_dim, dtype=torch.float32, device=kv.device)
                    abs_base = compress_abs - ratio + 1
                    for i in range(ratio):
                        abs_pos_i = abs_base + i
                        kv_pos = seq_start + (abs_pos_i - start_pos)
                        merged_kv[i] = kv[kv_pos, :head_dim].float()
                        merged_score[i] = score[kv_pos, :head_dim].float() + ape[abs_pos_i % ratio, :head_dim]

                compressed = (merged_kv * merged_score.softmax(dim=0)).sum(dim=0)
                write_pos = seq_start + (compress_abs - start_pos)
                compressed_kv[write_pos] = compressed.to(compressed_kv.dtype)

    return compressed_kv


class TestScoreKV:

    @pytest.fixture
    def head_dim(self):
        yield 128

    @pytest.fixture
    def num_states(self):
        yield 8

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def dtype(self):
        yield torch.bfloat16

    @pytest.fixture
    def overlap(self, request):
        yield request.param

    @pytest.fixture
    def ratio(self, request):
        yield request.param

    def _run_prefill_test(self, kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                          num_states, overlap, device, dtype):
        """Test score_kv for prefill scenarios."""
        B = len(kv_seqlens_list)
        coff = 1 + overlap
        D = coff * head_dim
        max_write = ratio * coff
        total_q = sum(q_seqlens_list)
        max_seqlen_q = max(q_seqlens_list)

        kv_seqlens = torch.tensor(kv_seqlens_list, dtype=torch.int32, device=device)
        cu_q_seqlens = torch.tensor([0] + list(q_seqlens_list), dtype=torch.int32,
                                    device=device).cumsum(0)

        kv = torch.randn(total_q, D, dtype=dtype, device=device)
        score = torch.randn(total_q, D, dtype=dtype, device=device)
        ape = torch.randn(ratio, D, dtype=torch.float32, device=device)

        state_ids = torch.arange(B, dtype=torch.int32, device=device)
        kv_state = torch.zeros(num_states, max_write, D, dtype=torch.float32, device=device)
        score_state = torch.full((num_states, max_write, D), float('-inf'),
                                 dtype=torch.float32, device=device)

        # if start_pos > 0, populate state with history
        from lmdeploy.pytorch.kernels.cuda.v4_compressor import fill_compress_state
        has_history = any(kv_seqlens_list[b] > q_seqlens_list[b] for b in range(B))
        if has_history:
            for b in range(B):
                history_len = kv_seqlens_list[b] - q_seqlens_list[b]
                if history_len > 0:
                    hist_kv = torch.randn(history_len, D, dtype=dtype, device=device)
                    hist_score = torch.randn(history_len, D, dtype=dtype, device=device)
                    hist_cu_q = torch.tensor([0, history_len], dtype=torch.int32, device=device)
                    hist_kvlen = torch.tensor([kv_seqlens_list[b] - q_seqlens_list[b]],
                                              dtype=torch.int32, device=device)
                    hist_sids = torch.tensor([b], dtype=torch.int32, device=device)
                    fill_compress_state(hist_kv, hist_score, ape, kv_state, score_state,
                                        hist_sids, hist_cu_q, hist_kvlen)

        # reference
        kv_state_ref = kv_state.clone()
        score_state_ref = score_state.clone()
        ref_compressed = _reference_score_kv(kv.clone(), score.clone(), ape, kv_state_ref,
                                             score_state_ref, state_ids, cu_q_seqlens,
                                             kv_seqlens, overlap)

        # kernel
        from lmdeploy.pytorch.kernels.cuda.v4_compressor import score_kv
        compressed_kv = torch.zeros(total_q, head_dim, dtype=dtype, device=device)
        kv_state_k = kv_state.clone()
        score_state_k = score_state.clone()
        score_kv(kv, score, ape, kv_state_k, score_state_k, state_ids,
                 cu_q_seqlens, kv_seqlens, compressed_kv, overlap, max_seqlen_q)

        torch.testing.assert_close(compressed_kv.float(),
                                   ref_compressed.float(),
                                   atol=1e-2, rtol=1e-2)

    def _run_decode_test(self, kvlen, ratio, head_dim, num_states, overlap, device, dtype):
        """Test score_kv for decode by simulating a full decode sequence.

        Creates full history, populates state via fill_compress_state, then tests score_kv on the last token.
        """
        coff = 1 + overlap
        D = coff * head_dim
        max_write = ratio * coff

        # full history kv/score (single batch)
        full_kv = torch.randn(kvlen, D, dtype=dtype, device=device)
        full_score = torch.randn(kvlen, D, dtype=dtype, device=device)
        ape = torch.randn(ratio, D, dtype=torch.float32, device=device)

        state_ids = torch.tensor([0], dtype=torch.int32, device=device)
        kv_state = torch.zeros(num_states, max_write, D, dtype=torch.float32, device=device)
        score_state = torch.full((num_states, max_write, D), float('-inf'),
                                 dtype=torch.float32, device=device)

        # populate state with full history
        from lmdeploy.pytorch.kernels.cuda.v4_compressor import fill_compress_state
        full_cu_q = torch.tensor([0, kvlen], dtype=torch.int32, device=device)
        full_kvlen = torch.tensor([kvlen], dtype=torch.int32, device=device)
        fill_compress_state(full_kv, full_score, ape, kv_state, score_state,
                            state_ids, full_cu_q, full_kvlen)

        # extract last token for score_kv
        last_kv = full_kv[-1:].clone()
        last_score = full_score[-1:].clone()
        last_cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
        last_kvlen = torch.tensor([kvlen], dtype=torch.int32, device=device)

        # reference
        kv_state_ref = kv_state.clone()
        score_state_ref = score_state.clone()
        ref_compressed = _reference_score_kv(last_kv.clone(), last_score.clone(), ape,
                                             kv_state_ref, score_state_ref, state_ids,
                                             last_cu_q, last_kvlen, overlap)

        # kernel
        from lmdeploy.pytorch.kernels.cuda.v4_compressor import score_kv
        compressed_kv = torch.zeros(1, head_dim, dtype=dtype, device=device)
        kv_state_k = kv_state.clone()
        score_state_k = score_state.clone()
        score_kv(last_kv, last_score, ape, kv_state_k, score_state_k, state_ids,
                 last_cu_q, last_kvlen, compressed_kv, overlap, 1)

        torch.testing.assert_close(compressed_kv.float(),
                                   ref_compressed.float(),
                                   atol=1e-2, rtol=1e-2)

    # ---- overlap=True, ratio=4, prefill ----

    @pytest.mark.parametrize('overlap', [True], indirect=True)
    @pytest.mark.parametrize('ratio', [4], indirect=True)
    @pytest.mark.parametrize('kv_seqlens_list, q_seqlens_list', [
        # prefill: no history (start_pos=0)
        ([8, 16], [8, 16]),
        ([4, 12], [4, 12]),
        # prefill: with history (start_pos>0)
        ([20, 48], [8, 16]),
        ([12, 24], [8, 16]),
    ])
    def test_prefill_overlap(self, kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                             num_states, overlap, device, dtype):
        self._run_prefill_test(kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                               num_states, overlap, device, dtype)

    # ---- overlap=True, ratio=4, decode ----

    @pytest.mark.parametrize('overlap', [True], indirect=True)
    @pytest.mark.parametrize('ratio', [4], indirect=True)
    @pytest.mark.parametrize('kvlen', [
        4,    # first emit (start_pos=3)
        8,    # second emit (start_pos=7)
        5,    # no emit (start_pos=4, (4+1)%4=1!=0)
        7,    # no emit
        12,   # third emit
    ])
    def test_decode_overlap(self, kvlen, ratio, head_dim, num_states, overlap,
                            device, dtype):
        self._run_decode_test(kvlen, ratio, head_dim, num_states, overlap, device, dtype)

    # ---- overlap=False, ratio=4, prefill ----

    @pytest.mark.parametrize('overlap', [False], indirect=True)
    @pytest.mark.parametrize('ratio', [4], indirect=True)
    @pytest.mark.parametrize('kv_seqlens_list, q_seqlens_list', [
        ([4, 8], [4, 8]),
        ([16], [16]),
        ([20, 40], [8, 16]),
    ])
    def test_prefill_no_overlap(self, kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                                num_states, overlap, device, dtype):
        self._run_prefill_test(kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                               num_states, overlap, device, dtype)

    # ---- overlap=False, ratio=4, decode ----

    @pytest.mark.parametrize('overlap', [False], indirect=True)
    @pytest.mark.parametrize('ratio', [4], indirect=True)
    @pytest.mark.parametrize('kvlen', [
        4,    # emit
        8,    # emit
        5,    # no emit
    ])
    def test_decode_no_overlap(self, kvlen, ratio, head_dim, num_states, overlap,
                               device, dtype):
        self._run_decode_test(kvlen, ratio, head_dim, num_states, overlap, device, dtype)

    # ---- overlap=False, ratio=128, prefill ----

    @pytest.mark.parametrize('overlap', [False], indirect=True)
    @pytest.mark.parametrize('ratio', [128], indirect=True)
    @pytest.mark.parametrize('kv_seqlens_list, q_seqlens_list', [
        ([256, 512], [256, 512]),
        ([512, 1024], [256, 128]),
    ])
    def test_prefill_ratio_128(self, kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                               num_states, overlap, device, dtype):
        self._run_prefill_test(kv_seqlens_list, q_seqlens_list, ratio, head_dim,
                               num_states, overlap, device, dtype)

    # ---- overlap=False, ratio=128, decode ----

    @pytest.mark.parametrize('overlap', [False], indirect=True)
    @pytest.mark.parametrize('ratio', [128], indirect=True)
    @pytest.mark.parametrize('kvlen', [
        128,    # emit
        256,    # emit
        129,    # no emit
    ])
    def test_decode_ratio_128(self, kvlen, ratio, head_dim, num_states, overlap,
                              device, dtype):
        self._run_decode_test(kvlen, ratio, head_dim, num_states, overlap, device, dtype)


def _reference_fill_compressed_kv(compressed_kv, kv_cache, cu_q_seqlens, kv_seqlens,
                                  block_offsets, compress_ratio, block_size):
    """Python reference for fill_compressed_kv, matching
    _write_compressed_cache_entries."""
    kv_cache = kv_cache.clone()
    B = kv_seqlens.size(0)
    entries_per_block = max(block_size // compress_ratio, 1)

    for b in range(B):
        seq_start = cu_q_seqlens[b].item()
        seq_end = cu_q_seqlens[b + 1].item()
        seqlen = seq_end - seq_start
        kvlen = kv_seqlens[b].item()
        start_pos = kvlen - seqlen

        if seqlen == 1:
            # Decode: emit at most 1 entry
            if (start_pos + 1) % compress_ratio != 0:
                continue
            p = start_pos // compress_ratio
            write_pos = seq_start
            token_pos = p * compress_ratio
            block_idx = token_pos // block_size
            block_off = p % entries_per_block
            if block_idx < block_offsets.size(1):
                phys_block = block_offsets[b, block_idx].item()
                kv_cache[phys_block, block_off] = compressed_kv[write_pos]
        else:
            # Prefill: iterate compression points
            first_compress = ((start_pos + compress_ratio) // compress_ratio) * compress_ratio - 1
            last_pos = start_pos + seqlen - 1
            if first_compress > last_pos:
                continue
            for compress_abs in range(first_compress, last_pos + 1, compress_ratio):
                p = compress_abs // compress_ratio
                write_pos = seq_start + (compress_abs - start_pos)
                token_pos = p * compress_ratio
                block_idx = token_pos // block_size
                block_off = p % entries_per_block
                if block_idx < block_offsets.size(1):
                    phys_block = block_offsets[b, block_idx].item()
                    kv_cache[phys_block, block_off] = compressed_kv[write_pos]

    return kv_cache


class TestFillCompressedKV:

    @pytest.fixture
    def head_dim(self):
        yield 128

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def dtype(self):
        yield torch.bfloat16

    @pytest.fixture
    def block_size(self):
        yield 128

    @pytest.fixture
    def compress_ratio(self, request):
        yield request.param

    def _run_prefill_test(self, kv_seqlens_list, q_seqlens_list, compress_ratio,
                          head_dim, block_size, device, dtype):
        B = len(kv_seqlens_list)
        overlap = compress_ratio == 4
        coff = 1 + overlap
        D = coff * head_dim
        max_write = compress_ratio * coff
        total_q = sum(q_seqlens_list)
        max_seqlen_q = max(q_seqlens_list)
        entries_per_block = max(block_size // compress_ratio, 1)

        kv_seqlens = torch.tensor(kv_seqlens_list, dtype=torch.int32, device=device)
        cu_q_seqlens = torch.tensor([0] + list(q_seqlens_list), dtype=torch.int32,
                                    device=device).cumsum(0)

        kv = torch.randn(total_q, D, dtype=dtype, device=device)
        score = torch.randn(total_q, D, dtype=dtype, device=device)
        ape = torch.randn(compress_ratio, D, dtype=torch.float32, device=device)

        state_ids = torch.arange(B, dtype=torch.int32, device=device)
        num_states = B
        kv_state = torch.zeros(num_states, max_write, D, dtype=torch.float32, device=device)
        score_state = torch.full((num_states, max_write, D), float('-inf'),
                                 dtype=torch.float32, device=device)

        from lmdeploy.pytorch.kernels.cuda.v4_compressor import fill_compress_state, score_kv
        has_history = any(kv_seqlens_list[b] > q_seqlens_list[b] for b in range(B))
        if has_history:
            for b in range(B):
                history_len = kv_seqlens_list[b] - q_seqlens_list[b]
                if history_len > 0:
                    hist_kv = torch.randn(history_len, D, dtype=dtype, device=device)
                    hist_score = torch.randn(history_len, D, dtype=dtype, device=device)
                    hist_cu_q = torch.tensor([0, history_len], dtype=torch.int32, device=device)
                    hist_kvlen = torch.tensor([history_len], dtype=torch.int32, device=device)
                    hist_sids = torch.tensor([b], dtype=torch.int32, device=device)
                    fill_compress_state(hist_kv, hist_score, ape, kv_state, score_state,
                                        hist_sids, hist_cu_q, hist_kvlen)

        compressed_kv = torch.zeros(total_q, head_dim, dtype=dtype, device=device)
        kv_state_k = kv_state.clone()
        score_state_k = score_state.clone()
        score_kv(kv, score, ape, kv_state_k, score_state_k, state_ids,
                 cu_q_seqlens, kv_seqlens, compressed_kv, overlap, max_seqlen_q)

        # Build block_offsets and kv_cache
        max_blocks = 64
        num_blocks = B * max_blocks
        block_offsets = torch.arange(num_blocks, device=device).reshape(B, max_blocks).long()
        kv_cache = torch.zeros(num_blocks, entries_per_block, head_dim, dtype=dtype, device=device)

        # Reference
        ref_cache = _reference_fill_compressed_kv(compressed_kv, kv_cache, cu_q_seqlens,
                                                   kv_seqlens, block_offsets, compress_ratio,
                                                   block_size)

        # Kernel
        kv_cache_k = kv_cache.clone()
        from lmdeploy.pytorch.kernels.cuda.v4_compressor import fill_compressed_kv
        fill_compressed_kv(compressed_kv, kv_cache_k, cu_q_seqlens, kv_seqlens,
                           block_offsets, compress_ratio, block_size, max_seqlen_q)

        torch.testing.assert_close(kv_cache_k.float(), ref_cache.float(), atol=1e-2, rtol=1e-2)

    def _run_decode_test(self, kvlen, compress_ratio, head_dim, block_size, device, dtype):
        overlap = compress_ratio == 4
        coff = 1 + overlap
        D = coff * head_dim
        max_write = compress_ratio * coff
        entries_per_block = max(block_size // compress_ratio, 1)

        full_kv = torch.randn(kvlen, D, dtype=dtype, device=device)
        full_score = torch.randn(kvlen, D, dtype=dtype, device=device)
        ape = torch.randn(compress_ratio, D, dtype=torch.float32, device=device)

        state_ids = torch.tensor([0], dtype=torch.int32, device=device)
        num_states = 1
        kv_state = torch.zeros(num_states, max_write, D, dtype=torch.float32, device=device)
        score_state = torch.full((num_states, max_write, D), float('-inf'),
                                 dtype=torch.float32, device=device)

        from lmdeploy.pytorch.kernels.cuda.v4_compressor import fill_compress_state, score_kv
        full_cu_q = torch.tensor([0, kvlen], dtype=torch.int32, device=device)
        full_kvlen = torch.tensor([kvlen], dtype=torch.int32, device=device)
        fill_compress_state(full_kv, full_score, ape, kv_state, score_state,
                            state_ids, full_cu_q, full_kvlen)

        # Last token
        last_kv = full_kv[-1:].clone()
        last_score = full_score[-1:].clone()
        last_cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
        last_kvlen = torch.tensor([kvlen], dtype=torch.int32, device=device)

        compressed_kv = torch.zeros(1, head_dim, dtype=dtype, device=device)
        kv_state_k = kv_state.clone()
        score_state_k = score_state.clone()
        score_kv(last_kv, last_score, ape, kv_state_k, score_state_k, state_ids,
                 last_cu_q, last_kvlen, compressed_kv, overlap, 1)

        # Build block_offsets and kv_cache
        max_blocks = 64
        num_blocks = max_blocks
        block_offsets = torch.arange(num_blocks, device=device).reshape(1, max_blocks).long()
        kv_cache = torch.zeros(num_blocks, entries_per_block, head_dim, dtype=dtype, device=device)

        # Reference
        ref_cache = _reference_fill_compressed_kv(compressed_kv, kv_cache, last_cu_q,
                                                   last_kvlen, block_offsets, compress_ratio,
                                                   block_size)

        # Kernel
        kv_cache_k = kv_cache.clone()
        from lmdeploy.pytorch.kernels.cuda.v4_compressor import fill_compressed_kv as kernel_fn
        kernel_fn(compressed_kv, kv_cache_k, last_cu_q, last_kvlen,
                  block_offsets, compress_ratio, block_size, 1)

        torch.testing.assert_close(kv_cache_k.float(), ref_cache.float(), atol=1e-2, rtol=1e-2)

    # ---- ratio=4, prefill ----

    @pytest.mark.parametrize('compress_ratio', [4], indirect=True)
    @pytest.mark.parametrize('kv_seqlens_list, q_seqlens_list', [
        ([8, 16], [8, 16]),
        ([4, 12], [4, 12]),
        ([20, 48], [8, 16]),
    ])
    def test_prefill_r4(self, kv_seqlens_list, q_seqlens_list, compress_ratio,
                        head_dim, block_size, device, dtype):
        self._run_prefill_test(kv_seqlens_list, q_seqlens_list, compress_ratio,
                               head_dim, block_size, device, dtype)

    # ---- ratio=4, decode ----

    @pytest.mark.parametrize('compress_ratio', [4], indirect=True)
    @pytest.mark.parametrize('kvlen', [4, 8, 5, 7, 12])
    def test_decode_r4(self, kvlen, compress_ratio, head_dim, block_size, device, dtype):
        self._run_decode_test(kvlen, compress_ratio, head_dim, block_size, device, dtype)

    # ---- ratio=128, prefill ----

    @pytest.mark.parametrize('compress_ratio', [128], indirect=True)
    @pytest.mark.parametrize('kv_seqlens_list, q_seqlens_list', [
        ([256, 512], [256, 512]),
        ([512, 1024], [256, 128]),
    ])
    def test_prefill_r128(self, kv_seqlens_list, q_seqlens_list, compress_ratio,
                          head_dim, block_size, device, dtype):
        self._run_prefill_test(kv_seqlens_list, q_seqlens_list, compress_ratio,
                               head_dim, block_size, device, dtype)

    # ---- ratio=128, decode ----

    @pytest.mark.parametrize('compress_ratio', [128], indirect=True)
    @pytest.mark.parametrize('kvlen', [128, 256, 129])
    def test_decode_r128(self, kvlen, compress_ratio, head_dim, block_size, device, dtype):
        self._run_decode_test(kvlen, compress_ratio, head_dim, block_size, device, dtype)
