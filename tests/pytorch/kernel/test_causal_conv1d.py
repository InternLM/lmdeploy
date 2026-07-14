import pytest
import torch


def do_test():
    try:
        import causal_conv1d  # noqa: F401
        import tilelang  # noqa: F401
        causal_conv1d_fn = causal_conv1d.causal_conv1d_fn  # noqa: F841
        causal_conv1d_update = causal_conv1d.causal_conv1d_update  # noqa: F841
        return True
    except Exception:
        return False


@pytest.mark.skipif(not do_test(), reason='tilelang or causal_conv1d is not available')
class TestCausalConv1dUpdate:

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def batch(self):
        yield 128

    @pytest.fixture
    def hidden_size(self):
        yield 2048

    @pytest.fixture
    def width(self):
        yield 4

    @pytest.fixture(params=[1, 4])
    def seqlen(self, request):
        yield request.param

    @pytest.fixture
    def x(self, batch, hidden_size, seqlen, device):
        yield torch.randn(batch, hidden_size, seqlen, device=device)

    @pytest.fixture
    def weight(self, hidden_size, width, device):
        yield torch.randn(hidden_size, width, device=device)

    @pytest.fixture
    def bias(self, hidden_size, device):
        yield torch.randn(hidden_size, device=device)

    @pytest.fixture(params=[True, False])
    def cache_seqlens(self, request, batch, device):
        if request.param:
            yield torch.randint(0, 4096, (batch, ), dtype=torch.int32, device=device)
        else:
            yield None

    @pytest.fixture(params=[True, False])
    def conv_state_indices(self, request, batch, device):
        if request.param:
            conv_state_indices = batch * 2 - 1 - torch.arange(0, batch * 2, 2, device=device)
            yield conv_state_indices.to(torch.int32)
        else:
            yield None

    @pytest.fixture
    def conv_state(self, batch, hidden_size, width, device, conv_state_indices):
        if conv_state_indices is not None:
            conv_state = torch.randn(batch * 4, hidden_size, width, device=device)
            conv_state = conv_state[::2]
        else:
            conv_state = torch.randn(batch, hidden_size, width, device=device)
        yield conv_state

    @pytest.fixture(params=[None, 'silu'])
    def activation(self, request):
        yield request.param

    def test_causal_conv1d_update(self, x, conv_state, weight, bias, activation, cache_seqlens, conv_state_indices):
        from causal_conv1d import causal_conv1d_update as causal_conv1d_update_gt

        from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_update

        conv_state_clone = conv_state.clone()
        out = causal_conv1d_update(x=x,
                                   conv_state=conv_state_clone,
                                   weight=weight,
                                   bias=bias,
                                   activation=activation,
                                   cache_seqlens=cache_seqlens,
                                   conv_state_indices=conv_state_indices)
        out_gt = causal_conv1d_update_gt(x=x,
                                         conv_state=conv_state,
                                         weight=weight,
                                         bias=bias,
                                         activation=activation,
                                         cache_seqlens=cache_seqlens,
                                         conv_state_indices=conv_state_indices)
        torch.testing.assert_close(out, out_gt, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(conv_state_clone, conv_state, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not do_test(), reason='tilelang or causal_conv1d is not available')
class TestCausalConv1dFn:

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def hidden_size(self):
        yield 2048

    @pytest.fixture
    def seqlen(self):
        yield 4096

    @pytest.fixture
    def seq_idx(self, seqlen, device):
        seq_idx = torch.zeros(seqlen, dtype=torch.int32, device=device)
        seq_idx[seqlen // 4 * 3:] = 1
        seq_idx = seq_idx.view(1, -1)
        yield seq_idx

    @pytest.fixture
    def x(self, hidden_size, seqlen, device):
        yield torch.randn(1, hidden_size, seqlen, device=device).transpose(1, 2).contiguous().transpose(1, 2)

    @pytest.fixture
    def weight(self, hidden_size, device):
        yield torch.randn(hidden_size, 4, device=device)

    @pytest.fixture
    def bias(self, hidden_size, device):
        yield torch.randn(hidden_size, device=device)

    @pytest.fixture(params=[None, 'silu'])
    def activation(self, request):
        yield request.param

    def test_causal_conv1d_fn(self, x, weight, bias, activation, seq_idx):
        from causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_gt

        from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_fn

        out = causal_conv1d_fn(x=x,
                               weight=weight,
                               bias=bias,
                               activation=activation,
                               return_final_states=False,
                               seq_idx=seq_idx)
        out_gt = causal_conv1d_fn_gt(x=x,
                                     weight=weight,
                                     bias=bias,
                                     activation=activation,
                                     return_final_states=False,
                                     seq_idx=seq_idx)
        torch.testing.assert_close(out, out_gt, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not do_test(), reason='tilelang or causal_conv1d is not available')
class TestCausalConv1dFnInitStates:
    """Test causal_conv1d_fn with per-sequence initial_states."""

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def hidden_size(self):
        yield 2048

    @pytest.fixture
    def width(self):
        yield 4

    @pytest.fixture(params=[None, 'silu'])
    def activation(self, request):
        yield request.param

    @pytest.fixture
    def weight(self, hidden_size, width, device):
        yield torch.randn(hidden_size, width, device=device)

    @pytest.fixture
    def bias(self, hidden_size, device):
        yield torch.randn(hidden_size, device=device)

    def _ref_conv1d(self, x_seq, weight, bias, init_state, activation):
        """Reference: prepend init_state and run F.conv1d per sequence."""
        dim = weight.shape[0]
        x_cat = torch.cat([init_state, x_seq], dim=-1).float()
        w = weight.float().unsqueeze(1)
        b = bias.float() if bias is not None else None
        out = torch.nn.functional.conv1d(x_cat, w, b, padding=0, groups=dim)
        out = out[..., :x_seq.shape[-1]]
        if activation in ('silu', 'swish'):
            out = torch.nn.functional.silu(out)
        return out.to(x_seq.dtype)

    def test_single_seq(self, hidden_size, width, weight, bias, activation, device):
        """Single sequence with non-zero initial states."""
        seqlen = 128
        x = torch.randn(1, hidden_size, seqlen, device=device).transpose(1, 2).contiguous().transpose(1, 2)
        seq_idx = torch.zeros(1, seqlen, dtype=torch.int32, device=device)
        init_states = torch.randn(1, hidden_size, width - 1, device=device, dtype=x.dtype)

        from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_fn

        out = causal_conv1d_fn(x=x, weight=weight, bias=bias, seq_idx=seq_idx, initial_states=init_states,
                               activation=activation)
        out_ref = self._ref_conv1d(x, weight, bias, init_states, activation)
        torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)

    def test_multi_seq(self, hidden_size, width, weight, bias, activation, device):
        """Multiple sequences packed, each with its own initial state."""
        seqlens = [100, 200, 50]
        n_seqs = len(seqlens)

        x_parts = [torch.randn(1, hidden_size, sl, device=device) for sl in seqlens]
        # Make channel-last stride like real lmdeploy usage.
        x_packed = torch.cat(x_parts, dim=-1).transpose(1, 2).contiguous().transpose(1, 2)

        seq_idx_parts = [torch.full((sl, ), i, dtype=torch.int32, device=device) for i, sl in enumerate(seqlens)]
        seq_idx = torch.cat(seq_idx_parts).unsqueeze(0)

        init_states = torch.randn(n_seqs, hidden_size, width - 1, device=device, dtype=x_packed.dtype)

        from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_fn

        out = causal_conv1d_fn(x=x_packed, weight=weight, bias=bias, seq_idx=seq_idx, initial_states=init_states,
                               activation=activation)

        # Build reference per-sequence.
        offset = 0
        for i, sl in enumerate(seqlens):
            x_seq = x_packed[:, :, offset:offset + sl]
            ref = self._ref_conv1d(x_seq, weight, bias, init_states[i:i + 1], activation)
            torch.testing.assert_close(out[:, :, offset:offset + sl], ref, rtol=1e-3, atol=1e-3)
            offset += sl

    def test_zero_init_matches_no_init(self, hidden_size, width, weight, bias, activation, device):
        """Zero initial_states should produce the same result as no
        initial_states."""
        seqlens = [80, 120]
        n_seqs = len(seqlens)

        x_packed = torch.randn(1, hidden_size, sum(seqlens), device=device).transpose(1, 2).contiguous().transpose(
            1, 2)
        seq_idx_parts = [torch.full((sl, ), i, dtype=torch.int32, device=device) for i, sl in enumerate(seqlens)]
        seq_idx = torch.cat(seq_idx_parts).unsqueeze(0)

        zero_states = torch.zeros(n_seqs, hidden_size, width - 1, device=device, dtype=x_packed.dtype)

        from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_fn

        out_with_init = causal_conv1d_fn(x=x_packed, weight=weight, bias=bias, seq_idx=seq_idx,
                                         initial_states=zero_states, activation=activation)
        out_no_init = causal_conv1d_fn(x=x_packed, weight=weight, bias=bias, seq_idx=seq_idx, initial_states=None,
                                       activation=activation)
        torch.testing.assert_close(out_with_init, out_no_init, rtol=1e-3, atol=1e-3)
