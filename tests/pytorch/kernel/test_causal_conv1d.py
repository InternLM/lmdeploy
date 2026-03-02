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
        yield 512

    @pytest.fixture
    def hidden_size(self):
        yield 2048

    @pytest.fixture
    def width(self):
        yield 4

    @pytest.fixture
    def x(self, batch, hidden_size, device):
        yield torch.randn(batch, hidden_size, 1, device=device)

    @pytest.fixture
    def weight(self, hidden_size, width, device):
        yield torch.randn(hidden_size, width, device=device)

    @pytest.fixture
    def conv_state(self, batch, hidden_size, width, device):
        conv_state = torch.randn(batch * 4, hidden_size, width, device=device)
        conv_state = conv_state[::2]
        yield conv_state

    @pytest.fixture
    def bias(self, hidden_size, device):
        yield torch.randn(hidden_size, device=device)

    @pytest.fixture
    def conv_state_indices(self, batch, device):
        conv_state_indices = batch * 2 - 1 - torch.arange(0, batch * 2, 2, device=device)
        yield conv_state_indices.to(torch.int32)

    @pytest.fixture(params=[None, 'silu'])
    def activation(self, request):
        yield request.param

    def test_causal_conv1d_update(self, x, conv_state, weight, bias, activation, conv_state_indices):
        from causal_conv1d import causal_conv1d_update as causal_conv1d_update_gt

        from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_update

        conv_state_clone = conv_state.clone()
        out = causal_conv1d_update(x=x,
                                   conv_state=conv_state_clone,
                                   weight=weight,
                                   bias=bias,
                                   activation=activation,
                                   conv_state_indices=conv_state_indices)
        out_gt = causal_conv1d_update_gt(x=x,
                                         conv_state=conv_state,
                                         weight=weight,
                                         bias=bias,
                                         activation=activation,
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
