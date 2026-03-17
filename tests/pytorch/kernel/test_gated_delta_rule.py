import pytest
import torch


def do_test():
    try:
        import tilelang  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def naive_recurrent_gdr(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1]**0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    o = o.to(dtype)
    if output_final_state:
        h = h.to(dtype)
    return o, h


@pytest.mark.skipif(not do_test(), reason='tilelang is not available')
class TestRecurrentGatedDeltaRule:

    @pytest.fixture(autouse=True)
    def auto_context(self):
        origin_dtype = torch.get_default_dtype()
        origin_device = torch.get_default_device()
        with torch.inference_mode():
            torch.set_default_dtype(torch.bfloat16)
            torch.set_default_device('cuda')
            try:
                yield
            finally:
                torch.set_default_dtype(origin_dtype)
                torch.set_default_device(origin_device)

    @pytest.fixture
    def batch(self):
        yield 512

    @pytest.fixture
    def num_heads(self):
        yield 16

    @pytest.fixture(params=[1, 4])
    def seqlen(self, request):
        yield request.param

    @pytest.fixture
    def head_dim(self):
        yield 128

    @pytest.fixture(params=[True, False])
    def use_qk_l2norm_in_kernel(self, request):
        yield request.param

    @pytest.fixture
    def q(self, batch, seqlen, num_heads, head_dim):
        yield torch.rand(batch, seqlen, num_heads, head_dim) - 0.5

    @pytest.fixture
    def k(self, batch, seqlen, num_heads, head_dim):
        yield torch.rand(batch, seqlen, num_heads, head_dim) - 0.5

    @pytest.fixture
    def v(self, batch, seqlen, num_heads, head_dim):
        yield torch.rand(batch, seqlen, num_heads, head_dim) - 0.5

    @pytest.fixture
    def g(self, batch, seqlen, num_heads):
        yield -2 * torch.rand(batch, seqlen, num_heads)

    @pytest.fixture
    def beta(self, batch, seqlen, num_heads):
        yield torch.rand(batch, seqlen, num_heads)

    @pytest.fixture
    def initial_state(self, batch, num_heads, head_dim):
        yield torch.rand(batch, num_heads, head_dim, head_dim) - 0.5

    @pytest.fixture
    def gt(self, q, k, v, g, beta, initial_state, use_qk_l2norm_in_kernel):
        state_copy = initial_state.clone()
        yield naive_recurrent_gdr(q,
                                  k,
                                  v,
                                  beta,
                                  g,
                                  initial_state=state_copy,
                                  output_final_state=True,
                                  use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    def test_fused_gated_delta_rule(self, q, k, v, g, beta, initial_state, use_qk_l2norm_in_kernel, gt):
        from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule
        state_copy = initial_state.clone()
        out, out_h = fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state_copy,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        gt_o, gt_h = gt
        torch.testing.assert_close(out, gt_o, atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(out_h, gt_h, atol=1e-2, rtol=1e-3)

    def test_circular_buffer(self, q, k, v, g, beta, seqlen, batch, num_heads, head_dim):
        """Test cache_seqlens circular buffer support."""
        from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

        # Build circular buffer state: [NUM_STATE, B, HV, K, V]
        num_states = seqlen + 2
        circular_state = torch.rand(num_states, batch, num_heads, head_dim, head_dim) - 0.5
        cache_seqlens = torch.randint(0, num_states * 3, (batch, ), dtype=torch.int32, device='cuda')

        # --- vectorized naive reference ---
        scale = 1 / (head_dim**0.5)
        rq = q.float() * scale
        rk = k.float()
        rv = v.float()
        rg = g.float()
        rb = beta.float()

        # read initial state per batch: slot = cache_seqlens[b] % num_states
        read_slots = (cache_seqlens % num_states).long()  # [B]
        ref_state = circular_state.clone().float()
        h = ref_state[read_slots, torch.arange(batch, device='cuda')]  # [B, HV, K, V]

        ref_out = torch.zeros_like(rv)
        expected_state = ref_state.clone()

        for t in range(seqlen):
            write_slots = (read_slots + 1 + t) % num_states  # [B]
            b_q = rq[:, t]  # [B, H, K]
            b_k = rk[:, t]  # [B, H, K]
            b_v = rv[:, t]  # [B, HV, V]
            b_g = rg[:, t]  # [B, HV]
            b_beta = rb[:, t]  # [B, HV]

            h = h * b_g.exp().unsqueeze(-1).unsqueeze(-1)
            hk = (h * b_k.unsqueeze(-1)).sum(-2)  # [B, HV, V]
            delta_v = (b_v - hk) * b_beta.unsqueeze(-1)
            h = h + b_k.unsqueeze(-1) * delta_v.unsqueeze(-2)
            ref_out[:, t] = torch.einsum('bhd,bhdm->bhm', b_q, h)

            # scatter write state into circular buffer
            expected_state[write_slots, torch.arange(batch, device='cuda')] = h

        ref_out = ref_out.to(q.dtype)

        # --- kernel under test ---
        state_copy = circular_state.clone()
        out, out_state = fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state_copy,
            output_final_state=True,
            cache_seqlens=cache_seqlens,
        )

        torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-4)
        # Only compare slots the kernel actually wrote to
        batch_idx = torch.arange(batch, device='cuda')
        for t in range(seqlen):
            write_slots = (read_slots + 1 + t) % num_states
            torch.testing.assert_close(out_state[write_slots, batch_idx].float(),
                                       expected_state[write_slots, batch_idx].to(out_state.dtype).float(),
                                       atol=1e-2,
                                       rtol=1e-3)
