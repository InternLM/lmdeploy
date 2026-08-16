from __future__ import annotations

import math
import random
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from .cases import LinearCase
from .linear import (
    Linear,
    Weight,
    activation_needs_quantize,
    dequantize_symm,
    device_context,
    link_experts,
    quantize_symm,
)
from .reference import (
    apply_block_fused_silu,
    block_pack_w1w3,
    compare_tensors,
    dense_gemm,
    fused_silu_block,
    moe_reference,
    quantize_symm_row_fp8,
)

# quant_vs_dequant: LlamaLinear vs torch on matched dequant oracle (incl. act
# quant round-trip when Linear quantizes activations). Gate on abs only.
# Weights are scaled by 0.1/sqrt(K) so |C| stays O(0.1) and bf16/fp16 ULP
# (~1e-3 at that magnitude) remains well under identity abs gates.
TOLERANCES = {
    ('bf16', 'bf16'): {'quant_vs_dequant': {'max_abs': 1e-2, 'mean_abs': 1e-3}},
    ('fp16', 'fp16'): {'quant_vs_dequant': {'max_abs': 1e-2, 'mean_abs': 1e-3}},
    ('bf16', 'fp8_e4m3'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}},
    ('fp16', 'fp8_e4m3'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}},
    ('fp16', 'uint4'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}},
    ('bf16', 'fp4_e2m1'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}},
    ('fp16', 'fp4_e2m1'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}},
}


def _weight_fill_scale(input_dim: int) -> float:
    return 0.1 / math.sqrt(max(input_dim, 1))

_TORCH_DTYPE = {
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
}


def sample_moe_routing(
    batch_size: int,
    expert_num: int,
    experts_per_token: int,
    device: torch.device,
    seed: int = 5489,
) -> dict[str, torch.Tensor]:
    """Port of testbed_v3 Route() layouts (f2n / en2f / offsets / scales).

    Sampling uses Python random.sample (not libstdc++ std::sample), but the
    buffer shapes and index algebra match Route() exactly.
    """
    # std::mt19937 default seed is 5489u
    rng = random.Random(seed)
    expert_ids: list[int] = []
    for _ in range(batch_size):
        expert_ids.extend(rng.sample(range(expert_num), experts_per_token))

    scales_cpu = torch.empty(batch_size * experts_per_token, dtype=torch.float32)
    for i in range(batch_size):
        tmp = [rng.uniform(1e-3, 1.0) for _ in range(experts_per_token)]
        s = sum(tmp)
        for e in range(experts_per_token):
            scales_cpu[e * batch_size + i] = tmp[e] / s

    count = [0] * expert_num
    f2i: list[list[int]] = [[] for _ in range(expert_num)]
    for i, eid in enumerate(expert_ids):
        count[eid] += 1
        f2i[eid].append(i)

    offsets_cpu = torch.empty(expert_num + 1, dtype=torch.int32)
    offsets_cpu[0] = 0
    for i in range(expert_num):
        offsets_cpu[i + 1] = offsets_cpu[i] + count[i]

    token_slots = len(expert_ids)
    f2n_cpu = torch.empty(token_slots, dtype=torch.int32)
    en2f_cpu = torch.empty(token_slots, dtype=torch.int32)
    i = 0
    for e in range(expert_num):
        for x in f2i[e]:
            f2n_cpu[i] = x // experts_per_token
            en = x % experts_per_token * batch_size + x // experts_per_token
            en2f_cpu[en] = i
            i += 1

    # Keep routing copies blocking so the active stream can consume them immediately.
    return {
        'f2n': f2n_cpu.to(device=device, non_blocking=False),
        'en2f': en2f_cpu.to(device=device, non_blocking=False),
        'offsets': offsets_cpu.to(device=device, non_blocking=False),
        'scales': scales_cpu.to(device=device, non_blocking=False),
    }


class _CudaStreamBoundary:

    def __init__(self, stream: torch.cuda.Stream, device: torch.device):
        self.stream = stream
        self.device = device

    @contextmanager
    def enter(self) -> Iterator[torch.cuda.Stream]:
        caller = torch.cuda.current_stream(self.device)
        if caller == self.stream:
            yield self.stream
            return

        self.stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self.stream):
                yield self.stream
        finally:
            caller.wait_stream(self.stream)

    def synchronize(self) -> None:
        self.stream.synchronize()

    def assert_active(self) -> None:
        assert torch.cuda.current_stream(self.device) == self.stream

    def clone_to_caller(self, tensor: torch.Tensor) -> torch.Tensor:
        caller = torch.cuda.current_stream(self.device)
        caller.wait_stream(self.stream)
        out = tensor.detach().clone()
        self.stream.wait_stream(caller)
        return out


class LinearFixture:
    def __init__(self, case: LinearCase, device: torch.device | None = None, *, force_nonnative_fp8: bool = False):
        self.case = case
        self.device = device or torch.device('cuda')
        self.force_nonnative_fp8 = force_nonnative_fp8
        self._ctx = device_context()
        self._ctx.__enter__()
        self._stream_boundary: _CudaStreamBoundary | None = _CudaStreamBoundary(
            torch.cuda.ExternalStream(self._ctx.stream_ptr, device=self.device),
            self.device,
        )
        self.linear: Linear | None = None
        self.w_original: Weight | None = None
        self.w_quant: Weight | None = None
        self.w_dequant: Weight | None = None
        self.e_original: list[Weight] = []
        self.e_quant: list[Weight] = []
        self.e_dequant: list[Weight] = []
        self.w_original_torch: torch.Tensor | list[torch.Tensor] | None = None
        self.w_dequant_torch: torch.Tensor | list[torch.Tensor] | None = None
        self.x_original: torch.Tensor | None = None
        self.x_dequant: torch.Tensor | None = None
        self.f2n: torch.Tensor | None = None
        self.en2f: torch.Tensor | None = None
        self.offsets: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.d_original: torch.Tensor | None = None
        self.d_dequant: torch.Tensor | None = None
        self.d_quant: torch.Tensor | None = None
        try:
            with self.on_tm_stream():
                self.linear = Linear()
            if case.expert_num > 0:
                self._build_moe_weights()
            else:
                self._build_dense_weights()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        # Join streams, then drop TM-backed torch tensors while the Context mempool
        # is still alive (from_dlpack / Forward outputs alias pool storage).
        if self._stream_boundary is not None:
            with self.on_tm_stream():
                pass
            self.sync_tm()
            torch.cuda.current_stream(self.device).synchronize()
        self.d_quant = None
        self.x_dequant = None
        self.d_original = None
        self.d_dequant = None
        self.x_original = None
        self.f2n = None
        self.en2f = None
        self.offsets = None
        self.scales = None
        self.w_original_torch = None
        self.w_dequant_torch = None
        # Destroy stream users before leaving the device Context.
        # Fused views alias expert storage — drop fused first, then experts.
        self.linear = None
        self.w_original = None
        self.w_quant = None
        self.w_dequant = None
        self.e_original = []
        self.e_quant = []
        self.e_dequant = []
        self._stream_boundary = None
        ctx = getattr(self, '_ctx', None)
        if ctx is not None:
            self._ctx = None
            ctx.__exit__(None, None, None)

    @contextmanager
    def on_tm_stream(self) -> Iterator[torch.cuda.Stream]:
        assert self._stream_boundary is not None
        with self._stream_boundary.enter() as stream:
            yield stream

    def sync_tm(self) -> None:
        """Host wait until Context::stream() is idle (e.g. before del / set_measure)."""
        assert self._stream_boundary is not None
        self._stream_boundary.synchronize()

    def _torch_dtype(self) -> torch.dtype:
        try:
            return _TORCH_DTYPE[self.case.data_type]
        except KeyError as e:
            raise NotImplementedError(f'torch_dtype_for_{self.case.data_type}') from e

    def _clone_tm_weight(self, w: Weight) -> torch.Tensor:
        """Clone TM weight storage into caller-stream-owned torch storage."""
        assert self._stream_boundary is not None
        return self._stream_boundary.clone_to_caller(w.weight_tensor())

    def _fill_weight_triple(
        self,
        w_original: Weight,
        w_quant: Weight,
        w_dequant: Weight,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize into w_quant / w_dequant; return torch dequant view (pre-
        prepare)."""
        c = self.case
        src = w.contiguous()
        with self.on_tm_stream():
            sp = self._ctx.stream_ptr
            w_original.copy_weight_from(src, stream_ptr=sp)
            if c.weight_type == c.data_type:
                w_quant.copy_weight_from(src, stream_ptr=sp)
                w_dequant.copy_weight_from(src, stream_ptr=sp)
            elif c.weight_type == 'fp8_e4m3':
                w_quant.quantize_symm_block_from(w_original)
                w_dequant.dequantize_symm_block_from(w_quant)
            elif c.weight_type in ('uint4', 'fp4_e2m1'):
                w_quant.quantize_groupwise_into(w_original, w_dequant)
            else:
                raise NotImplementedError(f'weight_type_{c.weight_type}')
        if c.weight_type == c.data_type:
            return w.clone()
        return self._clone_tm_weight(w_dequant)

    def _make_random_weight(self) -> torch.Tensor:
        """Random weight; fuse_silu uses the dtype-specific gate/up block
        layout."""
        c = self.case
        dtype = self._torch_dtype()
        scale = _weight_fill_scale(c.input_dim)
        if c.fuse_silu:
            inter = c.output_dim // 2
            w1 = torch.randn(c.input_dim, inter, device=self.device, dtype=dtype) * scale
            w3 = torch.randn(c.input_dim, inter, device=self.device, dtype=dtype) * scale
            return block_pack_w1w3(w1, w3, fused_silu_block(c.weight_type))
        return torch.randn(c.input_dim, c.output_dim, device=self.device, dtype=dtype) * scale

    def _apply_fuse_silu_epilogue(self, weight: Weight) -> None:
        from .linear import _tm

        weight.set_epilogue(_tm().Epilogue.kGatedSilu)

    def _allocate_weight_triple(self) -> tuple[Weight, Weight, Weight]:
        c = self.case
        with self.on_tm_stream():
            return (
                Weight(c.input_dim, c.output_dim, c.data_type, c.data_type, 0),
                Weight(c.input_dim, c.output_dim, c.data_type, c.weight_type, c.group_size),
                Weight(c.input_dim, c.output_dim, c.data_type, c.data_type, 0),
            )

    def _make_weight_triple(self) -> tuple[Weight, Weight, Weight, torch.Tensor, torch.Tensor]:
        c = self.case
        w_original, w_quant, w_dequant = self._allocate_weight_triple()
        w = self._make_random_weight()
        w_deq_torch = self._fill_weight_triple(w_original, w_quant, w_dequant, w)
        # Clone dequant before prepare(): groupwise/fp8 prepare may repack storage.
        with self.on_tm_stream():
            if self.force_nonnative_fp8:
                if c.weight_type != 'fp8_e4m3':
                    raise ValueError('force_nonnative_fp8_requires_fp8_weight')
                from .linear import _tm, to_tm_dtype
                dtype = to_tm_dtype(c.data_type)
                w_quant._impl.input_format = _tm().ResolveLinearWeightFormat(dtype, dtype, 1, 1)
            w_original.prepare()
            w_quant.prepare()
            w_dequant.prepare()
            if c.fuse_silu:
                self._apply_fuse_silu_epilogue(w_quant)
        return w_original, w_quant, w_dequant, w, w_deq_torch

    def _build_dense_weights(self) -> None:
        w_o, w_q, w_d, w_torch, w_deq = self._make_weight_triple()
        self.w_original = w_o
        self.w_quant = w_q
        self.w_dequant = w_d
        self.w_original_torch = w_torch
        self.w_dequant_torch = w_deq
        self.sync_tm()

    def _build_moe_weights(self) -> None:
        """Build per-expert quant Weights and torch reference tensors.

        Prepare is deferred until every expert has been quantized: preparing
        quantized weights can free/transpose scale storage and corrupt peers.
        """
        c = self.case
        orig_torch: list[torch.Tensor] = []
        deq_torch: list[torch.Tensor] = []
        for _ in range(c.expert_num):
            w_original, w_quant, w_dequant = self._allocate_weight_triple()
            w = self._make_random_weight()
            w_deq = self._fill_weight_triple(w_original, w_quant, w_dequant, w)
            orig_torch.append(w)
            deq_torch.append(w_deq)
            self.e_quant.append(w_quant)
            self.sync_tm()
            del w_original, w_dequant
        with self.on_tm_stream():
            for w_quant in self.e_quant:
                # Production MoE sets this via FfnWeight::prepare (is_expert_).
                # Required so GetConverters packs bf16/fp16 B for Config_F16_g.
                w_quant.set_grouped(True)
                w_quant.prepare()
                if c.fuse_silu:
                    self._apply_fuse_silu_epilogue(w_quant)
                self.sync_tm()
            fused = link_experts(self.e_quant)
            if c.fuse_silu:
                self._apply_fuse_silu_epilogue(fused)
            self.w_quant = fused
            self.sync_tm()
        self.w_original_torch = orig_torch
        self.w_dequant_torch = deq_torch

    def _prepare_x_dequant(self) -> None:
        """Match LlamaLinear GetOperandA: QuantizeSymm then DequantizeSymm.

        On SM90, FP8 weights derive input_format=fp8 even when the caller passes
        bf16 activations. The dequant oracle must use the same act quant round-trip
        or quant_vs_dequant is dominated by act error (~0.1 mean abs), not GEMM.
        """
        assert self.x_original is not None
        assert self.w_quant is not None
        if not activation_needs_quantize(self.w_quant):
            self.x_dequant = None
            return
        from .linear import _tm

        with self.on_tm_stream():
            x_tm = _tm().from_dlpack_with_strides(self.x_original)
            x_q, x_s = quantize_symm(x_tm)
            x_d = dequantize_symm(x_q, x_s)
            x_d_torch = torch.from_dlpack(x_d).to(dtype=self.x_original.dtype)
        # Clone so x_dequant never aliases Context-pool storage (same-dtype .to is a no-op).
        assert self._stream_boundary is not None
        self.x_dequant = self._stream_boundary.clone_to_caller(x_d_torch)

    def prepare_batch(self, batch_size: int) -> None:
        c = self.case
        dtype = self._torch_dtype()
        x_tokens = torch.randn(batch_size, c.input_dim, device=self.device, dtype=dtype)
        # testbed_v3 still feeds data_type activations into Forward; input_type
        # only changes how the dequant reference is formed. We always match
        # LlamaLinear's act quant via activation_needs_quantize().
        if c.input_type not in (c.data_type, 'fp8_e4m3'):
            raise NotImplementedError(f'input_type_{c.input_type}')
        if c.expert_num > 0:
            route = sample_moe_routing(batch_size, c.expert_num, c.experts_per_token, self.device)
            self.f2n = route['f2n']
            self.en2f = route['en2f']
            self.offsets = route['offsets']
            self.scales = route['scales']
            if c.moe_indexed:
                # w1/gate: token-major x + f2n indices.
                self.x_original = x_tokens
            else:
                # w2/down: expert-packed x, offsets only (matches moe_ffn_layer).
                self.x_original = x_tokens[self.f2n.long()].contiguous()
        else:
            self.f2n = None
            self.en2f = None
            self.offsets = None
            self.scales = None
            self.x_original = x_tokens
        self._prepare_x_dequant()

    def run_reference(self) -> None:
        assert self.x_original is not None
        assert self.w_original_torch is not None
        assert self.w_dequant_torch is not None
        c = self.case
        if c.expert_num > 0:
            assert self.offsets is not None
            assert self.scales is not None and self.en2f is not None
            assert isinstance(self.w_original_torch, list)
            assert isinstance(self.w_dequant_torch, list)
            x_deq = self.x_dequant if self.x_dequant is not None else self.x_original
            f2n = self.f2n if c.moe_indexed else None
            # Validate the grouped GEMM (packed expert-major outputs).
            self.d_original = moe_reference(
                self.x_original,
                self.w_original_torch,
                f2n,
                self.offsets,
                self.scales,
                self.en2f,
                c.experts_per_token,
                False,
            )
            self.d_dequant = moe_reference(
                x_deq,
                self.w_dequant_torch,
                f2n,
                self.offsets,
                self.scales,
                self.en2f,
                c.experts_per_token,
                False,
            )
        else:
            assert isinstance(self.w_original_torch, torch.Tensor)
            assert isinstance(self.w_dequant_torch, torch.Tensor)
            self.d_original = dense_gemm(self.x_original, self.w_original_torch)
            x_deq = self.x_dequant if self.x_dequant is not None else self.x_original
            self.d_dequant = dense_gemm(x_deq, self.w_dequant_torch)
        if c.fuse_silu:
            block = fused_silu_block(c.weight_type)
            self.d_original = apply_block_fused_silu(self.d_original, block)
            self.d_dequant = apply_block_fused_silu(self.d_dequant, block)
            # SM90 FP8 fused path quantizes after SiLU; reference matches QuantizeSymm.
            if c.weight_type == 'fp8_e4m3':
                _, _, self.d_original = quantize_symm_row_fp8(self.d_original)
                _, _, self.d_dequant = quantize_symm_row_fp8(self.d_dequant)

    def _forward_linear(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self._stream_boundary is not None
        self._stream_boundary.assert_active()
        assert self.linear is not None
        assert self.x_original is not None
        assert self.w_quant is not None
        c = self.case
        if c.expert_num > 0:
            assert self.offsets is not None
            indices = self.f2n if c.moe_indexed else None
            return self.linear.forward_moe(
                self.x_original, self.w_quant, indices, self.offsets)
        return self.linear.forward_dense(self.x_original, self.w_quant)

    def run_linear(self) -> None:
        with self.on_tm_stream():
            out, out_scales = self._forward_linear()
        # FP8 fused path is already dequantized to bf16 in Linear._pack_forward_result.
        assert self._stream_boundary is not None
        self.d_quant = self._stream_boundary.clone_to_caller(out)
        del out, out_scales
        with self.on_tm_stream():
            self.release_forward_result()

    def release_forward_result(self) -> None:
        assert self._stream_boundary is not None
        self._stream_boundary.assert_active()
        assert self.linear is not None
        self.linear.release_forward_result()

    def run_linear_forward(self) -> None:
        """Launch one forward while on_tm_stream() owns the active stream."""
        self._forward_linear()

    def compare(self) -> dict[str, dict[str, float]]:
        assert self.d_quant is not None
        assert self.d_dequant is not None
        assert self.d_original is not None
        return {
            'quant_vs_dequant': compare_tensors(self.d_quant, self.d_dequant),
            'quant_vs_original': compare_tensors(self.d_quant, self.d_original),
            'dequant_vs_original': compare_tensors(self.d_dequant, self.d_original),
        }

    def check_tolerances(self, metrics: dict[str, dict[str, float]]) -> None:
        key = (self.case.data_type, self.case.weight_type)
        gates = TOLERANCES.get(key)
        if gates is None:
            raise ValueError(f'no_tolerances_for_{key}')
        for pair, limits in gates.items():
            for metric_name, limit in limits.items():
                value = metrics[pair][metric_name]
                if not math.isfinite(value):
                    raise AssertionError(f'{pair}.{metric_name}={value} (non-finite)')
                if value > limit:
                    raise AssertionError(f'{pair}.{metric_name}={value} > {limit}')
