from __future__ import annotations

import math
import random
from contextlib import contextmanager

import torch

from lmdeploy.turbomind.weight_format import CompressedTensorFormat, FP8Format, MXFP4Format, TrivialFormat

from .cases import LinearCase
from .linear import Linear, Weight, _tm, _to_tm_dtype
from .reference import compare_tensors, dense_gemm, moe_reference, quantize_symm_row_fp8

TOLERANCES = {('bf16', 'bf16'): {'quant_vs_dequant': {'max_abs': 1e-2, 'mean_abs': 1e-3}}, ('fp16', 'fp16'): {'quant_vs_dequant': {'max_abs': 1e-2, 'mean_abs': 1e-3}}, ('bf16', 'fp8_e4m3'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}}, ('fp16', 'fp8_e4m3'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}}, ('bf16', 'uint4'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}}, ('fp16', 'uint4'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}}, ('bf16', 'fp4_e2m1'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}}, ('fp16', 'fp4_e2m1'): {'quant_vs_dequant': {'max_abs': 0.25, 'mean_abs': 0.05}}}

_TORCH_DTYPE = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp8_e4m3': torch.float8_e4m3fn}


def _weight_fill_scale(input_dim: int) -> float:
    return 0.1 / math.sqrt(max(input_dim, 1))


def sample_moe_routing(batch_size: int, expert_num: int, experts_per_token: int, device: torch.device, seed: int = 5489) -> dict[str, torch.Tensor]:
    """Create the routing layouts used by the grouped GEMM tests."""
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

    return {'f2n': f2n_cpu.to(device=device, non_blocking=False), 'en2f': en2f_cpu.to(device=device, non_blocking=False), 'offsets': offsets_cpu.to(device=device, non_blocking=False), 'scales': scales_cpu.to(device=device, non_blocking=False)}


class LinearFixture:

    def __init__(self, case: LinearCase, device: torch.device | None = None):
        self.case = case
        resolved = device or torch.device('cuda')
        if resolved.index is None:
            resolved = torch.device('cuda', torch.cuda.current_device())
        self.device = resolved

        self.linear: Linear | None = None
        self.weight_plan = None
        self.exec_plan = None
        self.w_quant: Weight | None = None
        self.w_original_torch = None
        self.w_dequant_torch = None

        self.x_original: torch.Tensor | None = None
        self.x_source: torch.Tensor | None = None
        self.x_dequant: torch.Tensor | None = None
        self.input_scales: torch.Tensor | None = None
        self.f2n: torch.Tensor | None = None
        self.en2f: torch.Tensor | None = None
        self.offsets: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None

        self.output: torch.Tensor | None = None
        self.output_scales: torch.Tensor | None = None
        self.d_original: torch.Tensor | None = None
        self.d_dequant: torch.Tensor | None = None
        self.d_quant: torch.Tensor | None = None
        self.returned_output: torch.Tensor | None = None
        self.returned_scales: torch.Tensor | None = None

        try:
            self.linear = Linear(self.device)
            self._build_weights()
        except Exception:
            self.close()
            raise

    @contextmanager
    def _quantization_context(self):
        with self.linear._activate():
            yield

    def _torch_dtype(self, name: str | None = None) -> torch.dtype:
        name = self.case.data_type if name is None else name
        try:
            return _TORCH_DTYPE[name]
        except KeyError as error:
            raise NotImplementedError(f'torch_dtype_for_{name}') from error

    def _weight_format(self):
        case = self.case
        if case.weight_type in ('bf16', 'fp16'):
            return TrivialFormat(weight_dtype=_to_tm_dtype(self._torch_dtype(case.weight_type)))
        if case.weight_type == 'fp8_e4m3':
            return FP8Format(block_out=128)
        if case.weight_type == 'uint4':
            return CompressedTensorFormat(block_in=case.group_size)
        if case.weight_type == 'fp4_e2m1':
            if case.group_size == 16:
                raise NotImplementedError('NVFP4 is not supported by the standalone Linear API')
            return MXFP4Format()
        raise NotImplementedError(f'weight_type_{case.weight_type}')

    def _make_source(self, output_dim: int) -> torch.Tensor:
        case = self.case
        dtype = self._torch_dtype()
        scale = _weight_fill_scale(case.input_dim)
        if case.weight_type == 'uint4':
            half_group = case.group_size // 2
            values = torch.randn(output_dim, case.input_dim // case.group_size, half_group, device=self.device, dtype=dtype)
            return torch.cat((values, -values), dim=-1).reshape(output_dim, case.input_dim) * scale
        return torch.randn(output_dim, case.input_dim, device=self.device, dtype=dtype) * scale

    def _make_weight_params(self, output_dim: int):
        case = self.case
        dtype = self._torch_dtype()
        source = self._make_source(output_dim).contiguous()
        original = source.t()

        if case.weight_type in ('bf16', 'fp16'):
            return source, None, None, original, original

        dequant = torch.empty_like(source)
        tm = _tm
        with self._quantization_context():
            if case.weight_type == 'fp8_e4m3':
                raw_weight = torch.empty_like(source, dtype=torch.float8_e4m3fn)
                raw_scales = torch.empty(((output_dim + 127) // 128, (case.input_dim + 127) // 128), dtype=torch.float32, device=self.device)
                tm.QuantizeSymmBlock(out=tm.from_dlpack_with_strides(raw_weight), scale=tm.from_dlpack_with_strides(raw_scales), src=tm.from_dlpack_with_strides(source))
                tm.DequantizeSymmBlock(out=tm.from_dlpack_with_strides(dequant), src=tm.from_dlpack_with_strides(raw_weight), scale=tm.from_dlpack_with_strides(raw_scales))
                return raw_weight, raw_scales, None, original, dequant.t()

            if case.weight_type == 'uint4':
                raw_weight = torch.empty((output_dim, case.input_dim // 8), dtype=torch.int32, device=self.device)
                quant = tm.from_dlpack(raw_weight, [output_dim, case.input_dim], tm.DataType.TYPE_UINT4)
                raw_scales = torch.empty((output_dim, case.input_dim // case.group_size), dtype=dtype, device=self.device)
                raw_zeros = torch.empty_like(raw_scales)
                tm.QuantizeGroupwise(quant=quant, scales=tm.from_dlpack_with_strides(raw_scales), zeros=tm.from_dlpack_with_strides(raw_zeros), dequant=tm.from_dlpack_with_strides(dequant), src=tm.from_dlpack_with_strides(source), group_size=case.group_size)
                return raw_weight, raw_scales, None, original, dequant.t()

            raw_blocks = torch.empty((output_dim, case.input_dim // 32, 16), dtype=torch.uint8, device=self.device)
            quant = tm.from_dlpack(raw_blocks, [output_dim, case.input_dim], tm.DataType.TYPE_FP4_E2M1)
            raw_scales = torch.empty((output_dim, case.input_dim // 32), dtype=torch.uint8, device=self.device)
            tm.QuantizeGroupwise(quant=quant, scales=tm.from_dlpack_with_strides(raw_scales), zeros=None, dequant=tm.from_dlpack_with_strides(dequant), src=tm.from_dlpack_with_strides(source), group_size=32)
            return raw_blocks, raw_scales, None, original, dequant.t()

    def _build_weights(self) -> None:
        case = self.case
        grouped = case.expert_num > 0
        weight_format = self._weight_format()
        plan = self.linear.get_weight_plan(weight_format=weight_format, dtype=self._torch_dtype(), input_dtype=self._torch_dtype(case.input_type), grouped=grouped, fusion_type='silu' if case.fuse_silu else None)
        self.weight_plan = plan

        if not grouped:
            if case.fuse_silu:
                projection_n = case.output_dim // 2
                gate_weight, gate_scales, gate_zeros, gate_original, gate_dequant = self._make_weight_params(projection_n)
                up_weight, up_scales, up_zeros, up_original, up_dequant = self._make_weight_params(projection_n)
                scales = None if gate_scales is None else (gate_scales, up_scales)
                zeros = None if gate_zeros is None else (gate_zeros, up_zeros)
                self.w_quant = self.linear.fuse_weight((gate_weight, up_weight), plan=plan, scales=scales, zeros=zeros)
                self.w_original_torch = (gate_original, up_original)
                self.w_dequant_torch = (gate_dequant, up_dequant)
                return

            weight, scales, zeros, original, dequant = self._make_weight_params(case.output_dim)
            self.w_quant = self.linear.prepare_weight(weight, plan=plan, scales=scales, zeros=zeros)
            self.w_original_torch = original
            self.w_dequant_torch = dequant
            return

        if case.fuse_silu:
            projection_n = case.output_dim // 2
            gate_weights = []
            gate_scales = []
            gate_zeros = []
            gate_originals = []
            gate_dequants = []
            up_weights = []
            up_scales = []
            up_zeros = []
            up_originals = []
            up_dequants = []
            for _ in range(case.expert_num):
                gate_weight, gate_scale, gate_zero, gate_original, gate_dequant = self._make_weight_params(projection_n)
                up_weight, up_scale, up_zero, up_original, up_dequant = self._make_weight_params(projection_n)
                gate_weights.append(gate_weight)
                gate_scales.append(gate_scale)
                gate_zeros.append(gate_zero)
                gate_originals.append(gate_original)
                gate_dequants.append(gate_dequant)
                up_weights.append(up_weight)
                up_scales.append(up_scale)
                up_zeros.append(up_zero)
                up_originals.append(up_original)
                up_dequants.append(up_dequant)
            scales = None if gate_scales[0] is None else (gate_scales, up_scales)
            zeros = None if gate_zeros[0] is None else (gate_zeros, up_zeros)
            self.w_quant = self.linear.fuse_weight((gate_weights, up_weights), plan=plan, scales=scales, zeros=zeros)
            self.w_original_torch = (gate_originals, up_originals)
            self.w_dequant_torch = (gate_dequants, up_dequants)
            return

        weights = []
        scales = []
        zeros = []
        originals = []
        dequants = []
        for _ in range(case.expert_num):
            weight, scale, zero, original, dequant = self._make_weight_params(case.output_dim)
            weights.append(weight)
            scales.append(scale)
            zeros.append(zero)
            originals.append(original)
            dequants.append(dequant)
        self.w_quant = self.linear.prepare_weight(weights, plan=plan, scales=None if scales[0] is None else scales, zeros=None if zeros[0] is None else zeros)
        self.w_original_torch = originals
        self.w_dequant_torch = dequants

    def _prepare_activation(self, x: torch.Tensor) -> None:
        case = self.case
        self.x_source = x
        if case.input_type == case.data_type:
            self.x_original = x
            self.x_dequant = None
            self.input_scales = None
            return
        if case.input_type != 'fp8_e4m3':
            raise NotImplementedError(f'input_type_{case.input_type}')

        rows = x.shape[0]
        groups = (x.shape[1] + 127) // 128
        aligned_rows = (rows + 3) // 4 * 4
        x_quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        x_scales = torch.empty_strided((groups, rows), (aligned_rows, 1), dtype=torch.float32, device=self.device)
        x_dequant = torch.empty_like(x)
        tm = _tm
        with self._quantization_context():
            tm.QuantizeSymm(out=tm.from_dlpack_with_strides(x_quant), scale=tm.from_dlpack_with_strides(x_scales), src=tm.from_dlpack_with_strides(x))
            tm.DequantizeSymm(out=tm.from_dlpack_with_strides(x_dequant), src=tm.from_dlpack_with_strides(x_quant), scale=tm.from_dlpack_with_strides(x_scales))
        self.x_original = x_quant
        self.x_dequant = x_dequant
        self.input_scales = x_scales

    def prepare_batch(self, batch_size: int) -> None:
        self._prepare_batch(batch_size)

    def _prepare_batch(self, batch_size: int) -> None:
        case = self.case
        x_tokens = torch.randn(batch_size, case.input_dim, device=self.device, dtype=self._torch_dtype())

        if case.expert_num > 0:
            route = sample_moe_routing(batch_size, case.expert_num, case.experts_per_token, self.device)
            self.f2n = route['f2n']
            self.en2f = route['en2f']
            self.offsets = route['offsets']
            self.scales = route['scales']
            x = x_tokens if case.moe_indexed else x_tokens[self.f2n.long()].contiguous()
        else:
            self.f2n = None
            self.en2f = None
            self.offsets = None
            self.scales = None
            x = x_tokens

        self._prepare_activation(x)
        if case.expert_num > 0:
            indices = self.f2n if case.moe_indexed else None
            self.exec_plan = self.linear.get_exec_plan(self.x_original, self.w_quant, offsets=self.offsets, indices=indices)
        else:
            self.exec_plan = self.linear.get_exec_plan(self.x_original, self.w_quant)
        self.output = None
        self.output_scales = None
        self.d_original = None
        self.d_dequant = None
        self.d_quant = None
        self.returned_output = None
        self.returned_scales = None

    @staticmethod
    def _silu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate_float = gate.float()
        return (gate_float * torch.sigmoid(gate_float) * up.float()).to(gate.dtype)

    def _reference_gemm(self, weights, *, dequant: bool):
        case = self.case
        x = self.x_dequant if dequant and self.x_dequant is not None else self.x_source
        if case.expert_num == 0:
            return dense_gemm(x, weights)

        indices = self.f2n if case.moe_indexed else None
        return moe_reference(x, weights, indices, self.offsets, self.scales, self.en2f, case.experts_per_token, False)

    def run_reference(self) -> None:
        self._run_reference()

    def _run_reference(self) -> None:
        case = self.case

        def evaluate(weights, *, dequant):
            if not case.fuse_silu:
                return self._reference_gemm(weights, dequant=dequant)
            gate, up = weights
            return self._silu(self._reference_gemm(gate, dequant=dequant), self._reference_gemm(up, dequant=dequant))

        self.d_original = evaluate(self.w_original_torch, dequant=False)
        self.d_dequant = evaluate(self.w_dequant_torch, dequant=True)
        if self.exec_plan.output_scales is not None:
            _, _, self.d_original = quantize_symm_row_fp8(self.d_original)
            _, _, self.d_dequant = quantize_symm_row_fp8(self.d_dequant)

    def run_linear_forward(self):
        return self._run_linear_forward()

    def _run_linear_forward(self):
        case = self.case
        if case.expert_num > 0:
            self.output, self.output_scales = self.linear.forward_moe(self.x_original, self.w_quant, exec_plan=self.exec_plan, offsets=self.offsets, indices=self.f2n if case.moe_indexed else None, input_scales=self.input_scales, out=self.output, out_scales=self.output_scales)
        else:
            self.output, self.output_scales = self.linear(self.x_original, self.w_quant, exec_plan=self.exec_plan, input_scales=self.input_scales, out=self.output, out_scales=self.output_scales)
        return self.output, self.output_scales

    def run_linear(self) -> None:
        output, output_scales = self._run_linear_forward()
        self.returned_output = output
        self.returned_scales = output_scales
        if output_scales is None:
            self.d_quant = output.clone()
            return

        self.d_quant = torch.empty(output.shape, dtype=self._torch_dtype(), device=self.device)
        tm = _tm
        with self._quantization_context():
            tm.DequantizeSymm(out=tm.from_dlpack_with_strides(self.d_quant), src=tm.from_dlpack_with_strides(output), scale=tm.from_dlpack_with_strides(output_scales))

    def tune(self) -> None:
        case = self.case
        indices = self.f2n if case.moe_indexed else None
        self.exec_plan, self.output, self.output_scales = self.linear.tune(self.x_original, self.w_quant, offsets=self.offsets, indices=indices, out=self.output, input_scales=self.input_scales, out_scales=self.output_scales)

    def compare(self) -> dict[str, dict[str, float]]:
        return {'quant_vs_dequant': compare_tensors(self.d_quant, self.d_dequant), 'quant_vs_original': compare_tensors(self.d_quant, self.d_original), 'dequant_vs_original': compare_tensors(self.d_dequant, self.d_original)}

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

    def close(self) -> None:
        with torch.cuda.device(self.device):
            self.exec_plan = None
            self.output = None
            self.output_scales = None
            self.d_quant = None
            self.d_original = None
            self.d_dequant = None
            self.returned_output = None
            self.returned_scales = None
            self.x_original = None
            self.x_source = None
            self.x_dequant = None
            self.input_scales = None
            self.f2n = None
            self.en2f = None
            self.offsets = None
            self.scales = None
            self.w_original_torch = None
            self.w_dequant_torch = None
            if self.w_quant is not None:
                self.w_quant.close()
                self.w_quant = None
            if self.linear is not None:
                self.linear.close()
                self.linear = None
            self.weight_plan = None
