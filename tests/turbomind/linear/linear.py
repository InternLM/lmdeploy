from __future__ import annotations

import ctypes
from collections.abc import Sequence

import torch

_TM = None


def _tm():
    global _TM
    if _TM is None:
        import _turbomind as tm
        _TM = tm
    return _TM


def is_available() -> bool:
    try:
        _tm()
        return True
    except Exception:
        return False


_DTYPE_TO_TM = {
    'bf16': 'TYPE_BF16',
    'fp16': 'TYPE_FP16',
    'fp8_e4m3': 'TYPE_FP8_E4M3',
    'uint4': 'TYPE_UINT4',
    'fp4_e2m1': 'TYPE_FP4_E2M1',
}


def to_tm_dtype(name: str):
    tm = _tm()
    key = _DTYPE_TO_TM[name]
    return getattr(tm.DataType, key)


def _resolve_block_sizes(weight_type: str, group_size: int) -> tuple[int, int]:
    # Match ResolveLinearWeightFormat rules in data_format.cc.
    if weight_type == 'fp8_e4m3':
        return 128, 128
    if weight_type in ('uint4', 'fp4_e2m1'):
        return (group_size or 1), 1
    return 1, 1


def quantize_symm_block(src, out=None, scale=None):
    """Thin wrapper over tm.QuantizeSymmBlock (Context stream).

    Returns (out, scale) TM tensors. Caller must Param.set them onto Weight when scale was newly allocated (typical for
    empty scales).
    """
    return _tm().QuantizeSymmBlock(out=out, scale=scale, src=src)


def dequantize_symm_block(src, scale, out=None):
    """Thin wrapper over tm.DequantizeSymmBlock (Context stream)."""
    return _tm().DequantizeSymmBlock(out=out, src=src, scale=scale)


def quantize_symm(src, out=None, scale=None):
    """Thin wrapper over tm.QuantizeSymm (row/group act quant, Context
    stream)."""
    return _tm().QuantizeSymm(out=out, scale=scale, src=src)


def dequantize_symm(src, scale, out=None):
    """Thin wrapper over tm.DequantizeSymm (Context stream)."""
    return _tm().DequantizeSymm(out=out, src=src, scale=scale)


def quantize_groupwise(
    *,
    quant,
    scales,
    dequant,
    src,
    group_size: int,
    zeros=None,
    rbits=None,
) -> None:
    """Thin wrapper over tm.QuantizeGroupwise (Context stream).

    Callers must pass K-major views (``.t()``) matching testbed_v3 GenerateWeight.
    """
    _tm().QuantizeGroupwise(
        quant=quant,
        scales=scales,
        zeros=zeros,
        dequant=dequant,
        src=src,
        rbits=rbits,
        group_size=group_size,
    )


def activation_needs_quantize(weight: Weight) -> bool:
    """True when LlamaLinear will QuantizeSymm activations for this weight."""
    return weight._impl.input_format.dtype != weight._impl.data_type


def tensor_data_ptr(tm_tensor) -> int:
    """Device pointer from a TurboMind Tensor `.data` capsule."""
    cap = tm_tensor.data
    fn = ctypes.pythonapi.PyCapsule_GetPointer
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = fn(cap, None)
    if not ptr:
        raise RuntimeError('null_tensor_data_ptr')
    return int(ptr)


def make_strided_ptrs(ptrs: Sequence[tuple[int, int]], dtype):
    """Build owned StridedPtr tensor via tm.MakeStridedPtrs (Context
    stream)."""
    return _tm().MakeStridedPtrs(list(ptrs), dtype)


def invoke_moe_dispatch(src: torch.Tensor, f2n: torch.Tensor, experts_per_token: int, out: torch.Tensor | None = None):
    tm = _tm()
    y = tm.invokeMoeDispatch(
        out=None if out is None else tm.from_dlpack_with_strides(out),
        src=tm.from_dlpack_with_strides(src),
        f2n=tm.from_dlpack_with_strides(f2n),
        expert_per_token=experts_per_token,
    )
    return torch.from_dlpack(y) if out is None else out


def invoke_moe_combine(
    out: torch.Tensor,
    src: torch.Tensor,
    scales: torch.Tensor,
    en2f: torch.Tensor,
    experts_per_token: int,
    *,
    bias: torch.Tensor | None = None,
    f2E: torch.Tensor | None = None,
    dst_scales: torch.Tensor | None = None,
    bscale: float = 1.0,
    dst_scale: float = 0.0,
) -> torch.Tensor:
    """Mirror testbed_v3 Run() invokeMoeCombine (out must be preallocated)."""
    tm = _tm()
    tm.invokeMoeCombine(
        out=tm.from_dlpack_with_strides(out),
        src=tm.from_dlpack_with_strides(src),
        bias=None if bias is None else tm.from_dlpack_with_strides(bias),
        scales=tm.from_dlpack_with_strides(scales),
        en2f=tm.from_dlpack_with_strides(en2f),
        f2E=None if f2E is None else tm.from_dlpack_with_strides(f2E),
        dst_scales=None if dst_scales is None else tm.from_dlpack_with_strides(dst_scales),
        experts_per_token=experts_per_token,
        bscale=bscale,
        dst_scale=dst_scale,
    )
    return out


def link_experts(experts: Sequence[Weight]) -> Weight:
    """Port of testbed_v3 / moe_weight LinkExperts into a fused Weight view.

    Experts must already be prepare()'d and kept alive for as long as the fused view is used (pointers alias expert
    storage). Call on the Context stream.
    """
    if not experts:
        raise ValueError('link_experts_requires_non_empty')
    e0 = experts[0]
    fused = Weight(
        e0._input_dim,
        e0._output_dim,
        e0._data_type,
        e0._weight_type,
        e0._group_size,
        has_bias=e0._has_bias,
    )
    e0._impl.copy_metadata_to(fused._impl)

    n = len(experts)
    fused._impl.k_desc.num = n
    fused._impl.q_desc.num = n

    weights: list[tuple[int, int]] = []
    scales: list[tuple[int, int]] = []
    for e in experts:
        weights.append((tensor_data_ptr(e.param_tensor('weight')), int(e._impl.k_desc.ld)))
        scales_t = e.param_tensor('scales')
        if scales_t:
            scales.append((tensor_data_ptr(scales_t), int(e._impl.q_desc.ld)))

    fused.set_param(
        'weight',
        make_strided_ptrs(weights, fused._impl.weight_format.dtype),
    )
    if scales:
        fused.set_param('scales', make_strided_ptrs(scales, e0.param_tensor('scales').type))
    fused._impl.k_desc.ld = 0
    fused._impl.q_desc.ld = 0
    fused._impl.k_desc.offsets = 0
    fused._impl.q_desc.offsets = 0
    return fused


class Weight:
    """Adapter over tm.LinearWeight.

    Instances use the TurboMind Context stream; destroy them before exiting the enclosing device_context().
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        data_type: str,
        weight_type: str,
        group_size: int = 0,
        has_bias: bool = False,
    ) -> None:
        tm = _tm()
        dt = to_tm_dtype(data_type)
        wt = to_tm_dtype(weight_type)
        block_in, block_out = _resolve_block_sizes(weight_type, group_size)
        cfg = tm.LinearConfig()
        cfg.input_dim = input_dim
        cfg.output_dim = output_dim
        cfg.data_type = dt
        cfg.format = tm.ResolveLinearWeightFormat(dt, wt, block_in, block_out)
        cfg.has_bias = has_bias
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._data_type = data_type
        self._weight_type = weight_type
        self._group_size = group_size
        self._has_bias = has_bias
        self._impl = tm.LinearWeight(cfg)
        self._impl.param('weight').alloc([input_dim, output_dim], wt)
        # Groupwise formats need scales (/zeros) in MN-major (K/g, N).
        if weight_type in ('uint4', 'fp4_e2m1'):
            if group_size <= 0:
                raise ValueError('groupwise_weight_requires_positive_group_size')
            if input_dim % group_size != 0:
                raise ValueError(f'input_dim_{input_dim}_not_divisible_by_group_size_{group_size}')
            scale_shape = [input_dim // group_size, output_dim]
            # uint4: f16/bf16 scales+zeros; fp4: ue8m0 scales (uint8).
            if weight_type == 'fp4_e2m1':
                scale_dtype = tm.DataType.TYPE_UINT8
            else:
                scale_dtype = dt
            self._impl.param('scales').alloc(scale_shape, scale_dtype)
            if weight_type == 'uint4':
                self._impl.param('zeros').alloc(scale_shape, scale_dtype)

    def set_grouped(self, grouped: bool) -> None:
        """Mark expert weights for grouped-GEMM layout conversion in prepare().

        Mirrors FfnWeight::prepare for is_expert_: without this, bf16/fp16 MoE
        weights stay flat and SM90 Config_F16_g (ibb + packed B) cannot match.
        """
        self._impl.set_grouped(grouped)

    def set_epilogue(self, epilogue) -> None:
        """Set GEMM epilogue (e.g. tm.Epilogue.kGatedSilu for fused SiLU)."""
        tm = _tm()
        self._impl.epilogue = epilogue
        # Mirror FfnWeight::prepare: SM90 FP8 fused SiLU writes FP8 + group scales.
        major, _ = torch.cuda.get_device_capability()
        if (epilogue == tm.Epilogue.kGatedSilu and self._weight_type == 'fp8_e4m3' and major == 9):
            self._impl.set_fp8_fused_silu_output()

    def prepare(self) -> None:
        self._impl.prepare()

    def weight_tensor(self) -> torch.Tensor:
        return torch.from_dlpack(self._impl.param('weight').get())

    def copy_weight_from(self, src: torch.Tensor, *, stream_ptr: int) -> None:
        """Copy contiguous ``src`` into this weight on ``stream_ptr`` (Context::stream).

        Must not use Tensor.copy_from: that path is default-stream cudaMemcpy and
        cannot participate in the harness's scoped Context-stream ownership.

        ``src`` must be materialized and contiguous before entering the Context-stream
        boundary. This method will not create caller-owned storage inside that scope.
        """
        if not src.is_contiguous():
            raise ValueError('copy_weight_from_requires_contiguous_src')
        tm = _tm()
        src_tm = tm.from_dlpack(src)
        tm.generic_copy_on_stream(src_tm, self.param_tensor('weight'), stream_ptr)

    def param_tensor(self, name: str):
        return self._impl.param(name).get()

    def set_param(self, name: str, tensor) -> None:
        self._impl.param(name).set(tensor)

    def quantize_symm_block_from(self, src_weight: Weight) -> None:
        out, scale = quantize_symm_block(
            src_weight.param_tensor('weight'),
            out=self.param_tensor('weight'),
            scale=None,
        )
        self.set_param('weight', out)
        self.set_param('scales', scale)

    def dequantize_symm_block_from(self, quant_weight: Weight) -> None:
        out = dequantize_symm_block(
            quant_weight.param_tensor('weight'),
            quant_weight.param_tensor('scales'),
            out=self.param_tensor('weight'),
        )
        self.set_param('weight', out)

    def quantize_groupwise_into(self, src_weight: Weight, dequant_weight: Weight) -> None:
        """Port of testbed_v3 GenerateWeight uint4/fp4 path (K-major via
        .t())."""
        zeros = self.param_tensor('zeros') if self._weight_type == 'uint4' else None
        quantize_groupwise(
            quant=self.param_tensor('weight').t(),
            scales=self.param_tensor('scales').t(),
            zeros=None if zeros is None else zeros.t(),
            dequant=dequant_weight.param_tensor('weight').t(),
            src=src_weight.param_tensor('weight').t(),
            group_size=self._group_size,
        )


class Linear:
    """Adapter over tm.LlamaLinear.

    Instances use the TurboMind Context stream; destroy them before exiting the enclosing device_context().
    """

    def __init__(self) -> None:
        self._impl = _tm().LlamaLinear()
        # TM tensors backing the latest forward dlpack views (Context pool).
        self._forward_keep_alive: tuple | None = None

    def forward_dense(
        self,
        x: torch.Tensor,
        weight: Weight,
        out: torch.Tensor | None = None,
        input_scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tm = _tm()
        xin = tm.from_dlpack_with_strides(x)
        oin = None if out is None else tm.from_dlpack_with_strides(out)
        iscales = None if input_scales is None else tm.from_dlpack_with_strides(input_scales)
        y, ys = self._impl.forward_dense(xin, weight._impl, oin, iscales, None)
        return self._pack_forward_result(y, ys)

    def forward_moe(
        self,
        x: torch.Tensor,
        weight: Weight,
        f2n: torch.Tensor | None,
        offsets: torch.Tensor,
        out: torch.Tensor | None = None,
        input_scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tm = _tm()
        y, ys = self._impl.forward_moe(
            tm.from_dlpack_with_strides(x),
            weight._impl,
            None if f2n is None else tm.from_dlpack_with_strides(f2n),
            tm.from_dlpack_with_strides(offsets),
            None if out is None else tm.from_dlpack_with_strides(out),
            None if input_scales is None else tm.from_dlpack_with_strides(input_scales),
            None,
        )
        return self._pack_forward_result(y, ys)

    def _pack_forward_result(self, y, ys) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return torch views of Context-stream results (FP8 dequantized to
        bf16).

        Views alias TM pool storage. Callers must transfer them through LinearFixture's stream boundary before releasing
        the backing TM tensors.
        """
        if ys:
            y_bf16 = dequantize_symm(y, ys)
            self._forward_keep_alive = (y, ys, y_bf16)
            y_t = torch.from_dlpack(y_bf16)
            if y_t.dtype != torch.bfloat16:
                raise TypeError(f'expected bf16 dequant output, got {y_t.dtype}')
            return y_t, torch.from_dlpack(ys)
        self._forward_keep_alive = (y, ys)
        return torch.from_dlpack(y), None

    def release_forward_result(self) -> None:
        self._forward_keep_alive = None

    def set_measure(self, on: bool) -> None:
        self._impl.set_measure(on)

    def import_records(self, path: str) -> int:
        return int(self._impl.import_records(path))

    def export_records(self, path: str) -> int:
        return int(self._impl.export_records(path))


def device_context():
    """Enter/exit a TurboMind device Context (stream).

    Objects that use the Context stream — notably Linear/LlamaLinear and Weight/LinearWeight — must be destroyed before
    exiting this context.
    """
    return _tm().create_device_context()
