# Copyright (c) OpenMMLab. All rights reserved.
"""Optional W4A16 backend backed by LMDeploy's bundled TurboMind.

The backend owns an architecture-specific weight layout while preserving LMDeploy's canonical AWQ parameters.
"""

import functools
import importlib
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

import lmdeploy
import lmdeploy.pytorch.distributed as dist

from ..awq_modules import LinearW4A16Impl
from .awq_modules import _is_turbomind_gemm_capability_supported


class _LinearRuntime:
    """Share one existing LlamaLinear workspace per CUDA stream."""

    def __init__(self, tm: ModuleType, device: torch.device,
                 stream: torch.cuda.Stream):
        self.device = device
        self.stream = stream
        self.context = tm.create_device_context(stream.cuda_stream)
        with torch.cuda.device(device), self.context:
            self.linear = tm.LlamaLinear()

    def forward(self, tm: ModuleType, x, weight, out):
        with torch.cuda.device(self.device), self.context:
            self.linear.forward_dense(
                tm.from_dlpack(x, stream=-1),
                weight,
                tm.from_dlpack(out, stream=-1),
            )

    def __del__(self):
        linear, self.linear = getattr(self, 'linear', None), None
        if linear is None:
            return
        try:
            with torch.cuda.device(self.device), self.context:
                del linear
        except Exception:
            pass


_runtime_pool = weakref.WeakValueDictionary()


def _get_runtime(tm: ModuleType, device: torch.device,
                 stream: torch.cuda.Stream):
    key = (device.index, stream.cuda_stream)
    runtime = _runtime_pool.get(key)
    if runtime is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Run one eager TurboMind W4A16 forward on this '
                               'CUDA stream before graph capture.')
        runtime = _LinearRuntime(tm, device, stream)
        _runtime_pool[key] = runtime
    return runtime


@dataclass
class _PreparedLinear:
    weight: object | None
    context: object
    stream: torch.cuda.Stream
    runtimes: dict[int, _LinearRuntime]
    device: torch.device

    def close(self):
        """Destroy Context-dependent objects under their original Context."""
        if self.weight is None:
            return
        weight, self.weight = self.weight, None
        with torch.cuda.device(self.device):
            for runtime in self.runtimes.values():
                if runtime.stream.cuda_stream != self.stream.cuda_stream:
                    self.stream.wait_stream(runtime.stream)
            self.runtimes.clear()
            with torch.cuda.stream(self.stream), self.context:
                del weight


@functools.lru_cache(maxsize=1)
def _load_turbomind() -> ModuleType:
    """Load the _turbomind extension built and shipped by LMDeploy."""
    lib_dir = (Path(lmdeploy.__file__).resolve().parent / 'lib').resolve()
    lib_dir_str = str(lib_dir)
    if lib_dir_str not in sys.path:
        sys.path.insert(0, lib_dir_str)

    try:
        tm = importlib.import_module('_turbomind')
    except (ImportError, OSError) as error:
        raise RuntimeError(
            'The TurboMind W4A16 backend requires LMDeploy\'s bundled '
            '_turbomind extension. Rebuild/install LMDeploy with TurboMind '
            f'enabled. Import error: {error}') from error

    module_file = getattr(tm, '__file__', None)
    if module_file is None or Path(module_file).resolve().parent != lib_dir:
        raise RuntimeError(
            'Loaded _turbomind is not LMDeploy\'s bundled extension: '
            f'{module_file!r}. Expected it under {lib_dir}. Remove the '
            'external turbomind package from this process and retry.')

    return tm


def _prepare_turbomind_linear(tm: ModuleType, in_features: int,
                              out_features: int, group_size: int,
                              qweight: torch.Tensor, scales: torch.Tensor,
                              qzeros: torch.Tensor):
    """Feed canonical AWQ weights to TurboMind's existing prepare path."""
    # Keep this import lazy so provider discovery does not require loading the
    # bundled TurboMind extension.
    from lmdeploy.turbomind.weight_format import AWQFormat

    weight_format = AWQFormat(block_in=group_size)
    source_params = {
        'weight': qweight,
        'scales': scales,
        'zeros': qzeros,
    }
    packed_params = {
        kind: weight_format.pack(
            weight_format.normalize(tensor.detach(), kind), kind)
        for kind, tensor in source_params.items()
    }
    buffers = {
        kind: packed.tensor.contiguous()
        for kind, packed in packed_params.items()
    }

    stream = torch.cuda.current_stream(qweight.device)
    stream_ptr = stream.cuda_stream
    context = tm.create_device_context(stream_ptr)
    with context:
        config = tm.LinearConfig()
        config.input_dim = in_features
        config.output_dim = out_features
        config.data_type = tm.DataType.TYPE_FP16
        config.format = weight_format.make_data_format(config.data_type)
        config.has_bias = False

        weight = tm.LinearWeight(config)
        for kind, packed in packed_params.items():
            src = tm.from_dlpack(buffers[kind], stream=-1)
            if packed.alloc_shape is not None:
                src = src.reinterpret(packed.alloc_dtype, packed.alloc_shape)
            weight.param(kind).set(src)
        weight.prepare()

    # Publish the private layout only after all normalization and prepare work
    # on the loading stream has completed. Forward streams therefore need no
    # weight-readiness event or cross-stream wait.
    stream.synchronize()
    return _PreparedLinear(weight=weight,
                           context=context,
                           stream=stream,
                           runtimes={},
                           device=qweight.device)


class TurbomindAwqLinearW4A16Impl(LinearW4A16Impl):
    """AWQ linear using LMDeploy's bundled TurboMind W4A16 operator."""

    def __init__(self, in_features: int, out_features: int, w_bit: int,
                 group_size: int):
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
        self._prepared: _PreparedLinear | None = None

    def _release_prepared(self):
        prepared, self._prepared = self._prepared, None
        if prepared is not None:
            prepared.close()

    def __del__(self):
        try:
            self._release_prepared()
        except Exception:
            # CUDA/Python may already be shutting down.
            pass

    def _validate_weights(self, qweight: torch.Tensor, scales: torch.Tensor,
                          qzeros: torch.Tensor, bias: torch.Tensor | None):
        if (self.w_bit != 4 or self.group_size != 128
                or self.in_features % 128 != 0 or self.out_features % 32 != 0):
            raise ValueError('The TurboMind prototype requires W4A16, group '
                             'size 128, K divisible by 128, and N divisible '
                             'by 32.')
        if qweight.device.type != 'cuda':
            raise ValueError('TurboMind weights must be CUDA tensors.')
        capability = torch.cuda.get_device_capability(qweight.device)
        if not _is_turbomind_gemm_capability_supported(capability):
            raise RuntimeError(f'Unsupported CUDA capability: {capability}.')

        expected = (
            ('qweight', qweight, (self.in_features, self.out_features // 8),
             torch.int32),
            ('scales', scales, (self.in_features // 128, self.out_features),
             torch.float16),
            ('qzeros', qzeros, (self.in_features // 128,
                                self.out_features // 8), torch.int32),
        )
        for name, tensor, shape, dtype in expected:
            if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                    or tensor.device != qweight.device):
                raise ValueError(f'Invalid TurboMind {name}: expected shape '
                                 f'{shape}, dtype {dtype}, and device '
                                 f'{qweight.device}.')
        if bias is not None and (tuple(bias.shape) != (self.out_features, )
                                 or bias.dtype != torch.float16
                                 or bias.device != qweight.device):
            raise ValueError('Invalid TurboMind bias tensor.')

    def update_weights(self,
                       qweight: torch.Tensor,
                       scales: torch.Tensor,
                       qzeros: torch.Tensor,
                       bias: torch.Tensor | None = None):
        """Build a private TurboMind layout while preserving AWQ parameters."""
        self._release_prepared()
        self._validate_weights(qweight, scales, qzeros, bias)

        tm = _load_turbomind()
        device = qweight.device
        with torch.cuda.device(device):
            self._prepared = _prepare_turbomind_linear(
                tm,
                self.in_features,
                self.out_features,
                self.group_size,
                qweight,
                scales,
                qzeros,
            )

        # The model/state_dict must remain in canonical checkpoint layout.
        return qweight, scales, qzeros, bias

    def forward(self,
                x,
                qweight: torch.Tensor,
                scales: torch.Tensor,
                qzeros: torch.Tensor,
                bias: torch.Tensor | None = None,
                all_reduce: bool = False,
                group: torch.distributed.ProcessGroup | None = None):
        """Run the prepared operator and preserve the CUDA AWQ API contract."""
        prepared = self._prepared
        if prepared is None:
            raise RuntimeError(
                'TurboMind W4A16 weights are not prepared; call update_weights() after loading weights.'
            )
        if x.size(-1) != self.in_features:
            raise ValueError(
                f'Expected input dim {self.in_features}, but got {x.size(-1)}.'
            )
        if x.device != prepared.device:
            raise ValueError(
                f'Input must be on {prepared.device}, but got {x.device}.')

        input_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.out_features, )

        with torch.cuda.device(prepared.device):
            op_input = x if input_dtype == torch.float16 else x.to(
                dtype=torch.float16)
            op_input = op_input.reshape(-1, self.in_features)
            if (not op_input.is_contiguous()
                    or op_input.storage_offset() != 0):
                op_input = op_input.clone(
                    memory_format=torch.contiguous_format)
            # TurboMind's native kernel and the canonical AWQ bias path both
            # operate in FP16.  Cast back only after applying bias so BF16 or
            # FP32 inputs do not promote a mixed BF16/FP16 add to FP32.
            if op_input.size(0) == 0:
                out = torch.empty((0, self.out_features),
                                  dtype=torch.float16,
                                  device=prepared.device)
            else:
                out = torch.empty((op_input.size(0), self.out_features),
                                  dtype=torch.float16,
                                  device=prepared.device)
                tm = _load_turbomind()
                stream = torch.cuda.current_stream(prepared.device)
                stream_ptr = stream.cuda_stream
                if op_input.is_cuda:
                    op_input.record_stream(stream)
                    out.record_stream(stream)
                runtime = prepared.runtimes.get(stream_ptr)
                if runtime is None:
                    runtime = _get_runtime(tm, prepared.device, stream)
                    prepared.runtimes[stream_ptr] = runtime
                runtime.forward(tm, op_input, prepared.weight, out)
            if bias is not None:
                out = out + bias
            out = out.reshape(out_shape)

            # Match the existing CUDA AWQ backend's 2D compatibility contract.
            if out.ndim == 2:
                out = out.unsqueeze(0)
            if input_dtype != torch.float16:
                out = out.to(dtype=input_dtype)
            if all_reduce:
                dist.all_reduce(out, group=group)
        return out
