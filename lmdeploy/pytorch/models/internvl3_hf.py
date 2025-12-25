# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import os
import threading
import torch
import torch.nn.functional as F
from packaging import version
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.models.utils.micro_batch import enable_micro_batch, split_batch
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn import LayerNorm, RMSNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_o_proj, build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.tools.stage_timing import NPUTimer, bump_forward_and_maybe_report, record_stage, stage_timing_enabled
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import build_model_from_hf_config
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin, vlm_model


def _get_env_int(name: str, default: int) -> int:
    """Read int from env with fallback."""
    val = os.getenv(name, None)
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default


def _get_env_bool(name: str, default: bool = False) -> bool:
    """Read bool from env with fallback."""
    val = os.getenv(name, None)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _infer_npu_device_index(device: Optional[torch.device]) -> Optional[int]:
    """Infer NPU device index from torch.device; return None if not NPU."""
    if device is None:
        return None
    try:
        if device.type != 'npu':
            return None
    except Exception:
        return None
    # torch.device('npu') has index=None; torch_npu will map it to current_device.
    return device.index


def _format_mib(num_bytes: int) -> float:
    return float(num_bytes) / 1024.0 / 1024.0


def _format_gib(num_bytes: int) -> float:
    """Format bytes to GiB (binary, 1GiB = 1024^3 bytes)."""
    return float(num_bytes) / 1024.0 / 1024.0 / 1024.0


_GLOBAL_NPU_PEAK_LOCK = threading.Lock()
_GLOBAL_NPU_PEAK_STATS_BY_TAG: Dict[str, Dict[str, int]] = {}


_ASCEND_GRAPH_RUNNER_CLS = None


def _is_ascend_cudagraph_capturing() -> bool:
    """Return True if Ascend cudagraph is capturing; best-effort/no hard dependency."""
    global _ASCEND_GRAPH_RUNNER_CLS
    if _ASCEND_GRAPH_RUNNER_CLS is None:
        try:
            # Keep consistent with torch_npu_ops.py imports
            from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph import AscendGraphRunner  # type: ignore
            _ASCEND_GRAPH_RUNNER_CLS = AscendGraphRunner
        except Exception:
            _ASCEND_GRAPH_RUNNER_CLS = False  # sentinel: unavailable

    if _ASCEND_GRAPH_RUNNER_CLS is False:
        return False
    try:
        return bool(getattr(_ASCEND_GRAPH_RUNNER_CLS, 'capturing', False))
    except Exception:
        return False


def _update_and_maybe_print_global_npu_peak(*,
                                           tag: str,
                                           device_index: int,
                                           peak_alloc_bytes: int,
                                           peak_rsv_bytes: int,
                                           delta_peak_alloc_bytes: int,
                                           delta_peak_rsv_bytes: int,
                                           before_alloc_bytes: int,
                                           before_rsv_bytes: int,
                                           after_alloc_bytes: int,
                                           after_rsv_bytes: int) -> None:
    """Update process-global max peak stats (bucketed by tag) and print only when it increases."""
    with _GLOBAL_NPU_PEAK_LOCK:
        prev = _GLOBAL_NPU_PEAK_STATS_BY_TAG.get(tag, None)
        if prev is None:
            prev = {
                'peak_alloc': 0,
                'peak_rsv': 0,
                'delta_peak_alloc': 0,
                'delta_peak_rsv': 0,
            }
            _GLOBAL_NPU_PEAK_STATS_BY_TAG[tag] = prev

        if (peak_alloc_bytes <= prev['peak_alloc'] and peak_rsv_bytes <= prev['peak_rsv']
                and delta_peak_alloc_bytes <= prev['delta_peak_alloc'] and delta_peak_rsv_bytes <= prev['delta_peak_rsv']):
            return

        prev['peak_alloc'] = max(prev['peak_alloc'], peak_alloc_bytes)
        prev['peak_rsv'] = max(prev['peak_rsv'], peak_rsv_bytes)
        prev['delta_peak_alloc'] = max(prev['delta_peak_alloc'], delta_peak_alloc_bytes)
        prev['delta_peak_rsv'] = max(prev['delta_peak_rsv'], delta_peak_rsv_bytes)

    try:
        import torch.distributed as dist  # type: ignore
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        rank = 0

    print(
        f"[LMDEPLOY_NPU_PEAK] rank={rank} device={device_index} tag={tag} "
        f"before_alloc={_format_gib(before_alloc_bytes):.3f}GiB "
        f"before_rsv={_format_gib(before_rsv_bytes):.3f}GiB "
        f"after_alloc={_format_gib(after_alloc_bytes):.3f}GiB "
        f"after_rsv={_format_gib(after_rsv_bytes):.3f}GiB "
        f"peak_alloc={_format_gib(peak_alloc_bytes):.3f}GiB "
        f"peak_rsv={_format_gib(peak_rsv_bytes):.3f}GiB "
        f"delta_peak_alloc={_format_gib(delta_peak_alloc_bytes):.3f}GiB "
        f"delta_peak_rsv={_format_gib(delta_peak_rsv_bytes):.3f}GiB "
        f"global_max_alloc={_format_gib(prev['peak_alloc']):.3f}GiB "
        f"global_max_rsv={_format_gib(prev['peak_rsv']):.3f}GiB "
        f"global_max_delta_alloc={_format_gib(prev['delta_peak_alloc']):.3f}GiB "
        f"global_max_delta_rsv={_format_gib(prev['delta_peak_rsv']):.3f}GiB"
    )


def _maybe_measure_npu_memory(tag: str, fn, device: Optional[torch.device] = None):
    """Measure NPU memory delta (allocated/reserved/peak) for a code segment.

    Enable with env: LMDEPLOY_NPU_MEM_DELTA=1
    """
    if not _get_env_bool('LMDEPLOY_NPU_MEM_DELTA', default=False):
        return fn()

    try:
        import torch_npu  # type: ignore
    except Exception:
        return fn()

    # If caller explicitly set a non-NPU device, skip.
    dev_index = _infer_npu_device_index(device)
    if dev_index is None:
        dev_index = torch_npu.npu.current_device()

    # Synchronize + reset peak stats so 'peak_*' is local to this segment.
    torch_npu.npu.synchronize(dev_index)
    torch_npu.npu.reset_peak_memory_stats(dev_index)
    before_alloc = torch_npu.npu.memory_allocated(dev_index)
    before_rsv = torch_npu.npu.memory_reserved(dev_index)

    out = fn()

    torch_npu.npu.synchronize(dev_index)
    after_alloc = torch_npu.npu.memory_allocated(dev_index)
    after_rsv = torch_npu.npu.memory_reserved(dev_index)
    peak_alloc = torch_npu.npu.max_memory_allocated(dev_index)
    peak_rsv = torch_npu.npu.max_memory_reserved(dev_index)
    free_b, total_b = torch_npu.npu.mem_get_info(dev_index)

    try:
        import torch.distributed as dist  # type: ignore
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        rank = 0

    print(
        f"######### [LMDEPLOY_NPU_MEM] rank={rank} {tag} "
        f"+alloc={_format_gib(after_alloc - before_alloc):.3f}GiB "
        f"+rsv={_format_gib(after_rsv - before_rsv):.3f}GiB "
        f"peak_alloc={_format_gib(peak_alloc):.3f}GiB "
        f"peak_rsv={_format_gib(peak_rsv):.3f}GiB "
        f"free={_format_gib(free_b):.3f}GiB/"
        f"{_format_gib(total_b):.3f}GiB"
    )
    return out


@torch.compile(dynamic=True)
def pre_rms_norm(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pre rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance_q = (q * q).sum(-1, keepdim=True)
    variance_k = (k * k).sum(-1, keepdim=True)
    variance = torch.stack([variance_q, variance_k], dim=0)
    return variance


@torch.compile(dynamic=True)
def post_rms_norm(q: torch.Tensor, k: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor,
                  variance: torch.Tensor, eps: float, embed_dim: int, dtype: torch.dtype):
    """Post rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance = variance / embed_dim + eps
    variance_q, variance_k = variance
    q = q * torch.rsqrt(variance_q)
    q = q.to(dtype) * weight_q
    k = k * torch.rsqrt(variance_k)
    k = k.to(dtype) * weight_k
    return q, k


class InternVLVisionPatchEmbeddings(nn.Module):
    """This class turns `pixel_values` of shape `(batch_size, num_channels,
    height, width)` into the initial `hidden_states` (patch embeddings) of
    shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(num_channels,
                                    hidden_size,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    dtype=dtype,
                                    device=device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                'Make sure that the channel dimension of the pixel values match with the one set in the configuration.')

        embeddings = self.projection(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings


class InternVLVisionEmbeddings(nn.Module):
    """Intern vision embedding."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.empty(1, 1, self.embed_dim, dtype=dtype, device=device))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.empty(1, 1, self.embed_dim, dtype=dtype, device=device))
        else:
            self.mask_token = None
        self.patch_embeddings = InternVLVisionPatchEmbeddings(config, dtype=dtype, device=device)

        self.num_positions = self.patch_embeddings.num_patches + 1

        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(
                torch.empty(1, self.num_positions, self.embed_dim, dtype=dtype, device=device))
        else:
            self.position_embeddings = None

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int):
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        target_dtype = embeddings.dtype
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]
        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.float().reshape(1, sqrt_num_positions, sqrt_num_positions,
                                                          -1).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed,
                                        size=(new_height, new_width),
                                        mode='bicubic',
                                        align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim).to(target_dtype)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embeddings(pixel_values)  # shape = [*, channel, width, height]
        batch_size = patch_embeds.shape[0]
        cls_token = self.cls_token.expand(batch_size, 1, -1)
        embeddings = torch.cat([cls_token, patch_embeds], dim=1)
        if self.position_embeddings is not None:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
            embeddings = embeddings + position_embeddings
        return embeddings


NORM2FN = {
    'rms_norm': RMSNorm,
    'layer_norm': LayerNorm,
}


class InternVLVisionAttention(nn.Module):
    """Intern vl attention."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv_proj = build_qkv_proj(
            self.embed_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            bias=config.attention_bias,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        self.use_qk_norm = config.use_qk_norm

        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                self.embed_dim,
                eps=config.layer_norm_eps,
                dtype=dtype,
                device=device,
                tp=True,
                align=self.head_dim,
            )
            self.k_norm = RMSNorm(
                self.embed_dim,
                eps=config.layer_norm_eps,
                dtype=dtype,
                device=device,
                tp=True,
                align=self.head_dim,
            )

        self.scale = self.head_dim**-0.5

        # o_proj
        self.projection_layer = build_o_proj(self.embed_dim,
                                             self.embed_dim,
                                             bias=True,
                                             quant_config=quantization_config,
                                             dtype=dtype,
                                             device=device,
                                             is_tp=True,
                                             tp_align_size=self.head_dim)

    def pre_rms_norm(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Pre rms norm."""
        return pre_rms_norm(q, k)

    def post_rms_norm(self, q: torch.Tensor, k: torch.Tensor, variance: torch.Tensor, dtype: torch.dtype):
        """Post rms norm."""
        eps = self.config.layer_norm_eps
        return post_rms_norm(q, k, self.q_norm.weight, self.k_norm.weight, variance, eps, self.embed_dim, dtype)

    def qkv_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        import lmdeploy.pytorch.distributed as dist
        q_shape = q.shape
        k_shape = k.shape
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)

        tp, _ = get_tp_world_rank()
        if tp == 1:
            q = self.q_norm(q).view(q_shape)
            k = self.k_norm(k).view(k_shape)
            return q, k

        # variance
        variance = self.pre_rms_norm(q, k)
        dist.all_reduce(variance)
        q, k = self.post_rms_norm(q, k, variance, q.dtype)
        q = q.view(q_shape)
        k = k.view(k_shape)

        return q, k

    def forward(self, hidden_states):
        """forward."""

        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        q, k, v = self.qkv_proj.split_qkv(qkv_states)

        if self.use_qk_norm:
            q, k = self.qkv_norm(q, k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # o proj
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(-2, -1)
        attn_output = self.projection_layer(attn_output)
        return attn_output


class InternVLVisionMLP(nn.Module):
    """Intern vl mlp."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()

        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.act = ACT2FN[config.hidden_act]

        self.fc1 = build_colwise_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
            dp_disable_tp=True,
        )

        self.fc2 = build_rowwise_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True,
            dp_disable_tp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVLVisionLayer(nn.Module):
    """Intern vision layer."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = getattr(config, 'norm_type', 'rms_norm')

        self.attention = InternVLVisionAttention(config, dtype=dtype, device=device)
        self.mlp = InternVLVisionMLP(config, dtype=dtype, device=device)
        self.layernorm_before = NORM2FN[self.norm_type](self.embed_dim,
                                                        eps=config.layer_norm_eps,
                                                        dtype=dtype,
                                                        device=device)
        self.layernorm_after = NORM2FN[self.norm_type](self.embed_dim,
                                                       eps=config.layer_norm_eps,
                                                       dtype=dtype,
                                                       device=device)

        self.lambda_1 = nn.Parameter(torch.empty(self.embed_dim, dtype=dtype, device=device))
        self.lambda_2 = nn.Parameter(torch.empty(self.embed_dim, dtype=dtype, device=device))

    @enable_micro_batch(param_name='hidden_states', index=0)
    def _attn(self, hidden_states):
        hidden_states = hidden_states + self.attention(self.layernorm_before(hidden_states).to(
            hidden_states[0].dtype)) * self.lambda_1
        return hidden_states

    @enable_micro_batch(param_name='hidden_states', index=0)
    def _mlp(self, hidden_states):
        hidden_states = hidden_states + self.mlp(self.layernorm_after(hidden_states).to(
            hidden_states.dtype)) * self.lambda_2
        return hidden_states

    def forward(
        self,
        hidden_states,
    ):
        hidden_states = self._attn(hidden_states)
        hidden_states = self._mlp(hidden_states)
        return hidden_states


class InternVLVisionEncoder(nn.Module):
    """Intern vision encoder."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [InternVLVisionLayer(config, dtype=dtype, device=device) for idx in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
    ):
        """forward."""
        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layer):
            layer_outputs = encoder_layer(hidden_states, )
            hidden_states = layer_outputs
        return hidden_states


@vlm_model
class InternVLVisionModel(nn.Module):
    """Intern vision model."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config

        self.embeddings = InternVLVisionEmbeddings(config, dtype=dtype, device=device)
        self.encoder = InternVLVisionEncoder(config, dtype=dtype, device=device)
        self.layernorm = None
        if not config.use_mean_pooling:
            self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype, device=device)

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_all: bool = True,
    ):
        """forward."""
        assert pixel_values.dim() == 4
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = hidden_states
        if self.layernorm is not None:
            last_hidden_state = self.layernorm(hidden_states)

        if return_all:
            return hidden_states, last_hidden_state
        return last_hidden_state


class InternVLMultiModalProjector(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        input_dim = config.vision_config.hidden_size * int(1 / config.downsample_ratio)**2
        self.layer_norm = LayerNorm(input_dim, eps=1e-5, dtype=dtype, device=device)

        quantization_config = getattr(config.text_config, 'quantization_config', None)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_1 = build_colwise_linear(
            input_dim,
            config.text_config.hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
            dp_disable_tp=True,
        )

        self.linear_2 = build_rowwise_linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True,
            dp_disable_tp=True,
        )

    def forward(self, image_features):
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class InternVLForConditionalGeneration(nn.Module, DeployModelMixin, CudaGraphMixin):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr


        # from .npu_memory_profiler import NPUMemoryProfiler
        # from pathlib import Path
        # self.npu_memory_profiler: NPUMemoryProfiler = NPUMemoryProfiler(profile_dir=Path('/data/tangzhiyi/intern-s1/test_models/mem_profile'))


        self.vision_tower = _maybe_measure_npu_memory(
            'vision_tower',
            lambda: InternVLVisionModel(config.vision_config, dtype=dtype, device=device),
            device=device,
        )

        # self.npu_memory_profiler.step(tag='vision_tower')
        self.multi_modal_projector = _maybe_measure_npu_memory(
            'multi_modal_projector',
            lambda: InternVLMultiModalProjector(config, dtype=dtype, device=device),
            device=device,
        )
        # self.npu_memory_profiler.step(tag='multi_modal_projector')
        self.language_model = _maybe_measure_npu_memory(
            'language_model',
            lambda: build_model_from_hf_config(config.text_config, dtype=dtype, device=device),
            device=device,
        )
        # self.npu_memory_profiler.step(tag='language_model')
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

        self.input_processor = InternVLProcessor(self.config, dtype)

        self.compile_vit = False

    def compile_model(self):
        torch_version = version.parse(torch.__version__)
        if torch_version < version.parse('2.5.0'):
            return

        tp, _ = get_tp_world_rank()
        if torch_version >= version.parse('2.6.0') and tp > 1:
            torch._inductor.config.reorder_for_compute_comm_overlap = True
            if isinstance(self.vision_tower, InternVLVisionModel):
                # Split ViT encoder forward into micro-batches to reduce peak activation memory.
                # Env override: LMDEPLOY_VIT_SPLIT_BATCH (default: 2). Set <=1 to disable.
                num_splits = _get_env_int('LMDEPLOY_VIT_SPLIT_BATCH', 2)
                if num_splits is None or num_splits <= 1:
                    num_splits = 1
                self.vision_tower.encoder.forward = split_batch(self.vision_tower.encoder.forward,
                                                                'inputs_embeds',
                                                                index=0,
                                                                num_splits=num_splits)

        self.get_image_features = torch.compile(self.get_image_features, mode='max-autotune-no-cudagraphs')
        self.compile_vit = True
        self.has_compiled_vit = False

    def _mark_dynamic_once(self, pixel_values, dims):
        """Call torch._dynamo.mark_dynamic to avoid recompile."""
        if not self.compile_vit or self.has_compiled_vit or pixel_values is None:
            return

        torch._dynamo.mark_dynamic(pixel_values, dims)
        self.has_compiled_vit = True

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.language_model.get_logits(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language_model.get_input_embeddings()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        """Obtains image last hidden states from the vision tower and apply
        multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`int` or `List[int]`):
                Layer index or list of layer indices to extract features from.
        Returns:
            vision_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`.
        """
        downsample_ratio = self.config.downsample_ratio
        if vision_feature_layer == -1:
            # Only need the final features; avoid returning an extra large tensor tuple.
            vision_features = self.vision_tower(pixel_values=pixel_values, return_all=False)
        else:
            hidden_states, last_hidden_state = self.vision_tower(pixel_values=pixel_values, return_all=True)
            vision_features = hidden_states[vision_feature_layer]
            # drop reference early (peak memory outside this function is dominated by vision_features)
            del last_hidden_state, hidden_states
        if vision_feature_select_strategy == 'default':
            vision_features = vision_features[:, 1:, :]

        # Calculate dimensions based on vision features
        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]

        # Reshape tensor to spatial dimensions
        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)

        # Apply downsampling using pixel shuffle
        vision_features = self.pixel_shuffle(vision_features, scale_factor=downsample_ratio)

        # Reshape tensor to prepare for projection
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])

        # Project features through multi-modal projector
        vision_features = self.multi_modal_projector(vision_features)

        return vision_features

    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
        """Perform pixel shuffle downsampling on vision features.

        Args:
            vision_features (`torch.Tensor`):
                Input tensor of shape (batch_size, width, height, channels).
            scale_factor (`float`, *optional*, defaults to `0.5`):
                Factor by which to downsample. Default is 0.5, which halves the dimensions.

        Returns:
            vision_features (`torch.Tensor`):
                Downsampled tensor of shape (batch_size, height*scale_factor,
                                                width*scale_factor, channels/(scale_factor^2)).
        """
        batch_size, width, height, channels = vision_features.size()

        inv = int(round(1.0 / scale_factor))
        if abs(inv * scale_factor - 1.0) > 1e-6:
            raise ValueError(f'scale_factor={scale_factor} must be a reciprocal of an integer.')

        if (height % inv) != 0 or (width % inv) != 0:
            raise ValueError(f'Height/width must be divisible by {inv} for proper downsampling.')

        h_out = height // inv
        w_out = width // inv
        c1 = channels * inv
        c2 = channels * inv * inv

        # TODO(tangzhiyi)
        # x = vision_features.view(batch_size, width, h_out, c1)   # (B, W, H/iv, C*iv)
        # x = x.transpose(1, 2)                                   # (B, H/iv, W, C*iv)
        # x = x.reshape(batch_size, h_out, w_out, c2)              # (B, H/iv, W/iv, C*iv^2)
        # x = x.transpose(1, 2)                                   # (B, W/iv, H/iv, C*iv^2)
        # 使用permute代替多次transpose+reshape（更高效）
        x = vision_features.reshape(batch_size, width, h_out, inv, channels)
        x = x.permute(0, 2, 3, 1, 4)  # (B, H_out, inv, W, C)
        x = x.reshape(batch_size, h_out, w_out, c2)


        return x.contiguous()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        pixel_values: Any = None,
        image_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        # Track peak memory of the multimodal fusion block (global max across forwards).
        # Enable with env: LMDEPLOY_NPU_FWD_PEAK=1
        _track_fwd_peak = _get_env_bool('LMDEPLOY_NPU_FWD_PEAK', default=False)
        _peak_dev_index = None
        _baseline_alloc = None
        _baseline_rsv = None
        if _track_fwd_peak:
            try:
                import torch_npu  # type: ignore
            except Exception:
                _track_fwd_peak = False

        if inputs_embeds is None and pixel_values is not None:
            # extract feature
            lang_embeds = self.get_input_embeddings()(input_ids)
            if _track_fwd_peak:
                try:
                    import torch_npu  # type: ignore
                    _peak_dev_index = _infer_npu_device_index(lang_embeds.device)
                    if _peak_dev_index is None:
                        _peak_dev_index = torch_npu.npu.current_device()
                    torch_npu.npu.synchronize(_peak_dev_index)
                    _baseline_alloc = int(torch_npu.npu.memory_allocated(_peak_dev_index))
                    _baseline_rsv = int(torch_npu.npu.memory_reserved(_peak_dev_index))
                    torch_npu.npu.reset_peak_memory_stats(_peak_dev_index)
                except Exception:
                    _track_fwd_peak = False
                    _peak_dev_index = None
                    _baseline_alloc = None
                    _baseline_rsv = None
            # Replace masked_scatter_ with index_copy_ to reduce peak memory and avoid
            # the masked_scatter slow-path (mask expansion / temporaries) on some backends.
            if image_mask is None:
                raise ValueError('image_mask must be provided when pixel_values is not None.')

            hidden = lang_embeds.size(-1)

            # Ensure we write into the original storage (view requires contiguous).
            # TODO(tangzhiyi)
            # if not lang_embeds.is_contiguous():
            #     lang_embeds = lang_embeds.contiguous()
            # lang_flat = lang_embeds.view(-1, hidden)  # [B*S, H]
            lang_flat = lang_embeds.reshape(-1, hidden)

            # Row-major order indices of image tokens (matches masked_scatter_ fill order).
            idx = image_mask.reshape(-1).nonzero(as_tuple=False).flatten()  # [M]

            # Pipeline fusion to reduce peak memory:
            # compute vision embeddings in chunks and write-back immediately, instead of
            # materializing full vit_embeds for all images.
            # Env override: LMDEPLOY_MM_CHUNK_SIZE (default: 4). Set <=0 to disable.
            mm_chunk_size = kwargs.get('mm_chunk_size', None)
            if mm_chunk_size is None:
                mm_chunk_size = _get_env_int('LMDEPLOY_MM_CHUNK_SIZE', 4)
            try:
                mm_chunk_size = int(mm_chunk_size)
            except Exception:
                mm_chunk_size = _get_env_int('LMDEPLOY_MM_CHUNK_SIZE', 4)
            # pixel_values can be a Tensor [N,3,H,W] or a list of per-image tensors [1,3,H,W]
            if isinstance(pixel_values, list):
                num_images = len(pixel_values)
            else:
                num_images = int(pixel_values.shape[0])
            if idx.numel() == 0 or num_images == 0:
                # nothing to fuse
                pass
            elif mm_chunk_size <= 0 or mm_chunk_size >= num_images:
                # original (non-pipelined) behavior: compute all at once
                _do_stage_timing = stage_timing_enabled() and (not _is_ascend_cudagraph_capturing())
                with NPUTimer(device=lang_embeds.device, sync=True, enabled=_do_stage_timing) as _tm:
                    if isinstance(pixel_values, list):
                        pv_all = torch.cat(pixel_values, dim=0)
                    else:
                        pv_all = pixel_values
                    self._mark_dynamic_once(pv_all, [0])
                    vit_embeds = self.get_image_features(
                        pv_all,
                        self.vision_feature_layer,
                        self.vision_feature_select_strategy,
                    )
                    vit_flat = vit_embeds.reshape(-1, hidden)  # [M, H]
                    if vit_flat.shape[0] != idx.numel():
                        raise ValueError(f'Image token count mismatch: mask has {idx.numel()} positions, '
                                         f'but vit_embeds has {vit_flat.shape[0]} tokens.')
                    # TODO(tangzhiyi)
                    # if vit_flat.dtype != lang_flat.dtype or vit_flat.device != lang_flat.device:
                    #     vit_flat = vit_flat.to(dtype=lang_flat.dtype, device=lang_flat.device)
                    lang_flat.index_copy_(0, idx, vit_flat)
                    del vit_embeds, vit_flat
                    if isinstance(pixel_values, list):
                        del pv_all
                if _do_stage_timing:
                    record_stage('vit', _tm.elapsed_s)
            else:
                # pipelined: process images in chunks, and consume idx in row-major order
                _do_stage_timing = stage_timing_enabled() and (not _is_ascend_cudagraph_capturing())
                with NPUTimer(device=lang_embeds.device, sync=True, enabled=_do_stage_timing) as _tm:
                    offset = 0
                    for start in range(0, num_images, mm_chunk_size):
                        end = min(num_images, start + mm_chunk_size)
                        if isinstance(pixel_values, list):
                            pv = torch.cat(pixel_values[start:end], dim=0)
                        else:
                            pv = pixel_values[start:end]
                        if start == 0:
                            self._mark_dynamic_once(pv, [0])
                        vit_chunk = self.get_image_features(
                            pv,
                            self.vision_feature_layer,
                            self.vision_feature_select_strategy,
                        )
                        vit_flat = vit_chunk.reshape(-1, hidden)
                        n = vit_flat.shape[0]
                        idx_chunk = idx[offset:offset + n]
                        if idx_chunk.numel() != n:
                            raise ValueError(f'Image token count mismatch: need {n} mask positions for chunk '
                                             f'[{start}:{end}], but got {idx_chunk.numel()} remaining.')
                        if vit_flat.dtype != lang_flat.dtype or vit_flat.device != lang_flat.device:
                            vit_flat = vit_flat.to(dtype=lang_flat.dtype, device=lang_flat.device)
                        lang_flat.index_copy_(0, idx_chunk, vit_flat)
                        offset += n
                        del pv, vit_chunk, vit_flat, idx_chunk
                    if offset != idx.numel():
                        raise ValueError(f'Image token count mismatch: mask has {idx.numel()} positions, '
                                         f'but consumed {offset} vision tokens.')
                if _do_stage_timing:
                    record_stage('vit', _tm.elapsed_s)

            # release vision-side temporaries ASAP to reduce peak
            pixel_values = None
            del idx, lang_flat

            inputs_embeds = lang_embeds
            input_ids = None

            if _track_fwd_peak and _peak_dev_index is not None and _baseline_alloc is not None and _baseline_rsv is not None:
                try:
                    import torch_npu  # type: ignore
                    torch_npu.npu.synchronize(_peak_dev_index)
                    after_alloc = torch_npu.npu.memory_allocated(_peak_dev_index)
                    after_rsv = torch_npu.npu.memory_reserved(_peak_dev_index)
                    peak_alloc = torch_npu.npu.max_memory_allocated(_peak_dev_index)
                    peak_rsv = torch_npu.npu.max_memory_reserved(_peak_dev_index)
                    delta_peak_alloc = max(0, int(peak_alloc) - int(_baseline_alloc))
                    delta_peak_rsv = max(0, int(peak_rsv) - int(_baseline_rsv))
                    _update_and_maybe_print_global_npu_peak(
                        tag='mm_fuse_711_812',
                        device_index=_peak_dev_index,
                        peak_alloc_bytes=int(peak_alloc),
                        peak_rsv_bytes=int(peak_rsv),
                        delta_peak_alloc_bytes=int(delta_peak_alloc),
                        delta_peak_rsv_bytes=int(delta_peak_rsv),
                        before_alloc_bytes=int(_baseline_alloc),
                        before_rsv_bytes=int(_baseline_rsv),
                        after_alloc_bytes=int(after_alloc),
                        after_rsv_bytes=int(after_rsv),
                    )
                except Exception:
                    pass

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Track peak memory of LM forward (prefill vs decode).
        # Enable with env: LMDEPLOY_NPU_FWD_PEAK=1
        _lm_track = _get_env_bool('LMDEPLOY_NPU_FWD_PEAK', default=False)
        _lm_dev_index = None
        _lm_before_alloc = None
        _lm_before_rsv = None
        _lm_tag = None
        if _lm_track:
            try:
                import torch_npu  # type: ignore
                _lm_dev_index = _infer_npu_device_index(inputs_embeds.device)
                if _lm_dev_index is None:
                    _lm_dev_index = torch_npu.npu.current_device()
                torch_npu.npu.synchronize(_lm_dev_index)
                _lm_before_alloc = int(torch_npu.npu.memory_allocated(_lm_dev_index))
                _lm_before_rsv = int(torch_npu.npu.memory_reserved(_lm_dev_index))
                torch_npu.npu.reset_peak_memory_stats(_lm_dev_index)
                _is_decoding = bool(getattr(attn_metadata, 'is_decoding', False))
                _lm_tag = 'lm_forward_decode' if _is_decoding else 'lm_forward_prefill'
            except Exception:
                _lm_track = False
                _lm_dev_index = None

        _do_stage_timing = stage_timing_enabled() and (not _is_ascend_cudagraph_capturing())
        with NPUTimer(device=inputs_embeds.device if inputs_embeds is not None else None,
                      sync=True,
                      enabled=_do_stage_timing) as _lm_timer:
            outputs = self.language_model.forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )
        # Stage timing stats (prefill vs decode)
        if _do_stage_timing:
            _is_decoding2 = bool(getattr(attn_metadata, 'is_decoding', False))
            record_stage('lm_decode' if _is_decoding2 else 'lm_prefill', _lm_timer.elapsed_s)

        if _lm_track and _lm_dev_index is not None and _lm_before_alloc is not None and _lm_before_rsv is not None and _lm_tag is not None:
            try:
                import torch_npu  # type: ignore
                torch_npu.npu.synchronize(_lm_dev_index)
                _lm_after_alloc = int(torch_npu.npu.memory_allocated(_lm_dev_index))
                _lm_after_rsv = int(torch_npu.npu.memory_reserved(_lm_dev_index))
                _lm_peak_alloc = int(torch_npu.npu.max_memory_allocated(_lm_dev_index))
                _lm_peak_rsv = int(torch_npu.npu.max_memory_reserved(_lm_dev_index))
                _lm_delta_peak_alloc = max(0, _lm_peak_alloc - int(_lm_before_alloc))
                _lm_delta_peak_rsv = max(0, _lm_peak_rsv - int(_lm_before_rsv))
                _update_and_maybe_print_global_npu_peak(
                    tag=_lm_tag,
                    device_index=_lm_dev_index,
                    peak_alloc_bytes=_lm_peak_alloc,
                    peak_rsv_bytes=_lm_peak_rsv,
                    delta_peak_alloc_bytes=_lm_delta_peak_alloc,
                    delta_peak_rsv_bytes=_lm_delta_peak_rsv,
                    before_alloc_bytes=int(_lm_before_alloc),
                    before_rsv_bytes=int(_lm_before_rsv),
                    after_alloc_bytes=_lm_after_alloc,
                    after_rsv_bytes=_lm_after_rsv,
                )
            except Exception:
                pass

        # One counter increment per forward (not per-stage).
        bump_forward_and_maybe_report(enabled=_do_stage_timing)
        return outputs

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = None

        # vision inputs
        pixel_values = None
        image_mask = None
        if context.input_multimodals is not None:
            pixel_values = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            # flatten batch
            pixel_values = [data for im_data in pixel_values for data in im_data]
            if len(pixel_values) > 0:
                image_token_id = pixel_values[0].meta['image_token_id']
                image_mask = input_ids == image_token_id
                # Keep as a list to avoid a large torch.cat peak; forward() will cat per chunk.
                pixel_values = [data.data for data in pixel_values]
            else:
                pixel_values = None
                image_mask = None

        # get inputs from context
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            vision_embedding_indexing = context.input_embedding_indexing
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            pixel_values=pixel_values,
            image_mask=image_mask,
            inputs_embeds=inputs_embeds,
        )

    def load_lora_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], adapter_id: int):
        """Load lora weights."""

        if hasattr(self.model.language_model, 'load_lora_weights'):
            return self.model.language_model.load_lora_weights(weights, adapter_id)
        else:
            from lmdeploy.pytorch.adapter.adapter import load_lora_weights

            return load_lora_weights(weights, adapter_id)

    def rename_weight(self, name: str) -> str:
        """Rename weight."""
        if name == 'lm_head.weight':
            return 'language_model.lm_head.weight'
        elif name.startswith('model.language_model.'):
            return 'language_model.model.' + name[len('model.language_model.'):]
        elif name.startswith('model.'):
            return name[len('model.'):]
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""

        lang_prefix = 'language_model.'
        lang_prefix_length = len(lang_prefix)
        new_weights = dict()
        params_dict = dict(self.named_parameters())
        vision_stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
        ]
        for name, loaded_weight in weights:

            if name.startswith(lang_prefix):
                new_key = name[lang_prefix_length:]
                new_weights[new_key] = loaded_weight
                continue

            for (param_name, weight_name, shard_id) in vision_stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

        self.language_model.load_weights(new_weights.items())

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class InternVLProcessor(BaseModelInputProcessor):
    """Internvl input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype

    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """Prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.dtype)
            offset = input_mm['offset']
            image_token_id = input_mm['image_token_id']
            num_pad = input_mm['image_tokens']
            if isinstance(num_pad, torch.Tensor):
                num_pad = num_pad.item()

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + num_pad,
                                       meta=dict(image_token_id=image_token_id))
            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result
