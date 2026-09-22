# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.vl.constants import Modality

from .kimi_k2_language import KimiK2ForCausalLM
from .kimi_k25_vision import KimiK25MultiModalProjector, KimiK25VisionTower
from .patch import get_build_model_context
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin


class KimiK25InputProcessor(BaseModelInputProcessor):
    """Convert Kimi processor outputs into scheduler multimodal spans."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None):
        token_id = getattr(config, 'media_placeholder_token_id', None)
        self.media_placeholder_token_id = None if token_id is None else int(token_id)
        self.dtype = dtype

    @staticmethod
    def _normalize_offset(offset: Any, image_tokens: int) -> tuple[int, int]:
        if isinstance(offset, torch.Tensor):
            offset = offset.tolist()
        if isinstance(offset, (tuple, list)):
            if len(offset) != 2:
                raise ValueError(f'Kimi image offset must be a (start, end) pair, got {offset!r}.')
            start, end = int(offset[0]), int(offset[1])
        else:
            start = int(offset)
            end = start + image_tokens
        if start < 0 or end < start:
            raise ValueError(f'Invalid Kimi image token span: ({start}, {end}).')
        return start, end

    def preprocess_input(
        self,
        input_ids: list[int],
        input_multimodals: list[dict[str, Any]] = None,
        **kwargs,
    ) -> PreprocessInputResult:
        """Validate the static-image contract and preserve one span per
        image."""
        del kwargs
        if input_multimodals is None or len(input_multimodals) == 0:
            return PreprocessInputResult(input_ids=input_ids, input_multimodals=input_multimodals)

        image_inputs = []
        input_ids_tensor = torch.as_tensor(input_ids, dtype=torch.long)
        previous_end = 0
        for image_index, input_mm in enumerate(input_multimodals):
            modality = input_mm.get('modality', Modality.IMAGE)
            if modality != Modality.IMAGE:
                raise NotImplementedError(
                    f'Kimi-K2.6 M5 supports static images only, got modality {modality!r} at index {image_index}.')

            try:
                pixel_values = input_mm['pixel_values']
                grid_thws = input_mm['grid_thws']
                image_token_id = int(input_mm['image_token_id'])
                image_tokens = input_mm['image_tokens']
                offset = input_mm['offset']
            except KeyError as error:
                raise ValueError(
                    f'Missing Kimi image processor field {error.args[0]!r} '
                    f'at index {image_index}.') from error
            configured_token_id = self.media_placeholder_token_id
            if configured_token_id is None:
                raise ValueError('Kimi config must define `media_placeholder_token_id` for image inference.')
            if image_token_id != int(configured_token_id):
                raise ValueError(
                    f'Kimi image token id {image_token_id} does not match config '
                    f'media_placeholder_token_id={configured_token_id}.')

            if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
                raise ValueError(
                    f'Kimi pixel_values must have shape [patches, 3, patch, patch], got '
                    f'{getattr(pixel_values, "shape", None)}.')
            if pixel_values.shape[1] != 3:
                raise ValueError(f'Kimi pixel_values must have three channels, got shape {tuple(pixel_values.shape)}.')

            grid_thws = torch.as_tensor(grid_thws, dtype=torch.long)
            if grid_thws.numel() != 3:
                raise ValueError(f'Each Kimi image requires one [t, h, w] grid, got shape {tuple(grid_thws.shape)}.')
            grid_thws = grid_thws.reshape(1, 3)
            t, h, w = (int(value) for value in grid_thws[0].tolist())
            if t != 1:
                raise NotImplementedError(f'Kimi-K2.6 M5 supports static images with t=1, got grid {(t, h, w)}.')
            if h <= 0 or w <= 0 or h % 2 or w % 2:
                raise ValueError(f'Kimi image grid height and width must be positive and even, got {(t, h, w)}.')
            expected_patches = t * h * w
            if pixel_values.shape[0] != expected_patches:
                raise ValueError(
                    f'Kimi pixel patch count mismatch: grid {(t, h, w)} requires {expected_patches}, '
                    f'got {pixel_values.shape[0]}.')

            if isinstance(image_tokens, torch.Tensor):
                image_tokens = image_tokens.item()
            image_tokens = int(image_tokens)
            expected_tokens = h * w // 4
            if image_tokens != expected_tokens:
                raise ValueError(
                    f'Kimi image token count mismatch: grid {(t, h, w)} requires {expected_tokens}, '
                    f'got {image_tokens}.')
            start, end = self._normalize_offset(offset, image_tokens)
            if start < previous_end:
                raise ValueError(
                    f'Kimi image spans must be non-overlapping and in prompt order; '
                    f'image {image_index} starts at {start} before {previous_end}.')
            if end - start != image_tokens:
                raise ValueError(
                    f'Kimi image span length {end - start} does not match image_tokens={image_tokens}.')
            if end > input_ids_tensor.numel():
                raise ValueError(
                    f'Kimi image span ({start}, {end}) exceeds the {input_ids_tensor.numel()} input tokens.')
            if not torch.all(input_ids_tensor[start:end] == image_token_id):
                raise ValueError(
                    f'Kimi image span ({start}, {end}) must contain only image token id {image_token_id}.')
            previous_end = end

            if self.dtype is not None and pixel_values.is_floating_point():
                pixel_values = pixel_values.to(dtype=self.dtype)
            image_inputs.append(
                MultiModalData(
                    data=pixel_values,
                    start=start,
                    end=end,
                    meta={
                        'grid_thws': grid_thws,
                        'image_token_id': image_token_id,
                    },
                ))

        return PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals={'image': image_inputs},
        )


class KimiK25ForConditionalGeneration(nn.Module, DeployModelMixin, CudaGraphMixin):
    """Kimi-K2.5/K2.6 vision-language wrapper backed by DeepSeek-V3."""

    packed_modules_mapping = {
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    _LANGUAGE_MODEL_PREFIX = 'language_model.'
    _SKIPPED_WEIGHT_PREFIXES = ('vision_tower.', 'mm_projector.')
    _UNSUPPORTED_MULTIMODAL_FORWARD_KEYS = frozenset({
        'pixel_values_videos',
        'image_grid_thw',
        'video_grid_thw',
        'image_features',
        'vision_inputs',
    })

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.packed_modules_mapping = {
            name: list(source_names)
            for name, source_names in type(self).packed_modules_mapping.items()
        }
        if not hasattr(config, 'text_config'):
            raise ValueError('KimiK25 config must define `text_config`.')

        quant_config = getattr(config.text_config, 'quantization_config', None)
        outer_quant_config = getattr(config, 'quantization_config', None)
        if quant_config is None:
            quant_config = outer_quant_config
        if isinstance(quant_config, Mapping):
            quant_method = quant_config.get('quant_method')
        else:
            quant_method = getattr(quant_config, 'quant_method', None)
        if quant_method == 'compressed-tensors':
            if getattr(config.text_config, 'quantization_config', None) is None:
                raise RuntimeError(
                    'Kimi-K2.6 compressed-tensors metadata must be defined on `text_config`; '
                    'outer-only metadata cannot drive routed-expert dispatch safely.')
            build_quant_config = get_build_model_context().quant_config
            if build_quant_config is None or build_quant_config.quant_method != 'compressed-tensors':
                raise RuntimeError(
                    'Kimi-K2.6 compressed-tensors construction requires the validated ModelConfig quantization '
                    'metadata in BuildModelContext.')

        self.config = config
        self.ctx_mgr = ctx_mgr
        build_context = get_build_model_context()
        self.language_model_only = build_context.language_model_only
        vision_config = getattr(config, 'vision_config', None)
        if not self.language_model_only and vision_config is None:
            raise ValueError('KimiK25 multimodal construction requires `vision_config`.')
        if self.language_model_only:
            self.vision_tower = nn.Identity()
            self.vision_tower._is_dummy_mod = True
            self.mm_projector = nn.Identity()
            self.mm_projector._is_dummy_mod = True
        else:
            self.vision_tower = KimiK25VisionTower(vision_config, dtype=dtype, device=device)
            self.mm_projector = KimiK25MultiModalProjector(vision_config, dtype=dtype, device=device)
        self.language_model = KimiK2ForCausalLM(
            config.text_config,
            ctx_mgr,
            dtype=dtype,
            device=device,
            prefix='language_model',
        )
        language_packed_mapping = getattr(self.language_model, 'packed_modules_mapping', {})
        fused_a_mapping = language_packed_mapping.get('fused_qkv_a_proj_with_mqa')
        if fused_a_mapping is not None:
            self.packed_modules_mapping['fused_qkv_a_proj_with_mqa'] = list(fused_a_mapping)
        self.input_processor = KimiK25InputProcessor(config, dtype=dtype)

    @staticmethod
    def _raise_for_multimodal_context(context: StepContext):
        if (context.input_multimodals is not None or context.input_embeddings is not None
                or context.vision_inputs is not None):
            raise NotImplementedError('Kimi-K2.6 multimodal inference is not implemented in the text-only milestone.')

    @classmethod
    def _raise_for_unsupported_multimodal_kwargs(cls, kwargs: dict[str, Any]):
        used_keys = sorted(key for key in cls._UNSUPPORTED_MULTIMODAL_FORWARD_KEYS if kwargs.get(key) is not None)
        if used_keys:
            joined_keys = ', '.join(used_keys)
            raise NotImplementedError(
                f'Kimi-K2.6 M5 does not support multimodal inputs ({joined_keys}).')

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run replicated MoonViT and project its merged patches to text
        width."""
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(dtype=target_dtype)
        image_features = self.vision_tower(pixel_values, grid_thws)
        return self.mm_projector(image_features)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        pixel_values: torch.Tensor = None,
        grid_thws: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """Inject projected image rows at media-pad positions during
        prefill."""
        self._raise_for_unsupported_multimodal_kwargs(kwargs)
        if self.language_model_only and any(value is not None for value in (pixel_values, grid_thws, image_mask)):
            raise NotImplementedError('Kimi-K2.6 multimodal inference is disabled by `language_model_only=True`.')

        image_arguments = (pixel_values, grid_thws, image_mask)
        if any(value is not None for value in image_arguments) and not all(
                value is not None for value in image_arguments):
            raise ValueError('Kimi image prefill requires pixel_values, grid_thws, and image_mask together.')

        if pixel_values is not None:
            if inputs_embeds is not None:
                raise ValueError('Kimi raw image inputs cannot be combined with precomputed `inputs_embeds`.')
            if input_ids is None or grid_thws is None or image_mask is None:
                raise ValueError('Kimi image prefill requires input_ids, pixel_values, grid_thws, and image_mask.')

            image_features = self.get_image_features(pixel_values, grid_thws)
            if len(image_features) == 0:
                raise ValueError('Kimi image prefill produced no image features.')
            image_features = torch.cat(image_features, dim=0)
            image_tokens = int(image_mask.sum().item())
            if image_features.shape[0] != image_tokens:
                raise ValueError(
                    f'Kimi projected feature rows ({image_features.shape[0]}) must equal '
                    f'media-pad positions ({image_tokens}).')

            inputs_embeds = self.get_input_embeddings()(input_ids)
            image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            inputs_embeds.masked_scatter_(image_mask[..., None], image_features)
            input_ids = None

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits with the delegated language-model head."""
        return self.language_model.get_logits(hidden_states)

    def get_input_embeddings(self):
        """Return the delegated language-model embeddings."""
        return self.language_model.get_input_embeddings()

    def get_outputs_cudagraph(self,
                              output_buffers: dict[str, torch.Tensor],
                              input_ids: torch.Tensor,
                              **kwargs):
        """Preserve language-model EAGLE auxiliary outputs on graph replay."""
        return self.language_model.get_outputs_cudagraph(
            output_buffers, input_ids=input_ids, **kwargs)

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        """Prepare text or image prefill; decode never reruns MoonViT."""
        if context is None:
            raise ValueError('`context` must be provided for Kimi-K2.6 generation.')
        if self.language_model_only:
            self._raise_for_multimodal_context(context)
        model_inputs = self.language_model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

        if self.language_model_only:
            return model_inputs

        image_data = []
        if context.input_multimodals is not None:
            per_request_images = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            image_data = [data for request_images in per_request_images for data in request_images]

        if image_data:
            is_decoding = getattr(context, 'is_decoding', getattr(context.attn_metadata, 'is_decoding', False))
            if is_decoding:
                raise ValueError('Kimi image payloads must not be replayed during decode.')
            if model_inputs.get('inputs_embeds') is not None:
                raise ValueError('Kimi raw image inputs cannot be combined with precomputed input embeddings.')

            image_token_ids = {int(data.meta['image_token_id']) for data in image_data}
            if len(image_token_ids) != 1:
                raise ValueError(f'Kimi image batch must use one media token id, got {sorted(image_token_ids)}.')
            image_token_id = next(iter(image_token_ids))
            grid_thws = torch.cat([data.meta['grid_thws'].reshape(-1, 3) for data in image_data], dim=0)
            pixel_values = torch.cat([data.data for data in image_data], dim=0)
            image_mask = model_inputs['input_ids'] == image_token_id
            expected_tokens = int(torch.sum(grid_thws[:, 1] * grid_thws[:, 2] // 4).item())
            actual_tokens = int(image_mask.sum().item())
            if actual_tokens != expected_tokens:
                raise ValueError(
                    f'Kimi media-pad positions ({actual_tokens}) do not match grid-derived image tokens '
                    f'({expected_tokens}).')
            model_inputs.update(
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                image_mask=image_mask,
            )

        vision_embeddings = context.input_embeddings
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if image_data:
                raise ValueError('Kimi raw images and externally supplied vision embeddings cannot be mixed.')
            vision_embedding_indexing = context.input_embedding_indexing
            if vision_embedding_indexing is None:
                raise ValueError('Kimi external vision embeddings require input_embedding_indexing.')
            if model_inputs.get('inputs_embeds') is None:
                model_inputs['inputs_embeds'] = self.get_input_embeddings()(model_inputs['input_ids'])
            model_inputs['inputs_embeds'][:, vision_embedding_indexing, :] = vision_embeddings.to(
                model_inputs['inputs_embeds'])

        return model_inputs

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        grid_thws: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        **kwargs,
    ):
        """Capture only text decode; media and embedding prefill stay eager."""
        if inputs_embeds is not None or any(value is not None for value in (pixel_values, grid_thws, image_mask)):
            return False
        if any(kwargs.get(key) is not None for key in self._UNSUPPORTED_MULTIMODAL_FORWARD_KEYS):
            return False
        return CudaGraphMixin.support_cuda_graph(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @classmethod
    def rename_weight(cls, name: str) -> str:
        """Normalize optional Hugging Face base-model prefixes lazily."""
        for prefix in ('language_model.', 'vision_tower.', 'mm_projector.'):
            model_prefix = f'model.{prefix}'
            if name.startswith(model_prefix):
                return name[len('model.'):]
        return name

    @classmethod
    def _iter_language_model_weights(
        cls,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield DeepSeek names without materializing a shard-sized mapping."""
        prefix_length = len(cls._LANGUAGE_MODEL_PREFIX)
        for name, loaded_weight in weights:
            if name.startswith(cls._LANGUAGE_MODEL_PREFIX):
                yield name[prefix_length:], loaded_weight
                continue
            if name.startswith(cls._SKIPPED_WEIGHT_PREFIXES):
                continue
            raise KeyError(f'Unexpected Kimi-K2.6 checkpoint weight: {name}')

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load language weights lazily and route replicated vision weights."""
        if self.language_model_only:
            language_weights = self._iter_language_model_weights(weights)
            self.language_model.load_weights(language_weights)
            return

        params_dict = {
            **{
                f'vision_tower.{name}': parameter
                for name, parameter in self.vision_tower.named_parameters()
            },
            **{
                f'mm_projector.{name}': parameter
                for name, parameter in self.mm_projector.named_parameters()
            },
        }
        language_prefix_length = len(self._LANGUAGE_MODEL_PREFIX)

        def iter_and_load_vision_weights():
            for name, loaded_weight in weights:
                if name.startswith(self._LANGUAGE_MODEL_PREFIX):
                    yield name[language_prefix_length:], loaded_weight
                    continue
                if name.startswith(self._SKIPPED_WEIGHT_PREFIXES):
                    try:
                        param = params_dict[name]
                    except KeyError:
                        raise KeyError(f'Unexpected Kimi-K2.6 vision checkpoint weight: {name}') from None
                    load_weight(param, loaded_weight)
                    continue
                raise KeyError(f'Unexpected Kimi-K2.6 checkpoint weight: {name}')

        language_weights = iter_and_load_vision_weights()
        self.language_model.load_weights(language_weights)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Return the scheduler-side Kimi image input processor."""
        return self.input_processor
