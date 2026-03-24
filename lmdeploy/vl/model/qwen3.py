# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
from transformers import AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


def check_qwen3_vl_deps_install():
    """Check dependencies for Qwen3-VL / Qwen3.5 (same vision stack as
    Qwen2-VL's ``check_qwen_vl_deps_install``).

    - **Transformers**: recent build with Qwen3-VL and Qwen3.5 classes (see Qwen3-VL model card on HF).
    - **Accelerate**: required for TurboMind split vision loading (`load_checkpoint_and_dispatch`).
    - **qwen-vl-utils** (optional): pip package ``qwen-vl-utils``; many upstream Qwen-VL recipes use it for
      video helpers. LMDeploy's Qwen3 preprocessor uses ``AutoProcessor`` only; warn if missing so users
      can align with `Qwen2VLModel` / official docs when needed.
    """
    try:
        from transformers import (  # noqa: F401
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )
    except ImportError:
        raise ImportError('please install a recent transformers with Qwen3-VL / Qwen3.5 support, e.g. '
                          'pip install git+https://github.com/huggingface/transformers.git')
    try:
        import accelerate  # noqa: F401
    except ImportError:
        raise ImportError('please install accelerate for TurboMind vision loading: pip install accelerate')
    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        logger.warning_once(
            'qwen-vl-utils is not installed. Install with `pip install qwen-vl-utils` if you use '
            'video pipelines or helpers from the Qwen-VL examples (optional for LMDeploy Qwen3 preprocess).')


def resolve_qwen_vl_family_automodel(arch: str) -> tuple[type, list[str]]:
    """Map HF architecture name to the model class and accelerate no-split
    vision block names.

    Qwen3-VL introduced this TurboMind split-vision path; Qwen3.5 reuses the same stack.
    """
    if arch == 'Qwen3VLForConditionalGeneration':
        from transformers import Qwen3VLForConditionalGeneration as AutoModelCls

        no_split = ['Qwen3VLVisionBlock', 'Qwen3VLMoeVisionBlock']
    elif arch == 'Qwen3VLMoeForConditionalGeneration':
        from transformers import Qwen3VLMoeForConditionalGeneration as AutoModelCls

        no_split = ['Qwen3VLVisionBlock', 'Qwen3VLMoeVisionBlock']
    elif arch == 'Qwen3_5ForConditionalGeneration':
        from transformers import Qwen3_5ForConditionalGeneration as AutoModelCls

        no_split = ['Qwen3_5VisionBlock', 'Qwen3_5MoeVisionBlock']
    elif arch == 'Qwen3_5MoeForConditionalGeneration':
        from transformers import Qwen3_5MoeForConditionalGeneration as AutoModelCls

        no_split = ['Qwen3_5VisionBlock', 'Qwen3_5MoeVisionBlock']
    else:
        raise ValueError(f'Unsupported Qwen VL family architecture: {arch}')
    return AutoModelCls, no_split


def load_qwen_vl_family_vision_backbone(
    model_path: str,
    hf_config: Any,
    with_llm: bool,
    max_memory: dict[int, int] | None,
) -> Any:
    """Load vision tower only (TurboMind path) for Qwen3-VL and Qwen3.5."""
    arch = hf_config.architectures[0]
    AutoModelCls, no_split = resolve_qwen_vl_family_automodel(arch)

    if with_llm:
        return AutoModelCls.from_pretrained(model_path, device_map='cpu')

    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

    with init_empty_weights():
        config = hf_config
        config.tie_word_embeddings = False
        if hasattr(config, 'text_config'):
            config.text_config.tie_word_embeddings = False
        model = AutoModelCls._from_config(config)
        del model.model.language_model
        del model.lm_head
        model.half()

    with disable_logging():
        load_checkpoint_and_dispatch(
            model=model,
            checkpoint=model_path,
            device_map='auto',
            max_memory=max_memory,
            no_split_module_classes=no_split,
            dtype=torch.half,
        )
    return model.model.eval()


@VISION_MODELS.register_module()
class Qwen3VLModel(VisionModel):
    """Qwen3VL model."""

    _arch = ['Qwen3VLForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration']

    def build_preprocessor(self):
        check_qwen3_vl_deps_install()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # image tokens
        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id

        # video tokens
        self.video_token = self.processor.video_token
        self.video_token_id = self.processor.video_token_id

        # vision start and end tokens
        self.vision_start_token = self.processor.vision_start_token
        self.vision_end_token = self.processor.vision_end_token

    def get_processor_args(self, mm_processor_kwargs: dict[str, Any] | None = None):
        min_pixels = self.processor.image_processor.size['shortest_edge']
        max_pixels = self.processor.image_processor.size['longest_edge']

        if mm_processor_kwargs is None:
            return min_pixels, max_pixels

        input_min_pixels = mm_processor_kwargs.get('min_pixels', None)
        input_max_pixels = mm_processor_kwargs.get('max_pixels', None)

        # boundary check for min_pixels and max_pixels
        if input_min_pixels is None:
            if input_max_pixels is not None:
                # only max_pixels is given in the input
                if input_max_pixels < min_pixels:
                    logger.warning(
                        f'input max_pixels {input_max_pixels} < default min_pixels {min_pixels}, fall back to default.')
                    return min_pixels, max_pixels
                max_pixels = input_max_pixels
        else:
            if input_max_pixels is None:
                # only min_pixels is given in the input
                if input_min_pixels > max_pixels:
                    logger.warning(
                        f'input min_pixels {input_min_pixels} > default max_pixels {max_pixels}, fall back to default.')
                    return min_pixels, max_pixels
            else:
                if input_min_pixels > input_max_pixels:
                    logger.warning(
                        f'input min_pixels {input_min_pixels} > max_pixels {input_max_pixels}, fall back to default.')
                    return min_pixels, max_pixels
                max_pixels = input_max_pixels
            min_pixels = input_min_pixels

        return min_pixels, max_pixels

    def _preprocess_image(self,
                          data: list[Any],
                          params: dict[str, Any],
                          mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:

        image = data.convert('RGB')
        min_pixels, max_pixels = self.get_processor_args(mm_processor_kwargs)

        result = self.processor.image_processor(images=image,
                                                size={
                                                    'shortest_edge': min_pixels,
                                                    'longest_edge': max_pixels
                                                },
                                                return_tensors='pt')
        merge_length = self.processor.image_processor.merge_size**2
        image_tokens = result['image_grid_thw'].prod(dim=1) // merge_length
        result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
        return result

    def _preprocess_video(self,
                          data: list[Any],
                          params: dict[str, Any],
                          mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:

        # TODO: zhouxinyu, apply transformers smart_resize using per-request kwargs
        metadata = params['video_metadata']
        video_kwargs = dict(return_metadata=True,
                            do_resize=True,
                            do_sample_frames=False,
                            video_metadata=metadata,
                            return_tensors='pt')
        result = self.processor.video_processor(videos=data, **video_kwargs)
        video_grid_thw = result['video_grid_thw']

        merge_length = self.processor.video_processor.merge_size**2
        if metadata.get('fps') is None:
            logger.warning_once('Qwen3VL: fps not found, defaulting to 24.')
            metadata['fps'] = metadata['fps'] or 24

        # if timestamps are not provided, calculate them
        curr_timestamp = self.processor._calculate_timestamps(
            metadata['frames_indices'],
            metadata['fps'],
            self.processor.video_processor.merge_size,
        )

        frame_seqlen = video_grid_thw[0][1:].prod() // merge_length
        result.update(curr_timestamp=curr_timestamp, frame_seqlen=frame_seqlen, video_token_id=self.video_token_id)
        return result

    def preprocess(self, messages: list[dict], mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:
        """Refer to `super().preprocess()` for spec."""
        outputs = []
        self.contains_video_input = False

        mm_items = self.collect_multimodal_items(messages)
        for modality, data, params in mm_items:
            result = {}
            if modality == Modality.IMAGE:
                result = self._preprocess_image(data, params, mm_processor_kwargs)
            elif modality == Modality.VIDEO:
                self.contains_video_input = True
                result = self._preprocess_video(data, params, mm_processor_kwargs)

            result.update(modality=modality)
            outputs.append(result)

        messages.append(dict(role='preprocess', content=outputs))
        return messages

    def proc_messages(self, messages, chat_template, sequence_start, chat_template_kwargs=None):
        """Apply chat template to get the prompt."""
        chat_template_kwargs = chat_template_kwargs or {}
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        messages = [x for x in messages if x['role'] not in ['preprocess', 'forward']]
        if VisionModel.IMAGE_TOKEN_included(messages):
            # backward compatibility
            for message in messages:
                role, content = message['role'], message['content']
                if role != 'user' or isinstance(content, str):
                    prompt_messages.append(message)
                    continue
                content = [x['text'] for x in content if x['type'] == 'text']
                prompt = ''.join(content)
                prompt = prompt.replace(IMAGE_TOKEN, f'<|vision_start|>{self.image_token}<|vision_end|>')
                prompt_messages.append(dict(role='user', content=prompt))
        else:
            prompt_messages = messages
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start, **chat_template_kwargs)
        return prompt, self.image_token

    def _ensure_turbomind_image_only(self, inputs: list[dict]):
        """TurboMind split vision currently supports image inputs only."""
        has_video = self.contains_video_input or any('video_grid_thw' in item for item in inputs)
        if has_video:
            raise NotImplementedError('TurboMind split vision for the Qwen3 VL family currently supports images only.')

    def to_pytorch_aux_video(self, messages, prompt, VIDEO_TOKEN, tokenizer, sequence_start):
        """Pack the video input to the compatible format with pytorch
        engine."""

        # collect all preprocessing result from messages
        preps = [x['content'] for x in messages if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        # split prompt into segments and validate data
        segs = prompt.split(self.vision_start_token + self.video_token + self.vision_end_token)
        assert len(segs) == len(preps) + 1, (f'the number of {self.video_token} is not equal '
                                             f'to input videos, {len(segs) - 1} vs {len(preps)}')

        # calculate the video token offset for each video
        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                frame_seqlen = preps[i - 1]['frame_seqlen']
                assert self.video_token_id == preps[i - 1]['video_token_id']

                video_grid_thw = preps[i - 1]['video_grid_thw']
                curr_timestamp = preps[i - 1]['curr_timestamp']

                # update prompt with timestamp index tokens and video pad tokens
                video_placeholder = ''
                for frame_idx in range(video_grid_thw[0][0]):
                    curr_time = curr_timestamp[frame_idx]
                    video_placeholder += f'<{curr_time:.1f} seconds>'
                    video_placeholder += (self.vision_start_token + '<|placeholder|>' * frame_seqlen +
                                          self.vision_end_token)

                video_placeholder = video_placeholder.replace('<|placeholder|>', self.video_token)
                video_token_ids = tokenizer.encode(video_placeholder)
                input_ids.extend(video_token_ids)

                preps[i - 1].update(video_tokens=len(video_token_ids))

            token_ids = tokenizer.encode(seg, add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   sequence_start,
                   chat_template_kwargs: dict | None = None,
                   **kwargs):
        """Return to the information needed by pytorch engine."""
        prompt, _ = self.proc_messages(messages, chat_template, sequence_start, chat_template_kwargs)

        if self.contains_video_input:
            return self.to_pytorch_aux_video(messages, prompt, self.video_token, tokenizer, sequence_start)
        else:
            return self.to_pytorch_aux(messages, prompt, self.image_token, tokenizer, sequence_start)

    def build_model(self):
        """Load vision tower for TurboMind split path (Qwen3-VL and Qwen3.5
        share the same stack)."""
        loaded = load_qwen_vl_family_vision_backbone(self.model_path, self.hf_config, self.with_llm,
                                                     self.max_memory)
        if self.with_llm:
            self.vl_model = loaded
        else:
            self.model = loaded

    @torch.no_grad()
    def forward(self, messages: list[dict], max_batch_size: int = 1) -> list[dict]:
        """Run vision encoder for TurboMind split path (shared Qwen3 VL
        family)."""
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess'][0]
        self._ensure_turbomind_image_only(inputs)
        dtype = torch.half
        device = next(self.model.visual.parameters()).device
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            pixel_values = [x['pixel_values'].type(dtype) for x in inputs[idx:idx + max_batch_size]]
            image_grid_thw = [x['image_grid_thw'] for x in inputs[idx:idx + max_batch_size]]
            pixel_values = torch.cat(pixel_values, dim=0).to(device)
            image_grid_thw = torch.cat(image_grid_thw, dim=0).to(device)
            image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
            if hasattr(image_embeds, 'pooler_output'):
                image_embeds = image_embeds.pooler_output
            merge_length = self.processor.image_processor.merge_size**2
            split_size = image_grid_thw.prod(dim=1) // merge_length
            image_embeds = image_embeds.split(split_size.tolist())
            outputs.extend(image_embeds)
        messages.append(dict(role='forward', content=outputs))
        return messages

    @staticmethod
    def get_mrope_info(seq_len: int, grid_thws: list[tuple] | None = None, ranges: list[tuple] | None = None):
        mrope_position_ids = [torch.arange(ranges[0][0]).expand(3, -1)]
        st_idx = ranges[0][0]
        for i, (grid_thw, embedding_range) in enumerate(zip(grid_thws, ranges)):
            llm_grid_t, llm_grid_h, llm_grid_w = grid_thw
            llm_grid_h //= 2
            llm_grid_w //= 2
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            mrope_position_ids.append(torch.stack([t_index, h_index, w_index]) + st_idx)
            st_idx += max(llm_grid_h, llm_grid_w)
            if i < len(ranges) - 1:
                text_len = ranges[i + 1][0] - ranges[i][1]
            else:
                text_len = seq_len - embedding_range[1]
            mrope_position_ids.append(torch.arange(text_len).expand(3, -1) + st_idx)
            st_idx += text_len
        mrope_position_ids = torch.cat(mrope_position_ids, dim=-1)
        mrope_position_delta = torch.tensor([st_idx - seq_len], dtype=torch.long)
        return mrope_position_ids, mrope_position_delta

    def to_turbomind(self,
                     messages,
                     chat_template,
                     tokenizer,
                     sequence_start,
                     chat_template_kwargs: dict | None = None,
                     **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start, chat_template_kwargs)
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess'][0]
        self._ensure_turbomind_image_only(inputs)
        info = super().to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
        grid_thws = [x['image_grid_thw'].tolist()[0] for x in inputs]
        seq_len = len(info['input_ids'])
        ranges = info['input_embedding_ranges']
        mrope_position_ids, mrope_position_delta = self.get_mrope_info(seq_len, grid_thws, ranges)
        meta = dict(mrope_position_ids=mrope_position_ids, mrope_position_delta=mrope_position_delta)
        info.update(dict(input_meta=meta))
        return info
