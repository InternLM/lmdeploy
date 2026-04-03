# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
from transformers import AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel

logger = get_logger('lmdeploy')


def check_transformers():
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3OmniModel(VisionModel):
    """Qwen3Omni model."""

    _arch = ['Qwen3OmniMoeForConditionalGeneration']

    def build_preprocessor(self):
        check_transformers()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer

        # image tokens
        self.image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(self.image_token)[-1]

        # video tokens
        self.video_token = self.processor.video_token
        self.video_token_id = tokenizer.encode(self.video_token)[-1]

        # audio tokens
        self.audio_token = self.processor.audio_token
        self.audio_token_id = tokenizer.encode(self.audio_token)[-1]

    def resolve_size_params(self, processor, mm_processor_kwargs: dict[str, Any] | None = None):
        default_min = processor.size['shortest_edge']
        default_max = processor.size['longest_edge']

        if not mm_processor_kwargs:
            return {'shortest_edge': default_min, 'longest_edge': default_max}

        min_pixels = mm_processor_kwargs.get('min_pixels', default_min)
        max_pixels = mm_processor_kwargs.get('max_pixels', default_max)

        if min_pixels > max_pixels:
            logger.warning(f'min_pixels {min_pixels} > max_pixels {max_pixels}, falling back to defaults.')
            return {'shortest_edge': default_min, 'longest_edge': default_max}

        return {'shortest_edge': min_pixels, 'longest_edge': max_pixels}

    def _preprocess_image(self,
                          data: list[Any],
                          params: dict[str, Any],
                          mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:

        size = self.resolve_size_params(self.processor.image_processor, mm_processor_kwargs)
        result = self.processor.image_processor(images=data, size=size, return_tensors='pt')
        merge_length = self.processor.image_processor.merge_size**2
        image_tokens = result['image_grid_thw'].prod(dim=1) // merge_length
        result.update(dict(image_size=data.size, mm_token_num=image_tokens, image_token_id=self.image_token_id))
        return result

    def _preprocess_video(self,
                          data: list[Any],
                          params: dict[str, Any],
                          mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:

        metadata = params['video_metadata']
        if metadata.get('fps') is None or metadata['fps'] <= 0:
            logger.warning('Qwen3Omni: fps not found or invalid, fallback to 24.')
            metadata['fps'] = 24
        size = self.resolve_size_params(self.processor.video_processor, mm_processor_kwargs)

        # do_resize = True, we leave resize to hf processor
        # do_sample_frames = False, we already sample frames in video loader, avoid duplicates in hf processor
        result = self.processor.video_processor(videos=data,
                                                size=size,
                                                return_metadata=True,
                                                do_resize=True,
                                                do_sample_frames=False,
                                                video_metadata=metadata,
                                                return_tensors='pt')

        merge_length = self.processor.video_processor.merge_size**2
        video_grid_thw = result['video_grid_thw']
        second_per_grid = self.processor.video_processor.temporal_patch_size / metadata.get('fps', 1.0)
        frame_seqlen = video_grid_thw[0][1:].prod() // merge_length
        video_tokens = video_grid_thw[0].prod() // merge_length  # T*H*W / merge^2
        result.update(frame_seqlen=frame_seqlen,
                      mm_token_num=video_tokens,
                      second_per_grid=second_per_grid,
                      video_token_id=self.video_token_id)
        return result

    def _get_feat_extract_output_lengths(self, input_lengths):
        """Computes the output length of the convolutional layers and the
        output length of the audio encoder."""

        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def _preprocess_audio(self,
                          data: list[Any],
                          params: dict[str, Any],
                          mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:
        audio, original_sr = data
        # WhisperFeatureExtractor was trained using a fixed sampling rate of 16000
        sr = self.processor.feature_extractor.sampling_rate
        truncation = mm_processor_kwargs.get('truncation', False) if mm_processor_kwargs else False

        result = self.processor.feature_extractor(audio,
                                                  sampling_rate=sr,
                                                  padding=True,
                                                  truncation=truncation,
                                                  return_attention_mask=True,
                                                  return_tensors='pt')

        feature_attention_mask = result.get('attention_mask')
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        audio_output_length = self._get_feat_extract_output_lengths(audio_feature_lengths)
        audio_tokens = audio_output_length

        result.update(
            dict(mm_token_num=audio_tokens,
                 audio_feature_lengths=audio_feature_lengths,
                 audio_token_id=self.audio_token_id))
        return result

    def preprocess(self, messages: list[dict], mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:
        """Refer to `super().preprocess()` for spec."""
        outputs = []
        self.contains_video_input = False
        self.contains_audio_input = False

        mm_items = self.collect_multimodal_items(messages)
        for modality, data, params in mm_items:
            result = {}
            if modality == Modality.IMAGE:
                result = self._preprocess_image(data, params, mm_processor_kwargs)
            elif modality == Modality.VIDEO:
                self.contains_video_input = True
                result = self._preprocess_video(data, params, mm_processor_kwargs)
            elif modality == Modality.AUDIO:
                self.contains_audio_input = True
                result = self._preprocess_audio(data, params, mm_processor_kwargs)

            result.update(modality=modality)
            outputs.append(result)

        messages.append(dict(role='preprocess', content=outputs))
        return messages

    def proc_messages(self, messages, chat_template, sequence_start, chat_template_kwargs=None):
        """Apply chat template to get the prompt."""
        chat_template_kwargs = chat_template_kwargs or {}
        messages = [x for x in messages if x['role'] not in ['preprocess', 'forward']]
        prompt = chat_template.messages2prompt(messages, sequence_start, **chat_template_kwargs)

        mm_placeholder = self.image_token
        if self.contains_video_input:
            mm_placeholder = self.video_token
        elif self.contains_audio_input:
            mm_placeholder = self.audio_token

        return prompt, mm_placeholder

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   sequence_start,
                   chat_template_kwargs: dict | None = None,
                   **kwargs):
        """Return to the information needed by pytorch engine."""
        prompt, mm_placeholder = self.proc_messages(messages, chat_template, sequence_start, chat_template_kwargs)
        return self.to_pytorch_aux(messages, prompt, mm_placeholder, tokenizer, sequence_start)
