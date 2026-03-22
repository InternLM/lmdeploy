# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional

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

    def get_processor_args(self, mm_processor_kwargs: Optional[Dict[str, Any]] = None):
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
                          data: List[Any],
                          params: Dict[str, Any],
                          mm_processor_kwargs: Dict[str, Any] | None = None) -> List[Dict]:

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
        result.update(dict(image_size=image.size, mm_token_num=image_tokens, image_token_id=self.image_token_id))
        return result

    def _preprocess_video(self,
                          data: List[Any],
                          params: Dict[str, Any],
                          mm_processor_kwargs: Dict[str, Any] | None = None) -> List[Dict]:

        # TODO: zhouxinyu, apply transformers smart_resize using per-request kwargs
        metadata = params['video_metadata']
        video_kwargs = dict(return_metadata=True,
                            do_resize=True,
                            do_sample_frames=False,
                            video_metadata=metadata,
                            return_tensors='pt')

        # TODO: update from mm_processor_kwargs when needed
        video_kwargs.update(size={
            'shortest_edge': 128 * 32 * 32,
            'longest_edge': 768 * 32 * 32,
        })
        result = self.processor.video_processor(videos=data, **video_kwargs)
        video_grid_thw = result['video_grid_thw']

        merge_length = self.processor.video_processor.merge_size**2
        if metadata.get('fps') is None:
            logger.warning_once('Qwen3VL: fps not found, defaulting to 24.')
            metadata['fps'] = metadata['fps'] or 24

        # TODO: update fps from video kwargs, refer to transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py
        second_per_grid = self.processor.video_processor.temporal_patch_size / video_kwargs.get('fps', 1.0)

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
                          data: List[Any],
                          params: Dict[str, Any],
                          mm_processor_kwargs: Dict[str, Any] | None = None) -> List[Dict]:
        audio, original_sr = data
        # NOTE: WhisperFeatureExtractor was trained using a fixed sampling rate of 16000
        # TODO: zhouxinyu, get truncation from mm_processor_kwargs when needed
        sr = self.processor.feature_extractor.sampling_rate
        audio_kwargs = {
            'sampling_rate': sr,
            'padding': True,
            'truncation': False,
            'return_attention_mask': True,
            'return_tensors': 'pt'
        }
        result = self.processor.feature_extractor(audio, **audio_kwargs)
        feature_attention_mask = result.get('attention_mask')
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        audio_output_length = self._get_feat_extract_output_lengths(audio_feature_lengths)
        audio_tokens = audio_output_length

        result.update(
            dict(mm_token_num=audio_tokens,
                 audio_feature_lengths=audio_feature_lengths,
                 audio_token_id=self.audio_token_id))
        return result

    def preprocess(self, messages: List[Dict], mm_processor_kwargs: Dict[str, Any] | None = None) -> List[Dict]:
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
                   chat_template_kwargs: Dict | None = None,
                   **kwargs):
        """Return to the information needed by pytorch engine."""
        prompt, mm_placeholder = self.proc_messages(messages, chat_template, sequence_start, chat_template_kwargs)
        return self.to_pytorch_aux(messages, prompt, mm_placeholder, tokenizer, sequence_start)
