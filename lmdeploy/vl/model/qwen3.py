# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List

import torch
from transformers import AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel

logger = get_logger('lmdeploy')


def check_transformers():
    try:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3VLModel(VisionModel):
    """Qwen3VL model."""

    _arch = ['Qwen3VLForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration']

    def build_preprocessor(self):
        check_transformers()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # image tokens
        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id

        # video tokens
        self.contains_video_input = False
        self.video_token = self.processor.video_token
        self.video_token_id = self.processor.video_token_id
        self.vision_start_token = self.processor.vision_start_token
        self.vision_end_token = self.processor.vision_end_token

    def get_processor_args(self, mm_processor_kwargs: Dict[str, Any] | None = None):
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
        result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
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

    def preprocess(self, messages: List[Dict], mm_processor_kwargs: Dict[str, Any] | None = None) -> List[Dict]:
        """Refer to `super().preprocess()` for spec."""
        outputs = []

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
                   chat_template_kwargs: Dict | None = None,
                   **kwargs):
        """Return to the information needed by pytorch engine."""
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start, chat_template_kwargs)

        if self.contains_video_input:
            return self.to_pytorch_aux_video(messages, prompt, self.video_token, tokenizer, sequence_start)
        else:
            return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def build_model(self):
        # TODO: implement for turbomind
        pass

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        # TODO: implement for turbomind
        pass

    def to_turbomind(self,
                     messages,
                     chat_template,
                     tokenizer,
                     sequence_start,
                     chat_template_kwargs: Dict | None = None,
                     **kwargs):
        # TODO: implement for turbomind
        pass
