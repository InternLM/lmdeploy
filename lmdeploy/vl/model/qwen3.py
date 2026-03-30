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
        self.video_token = self.processor.video_token
        self.video_token_id = self.processor.video_token_id

        # vision start and end tokens
        self.vision_start_token = self.processor.vision_start_token
        self.vision_end_token = self.processor.vision_end_token

    def resolve_size_params(self, processor, mm_processor_kwargs: dict[str, Any] | None = None):
        default_min = processor.size['shortest_edge']
        default_max = processor.size['longest_edge']

        if not mm_processor_kwargs:
            return processor.size

        min_pixels = mm_processor_kwargs.get('min_pixels', default_min)
        max_pixels = mm_processor_kwargs.get('max_pixels', default_max)

        if min_pixels > max_pixels:
            logger.warning(f'min_pixels {min_pixels} > max_pixels {max_pixels}, falling back to defaults.')
            return processor.size

        return {'shortest_edge': min_pixels, 'longest_edge': max_pixels}

    def _preprocess_image(self,
                          data: list[Any],
                          params: dict[str, Any],
                          mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:

        size = self.resolve_size_params(self.processor.image_processor, mm_processor_kwargs)
        result = self.processor.image_processor(images=data, size=size, return_tensors='pt')
        merge_length = self.processor.image_processor.merge_size**2
        image_tokens = result['image_grid_thw'].prod(dim=1) // merge_length
        result.update(dict(image_size=data.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
        return result

    def _preprocess_video(self,
                          data: list[Any],
                          params: dict[str, Any],
                          mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:

        metadata = params['video_metadata']
        if metadata.get('fps') is None or metadata['fps'] <= 0:
            logger.warning('Qwen3VL: fps not found or invalid, fallback to 24.')
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
        frame_seqlen = video_grid_thw[0][1:].prod() // merge_length
        curr_timestamp = self.processor._calculate_timestamps(
            metadata['frames_indices'],
            metadata['fps'],
            self.processor.video_processor.merge_size,
        )

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
        return prompt, None

    def to_pytorch_aux_video(self, messages, prompt, VIDEO_TOKEN, tokenizer, sequence_start):
        """Pack the video input to the compatible format with pytorch engine.

        Each video is split into per-frame (temporal step) entries so that the timestamp text tokens between frames get
        sequential mrope positions and each frame's video-pad tokens get independent 3D spatial positions.
        """

        # collect all preprocessing result from messages
        preps = [x['content'] for x in messages if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        # split prompt into segments and validate data
        segs = prompt.split(self.vision_start_token + self.video_token + self.vision_end_token)
        assert len(segs) == len(preps) + 1, (f'the number of {self.video_token} is not equal '
                                             f'to input videos, {len(segs) - 1} vs {len(preps)}')

        # calculate the video token offset for each frame
        input_ids = []
        frame_preps = []

        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                video_prep = preps[i - 1]
                frame_seqlen = video_prep['frame_seqlen']
                curr_timestamp = video_prep['curr_timestamp']
                video_grid_thw = video_prep['video_grid_thw']
                pixel_values_videos = video_prep['pixel_values_videos']
                assert self.video_token_id == video_prep['video_token_id']

                t, h, w = video_grid_thw[0].tolist()

                # each temporal step becomes an independent multimodal entry
                for frame_idx in range(t):
                    curr_time = curr_timestamp[frame_idx]

                    # timestamp text + vision_start (regular text tokens)
                    prefix = f'<{curr_time:.1f} seconds>' + self.vision_start_token
                    prefix_ids = tokenizer.encode(prefix, add_bos=False)
                    input_ids.extend(prefix_ids)

                    # video pad tokens for this frame
                    frame_offset = len(input_ids)
                    input_ids.extend([self.video_token_id] * frame_seqlen)

                    # vision_end (regular text token)
                    suffix_ids = tokenizer.encode(self.vision_end_token, add_bos=False)
                    input_ids.extend(suffix_ids)

                    # since we use timestamps to separate videos
                    # like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>
                    # the video_grid_thw should also be split, becomes [1, h, w] for each frame
                    frame_preps.append(
                        dict(
                            offset=frame_offset,
                            video_tokens=frame_seqlen,
                            pixel_values_videos=pixel_values_videos[frame_idx * h * w:(frame_idx + 1) * h * w],
                            video_grid_thw=torch.tensor([[1, h, w]]),
                            video_token_id=self.video_token_id,
                            modality=video_prep['modality'],
                        )
                    )

            token_ids = tokenizer.encode(seg, add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=frame_preps)

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
        # TODO: implement for turbomind
        pass

    @torch.no_grad()
    def forward(self, messages: list[dict], max_batch_size: int = 1) -> list[dict]:
        # TODO: implement for turbomind
        pass

    def to_turbomind(self,
                     messages,
                     chat_template,
                     tokenizer,
                     sequence_start,
                     chat_template_kwargs: dict | None = None,
                     **kwargs):
        # TODO: implement for turbomind
        pass
