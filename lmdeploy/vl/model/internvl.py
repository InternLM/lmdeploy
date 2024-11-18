# Copyright (c) OpenMMLab. All rights reserved.

import itertools
from typing import Dict, List

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoModel, CLIPImageProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')

IMAGE_TOKEN = '<IMAGE_TOKEN>'


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    """copy from https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    """copy from https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


@VISION_MODELS.register_module()
class InternVLVisionModel(VisonModel):
    """InternVL vision model."""

    _arch = 'InternVLChatModel'

    def build_preprocessor(self):
        # load empty model from its config
        from accelerate import init_empty_weights
        with init_empty_weights():
            self.config = self.hf_config
            # transformers below 4.37.0 may raise error about flash_attn
            self.config.llm_config.attn_implementation = 'eager'
            self.model = AutoModel.from_config(self.config,
                                               trust_remote_code=True)
            if not self.with_llm:
                del self.model.language_model
            else:
                self.vl_model = self.model

        dynamic_image_size = getattr(self.config, 'dynamic_image_size', False)
        image_processor = None
        try:
            image_processor = CLIPImageProcessor.from_pretrained(
                self.model_path)
        except OSError:
            pass

        if dynamic_image_size or image_processor is None:
            logger.info('using InternVL-Chat-V1-5 vision preprocess')
            MEAN = (0.485, 0.456, 0.406)
            STD = (0.229, 0.224, 0.225)
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            input_size = self.config.vision_config.image_size
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')
                         if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size),
                         interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            self.processor = self._preprocess_v1_5
            self._forward_func = self._forward_v1_5
        else:
            self.processor = self._preprocess
            self.image_processor = image_processor
            self._forward_func = self._forward

    def build_model(self):
        """Load model."""

        self.model.half()
        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=self.model,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                max_memory=self.max_memory,
                no_split_module_classes=['InternVisionEncoderLayer'],
                dtype=torch.half)

        # We need eval mode to freeze the weights in model, thus,
        # avoid randomness in inference.
        self.model = self.model.eval()

    def _preprocess_v1_5(self, image: Image, params: Dict = None):
        image_res = {'low': 6, 'medium': 12, 'high': 24}
        max_num = params.get('max_dynamic_patch')
        if max_num is None or not isinstance(max_num, int):
            res_key = params.get('detail', 'default')
            max_num = image_res.get(res_key, self.config.max_dynamic_patch)
        out = dynamic_preprocess(
            image,
            min_num=self.config.min_dynamic_patch,
            max_num=max_num,
            image_size=self.config.vision_config.image_size,
            use_thumbnail=self.config.use_thumbnail)
        pixel_values = [self.transform(x) for x in out]
        # (patch) x c x h x w
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def _forward_v1_5(self, inputs):
        """forward for internvl-chat-v1-5."""
        assert all(x.get('pixel_values') is not None for x in inputs)
        outputs = [x['pixel_values'] for x in inputs]
        split = [x['pixel_values'].shape[0] for x in inputs]
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.to(self.model.device, dtype=torch.float16)
        outputs = self.model.extract_feature(outputs)
        outputs = torch.split(outputs, split, dim=0)
        outputs = [x.reshape(-1, x.shape[-1]) for x in outputs]
        return outputs

    def _preprocess(self, image: Image, params: Dict = None):
        """forward for internvl-chat-v1-1, internvl-chat-v1-2."""
        pixel_values = self.image_processor(images=image,
                                            return_tensors='pt').pixel_values
        return pixel_values

    def _forward(self, inputs):
        """forward for internvl-chat-v1-1, internvl-chat-v1-2."""
        assert all(x.get('pixel_values') is not None for x in inputs)
        outputs = [x['pixel_values'] for x in inputs]
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.to(self.model.device, dtype=torch.float16)
        outputs = self.model.extract_feature(outputs)
        outputs = torch.split(outputs, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        assert isinstance(messages, List)
        assert isinstance(messages[-1]['content'], List)
        content = [
            item['text'] for item in messages[-1]['content']
            if item['type'] == 'text'
        ]
        if len(content) > 1:
            logger.warning(f'There are {len(content)} text in {content}. '
                           'Only the first one is considered')

        # get images and their corresponding preprocess parameters from
        # messages, and perform preprocessing
        outputs = []
        for item in messages[-1]['content']:
            item_type = item['type']
            if item_type == 'image':
                image = item['image'].convert('RGB')
                params = {
                    key: item[key]
                    for key in item.keys() if key not in {'type', 'image'}
                }
                pixel_values = self.processor(image, params)
                outputs.append(dict(pixel_values=pixel_values))
        return outputs

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        """forward vision model to get vision embedding
        Args:
            inputs (List[Dict]): the output of `preprocess`
        """
        return self._forward_func(inputs)

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        # apply chat template to get the prompt
        prompt_messages = []
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            n_images = [
                1 for item in message['content'] if item['type'] == 'image'
            ]
            n_images = sum(n_images)
            content = [
                item['text'] for item in message['content']
                if item['type'] == 'text'
            ]
            content = f'<img>{IMAGE_TOKEN * n_images}</img>\n' + content[0]
            prompt_messages.append(dict(role='user', content=content))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)

        # collect all preprocessing result from messages
        preps = [
            message.pop('preprocess') for message in messages
            if 'preprocess' in message.keys()
        ]
        segs = prompt.split(IMAGE_TOKEN)
        # flatten the list
        preps = list(itertools.chain(*preps))
        assert len(segs) == len(preps) + 1, (
            f'the number of {IMAGE_TOKEN} is not equal '
            f'to input images, {len(segs) - 1} vs {len(preps)}')

        return prompt, segs, preps

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, segs, preps = self.proc_messages(messages, chat_template,
                                                 sequence_start)

        # calculate the image token offset for each image
        input_ids = []
        IMAGE_DUMMY_TOKEN_INDEX = 0
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                # TODO(hardcode 256)
                image_dim = preps[i - 1]['pixel_values'].shape[0] * 256
                input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_dim)
            token_ids = tokenizer.encode(seg,
                                         add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, segs, features = self.proc_messages(messages, chat_template,
                                                    sequence_start)
        features = [x.cpu().numpy() for x in features]

        # tokenizer prompt, and get input_embeddings and input_embedding_ranges
        input_ids = []
        begins = []
        ends = []
        IMAGE_DUMMY_TOKEN_INDEX = 0
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(features):
                image_dim = features[i - 1].shape[0]
                begins.append(len(input_ids))
                ends.append(begins[-1] + image_dim)
                input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_dim)
            seg_ids = tokenizer.encode(seg,
                                       add_bos=((i == 0) and sequence_start))
            input_ids.extend(seg_ids)
        ranges = np.stack([begins, ends], axis=1).tolist()
        return dict(prompt=prompt,
                    input_ids=input_ids,
                    input_embeddings=features,
                    input_embedding_ranges=ranges)
