# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """copy from https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
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
class InternVLVisionModel(VisionModel):
    """InternVL vision model."""

    _arch = 'InternVLChatModel'

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = ''):
        super().__init__(model_path, with_llm, max_memory, hf_config, backend)
        self.image_token = '<IMG_CONTEXT>'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

    def build_preprocessor(self):
        self.config = self.hf_config
        dynamic_image_size = getattr(self.config, 'dynamic_image_size', False)
        image_processor = None
        try:
            image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
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
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            self.processor = self._preprocess_v1_5
            self._forward_func = self._forward_v1_5
        else:
            self.processor = self._preprocess
            self.image_processor = image_processor
            self._forward_func = self._forward

        force_image_size = self.hf_config.force_image_size
        patch_size = self.hf_config.vision_config.patch_size
        downsample_ratio = self.hf_config.downsample_ratio
        self.image_tokens_per_patch = int((force_image_size // patch_size)**2 * (downsample_ratio**2))

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        from accelerate import init_empty_weights
        with init_empty_weights():
            # transformers below 4.37.0 may raise error about flash_attn
            self.config.llm_config.attn_implementation = 'eager'
            model = AutoModel.from_config(self.config, trust_remote_code=True)
            self.vl_model = model
            if not self.with_llm:
                del model.language_model

        model.half()
        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(model=model,
                                         checkpoint=self.model_path,
                                         device_map='auto' if not self.with_llm else {'': 'cpu'},
                                         max_memory=self.max_memory,
                                         no_split_module_classes=['InternVisionEncoderLayer'],
                                         dtype=torch.half)

        # We need eval mode to freeze the weights in model, thus,
        # avoid randomness in inference.
        self.model = model.eval()

    def _preprocess_v1_5(self, image, params=None):
        image_res = {'low': 6, 'medium': 12, 'high': 24}
        max_num = params.get('max_dynamic_patch')
        if max_num is None or not isinstance(max_num, int):
            res_key = params.get('detail', 'default')
            max_num = image_res.get(res_key, self.config.max_dynamic_patch)
        out = dynamic_preprocess(image,
                                 min_num=self.config.min_dynamic_patch,
                                 max_num=max_num,
                                 image_size=self.config.vision_config.image_size,
                                 use_thumbnail=self.config.use_thumbnail)
        pixel_values = [self.transform(x) for x in out]
        # (patch) x c x h x w
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def _forward_v1_5(self, inputs, max_batch_size):
        """Forward for internvl-chat-v1-5."""
        assert all(x.get('pixel_values') is not None for x in inputs)
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            pixel_values = [x['pixel_values'] for x in inputs[idx:idx + max_batch_size]]
            split = [x.shape[0] for x in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            pixel_values = pixel_values.to(self.model.device, dtype=torch.float16)
            logger.info(f'vision forward shape: {pixel_values.shape}')
            feats = self.model.extract_feature(pixel_values)
            feats = torch.split(feats, split, dim=0)
            outputs.extend([x.reshape(-1, x.shape[-1]) for x in feats])
        return outputs

    def _preprocess(self, image, params=None):
        """Forward for internvl-chat-v1-1, internvl-chat-v1-2."""
        pixel_values = self.image_processor(images=image, return_tensors='pt').pixel_values
        return pixel_values

    def _forward(self, inputs, max_batch_size):
        """Forward for internvl-chat-v1-1, internvl-chat-v1-2."""
        assert all(x.get('pixel_values') is not None for x in inputs)
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            pixel_values = [x['pixel_values'] for x in inputs[idx:idx + max_batch_size]]
            pixel_values = torch.cat(pixel_values, dim=0)
            pixel_values = pixel_values.to(self.model.device, dtype=torch.float16)
            logger.info(f'vision forward shape: {pixel_values.shape}')
            feats = self.model.extract_feature(pixel_values)
            feats = torch.split(feats, 1, dim=0)
            outputs.extend([x.squeeze() for x in feats])
        return outputs

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refers to `super.preprocess() for spec."""
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            pixel_values = self.processor(image, params)
            image_tokens = (pixel_values.shape[0] * self.image_tokens_per_patch)
            outputs.append(
                dict(pixel_values=pixel_values,
                     image_tokens=image_tokens,
                     image_token_id=self.image_token_id,
                     image_size=image.size))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        """Extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the outputs of `preprocess`
            max_batch_size(int): the max batch size when forwarding vision
                model
        Return:
            the message list with forwarding results included
        """
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess']
        inputs = inputs[0]
        outputs = self._forward_func(inputs, max_batch_size)
        messages.append(dict(role='forward', content=outputs))
        return messages

    def proc_messages(
        self,
        messages,
        chat_template,
        sequence_start,
        tools: Optional[List[object]] = None,
        chat_template_kwargs: Optional[Dict] = None,
    ):
        chat_template_kwargs = chat_template_kwargs or {}
        """Apply chat template to get the prompt."""
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
                prompt = prompt.replace(f'{IMAGE_TOKEN}', f'<img>{self.image_token}</img>')
                prompt_messages.append(dict(role='user', content=prompt))
        else:
            for message in messages:
                role, content = message['role'], message['content']
                if role != 'user' or isinstance(content, str):
                    prompt_messages.append(message)
                    continue
                _content = []
                for item in content:
                    item_type = item['type']
                    if item_type == 'text':
                        _content.append(item['text'])
                    elif item_type in ['image', 'image_url']:
                        _content.append(f'<img>{self.image_token}</img>\n')
                    else:
                        raise ValueError(f'Unsupported message type: {item["type"]}')
                prompt_messages.append(dict(role='user', content=''.join(_content)))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start, tools=tools, **chat_template_kwargs)
        return prompt, self.image_token

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   sequence_start,
                   tools: Optional[List[object]] = None,
                   chat_template_kwargs: Optional[Dict] = None,
                   **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages,
                                                 chat_template,
                                                 sequence_start,
                                                 tools=tools,
                                                 chat_template_kwargs=chat_template_kwargs)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self,
                     messages,
                     chat_template,
                     tokenizer,
                     sequence_start,
                     tools: Optional[List[object]] = None,
                     chat_template_kwargs: Optional[Dict] = None,
                     **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages,
                                                 chat_template,
                                                 sequence_start,
                                                 tools=tools,
                                                 chat_template_kwargs=chat_template_kwargs)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
