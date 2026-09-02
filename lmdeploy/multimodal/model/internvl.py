# Copyright (c) OpenMMLab. All rights reserved.

from argparse import Namespace

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor

from lmdeploy.multimodal.model.base import VISION_MODELS, VisionModel
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _normalize_vision_config(vision_config):
    """Normalize vision_config to support both dict and object forms."""
    if isinstance(vision_config, dict):
        return Namespace(**vision_config)
    return vision_config


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
    _turbomind_native_vision = True

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = '',
                 trust_remote_code: bool = False):
        super().__init__(model_path, with_llm, max_memory, hf_config, backend, trust_remote_code=trust_remote_code)
        self.image_token = '<IMG_CONTEXT>'
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=trust_remote_code,
                                                  use_fast=False)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

    def build_preprocessor(self, trust_remote_code: bool = False):
        self.config = self.hf_config
        self.vision_config = _normalize_vision_config(self.config.vision_config)
        dynamic_image_size = getattr(self.config, 'dynamic_image_size', False)
        image_processor = None
        try:
            image_processor = CLIPImageProcessor.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        except OSError:
            pass

        if dynamic_image_size or image_processor is None:
            logger.info('using InternVL-Chat-V1-5 vision preprocess')
            MEAN = (0.485, 0.456, 0.406)
            STD = (0.229, 0.224, 0.225)
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            input_size = self.vision_config.image_size
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            self.processor = self._preprocess_v1_5
        else:
            self.processor = self._preprocess
            self.image_processor = image_processor

        force_image_size = self.hf_config.force_image_size
        patch_size = self.vision_config.patch_size
        downsample_ratio = self.hf_config.downsample_ratio
        self.image_tokens_per_patch = int((force_image_size // patch_size)**2 * (downsample_ratio**2))

    def build_model(self, trust_remote_code: bool = False):
        """Load the whole VLM for quantization."""
        # transformers below 4.37.0 may raise error about flash_attn
        self.config.llm_config.attn_implementation = 'eager'
        self.vl_model = AutoModel.from_pretrained(self.model_path,
                                                  config=self.config,
                                                  device_map='cpu',
                                                  dtype=torch.half,
                                                  trust_remote_code=trust_remote_code).eval()

    def _preprocess_v1_5(self, image, params=None):
        image_res = {'low': 6, 'medium': 12, 'high': 24}
        max_num = params.get('max_dynamic_patch')
        if max_num is None or not isinstance(max_num, int):
            res_key = params.get('detail', 'default')
            max_num = image_res.get(res_key, self.config.max_dynamic_patch)
        out = dynamic_preprocess(image,
                                 min_num=self.config.min_dynamic_patch,
                                 max_num=max_num,
                                 image_size=self.vision_config.image_size,
                                 use_thumbnail=self.config.use_thumbnail)
        pixel_values = [self.transform(x) for x in out]
        # (patch) x c x h x w
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def _preprocess(self, image, params=None):
        """Preprocess for internvl-chat-v1-1 and internvl-chat-v1-2."""
        pixel_values = self.image_processor(images=image, return_tensors='pt').pixel_values
        return pixel_values

    def preprocess(self, messages: list[dict]) -> list[dict]:
        """Refers to `super.preprocess() for spec."""
        images = self.collect_multimodal_items(messages)
        outputs = []
        for modality, image, params in images:
            pixel_values = self.processor(image, params)
            image_tokens = (pixel_values.shape[0] * self.image_tokens_per_patch)
            outputs.append(
                dict(pixel_values=pixel_values.to(self.mm_feature_dtype),
                     image_tokens=image_tokens,
                     image_token_id=self.image_token_id,
                     image_size=image.size))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    def proc_messages(
        self,
        messages,
        chat_template,
        tools: list[object] | None = None,
        chat_template_kwargs: dict | None = None,
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
        prompt = chat_template.messages2prompt(prompt_messages, tools=tools, **chat_template_kwargs)
        return prompt, self.image_token

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   tools: list[object] | None = None,
                   chat_template_kwargs: dict | None = None,
                   **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages,
                                                 chat_template,
                                                 tools=tools,
                                                 chat_template_kwargs=chat_template_kwargs)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer)
