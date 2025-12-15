# Copyright (c) OpenMMLab. All rights reserved.

import enum
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel
from lmdeploy.vl.model.utils import add_device_hook, disable_logging, rewrite_ctx

logger = get_logger('lmdeploy')


def check_xcomposer_install():
    try:
        # WARNING! we have to do this otherwise the model_type is wrong for
        # xcomposer2d5
        import decord  # noqa: F401
    except ImportError:
        raise ImportError("No module named 'decord'. Please install decord by `pip install decord`"  # noqa
                          )


class ModelType(enum.Enum):
    """Request type."""
    XCOMPOSER2 = enum.auto()
    XCOMPOSER2_4KHD = enum.auto()
    XCOMPOSER2D5 = enum.auto()


def get_xcomposer_type(model_path: str) -> Tuple[ModelType, Any]:
    """Get xcomposer type."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    match_modules = {
        'ixc_utils.Image_transform': ModelType.XCOMPOSER2D5,
        'ixc_utils.HD_transform': ModelType.XCOMPOSER2_4KHD
    }
    for key, value in match_modules.items():
        try:
            module = get_class_from_dynamic_module(key, model_path)
            return value, module
        except Exception:
            pass
    return ModelType.XCOMPOSER2, None


def _CLIPVisionModel_from_pretrained(vision_tower_name):
    from transformers import CLIPVisionConfig, CLIPVisionModel
    config = CLIPVisionConfig.from_pretrained(vision_tower_name)
    model = CLIPVisionModel._from_config(config)
    return model


@contextmanager
def init_empty_vit(model_path):
    """Skip download vision model."""
    origin_func_path = [
        'transformers.CLIPVisionModel.from_pretrained',
    ]
    rewrite_func = [
        _CLIPVisionModel_from_pretrained,
    ]

    model_type, _ = get_xcomposer_type(model_path)
    if model_type == ModelType.XCOMPOSER2D5:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.utils import TRANSFORMERS_DYNAMIC_MODULE_NAME
        _ = get_class_from_dynamic_module('modeling_internlm_xcomposer2.get_font', model_path)
        folder = model_path.rstrip(os.sep).split(os.sep)[-1]
        module_path = '.'.join([TRANSFORMERS_DYNAMIC_MODULE_NAME, folder, 'modeling_internlm_xcomposer2'])
        origin_get_font_func = getattr(sys.modules[module_path], 'get_font')
        origin_func_path.append(origin_get_font_func)
        rewrite_func.append(lambda: None)

    with rewrite_ctx(origin_func_path, rewrite_func):
        yield


@VISION_MODELS.register_module()
class Xcomposer2VisionModel(VisionModel):
    """InternLM-Xcomposer2 vision model."""

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = ''):
        model_path = model_path.rstrip(os.sep)
        super().__init__(model_path, with_llm, max_memory, hf_config, backend)
        check_xcomposer_install()
        self.model_type, self.module = get_xcomposer_type(self.model_path)
        logger.info(f'matching type of {self.model_type}')

    @classmethod
    def match(cls, config: AutoConfig):
        """Check whether the config match the model."""
        arch = config.architectures[0] if config.architectures else None
        target = 'InternLMXComposer2ForCausalLM'
        if arch == target:
            return True
        for _, v in getattr(config, 'auto_map', {}).items():
            if target in v:
                return True
        return False

    def build_preprocessor(self):

        import torchvision.transforms as transforms
        from torchvision.transforms.functional import InterpolationMode

        if self.model_type in [ModelType.XCOMPOSER2D5, ModelType.XCOMPOSER2_4KHD]:
            self.HD_transform = self.module
            self.vis_processor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            self.preprocess_func = (self._preprocess_2d5
                                    if self.model_type == ModelType.XCOMPOSER2D5 else self._preprocess_4khd_7b)
        else:
            self.vis_processor = transforms.Compose([
                transforms.Resize((self.hf_config.img_size, self.hf_config.img_size),
                                  interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            self.preprocess_func = self._preprocess_7b

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        from accelerate import init_empty_weights
        with init_empty_weights(), warnings.catch_warnings(), \
                init_empty_vit(self.model_path):
            warnings.simplefilter('ignore')
            config = self.hf_config
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            model.vit.load_model()
            model.vit.resize_pos()
            if hasattr(self.hf_config, 'img_size'):
                model.vit.vision_tower.vision_model.embeddings.image_size = \
                    self.hf_config.img_size
            model.vit.vision_tower.vision_model.post_layernorm.to_empty(device='cpu').half()
            self.vl_model = model
            if not self.with_llm:
                del model.model
                del model.output

        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        max_memory = get_balanced_memory(model,
                                         max_memory=self.max_memory,
                                         dtype=torch.half,
                                         no_split_module_classes=['CLIPEncoderLayer'])
        device_map = infer_auto_device_map(model,
                                           no_split_module_classes=['CLIPEncoderLayer'],
                                           max_memory=max_memory,
                                           dtype=torch.half)
        # make all tensor on same device for postprocess
        if 'plora_glb_GN' in device_map:
            device_map['plora_sub_GN'] = device_map['plora_glb_GN']

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(model=model,
                                         checkpoint=self.model_path,
                                         device_map=device_map if not self.with_llm else {'': 'cpu'},
                                         no_split_module_classes=['CLIPEncoderLayer'],
                                         dtype=torch.half)

        if 'plora_glb_GN' in device_map:
            add_device_hook(model.vit.vision_tower.vision_model.encoder.layers[-1], device_map['plora_glb_GN'],
                            lambda x: (x[0].to(device=device_map['plora_glb_GN']), ))

        self.model = model.eval()

    def _preprocess_2d5(self, image: Image, params: Dict) -> Dict:
        """Image preprocessing for internlm-xcomposer2d5-7b."""
        hd_num = params.get('hd_num', 24)
        image = self.HD_transform(image, hd_num=hd_num)
        pixel_values = self.vis_processor(image).unsqueeze(0).half()
        w, h = image.size
        w, h = w // 560, h // 560
        n_token_per_image = int((h * w + 1) * 400 + 1 + (h + 1) * 20)
        return pixel_values, n_token_per_image

    def _preprocess_7b(self, image: Image, params: Dict) -> Dict:
        """Image preprocessing for internlm-xcomposer2-7b."""
        pixel_values = self.vis_processor(image).unsqueeze(0).half()
        return pixel_values, 256

    def _preprocess_4khd_7b(self, image: Image, params: Dict) -> Dict:
        """Image preprocessing for internlm-xcomposer2-4khd-7b."""
        image = self.HD_transform(image, hd_num=25)
        pixel_values = self.vis_processor(image).unsqueeze(0).half()
        w, h = image.size
        w, h = w // 336, h // 336
        n_token_per_image = int((h * w + 1) * 144 + 1 + (h + 1) * 12)
        return pixel_values, n_token_per_image

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refer to `super().preprocess() for spec."""
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            pixel_values, n_token = self.preprocess_func(image, params)
            outputs.append(
                dict(pixel_values=pixel_values,
                     image_size=image.size,
                     image_tokens=n_token,
                     image_token_id=self.image_token_id))
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
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            if self.model_type in [ModelType.XCOMPOSER2D5, ModelType.XCOMPOSER2_4KHD]:
                pixel_values = [x['pixel_values'] for x in inputs[idx:idx + max_batch_size]]
                embeds, split = self.model.vit(pixel_values, self.model.plora_glb_GN, self.model.plora_sub_GN)
                embeds = self.model.vision_proj(embeds)
                embeds = torch.split(embeds, split, dim=1)
                embeds = [x.squeeze() for x in embeds]
            else:
                pixel_values = [x['pixel_values'] for x in inputs[idx:idx + max_batch_size]]
                pixel_values = torch.cat(pixel_values, dim=0)
                logger.info(f'vision forward shape: {pixel_values.shape}')
                embeds = self.model.vit(pixel_values)
                embeds = self.model.vision_proj(embeds)
                embeds = torch.split(embeds, 1, dim=0)
                embeds = [x.squeeze() for x in embeds]
            outputs.extend(embeds)
        messages.append(dict(role='forward', content=outputs))
        return messages

    @staticmethod
    def proc_messages(messages, chat_template, sequence_start, model_type):
        """Apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        prefix_image_token = ''
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            n_images = len([1 for x in message['content'] if x['type'] == 'image'])
            content = [item['text'] for item in message['content'] if item['type'] == 'text']
            if IMAGE_TOKEN not in content[0]:
                if model_type == ModelType.XCOMPOSER2D5:
                    if n_images == 1:
                        prefix_image_token, prompt = IMAGE_TOKEN, content[0]
                    else:
                        prompt = ''.join([f'Image{i+1} {IMAGE_TOKEN}; ' for i in range(n_images)]) + content[0]
                else:
                    prompt = ''.join([IMAGE_TOKEN] * n_images) + content[0]
            else:
                prompt = content[0]
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = prefix_image_token + chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start, self.model_type)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start, self.model_type)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
