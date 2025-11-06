# Copyright (c) OpenMMLab. All rights reserved.

# TODO: may consider separate similar to transformers
# - Internvl
#     - configuration_internvl.py
#     - modeling_internvl.py
#     - processing_internvl.py

# but this may bring too many files, so currently we just put all things together

from typing import Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.utils import disable_logging

from .base import BASE_MODELS, BaseModel

logger = get_logger('lmdeploy')


class InternVLImagesKwargs(ImagesKwargs, total=False):
    crop_to_patches: Optional[bool]
    min_patches: Optional[int]
    max_patches: Optional[int]


class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: InternVLImagesKwargs
    _defaults = {
        'text_kwargs': {
            'padding': False,
        },
        'images_kwargs': {
            'crop_to_patches': True,
        },
        'videos_kwargs': {},
    }


@BASE_MODELS.register_module()
class InternVL3VisionModel(BaseModel):
    """Internvl3 vision model."""

    _arch = ['InternVLForConditionalGeneration', 'InternS1ForConditionalGeneration']

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = ''):
        super().__init__(model_path, with_llm, max_memory, hf_config, backend)
        self.arch = hf_config.architectures[0]

    def build_preprocessor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer = self.processor.tokenizer
        self.image_token_id = tokenizer.context_image_token_id
        self.image_tokens_per_patch = self.processor.image_seq_length
        self.tokenizer_init_kwargs = tokenizer.init_kwargs

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        from accelerate import init_empty_weights
        with init_empty_weights():
            if self.arch == 'InternVLForConditionalGeneration':
                model = AutoModel.from_config(self.hf_config, trust_remote_code=True)
                # if not self.with_llm:
                #     print('delete language model')
                del model.language_model
            elif self.arch == 'InternS1ForConditionalGeneration':
                model = AutoModelForCausalLM.from_config(self.hf_config, trust_remote_code=True)
                # if not self.with_llm:
                #     print('delete language model')
                del model.model.language_model
            else:
                raise ValueError(f'unsupported model arch {self.arch}')

        model.half()
        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                #  device_map='auto' if not self.with_llm else {'': 'cpu'},
                device_map='auto',
                max_memory=self.max_memory,
                no_split_module_classes=['InternVLVisionLayer', 'InternS1VisionLayer'],
                dtype=torch.half)
        # We need eval mode to freeze the weights in model, thus,
        # avoid randomness in inference.
        self.model = model.eval()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refers to `super.preprocess() for spec."""
        from transformers.image_utils import make_flat_list_of_images
        output_kwargs = self.processor._merge_kwargs(
            InternVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer_init_kwargs,
            **{
                'return_tensors': 'pt',
                'add_special_tokens': False
            },
        )
        images = self.collect_images(messages)
        images = [image.convert('RGB') for image, _ in images]
        num_image = len(images)
        images = make_flat_list_of_images(images)
        image_inputs = self.processor.image_processor(images, **output_kwargs['images_kwargs'])
        image_num_patches = image_inputs.pop('num_patches').cpu().numpy().tolist()
        image_pixel_values = image_inputs.pop('pixel_values')
        outputs = []
        cum_num_patches = 0
        for idx in range(num_image):
            cur_num_patches = image_num_patches[idx]
            pixel_values = image_pixel_values[cum_num_patches:cum_num_patches + cur_num_patches, ...]
            cum_num_patches += cur_num_patches
            data = dict(pixel_values=pixel_values,
                        image_tokens=self.image_tokens_per_patch * cur_num_patches,
                        image_token_id=self.image_token_id)
            outputs.append(data)

        return outputs

    @torch.no_grad()
    def forward(self, processed_inputs: List[Dict]) -> torch.Tensor:
        # FIXME: consider batch?
        outputs = []
        pixel_values = [x['pixel_values'] for x in processed_inputs]
        split = [x.shape[0] for x in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)
        pixel_values = pixel_values.to(dtype=torch.float16)
        logger.info(f'vision forward shape: {pixel_values.shape}')
        feats = self.model.get_image_features(
            pixel_values,
            vision_feature_layer=self.hf_config.vision_feature_layer,
            vision_feature_select_strategy=self.hf_config.vision_feature_select_strategy,
        )
        feats = torch.split(feats, split, dim=0)
        outputs.extend([x.reshape(-1, x.shape[-1]) for x in feats])
        return outputs

    @staticmethod
    def proc_messages(
        messages,
        chat_template,
        sequence_start,
        tools: Optional[List[object]] = None,
        enable_thinking: Optional[bool] = None,
    ):
        """Apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['preprocess', 'forward']:
                continue
            n_images = len([1 for x in message['content'] if x['type'] == 'image'])
            content = [x.get('text', '') for x in message['content'] if x['type'] == 'text']
            prompt = content[0]
            if IMAGE_TOKEN in prompt and f'<img>{IMAGE_TOKEN}' not in prompt:
                prompt = prompt.replace(f'{IMAGE_TOKEN}', f'<img>{IMAGE_TOKEN}</img>')
                prompt = prompt.replace('</img><img>', '')
                prompt = prompt.replace('<img><img>', '<img>')
                prompt = prompt.replace('</img></img>', '</img>')
            elif IMAGE_TOKEN not in prompt:
                prompt = f'<img>{IMAGE_TOKEN * n_images}</img>\n' + prompt
            else:
                pass
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages,
                                               sequence_start,
                                               tools=tools,
                                               enable_thinking=enable_thinking)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   sequence_start,
                   tools: Optional[List[object]] = None,
                   enable_thinking: Optional[bool] = None,
                   **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages,
                                                 chat_template,
                                                 sequence_start,
                                                 tools=tools,
                                                 enable_thinking=enable_thinking)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self,
                     messages,
                     chat_template,
                     tokenizer,
                     sequence_start,
                     tools: Optional[List[object]] = None,
                     enable_thinking: Optional[bool] = None,
                     **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages,
                                                 chat_template,
                                                 sequence_start,
                                                 tools=tools,
                                                 enable_thinking=enable_thinking)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
