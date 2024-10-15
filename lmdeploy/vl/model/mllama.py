# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch
from PIL.Image import Image

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


def check_transformers():
    """check qwen_vl_utils."""
    try:
        from transformers import MllamaForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError(
            'please install latest transformers by '
            'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class MllamaVLModel(VisonModel):
    """llama3.2 model."""

    _arch = 'MllamaForConditionalGeneration'

    def build_model(self):
        check_transformers()

        from accelerate import init_empty_weights
        with init_empty_weights():
            config = self.hf_config
            config.quantization_config = {}  # disable vision part quantization
            # disable accelerate check_tied_parameters_in_config
            # for Qwen2-VL-2B-Instruct
            config.tie_word_embeddings = False

            from transformers import MllamaForConditionalGeneration
            model = MllamaForConditionalGeneration._from_config(config)
            if not self.with_llm:
                del model.language_model
            else:
                self.vl_model = model
            model.half()

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                max_memory=self.max_memory,
                dtype=torch.bfloat16)

        self.model = model.eval()

        # processor
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.image_token_id = 128256

    @torch.no_grad()
    def forward(self,
                images: List[Image],
                params: List[Dict] = None) -> List[torch.Tensor]:
        """forward."""
        # only support image input
        if params is not None:
            assert len(images) == len(
                params), 'different length of images and params'
        else:
            params = [{}] * len(images)

        image_inputs = self.processor.image_processor(images=images,
                                                      return_tensors='pt')
        pixel_values = image_inputs['pixel_values'].to(
            self.model.vision_model.device)
        pixel_values = pixel_values.type(self.model.vision_model.dtype)
        aspect_ratio_ids = image_inputs['aspect_ratio_ids'].to(
            self.model.vision_model.device)
        aspect_ratio_mask = image_inputs['aspect_ratio_mask'].to(
            self.model.vision_model.device)
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True)
        cross_attention_states = vision_outputs[0]
        cross_attention_states = self.model.multi_modal_projector(
            cross_attention_states)
        _, bsz, _, _, image_token_dim = tuple(cross_attention_states.shape)
        cross_attention_states = cross_attention_states.view(
            bsz, -1, image_token_dim).split([1] * len(images))
        return cross_attention_states
