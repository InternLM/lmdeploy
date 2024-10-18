# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch
from PIL.Image import Image

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


def check_qwen_vl_deps_install():
    """check qwen_vl_utils."""
    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        raise ImportError(
            'please install qwen_vl_utils by pip install qwen_vl_utils'  # noqa: E501
        )
    try:
        from transformers import Qwen2VLForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError(
            'please install latest transformers by '
            'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen2VLModel(VisonModel):
    """Qwen2VL model."""

    _arch = 'Qwen2VLForConditionalGeneration'

    def build_model(self):
        check_qwen_vl_deps_install()

        from accelerate import init_empty_weights
        with init_empty_weights():
            config = self.hf_config
            config.quantization_config = {}  # disable vision part quantization
            # disable accelerate check_tied_parameters_in_config
            # for Qwen2-VL-2B-Instruct
            config.tie_word_embeddings = False

            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration._from_config(config)
            if not self.with_llm:
                del model.model
                del model.lm_head
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
                no_split_module_classes=['Qwen2VLVisionBlock'],
                dtype=torch.half)

        self.model = model.eval()

        # processor
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)

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

        from qwen_vl_utils import process_vision_info
        images = [x.convert('RGB') for x in images]
        content = []
        optional_keys = [
            'resized_height', 'resized_width', 'min_pixels', 'max_pixels'
        ]
        for image, param in zip(images, params):
            item = dict(type='image', image=image)
            item.update({k: param[k] for k in optional_keys if k in param})
            content.append(item)
        messages = [dict(content=content)]
        image_inputs, _ = process_vision_info(messages)
        image_inputs = self.processor.image_processor(images=image_inputs,
                                                      videos=None,
                                                      return_tensors='pt')
        pixel_values = image_inputs['pixel_values'].to(
            self.model.visual.get_device())
        image_grid_thw = image_inputs['image_grid_thw'].to(
            self.model.visual.get_device())
        pixel_values = pixel_values.type(self.model.visual.get_dtype())
        image_embeds = self.model.visual(pixel_values,
                                         grid_thw=image_grid_thw).cpu()
        merge_length = self.processor.image_processor.merge_size**2
        split_size = image_inputs['image_grid_thw'].prod(dim=1) // merge_length
        image_embeds = image_embeds.split(split_size.tolist())

        outputs = []
        for i, embeddings in enumerate(image_embeds):
            outputs.append(
                dict(embeddings=embeddings,
                     grid_thw=image_inputs['image_grid_thw'][i].tolist()))
        return outputs
