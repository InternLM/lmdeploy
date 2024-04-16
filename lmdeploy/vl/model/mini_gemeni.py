# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from typing import List

import torch
from PIL.Image import Image

from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import load_model_from_weight_files


def check_mini_gemini_install():
    """check mini gemini install."""
    try:
        import minigemini  # noqa: F401
    except ImportError:
        raise ImportError(
            'To use MiniGeminiVisionModel, please install minigemini by '
            'pip install git+https://github.com/dvlab-research/MiniGemini.git'
            ' --no-deps')


class MiniGeminiVisionModel(VisonModel):
    """Qwen vision model."""

    def __init__(self, model_path, device='cuda:0'):
        self.model_path = model_path
        self.device = device
        self.build_model()

    def build_model(self):
        check_mini_gemini_install()
        # empty init
        from accelerate import init_empty_weights
        from minigemini.mm_utils import process_images
        from minigemini.model import MiniGeminiLlamaForCausalLM
        with init_empty_weights():
            warnings.simplefilter('ignore')
            model = MiniGeminiLlamaForCausalLM.from_pretrained(self.model_path)
            del model.lm_head
            del model.model.embed_tokens
            del model.model.layers
            del model.model.norm

        # # load weight
        with torch.device('cpu'):
            model.to_empty(device='cpu')
            vision_tower = model.get_vision_tower()
            vision_tower.is_loaded = False
            vision_tower.load_model()
            vision_tower_aux = model.get_vision_tower_aux()
            vision_tower_aux.is_loaded = False
            vision_tower_aux.load_model()
        load_model_from_weight_files(model, self.model_path)
        model.to(self.device).eval().half()
        setattr(model.config, 'model_path', self.model_path)
        model.get_model().initialize_uni_modules(model.config, for_eval=True)

        self.model = model
        self.vision_tower = model.model.vision_tower
        self.mm_projector = model.model.mm_projector

        image_processor = self.vision_tower.image_processor
        if hasattr(model.config, 'image_size_aux'):
            if not hasattr(image_processor, 'image_size_raw'):
                image_processor.image_size_raw = image_processor.crop_size.copy(  # noqa
                )
            image_processor.crop_size['height'] = model.config.image_size_aux
            image_processor.crop_size['width'] = model.config.image_size_aux
            image_processor.size['shortest_edge'] = model.config.image_size_aux
        self.image_processor = image_processor
        self.process_images = process_images

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        outputs = [x.convert('RGB') for x in images]
        image_tensor = self.process_images(outputs, self.image_processor,
                                           self.model.config)
        image_grid = getattr(self.model.config, 'image_grid', 1)
        if hasattr(self.model.config, 'image_size_aux'):
            raw_shape = [
                self.image_processor.image_size_raw['height'] * image_grid,
                self.image_processor.image_size_raw['width'] * image_grid
            ]
            image_tensor_aux = image_tensor
            image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                           size=raw_shape,
                                                           mode='bilinear',
                                                           align_corners=False)
        else:
            image_tensor_aux = []

        if image_grid >= 2:
            raw_image = image_tensor.reshape(
                3, image_grid, self.image_processor.image_size_raw['height'],
                image_grid, self.image_processor.image_size_raw['width'])
            raw_image = raw_image.permute(1, 3, 0, 2, 4)
            raw_image = raw_image.reshape(
                -1, 3, self.image_processor.image_size_raw['height'],
                self.image_processor.image_size_raw['width'])

            if getattr(self.model.config, 'image_global', False):
                global_image = image_tensor
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(
                    global_image,
                    size=[
                        self.image_processor.image_size_raw['height'],
                        self.image_processor.image_size_raw['width']
                    ],
                    mode='bilinear',
                    align_corners=False)
                # [image_crops, image_global]
                raw_image = torch.cat([raw_image, global_image], dim=0)
            image_tensor = raw_image.contiguous()
            image_tensor = image_tensor.unsqueeze(0)

        if type(image_tensor) is list:
            image_tensor = [
                image.to(self.model.device, dtype=torch.float16)
                for image in image_tensor
            ]
            image_tensor_aux = [
                image.to(self.model.device, dtype=torch.float16)
                for image in image_tensor_aux
            ]
        else:
            image_tensor = image_tensor.to(self.model.device,
                                           dtype=torch.float16)
            image_tensor_aux = image_tensor_aux.to(self.model.device,
                                                   dtype=torch.float16)

        images_embeds = self.model.encode_images(image_tensor,
                                                 image_tensor_aux)

        outputs = torch.split(images_embeds, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs
