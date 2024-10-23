# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch
from PIL.Image import Image
from transformers import AutoModel, AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class MolmoVisionModel(VisonModel):
    """molmo's vision model."""

    _arch = 'MolmoForCausalLM'

    def build_model(self):
        """Load model."""
        from accelerate import init_empty_weights
        with init_empty_weights():
            config = self.hf_config
            model = AutoModel.from_config(config, trust_remote_code=True)
            if not self.with_llm:
                for key in ['emb_drop', 'ln_f', 'blocks']:
                    del model.model.transformer[key]
                # get `wte.new_embedding` parameters, which will be
                # used to perform image token embbeding later on
                self.token_embedding = model.model.transformer.wte
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
                no_split_module_classes=['ResidualAttentionBlock'],
                dtype=torch.half)

        # We need eval mode to freeze the weights in model, thus,
        # avoid randomness in inference.
        self.model = model.eval()
        self.config = config
        # TODO: get embedding model

        self.image_processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto')

    def preprocess(self, images: List[Image], params: List[Dict] = None):
        # TODO
        pass

    @torch.no_grad()
    def forward(self,
                images: List[Image],
                params: List[Dict] = None) -> List[torch.Tensor]:
        # TODO
        images = [x.convert('RGB') for x in images]
        self.preprocess(images)
        # return self._forward_func(images, params)
