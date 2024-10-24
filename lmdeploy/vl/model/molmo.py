# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import numpy as np
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
                no_split_module_classes=[
                    'ResidualAttentionBlock', 'Embedding'
                ],
                dtype=torch.half)

        # We need eval mode to freeze the weights in model, thus,
        # avoid randomness in inference.
        self.model = model.eval()
        self.config = config
        # TODO: get embedding model

        processor = AutoProcessor.from_pretrained(self.model_path,
                                                  trust_remote_code=True,
                                                  torch_dtype='auto',
                                                  device_map='auto')
        self.image_processor = processor.image_processor

    def preprocess(self, images: List[Image], params: List[Dict] = None):
        images = [np.array(x.convert('RGB')) for x in images]
        image_idx = [-1] * len(images)

        DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
        DEFAULT_IM_START_TOKEN = '<im_start>'
        DEFAULT_IM_END_TOKEN = '<im_end>'
        DEFAULT_IM_COL_TOKEN = '<im_col>'

        image_patch_token_id = self.image_processor.special_token_ids[
            DEFAULT_IMAGE_PATCH_TOKEN]
        image_col_token_id = self.image_processor.special_token_ids[
            DEFAULT_IM_COL_TOKEN]
        image_start_token_id = self.image_processor.special_token_ids[
            DEFAULT_IM_START_TOKEN]
        image_end_token_id = self.image_processor.special_token_ids[
            DEFAULT_IM_END_TOKEN]
        out = self.image_processor.multimodal_preprocess(
            images=images,
            image_idx=image_idx,
            tokens=np.asarray([]).astype(np.int32),
            sequence_length=0,  # unused parameter
            image_patch_token_id=image_patch_token_id,
            image_col_token_id=image_col_token_id,
            image_start_token_id=image_start_token_id,
            image_end_token_id=image_end_token_id,
        )
        return out

    @torch.no_grad()
    def forward(self,
                images: List[Image],
                params: List[Dict] = None) -> List[torch.Tensor]:
        self.preprocess(images)
        # return self._forward_func(images, params)
