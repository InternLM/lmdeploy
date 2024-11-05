# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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
        # import pdb; pdb.set_trace()
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        with init_empty_weights():
            config = self.hf_config
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
            if not self.with_llm:
                for key in ['emb_drop', 'ln_f', 'blocks', 'ff_out']:
                    del model.model.transformer[key]
                # get `wte.new_embedding` parameters, which will be
                # used to perform image token embbeding later on
                self.token_embedding = model.model.transformer.wte
            else:
                self.vl_model = model

        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                max_memory=self.max_memory,
                no_split_module_classes=[
                    'ResidualAttentionBlock', 'Embedding'
                ])

        # We need eval mode to freeze the weights in model, thus,
        # avoid randomness in inference.
        self.model = model.eval()
        self.config = config

        self.processor = AutoProcessor.from_pretrained(self.model_path,
                                                       trust_remote_code=True,
                                                       torch_dtype='auto',
                                                       device_map='auto')

    @torch.no_grad()
    def forward(self,
                images: List[Image],
                params: List[Dict] = None) -> List[torch.Tensor]:
        """forward the model with given input.

        Args:
            images (List): [None]
            messages (List):
        """

        messages = params[0]
        assert isinstance(messages, List)

        results = []
        prompts = ''
        for message in messages:
            if 'images' in message.keys():
                # preprocess images. The output is a dict
                inputs = self.processor.process(images=message['images'],
                                                text=message['content'])
                inputs = {
                    k: v.to(self.model.device).unsqueeze(0)
                    for k, v in inputs.items()
                }
                input_ids = inputs['input_ids']
                images = inputs[
                    'images']  # (batch_size, num_image, num_patch, d_model)
                image_input_idx = inputs[
                    'image_input_idx']  # (batch_size, num_image, num_patch)
                image_masks = inputs['image_masks']
                batch_size, seq_len = input_ids.size()
                assert batch_size == 1

                # Get embeddings of input.
                if input_ids is not None:
                    input_ids = input_ids * (input_ids != -1).to(
                        input_ids.dtype)
                embeddings = self.model.model.transformer.wte(input_ids)
                image_features, _ = self.model.model.vision_backbone(
                    images, image_masks)
                num_image, num_patch = image_features.shape[1:3]
                assert image_input_idx.shape == (batch_size, num_image,
                                                 num_patch)

                # insert the image feature into the embedding.
                image_features = image_features.view(batch_size,
                                                     num_image * num_patch, -1)
                image_input_idx = image_input_idx.view(batch_size,
                                                       num_image * num_patch)

                valid = image_input_idx >= 0
                batch_idx = torch.arange(batch_size, device=embeddings.device)
                batch_idx = torch.tile(batch_idx[:, None],
                                       [1, image_features.shape[1]])
                image_features = image_features.to(embeddings.device)
                # print(f'>> molmo forward image ...')
                # print(f'image_features.shape: {image_features.shape}')
                # print(f'image_input_idx.shape: {image_input_idx.shape}')
                # print(f'batch_idx[valid]: {batch_idx[valid]}')
                embeddings[batch_idx[valid],
                           image_input_idx[valid]] += image_features[valid]
                results.append(input_ids.flatten().tolist(),
                               embeddings.flatten())
            else:
                role = message['role']
                content = message['content']
                assert isinstance(content, str)
                prompt = ''
                if role == 'user':
                    prompt = f'User: {content} '
                elif role == 'assistant':
                    prompt = f'Assistant:{content}'
                else:
                    assert 0, f'molmo does not support role {role}, message is {message}'  # noqa
                input_ids = self.processor.tokenizer.encode(
                    prompt, add_special_tokens=False)
                results.append((input_ids, None))
                prompts += prompt
        # concat input_ids from results, calculate the range in the input_ids
        # where embeddings will be copied to
        # import pdb; pdb.set_trace()
        input_ids = []
        input_embeddings = []
        input_embedding_ranges = []
        for result in results:
            input_ids += result[0]
            if results[1] is not None:
                input_embeddings.append(results[1])
                start = len(input_ids)
                end = start + result[1].shape[0]
                input_embedding_ranges.append((start, end))
        return (prompts, input_ids, input_embeddings, input_embedding_ranges)
