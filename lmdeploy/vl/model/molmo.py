# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class MolmoVisionModel(VisonModel):
    """molmo's vision model."""

    _arch = 'MolmoForCausalLM'

    def build_model(self):
        """Load model."""
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        with init_empty_weights():
            config = self.hf_config
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
            if not self.with_llm:
                # Remove nn modules other than embedding from the LLM model
                for key in ['emb_drop', 'ln_f', 'blocks', 'ff_out']:
                    del model.model.transformer[key]
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
                params: List[Dict] = None) -> List[Dict]:
        """forward the model with given input.

        Args:
            images (List): [None] it is not used
            params (List): the inputs after precessing GPT4V messages in
                `MolmoChatTemplateWrapper`. Its format is like the following:
                [[
                    {'role': 'user', 'content': 'user prompt'},
                    {'role': 'asssistant', 'content': 'assistant prompt'},
                    {'role': 'user', 'content': 'user prompt', 'images': [PIL image list]},
                    ...
                ]]
        """  # noqa

        messages = params[0]
        assert isinstance(messages, List)
        # append an assistant message to `messages`
        messages.append(dict(role='assistant', content=''))
        # results is a list of tuple(input_ids, embeddings)
        results = []
        # the concat prompt. It is not used during inference but to adhere the
        # interface definition of `_get_prompt_input` in `class VLAsyncEngine`
        prompts = ''
        # Prepend BOS
        # qwen2 and olmo do not have a BOS, and instead use EOS as a generic
        # separator token.
        bos = (self.processor.tokenizer.bos_token_id
               or self.processor.tokenizer.eos_token_id)
        results.append(([bos], None))
        for i, message in enumerate(messages):
            if 'images' in message.keys():
                prompts += ' User: ' + (IMAGE_TOKEN + '\n') * len(
                    message['images']) + message['content']
                prompt = f' User: {message["content"]}'
                tokens = self.processor.tokenizer.encode(
                    prompt, add_special_tokens=False)
                # preprocess images. The output is a dict
                inputs = self.processor.process(images=message['images'],
                                                tokens=tokens)
                inputs = {
                    k: v.to(self.model.device).unsqueeze(0)
                    for k, v in inputs.items()
                }
                input_ids = inputs['input_ids']
                # remove the bos from input_ids which is prepended by molmo's
                # processor
                input_ids = input_ids[:, 1:]
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
                embeddings[batch_idx[valid],
                           image_input_idx[valid]] += image_features[valid]
                assert embeddings.shape[:2] == (batch_size, seq_len)
                results.append((input_ids.flatten().tolist(), embeddings))
            else:
                role = message['role']
                content = message['content']
                assert isinstance(content, str)
                prompt = ''
                if role == 'user':
                    prompt = f' User: {content}'
                elif role == 'assistant':
                    prompt = f' Assistant:{content}'
                else:
                    assert 0, f'molmo does not support role {role}, message is {message}'  # noqa
                input_ids = self.processor.tokenizer.encode(
                    prompt, add_special_tokens=False)
                results.append((input_ids, None))
                prompts += prompt

        # concat input_ids from results, calculate the range in the input_ids
        # where embeddings will be copied to
        input_ids = []
        input_embeddings = []
        input_embedding_ranges = []
        start = 0
        for _input_ids, _embeddings in results:
            if _embeddings is not None:
                input_embeddings.append(_embeddings.cpu())
                end = start + len(_input_ids)
                input_embedding_ranges.append((start, end))
            input_ids += _input_ids
            start += len(_input_ids)
        return [
            dict(prompt=prompts,
                 input_ids=input_ids,
                 input_embeddings=input_embeddings,
                 input_embedding_ranges=input_embedding_ranges)
        ]
