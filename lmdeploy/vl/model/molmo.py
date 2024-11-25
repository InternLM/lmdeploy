# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class MolmoVisionModel(VisonModel):
    """molmo's vision model."""

    _arch = 'MolmoForCausalLM'

    def build_preprocessor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path,
                                                       trust_remote_code=True,
                                                       torch_dtype='auto',
                                                       device_map='auto')

    def build_model(self):
        """Load model."""
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(self.hf_config,
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

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refer to the `super.preprocess() for spec."""
        for i, message in enumerate(messages):
            if not isinstance(message['content'], List):
                continue
            images = [
                x['image'].convert('RGB') for x in message['content']
                if x['type'] == 'image'
            ]
            content = [
                x['text'] for x in message['content'] if x['type'] == 'text'
            ]
            prompt = f' User: {content[0]}'
            tokens = self.processor.tokenizer.encode(prompt,
                                                     add_special_tokens=False)
            # preprocess images. The output is a dict, which is
            # {
            #     'input_ids': torch.Tensor,
            #     'images': torch.Tensor, # (n_patch, d_model)
            #     'image_input_idx': torch.Tensor, # (n_patch, d_model)
            #     'image_masks': torch.Tensor,  # (n_patch, d_model)
            # }
            result = self.processor.process(images=images, tokens=tokens)
            # remove the bos from input_ids which is prepended by molmo's
            # processor
            input_ids = result['input_ids'][1:]
            result.update(input_ids=input_ids)
            messages[i].update(preprocess=result)
        return messages

    @torch.no_grad()
    def forward(self, messages: List[Dict]) -> List[Dict]:
        """extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the outputs of `preprocess`
        Return:
            the message list with forwarding results included
        """
        for i, message in enumerate(messages):
            if 'preprocess' not in message.keys():
                continue
            inputs = message['preprocess']
            # get input_ids of embedding
            inputs = {
                k: v.to(self.model.device).unsqueeze(0)
                for k, v in inputs.items()
            }
            input_ids = inputs['input_ids']
            # (batch_size, num_image, num_patch, d_model)
            images = inputs['images']
            # (batch_size, num_image, num_patch)
            image_input_idx = inputs['image_input_idx']
            image_masks = inputs['image_masks']
            batch_size, seq_len = input_ids.size()
            assert batch_size == 1
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
            embeddings = self.model.model.transformer.wte(input_ids)
            image_features, _ = self.model.model.vision_backbone(
                images, image_masks)
            num_image, num_patch = image_features.shape[1:3]
            assert image_input_idx.shape == (batch_size, num_image, num_patch)

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
            messages[i].update(
                dict(forward=dict(input_ids=input_ids.flatten(),
                                  embeddings=embeddings)))
        return messages

    def proc_messages(cls, messages):
        prompt = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            role, content = message['role'], message['content']
            if role == 'images':
                continue
            if isinstance(content, List):
                n_images = len([1 for x in content if x['type'] == 'image'])
                content = [x['text'] for x in content if x['type'] == 'text']
                prompt.append(' User: ' + (IMAGE_TOKEN + '\n') * n_images +
                              content[0])
            else:
                if role == 'user':
                    prompt.append(f' User: {content}')
                elif role == 'assistant':
                    prompt.append(f' Assistant:{content}')
                else:
                    assert 0, f'molmo does not support role {role}, message is {message}'  # noqa
        prompt.append(' Assistant:')
        return ''.join(prompt)

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        assert 0, 'molmo is not supported by pytorch engine'

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        # results is a list of tuple(input_ids, embeddings)
        results = []
        # Prepend BOS
        # qwen2 and olmo do not have a BOS, and instead use EOS as a generic
        # separator token.
        bos = (self.processor.tokenizer.bos_token_id
               or self.processor.tokenizer.eos_token_id)
        results.append(([bos], None))

        for i, message in enumerate(messages):
            role, content = message['role'], message['content']
            if role == 'images':
                continue
            if isinstance(content, List):
                forward_result = message.pop('forward')
                input_ids = forward_result['input_ids']
                embeddings = forward_result['embeddings']
                results.append((input_ids.tolist(), embeddings))
            else:
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

        prompt = self.proc_messages(messages)
        return dict(prompt=prompt,
                    input_ids=input_ids,
                    input_embeddings=input_embeddings,
                    input_embedding_ranges=input_embedding_ranges)
