# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import numpy as np

from lmdeploy.pytorch.check_env import try_import_deeplink
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX, IMAGE_TOKEN
from lmdeploy.vl.engine import ImageEncoder
from lmdeploy.vl.templates import VLPromptType, get_vl_prompt_template

logger = get_logger('lmdeploy')


class VLAsyncEngine(AsyncEngine):
    """Visual Language Async inference engine."""

    def __init__(self, model_path: str, **kwargs) -> None:
        vision_config = kwargs.pop('vision_config', None)
        backend_config = kwargs.get('backend_config', None)
        if kwargs.get('backend', '') == 'pytorch':
            try_import_deeplink(backend_config.device_type)
        self.vl_encoder = ImageEncoder(model_path,
                                       vision_config,
                                       backend_config=backend_config)
        super().__init__(model_path, **kwargs)
        if self.model_name == 'base':
            raise RuntimeError(
                'please specify chat template as guided in https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html#set-chat-template'  # noqa: E501
            )
        self.vl_prompt_template = get_vl_prompt_template(
            model_path, self.chat_template, self.model_name)

    def _convert_prompts(self,
                         prompts: Union[VLPromptType, List[Dict],
                                        List[VLPromptType], List[List[Dict]]]):
        """convert prompts to openai format."""
        if isinstance(prompts, str) or isinstance(prompts, tuple):
            _prompts = self.vl_prompt_template.prompt_to_messages(prompts)
        elif isinstance(prompts[0], tuple) or isinstance(prompts[0], str):
            _prompts = [
                self.vl_prompt_template.prompt_to_messages(x) for x in prompts
            ]
        else:
            _prompts = prompts
        return _prompts

    async def _get_prompt_input(self,
                                prompt: Dict,
                                do_preprocess: bool,
                                sequence_start: bool,
                                adapter_name: str,
                                tools: Optional[List[object]] = None,
                                **kwargs):
        """get input_ids, embeddings and offsets."""
        if do_preprocess:
            decorated = self.vl_prompt_template.messages2prompt(
                prompt, sequence_start)
        else:
            decorated = prompt
        segs = decorated.split(IMAGE_TOKEN)

        results = {}
        input_ids = []
        from lmdeploy.vl.templates import (MllamaTempateWrapper,
                                           MolmoChatTemplateWrapper,
                                           Qwen2VLChatTemplateWrapper)
        ranges = None
        grid_thws = None
        if len(segs) > 1:
            # yapf: disable
            images_with_kwargs = await self.vl_prompt_template.async_collect_pil_images(prompt)  # noqa: E501
            # yapf: enable
            features = []
            if len(images_with_kwargs) > 0:
                images, image_kwargs = list(zip(*images_with_kwargs))
                features = await self.vl_encoder.async_infer(
                    images, image_kwargs)

                from lmdeploy.vl.templates import MiniCPMVTempateWrapper
                if isinstance(self.vl_prompt_template, MiniCPMVTempateWrapper):
                    decorated, features = self.vl_prompt_template.update_image_token(  # noqa: E501
                        decorated, features)
                    segs = decorated.split(IMAGE_TOKEN)

                if isinstance(self.vl_prompt_template,
                              Qwen2VLChatTemplateWrapper):
                    grid_thws = [x['grid_thw'] for x in features]
                    features = [x['embeddings'] for x in features]

                if isinstance(self.vl_prompt_template, MllamaTempateWrapper):
                    # llama3.2 just encode <|image|> and inference
                    decorated = decorated.replace(IMAGE_TOKEN, '<|image|>')
                    input_ids = self.tokenizer.encode(decorated,
                                                      add_bos=sequence_start)
                    results['input_ids'] = input_ids
                    results['prompt'] = decorated
                    assert len(features)
                    results['cross_attention_states'] = features[0]
                    return results

                if isinstance(self.vl_prompt_template,
                              MolmoChatTemplateWrapper):
                    return features[0]

            features = [x.cpu().numpy() for x in features]
            input_ids = []
            begins = []
            ends = []
            if len(segs) != len(features) + 1:
                logger.error(
                    f'the number of {IMAGE_TOKEN} is not equal '
                    f'to input images, {len(segs) - 1} vs {len(features)}')
                features = features[:len(segs) - 1]
            for i, seg in enumerate(segs):
                if i > 0 and i <= len(features):
                    image_dim = features[i - 1].shape[0]
                    begins.append(len(input_ids))
                    ends.append(begins[-1] + image_dim)
                    input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_dim)
                seg_ids = self.tokenizer.encode(seg,
                                                add_bos=((i == 0)
                                                         and sequence_start))
                input_ids.extend(seg_ids)
            ranges = np.stack([begins, ends], axis=1).tolist()
            results['input_embeddings'] = features or None
            results['input_embedding_ranges'] = ranges or None
        else:
            input_ids = self.tokenizer.encode(decorated,
                                              add_bos=sequence_start)

        if isinstance(self.vl_prompt_template, Qwen2VLChatTemplateWrapper):
            # TODO: refactor _get_prompt_input function
            mrope_position_ids, mrope_position_delta = \
                self.vl_prompt_template.get_mrope_info(
                    len(input_ids), grid_thws=grid_thws,
                    embedding_ranges=ranges)
            results['mrope_position_ids'] = mrope_position_ids
            results['mrope_position_delta'] = mrope_position_delta

        results['input_ids'] = input_ids
        results['prompt'] = decorated
        return results

    def batch_infer(self, prompts: Union[VLPromptType, List[Dict],
                                         List[VLPromptType], List[List[Dict]]],
                    **kwargs):
        """Inference a batch of prompts."""
        prompts = self._convert_prompts(prompts)
        return super().batch_infer(prompts, **kwargs)

    def stream_infer(self, prompts: Union[VLPromptType, List[Dict],
                                          List[VLPromptType],
                                          List[List[Dict]]], **kwargs):
        """Inference a batch of prompts with stream mode."""
        prompts = self._convert_prompts(prompts)
        return super().stream_infer(prompts, **kwargs)

    def __call__(self, prompts: Union[VLPromptType, List[Dict],
                                      List[VLPromptType], List[List[Dict]]],
                 **kwargs):
        """Inference a batch of prompts."""
        return super().__call__(prompts, **kwargs)

    def chat(self, prompts: VLPromptType, **kwargs):
        """chat."""
        _prompts = self._convert_prompts(prompts)
        sess = super().chat(_prompts, **kwargs)

        # recover prompts & history
        sess._prompt = prompts
        last_round = sess.history[-1]
        sess.history[-1] = (prompts, last_round[-1])
        return sess
