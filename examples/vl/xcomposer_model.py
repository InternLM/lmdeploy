import os
# from safetensors.torch import load_file
from collections.abc import Sequence
from glob import glob

import numpy as np
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lmdeploy.model import MODELS, BaseChatTemplate

meta_instruction = """meta instruction
You are an AI assistant whose name is 浦语.
- 浦语 is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- 浦语 can understand and communicate fluently in the language chosen by the user such as English and 中文.
conversation
"""  # noqa


@MODELS.register_module(name='internlm-xcomposer-7b')
class InternLMXComposerTemplate(BaseChatTemplate):
    """Internlm xcomposer chat template."""

    def __init__(self,
                 meta_instruction=meta_instruction,
                 user=' <|User|>: ',
                 assistant=' <|Bot|>: ',
                 eoh='<TOKENS_UNUSED_0>',
                 eoa='<TOKENS_UNUSED_1>',
                 stop_words=['<TOKENS_UNUSED_0>', '<TOKENS_UNUSED_1>'],
                 image_placeholder='<Img><ImageHere></Img>',
                 **kwargs):
        super().__init__(**kwargs)
        self.meta_instruction = meta_instruction
        self.user = user
        self.assistant = assistant
        self.eoh = eoh
        self.eoa = eoa
        self.stop_words = stop_words
        self.image_placeholder = image_placeholder

    def _concat_image_info(self, prompt):
        """Append image placeholder."""
        if isinstance(prompt, str):
            return prompt
        prompt, nimg = prompt
        assert nimg <= 1
        if nimg == 1:
            prompt = f'{self.image_placeholder}{prompt}'
        return prompt

    def get_prompt(self, prompt, sequence_start=True):
        """Apply chat template to prompt."""
        prompt = self._concat_image_info(prompt)
        return super().get_prompt(prompt, sequence_start)

    def messages2prompt(self, messages, sequence_start=True):
        """Apply chat template to history."""
        if isinstance(messages, str) or isinstance(messages[0], str):
            return self.get_prompt(messages, sequence_start)
        box_map = dict(user=self.user,
                       assistant=self.assistant,
                       system=self.system)
        eox_map = dict(user=self.eoh,
                       assistant=self.eoa + self.stop_word_suffix,
                       system=self.eosys)
        ret = ''
        if self.meta_instruction is not None:
            if len(messages) and messages[0]['role'] != 'system':
                ret += f'{self.system}{self.meta_instruction}{self.eosys}'
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'user' and not isinstance(content, str):
                assert isinstance(content, Sequence)
                assert all(isinstance(item, dict) for item in content)
                content = [content[0]['text'], len(content) - 1]
            content = self._concat_image_info(content)
            ret += f'{box_map[role]}{content}{eox_map[role]}'
        ret += f'{self.assistant}'
        return ret


class InternLMXComposer:
    """Internlm-xcomposer preprocessor to prepare the inputs for a model."""

    def __init__(self, pretrained_model_name_or_path, **kwargs):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.decorator = InternLMXComposerTemplate(**kwargs)
        self._load_model()

    def _load_model(self):
        path = self.pretrained_model_name_or_path
        if not os.path.exists(path):
            path = snapshot_download(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        with init_empty_weights():
            config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            config.num_hidden_layers = 0  # speedup
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
            model.internlm_model = None
            model.to_empty(device='cpu')
            named_parameters = set()
            for key, _ in model.named_parameters():
                named_parameters.add(key)
            # TODO: load bin according to index.json
            bins = glob(os.path.join(path, '*.bin'))
            # bins = glob(os.path.join(path, '*.safetensors'))
            for bin in bins:
                dt = torch.load(bin, map_location='cpu')
                # dt = load_file(bin)
                missed, _ = model.load_state_dict(dt, strict=False)
                named_parameters.difference_update(set(missed))
            assert len(
                named_parameters) == 0, f'missing keys: {named_parameters}'
            self.model = model.to('cuda').eval()

    @torch.no_grad()
    def encode_img(self, paths):
        """Extract image features."""
        if len(paths) == 0:
            return None
        features = []
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for path in paths:
                out = self.model.encode_img(path)
                features.append(out.squeeze().cpu().numpy())
        return features

    def _to_inputs(self, decorate_text, image_paths, sequence_start):
        features = self.encode_img(image_paths)
        input_ids = []
        ranges = None
        begins = []
        segs = decorate_text.split(self.decorator.image_placeholder)
        image_dim = features[-1].shape[0] if features is not None else 0
        for i, seg in enumerate(segs):
            if i > 0:
                begins.append(len(input_ids))
                input_ids.extend([0] * image_dim)
            seg_ids = self.tokenizer.encode(
                seg, add_special_tokens=((i == 0) and sequence_start))
            input_ids.extend(seg_ids)
        if features is not None:
            ends = np.array(begins) + image_dim
            ranges = np.stack([begins, ends], axis=1).tolist()
        return input_ids, features, ranges

    def prepare_query(self, query, sequence_start=True):
        """Convert query to input_ids, features and the ranges of features to
        input_ids."""
        image_paths = []
        if not isinstance(query, str):
            query, image_paths = query[0], query[1:]
            if len(image_paths) > 1:
                print('does not support multiple images, use last one.')
                image_paths = image_paths[-1:]
        decorate_text = self.decorator.get_prompt((query, len(image_paths)))
        return self._to_inputs(decorate_text, image_paths, sequence_start)

    def prepare_message(self, messages):
        """Convert messages to input_ids, features and the ranges of features
        to input_ids."""
        decorate_text = self.decorator.messages2prompt(messages, True)
        image_paths = []
        for msg in messages:
            if msg['role'] == 'user':
                content = msg['content']
                if isinstance(content, str):
                    continue
                for item in content:
                    if item['type'] == 'image_url':
                        url = item['image_url']['url']
                        image_paths.append(url)
        return self._to_inputs(decorate_text, image_paths, True)
