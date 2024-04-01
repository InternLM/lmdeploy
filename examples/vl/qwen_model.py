import os
from glob import glob

import numpy as np
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lmdeploy.model import MODELS, Qwen7BChat


@MODELS.register_module(name='qwen-vl-chat')
class QwenVLChatTemplate(Qwen7BChat):
    """Qwen vl chat template."""

    def __init__(self,
                 session_len=8192,
                 top_p=0.3,
                 top_k=None,
                 temperature=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.session_len = session_len
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

    def _concat_image_info(self, prompt):
        """Append image placeholder."""
        if isinstance(prompt, str):
            return prompt
        prompt, nimg = prompt
        res = ''
        for i in range(nimg):
            res += f'Picture {str(i)}:<img>placeholder</img>\n'
        prompt = res + prompt
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
                       assistant=self.eoa + self.separator,
                       system=self.eosys)
        ret = ''
        if self.meta_instruction is not None:
            if len(messages) and messages[0]['role'] != 'system':
                ret += f'{self.system}{self.meta_instruction}{self.eosys}'
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'user' and not isinstance(content, str):
                content = [content[0]['text'], len(content) - 1]
                content = self._concat_image_info(content)
            ret += f'{box_map[role]}{content}{eox_map[role]}'
        ret += f'{self.assistant}'
        return ret


class QwenVLChat:
    """Qwen vl preprocessor to prepare the inputs for a model."""

    def __init__(self, pretrained_model_name_or_path, **kwargs):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.decorator = QwenVLChatTemplate(**kwargs)
        self._load_model()

    def _load_model(self):
        path = self.pretrained_model_name_or_path
        if not os.path.exists(path):
            path = snapshot_download(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        with init_empty_weights():
            config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
            del model.lm_head
            for key in ['wte', 'h', 'ln_f']:
                setattr(model.transformer, key, None)
            model.to_empty(device='cpu')
            named_parameters = set()
            for key, _ in model.named_parameters():
                named_parameters.add(key)
            # TODO: load bin according to index.json
            bins = glob(os.path.join(path, '*.bin'))
            for bin in bins:
                dt = torch.load(bin, map_location='cpu')
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
        # with torch.cuda.amp.autocast(dtype=torch.float16):
        features = self.model.transformer.visual.encode(paths).float()
        features = [x.cpu().numpy() for x in features]
        return features

    def _to_inputs(self, decorate_text, image_paths, sequence_start):
        features = self.encode_img(image_paths)
        input_ids = self.tokenizer.encode(decorate_text)
        ranges = None
        if features is not None:
            input_ids_arr = np.array(input_ids)
            begins = np.where(
                input_ids_arr == self.tokenizer.img_start_id)[0] + 1
            ends = np.where(input_ids_arr == self.tokenizer.img_end_id)[0]
            ranges = np.stack([begins, ends], axis=1)
            assert len(features) == len(ranges)
        return input_ids, features, ranges

    def prepare_query(self, query, sequence_start=True):
        """Convert query to input_ids, features and the ranges of features to
        input_ids."""
        image_paths = []
        if not isinstance(query, str):
            query, image_paths = query[0], query[1:]
        decorate_text = self.decorator.get_prompt((query, len(image_paths)),
                                                  sequence_start)
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
