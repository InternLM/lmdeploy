import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from xcomposer_model import InternLMXComposerTemplate  # noqa

model = AutoModel.from_pretrained('internlm/internlm-xcomposer-7b',
                                  trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer-7b',
                                          trust_remote_code=True)

internlm_model = model.internlm_model

lora_layers = [
    'self_attn.q_proj', 'self_attn.v_proj', 'mlp.down_proj', 'mlp.up_proj'
]


def get_attr(m, key):
    keys = key.split('.')
    for key in keys:
        m = getattr(m, key)
    return m


# merge lora
for i in range(len(internlm_model.model.layers)):
    layer = internlm_model.model.layers[i]
    for key in lora_layers:
        lora_linear = get_attr(layer, key)
        lora_b = lora_linear.lora_B
        lora_a = lora_linear.lora_A
        w_ba = torch.matmul(lora_b.weight, lora_a.weight)
        lora_linear.weight.data += w_ba.data

# save model
cur_folder = Path(__file__).parent
dst_path = os.path.join(cur_folder, 'internlm_model')
internlm_model.save_pretrained(dst_path)
tokenizer.save_pretrained(dst_path)
