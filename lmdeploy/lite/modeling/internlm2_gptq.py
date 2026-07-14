# Copyright (c) OpenMMLab. All rights reserved.
from auto_gptq.modeling import BaseGPTQForCausalLM


class InternLM2GPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = 'InternLM2DecoderLayer'
    layers_block_name = 'model.layers'
    outside_layer_modules = ['model.tok_embeddings', 'model.norm']
    inside_layer_modules = [
        ['attention.wqkv'],
        ['attention.wo'],
        ['feed_forward.w3', 'feed_forward.w1'],
        ['feed_forward.w2'],
    ]
