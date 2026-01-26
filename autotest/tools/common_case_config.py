TURBOMIND = 'turbomind'
PYTORCH = 'pytorch'

TURBOMIND_PR_TEST_LLM_GPU2 = [{
    'model': 'internlm/internlm2_5-20b-chat',
    'backend': TURBOMIND,
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'internlm/internlm2_5-20b-chat-inner-4bits',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}]

TURBOMIND_PR_TEST_LLM_GPU1 = [{
    'model': 'OpenGVLab/InternVL3-8B',
    'backend': TURBOMIND,
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3-8B',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_PR_TEST_MLLM_GPU1 = [{
    'model': 'liuhaotian/llava-v1.6-vicuna-7b',
    'backend': TURBOMIND,
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL2-4B',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3-8B',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_FALLBACK_TEST_LLM_GPU1 = [{
    'model': 'microsoft/Phi-4-mini-instruct',
    'backend': TURBOMIND,
    'communicator': 'cuda-ipc',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_FALLBACK_TEST_LLM_GPU2 = [{
    'model': 'google/gemma-2-27b-it',
    'backend': TURBOMIND,
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'deepseek-ai/deepseek-moe-16b-chat',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_FALLBACK_TEST_MLLM_GPU1 = [{
    'model': 'microsoft/Phi-4-mini-instruct',
    'backend': TURBOMIND,
    'communicator': 'cuda-ipc',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'THUDM/glm-4v-9b',
    'backend': TURBOMIND,
    'communicator': 'cuda-ipc',
    'quant_policy': 4,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'THUDM/glm-4v-9b-inner-4bits',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL2-4B',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_LOGPROBS_TEST_LLM_GPU2 = [{
    'model': 'internlm/internlm2_5-20b-chat',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3-38B',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}]

BASE_MODELSCOPE_CONFIG = [{
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {},
    'env': {
        'LMDEPLOY_USE_MODELSCOPE': 'True'
    }
}, {
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {},
    'env': {
        'LMDEPLOY_USE_MODELSCOPE': 'True'
    }
}]

MODELSCOPE_CONFIG = [{
    **item, 'backend': TURBOMIND
} for item in BASE_MODELSCOPE_CONFIG] + [{
    **item, 'backend': PYTORCH
} for item in BASE_MODELSCOPE_CONFIG]

PYTORCH_LORA_TEST_LLM_GPU1 = [{
    'model': 'meta-llama/Llama-2-7b-chat-hf',
    'backend': PYTORCH,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'adapters': ['lora/Llama2-Chinese-7b-Chat-LoRA']
    }
}]

PYTORCH_LORA_TEST_LLM_GPU2 = [{
    'model': 'baichuan-inc/Baichuan2-13B-Chat',
    'backend': PYTORCH,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'adapters': ['a=lora/2024-01-25_self_dup', 'b=lora/2024-01-25_self']
    }
}]

PYTORCH_PR_TEST_LLM_GPU2 = [{
    'model': 'internlm/internlm2_5-20b-chat',
    'backend': PYTORCH,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'backend': PYTORCH,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}]

PYTORCH_PR_TEST_LLM_GPU1 = [{
    'model': 'meta-llama/Meta-Llama-3-1-8B-Instruct',
    'backend': PYTORCH,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3-8B',
    'backend': PYTORCH,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

BASE_TOOLCALL_TEST_LLM = [{
    'model': 'internlm/internlm2_5-7b-chat',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'tool-call-parser': 'internlm'
    }
}, {
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'tool-call-parser': 'qwen'
    }
}, {
    'model': 'internlm/internlm2_5-20b-chat',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'tool-call-parser': 'internlm'
    }
}, {
    'model': 'meta-llama/Meta-Llama-3-1-70B-Instruct',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 4
    },
    'extra_params': {
        'tool-call-parser': 'llama3'
    }
}, {
    'model': 'Qwen/Qwen2.5-72B-Instruct',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 4
    },
    'extra_params': {
        'tool-call-parser': 'qwen'
    }
}]

BASE_REASONING_TEST_LLM = [{
    'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'reasoning-parser': 'deepseek-r1'
    }
}, {
    'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'reasoning-parser': 'deepseek-r1'
    }
}]

TOOLCALL_TEST_LLM = [{
    **item, 'backend': TURBOMIND
} for item in BASE_TOOLCALL_TEST_LLM] + [{
    **item, 'backend': PYTORCH
} for item in BASE_TOOLCALL_TEST_LLM]

REASONING_TEST_LLM = [{
    **item, 'backend': TURBOMIND
} for item in BASE_REASONING_TEST_LLM] + [{
    **item, 'backend': PYTORCH
} for item in BASE_REASONING_TEST_LLM]
