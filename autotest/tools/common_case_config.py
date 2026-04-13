TURBOMIND_PR_TEST_LLM_GPU2 = [{
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}]

TURBOMIND_PR_TEST_LLM_GPU1 = [{
    'model': 'Qwen/Qwen3-0.6B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-0.6B-inner-4bits',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-8B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_PR_TEST_MLLM_GPU1 = [{
    'model': 'OpenGVLab/InternVL3-8B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3-8B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_PR_TEST_MLLM_GPU2 = [{
    'model': 'OpenGVLab/InternVL3_5-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3_5-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}]

TURBOMIND_FALLBACK_TEST_LLM_GPU1 = [{
    'model': 'THUDM/cogvlm-chat-hf',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'microsoft/Phi-3.5-vision-instruct',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_FALLBACK_TEST_LLM_GPU2 = [{
    'model': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_FALLBACK_TEST_MLLM_GPU1 = [{
    'model': 'THUDM/glm-4v-9b',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 4,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'THUDM/glm-4v-9b',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_LOGPROBS_TEST_LLM_GPU2 = [{
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3-38B',
    'backend': 'turbomind',
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
    **item, 'backend': 'turbomind'
} for item in BASE_MODELSCOPE_CONFIG] + [{
    **item, 'backend': 'pytorch'
} for item in BASE_MODELSCOPE_CONFIG]

PYTORCH_LORA_TEST_LLM_GPU1 = [{
    'model': 'meta-llama/Llama-2-7b-chat-hf',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'adapters': {
            'default': 'lora/Llama2-Chinese-7b-Chat-LoRA'
        }
    }
}]

PYTORCH_LORA_TEST_LLM_GPU2 = [{
    'model': 'baichuan-inc/Baichuan2-13B-Chat',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'adapters': {
            'a': 'lora/2024-01-25_self_dup',
            'b': 'lora/2024-01-25_self'
        }
    }
}]

PYTORCH_PR_TEST_LLM_GPU2 = [{
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}]

PYTORCH_PR_TEST_LLM_GPU1 = [{
    'model': 'meta-llama/Meta-Llama-3-1-8B-Instruct',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-0.6B',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

BASE_TOOLCALL_TEST_LLM = [{
    'model': 'Qwen/Qwen3-8B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'tool-call-parser': 'qwen'
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
    'model': 'Qwen/Qwen3-30B-A3B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'tool-call-parser': 'qwen'
    }
}]

BASE_REASONING_TEST_LLM = [{
    'model': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'reasoning-parser': 'qwen-qwq'
    }
}, {
    'model': 'Qwen/Qwen3-30B-A3B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'reasoning-parser': 'qwen-qwq'
    }
}]

TOOLCALL_TEST_LLM = [{
    **item, 'backend': 'turbomind'
} for item in BASE_TOOLCALL_TEST_LLM] + [{
    **item, 'backend': 'pytorch'
} for item in BASE_TOOLCALL_TEST_LLM]

REASONING_TEST_LLM = [{
    **item, 'backend': 'turbomind'
} for item in BASE_REASONING_TEST_LLM] + [{
    **item, 'backend': 'pytorch'
} for item in BASE_REASONING_TEST_LLM]

BASE_SPECULATIVE_DECODING_PIPELINE_TEST_LLM = [{
    'model': 'meta-llama/Meta-Llama-3-1-8B-Instruct',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'max_batch_size': 128,
        'speculative_config': {
            'method': 'eagle3',
            'num_speculative_tokens': 3,
            'model': 'yuhuili/EAGLE3-LLaMA3.1-Instruct-8B'
        }
    }
}, {
    'model': 'zai-org/GLM-4.7-Flash',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'max_batch_size': 128,
        'speculative_config': {
            'method': 'deepseek_mtp',
            'num_speculative_tokens': 3
        }
    }
}, {
    'model': 'Qwen/Qwen3.5-35B-A3B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'max_batch_size': 256,
        'reasoning_parser': 'qwen-qwq',
        'speculative_config': {
            'method': 'qwen3_5_mtp',
            'num_speculative_tokens': 4
        }
    }
}, {
    'model': 'Qwen/Qwen3.5-35B-A3B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'max_batch_size': 256,
        'reasoning_parser': 'qwen-qwq',
        'max_prefill_token_num': 1024,
        'model_format': 'fp8',
        'speculative_config': {
            'method': 'qwen3_5_mtp',
            'num_speculative_tokens': 4
        }
    }
}, {
    'model': 'Qwen/Qwen3.5-35B-A3B-FP8',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'max_batch_size': 256,
        'reasoning_parser': 'qwen-qwq',
        'speculative_config': {
            'method': 'qwen3_5_mtp',
            'num_speculative_tokens': 4
        }
    }
}]

SPECULATIVE_DECODING_PIPELINE_TEST_LLM = [{
    **item, 'backend': 'pytorch'
} for item in BASE_SPECULATIVE_DECODING_PIPELINE_TEST_LLM]

BASE_SPECULATIVE_DECODING_RESTFUL_TEST_LLM = [{
    'model': 'meta-llama/Meta-Llama-3-1-8B-Instruct',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'speculative-draft-model': 'yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
        'speculative-algorithm': 'eagle3',
        'speculative-num-draft-tokens': 3,
        'max-batch-size': 128
    }
}, {
    'model': 'deepseek-ai/DeepSeek-V3',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 16
    },
    'extra_params': {
        'speculative-algorithm': 'deepseek_mtp',
        'speculative-num-draft-tokens': 3,
        'max-batch-size': 128
    }
}, {
    'model': 'zai-org/GLM-4.7-Flash',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'speculative-algorithm': 'deepseek_mtp',
        'speculative-num-draft-tokens': 3,
        'max-batch-size': 128
    }
}, {
    'model': 'Qwen/Qwen3.5-35B-A3B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'reasoning-parser': 'qwen-qwq',
        'speculative-algorithm': 'qwen3_5_mtp',
        'speculative-num-draft-tokens': 4,
        'max-batch-size': 256
    }
}, {
    'model': 'Qwen/Qwen3.5-35B-A3B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'reasoning-parser': 'qwen-qwq',
        'speculative-algorithm': 'qwen3_5_mtp',
        'speculative-num-draft-tokens': 4,
        'max-batch-size': 256,
        'max-prefill-token-num': 1024,
        'model-format': 'fp8'
    }
}, {
    'model': 'Qwen/Qwen3.5-35B-A3B-FP8',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'reasoning-parser': 'qwen-qwq',
        'speculative-algorithm': 'qwen3_5_mtp',
        'speculative-num-draft-tokens': 4,
        'max-batch-size': 256
    }
}]

SPECULATIVE_DECODING_RESTFUL_TEST_LLM = [{
    **item, 'backend': 'pytorch'
} for item in BASE_SPECULATIVE_DECODING_RESTFUL_TEST_LLM]
