import os

DEFAULT_PORT = 23333
DEFAULT_SERVER = os.getenv('MASTER_ADDR', '127.0.0.1')
PROXY_PORT = 8000

EVAL_CONFIGS = {
    'default': {
        'query_per_second': 4,
        'max_out_len': 64000,
        'max_seq_len': 65536,
        'batch_size': 500,
        'temperature': 0.6,
    },
    'default-32k': {
        'query_per_second': 4,
        'max_out_len': 32768,
        'max_seq_len': 65536,
        'batch_size': 500,
        'temperature': 0.6,
    },
    'gpt': {
        'query_per_second': 4,
        'max_out_len': 64000,
        'max_seq_len': 65536,
        'batch_size': 500,
        'temperature': 0.6,
        'openai_extra_kwargs': {
            'reasoning_effort': 'high',
        }
    },
    'gpt-32k': {
        'query_per_second': 4,
        'max_out_len': 32768,
        'max_seq_len': 65536,
        'batch_size': 500,
        'temperature': 0.6,
        'openai_extra_kwargs': {
            'reasoning_effort': 'high',
        }
    }
}

MLLM_EVAL_CONFIGS = {
    'default': {},
    'internvl': {
        'repetition-penalty': 1.0,
        'top-p': 0.8,
        'top-k': 20,
        'temperature': 0.7,
    }
}

BACKEND_LIST = ['turbomind', 'pytorch']

RESTFUL_MODEL_LIST_LATEST = [
    'Qwen/Qwen3-32B', 'Qwen/Qwen3-30B-A3B', 'OpenGVLab/InternVL3_5-30B-A3B', 'OpenGVLab/InternVL3-38B',
    'Qwen/Qwen3-VL-8B-Instruct', 'internlm/internlm3-8b-instruct', 'meta-llama/Llama-3.2-3B-Instruct',
    'Qwen/Qwen3-VL-30B-A3B-Instruct'
]

RESTFUL_MODEL_LIST_LEGACY = ['internlm/internlm2_5-20b']

_IS_LEGACY = 'legacy' in os.getenv('TEST_ENV', '')

RESTFUL_MODEL_LIST = RESTFUL_MODEL_LIST_LEGACY if _IS_LEGACY else RESTFUL_MODEL_LIST_LATEST

TOOL_REASONING_MODEL_LIST_LATEST = [
    'Qwen/Qwen3-8B-FP8',
    'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'Qwen/Qwen2.5-7B-Instruct',
]

TOOL_REASONING_MODEL_LIST_LEGACY = [
    'internlm/Intern-S1-Pro-FP8',
]

TOOL_REASONING_MODEL_LIST = TOOL_REASONING_MODEL_LIST_LEGACY if _IS_LEGACY else TOOL_REASONING_MODEL_LIST_LATEST

RESTFUL_BASE_MODEL_LIST = [
    'Qwen/Qwen3-8B-Base', 'internlm/internlm2_5-20b', 'Qwen/Qwen3-4B', 'internlm/internlm3-8b-instruct'
]

SUFFIX_INNER_AWQ = '-inner-4bits'
SUFFIX_INNER_GPTQ = '-inner-gptq'
SUFFIX_INNER_W8A8 = '-inner-w8a8'

EVAL_RUN_CONFIG = {
    'model': 'Qwen/Qwen2.5-32B-Instruct',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'server-name': DEFAULT_SERVER,
        'session-len': 76000,
        'cache-max-entry-count': 0.7
    }
}
