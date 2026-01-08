DEFAULT_PORT = 23333
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

BACKEND_LIST = ['turbomind', 'pytorch']

RESTFUL_MODEL_LIST = [
    'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-VL-2B-Instruct', 'Qwen/Qwen3-30B-A3B', 'internlm/Intern-S1',
    'internlm/internlm2_5-20b-chat', 'internlm/internlm2_5-20b', 'Qwen/Qwen3-32B', 'OpenGVLab/InternVL3_5-30B-A3B',
    'OpenGVLab/InternVL3-38B', 'Qwen/Qwen3-VL-8B-Instruct', 'internlm/internlm3-8b-instruct',
    'meta-llama/Llama-3.2-3B-Instruct', 'Qwen/Qwen3-VL-30B-A3B-Instruct'
]

RESTFUL_BASE_MODEL_LIST = ['Qwen/Qwen3-8B-Base', 'internlm/internlm2_5-20b', 'Qwen/Qwen3-4B']
