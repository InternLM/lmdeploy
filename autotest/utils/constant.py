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
    'default-2batch': {
        'query_per_second': 4,
        'max_out_len': 64000,
        'max_seq_len': 65536,
        'batch_size': 2,
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
    },
    'gpt-2batch': {
        'query_per_second': 4,
        'max_out_len': 64000,
        'max_seq_len': 65536,
        'batch_size': 2,
        'temperature': 0.6,
        'openai_extra_kwargs': {
            'reasoning_effort': 'high',
        }
    },
    'sdar': {
        'query_per_second': 4,
        'max_out_len': 64000,
        'max_seq_len': 65536,
        'batch_size': 500,
        'temperature': 1.0,
        'openai_extra_kwargs': {
            'top_p': 1.0,
        },
        'extra_body': {
            'top_k': 0,
        }
    },
    'sdar-32k': {
        'query_per_second': 4,
        'max_out_len': 32768,
        'max_seq_len': 65536,
        'batch_size': 500,
        'temperature': 1.0,
        'openai_extra_kwargs': {
            'top_p': 1.0,
        },
        'extra_body': {
            'top_k': 0,
        }
    },
    'sdar-2batch': {
        'query_per_second': 4,
        'max_out_len': 64000,
        'max_seq_len': 65536,
        'batch_size': 2,
        'temperature': 1.0,
        'openai_extra_kwargs': {
            'top_p': 1.0,
        },
        'extra_body': {
            'top_k': 0,
        }
    },
    'intern-s1-pro': {
        'query_per_second': 4,
        'max_out_len': 64000,
        'max_seq_len': 65536,
        'batch_size': 500,
        'temperature': 0.8,
        'openai_extra_kwargs': {
            'top_p': 0.95,
        },
        'extra_body': {
            'top_k': 50,
            'min_p': 0.0,
        }
    },
    'intern-s1-pro-32k': {
        'query_per_second': 4,
        'max_out_len': 32768,
        'max_seq_len': 65536,
        'batch_size': 500,
        'temperature': 0.8,
        'openai_extra_kwargs': {
            'top_p': 0.95,
        },
        'extra_body': {
            'top_k': 50,
            'min_p': 0.0,
        }
    },
    'intern-s1-pro-2batch': {
        'query_per_second': 4,
        'max_out_len': 64000,
        'max_seq_len': 65536,
        'batch_size': 2,
        'temperature': 0.8,
        'openai_extra_kwargs': {
            'top_p': 0.95,
        },
        'extra_body': {
            'top_k': 50,
            'min_p': 0.0,
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

SLEEP_WAKEUP_MODEL_LIST = [
    'Qwen/Qwen3.5-35B-A3B',
    'Qwen/Qwen3.5-35B-A3B-FP8',
    'Qwen/Qwen3.5-122B-A10B',
]

SLEEP_WAKEUP_BACKENDS = ['pytorch', 'turbomind']
SLEEP_WAKEUP_UPDATE_WEIGHTS_BACKENDS = ['pytorch', 'turbomind']

BACKEND_LIST = ['turbomind', 'pytorch']

RESTFUL_MODEL_LIST = [
    'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-VL-2B-Instruct', 'Qwen/Qwen3-30B-A3B', 'internlm/Intern-S1',
    'internlm/internlm2_5-20b', 'Qwen/Qwen3-32B', 'OpenGVLab/InternVL3_5-30B-A3B', 'OpenGVLab/InternVL3-38B',
    'Qwen/Qwen3-VL-8B-Instruct', 'internlm/internlm3-8b-instruct', 'meta-llama/Llama-3.2-3B-Instruct',
    'Qwen/Qwen3-VL-30B-A3B-Instruct'
]

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

# Qwen3-VL tomb demo (REST ``mm_processor`` + pipeline video): MCQ JSON without a labelled answer field.
MM_DEMO_TOMB_MCQ_JSON_BLOCK = """{
  "question": "How many porcelain jars were discovered in the niches located in the primary chamber of the tomb?",
  "options": [
    "A. 4.",
    "B. 9.",
    "C. 5.",
    "D. 13."
  ]
}"""
MM_DEMO_TOMB_USER_PROMPT = (
    'You are given a multiple-choice problem as JSON (question and options only; there is no answer field). '
    'Watch the entire video, pick the best option from what you see, then reply briefly with the letter '
    '(A, B, C, or D) first and at most one short sentence. Do not output long step-by-step reasoning; '
    'keep the final reply concise.\n\n' + MM_DEMO_TOMB_MCQ_JSON_BLOCK)
