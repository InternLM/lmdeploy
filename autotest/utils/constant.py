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
    'gpt': {
        'query_per_second': 4,
        'max_out_len': 64000,
        'max_seq_len': 65536,
        'batch_size': 500,
        'temperature': 0.6,
        'openai_extra_kwargs': {
            'reasoning_effort': 'high',
        }
    }
}
