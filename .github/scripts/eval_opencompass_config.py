from mmengine.config import read_base
from opencompass.models.turbomind import PytorchModel, TurboMindModel

with read_base():
    # choose a list of datasets
    # from .datasets.ceval.ceval_gen_5f30c7 import \
    #     ceval_datasets  # noqa: F401, E501
    # from .datasets.crowspairs.crowspairs_gen_381af0 import \
    #     crowspairs_datasets  # noqa: F401, E501
    # from .datasets.gsm8k.gsm8k_gen_1d7fe4 import \
    #     gsm8k_datasets  # noqa: F401, E501
    # from .datasets.mmlu.mmlu_gen_a484b3 import \
    #     mmlu_datasets  # noqa: F401, E501
    from .datasets.race.race_gen_69ee4f import \
        race_datasets  # noqa: F401, E501
    # from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import \
    #     WiC_datasets  # noqa: F401, E501
    # from .datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import \
    #     WSC_datasets  # noqa: F401, E501
    # from .datasets.triviaqa.triviaqa_gen_2121ce import \
    #     triviaqa_datasets  # noqa: F401, E501
    # and output the results in a chosen format
    from .summarizers.medium import summarizer  # noqa: F401, E501

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

internlm_meta_template = dict(round=[
    dict(role='HUMAN', begin='<|User|>:', end='\n'),
    dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
],
                              eos_token_id=103028)

llama2_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='[INST] ', end=' [/INST]'),
        dict(role='BOT', generate=True),
    ],
    eos_token_id=2,
)
qwen_meta_template = dict(round=[
    dict(role='HUMAN', begin='\n<|im_start|>user\n', end='<|im_end|>'),
    dict(role='BOT',
         begin='\n<|im_start|>assistant\n',
         end='<|im_end|>',
         generate=True),
], )

baichuan2_meta_template = dict(round=[
    dict(role='HUMAN', begin='<reserved_106>'),
    dict(role='BOT', begin='<reserved_107>', generate=True),
], )

# config for internlm-chat-7b
tb_internlm_chat_7b = dict(
    type=TurboMindModel,
    abbr='internlm-chat-7b-turbomind',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=32,
    concurrency=32,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

tb_internlm_chat_7b_w4a16 = dict(
    type=TurboMindModel,
    abbr='internlm-chat-7b-w4-turbomind',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=32,
    concurrency=32,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

# config for internlm-chat-7b-w4kv8 model
tb_internlm_chat_7b_w4kv8 = dict(
    type=TurboMindModel,
    abbr='internlm-chat-7b-w4kv8-turbomind',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=32,
    concurrency=32,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

pt_internlm_chat_7b = dict(
    type=PytorchModel,
    abbr='internlm-chat-7b-pytorch',
    path='internlm-chat-7b',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=16,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

pt_internlm_chat_7b_w8a8 = dict(
    type=PytorchModel,
    abbr='internlm-chat-7b-pytorch-w8a8',
    path='internlm-chat-7b-w8a8',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=16,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

# config for internlm-chat-20b
tb_internlm_chat_20b = dict(
    type=TurboMindModel,
    abbr='internlm-chat-20b-turbomind',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=8,
    concurrency=8,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

# config for internlm-chat-20b-w4 model
tb_internlm_chat_20b_w4 = dict(
    type=TurboMindModel,
    abbr='internlm-chat-20b-w4-turbomind',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=16,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

# config for internlm-chat-20b-w4kv8 model
tb_internlm_chat_20b_w4kv8 = dict(
    type=TurboMindModel,
    abbr='internlm-chat-20b-w4kv8-turbomind',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=16,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

# config for llama2-chat-7b
tb_llama2_chat_7b = dict(
    type=TurboMindModel,
    abbr='llama2-chat-7b-turbomind',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=32,
    meta_template=llama2_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

# config for qwen-chat-7b
tb_qwen_chat_7b = dict(
    type=TurboMindModel,
    abbr='qwen-chat-7b-turbomind',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=16,
    meta_template=qwen_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

tb_qwen_chat_7b_w4a16 = dict(
    type=TurboMindModel,
    abbr='qwen-chat-7b-turbomind-w4a16',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=32,
    meta_template=qwen_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

tb_qwen_chat_7b_w4kv8 = dict(
    type=TurboMindModel,
    abbr='qwen-chat-7b-turbomind-w4kv8',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=32,
    concurrency=32,
    meta_template=qwen_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

pt_qwen_chat_7b = dict(
    type=PytorchModel,
    abbr='qwen-chat-7b-pytorch',
    path='./Qwen-7B-Chat',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=16,
    meta_template=qwen_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

pt_qwen_chat_7b_w8a8 = dict(
    type=PytorchModel,
    abbr='qwen-chat-7b-pytorch-w8a8',
    path='./Qwen-7B-Chat-w8a8',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=16,
    meta_template=qwen_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

# config for qwen-chat-7b
tb_qwen_chat_14b = dict(
    type=TurboMindModel,
    abbr='qwen-chat-14b-turbomind',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=32,
    meta_template=qwen_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

# config for baichuan2-chat-7b
tb_baichuan2_chat_7b = dict(
    type=TurboMindModel,
    abbr='baichuan2-chat-7b-turbomind',
    path='./turbomind',
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=32,
    meta_template=baichuan2_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)
