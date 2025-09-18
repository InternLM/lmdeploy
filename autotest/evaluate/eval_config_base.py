from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_4b5a83 import gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_few_shot_ppl import race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import \
        winogrande_datasets  # noqa: F401, E501

race_datasets = [race_datasets[1]]
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

MODEL_NAME = 'internlm2_5-1_8b'
MODEL_PATH = '/nvme/qa_test_models/internlm/internlm2_5-1_8b'
API_BASE = 'http://127.0.0.1:23333/v1'

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

models = [
    dict(
        type=OpenAISDK,
        abbr=f'{MODEL_NAME}-lmdeploy-api',
        openai_api_base=API_BASE,
        key='EMPTY',
        path=MODEL_PATH,
        meta_template=api_meta_template,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, communicator='native'),
        temperature=0.1,
    )
]

summarizer = dict(
    dataset_abbrs=[
        ['gsm8k', 'accuracy'],
        ['GPQA_diamond', 'accuracy'],
        ['race-high', 'accuracy'],
        ['winogrande', 'accuracy'],
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
