from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups  # noqa: F401, E501

mmlu_datasets = [
    x for x in mmlu_datasets if x['abbr'].replace('lukaemon_mmlu_', '') in [
        'business_ethics', 'clinical_knowledge', 'college_medicine', 'global_facts', 'human_aging', 'management',
        'marketing', 'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting',
        'professional_medicine', 'virology'
    ]
]

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

MODEL_NAME = ''
MODEL_PATH = ''
API_BASE = ''

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

models = [
    dict(type=OpenAISDK,
         abbr=f'{MODEL_NAME}-lmdeploy-api',
         openai_api_base=API_BASE,
         key='EMPTY',
         path=MODEL_PATH,
         meta_template=api_meta_template,
         max_out_len=32768,
         batch_size=500,
         temperature=0.1,
         pred_postprocessor=dict(type=extract_non_reasoning_content))
]

summarizer = dict(
    dataset_abbrs=[
        ['mmlu', 'naive_average'],
        ['gsm8k', 'accuracy'],
        'mmlu-other',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

for item in datasets:
    if 'max_out_len' in item['infer_cfg']['inferencer']:
        del item['infer_cfg']['inferencer']['max_out_len']
