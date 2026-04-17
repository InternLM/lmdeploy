# flake8: noqa

from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferConcurrentTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    from opencompass.configs.datasets.ruler.ruler_512k_gen import (
        ruler_datasets as ruler_512k_datasets,
    )
    from opencompass.configs.summarizers.groups.ruler import (
        ruler_summary_groups as _ruler_summary_groups_all,
    )

ruler_summary_groups = [
    g for g in _ruler_summary_groups_all if g.get('name') == 'ruler_512k'
]

#######################################################################
#                         Model Configuration                         #
#######################################################################

MODEL_NAME = ''
MODEL_PATH = ''
API_BASE = ''
JUDGE_MODEL_NAME = ''
JUDGE_MODEL_PATH = ''
JUDGE_API_BASE = ''

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# Use OpenAISDK to configure LMDeploy OpenAI interface
models = [
    dict(type=OpenAISDK,
         abbr=f'{MODEL_NAME}',
         path=MODEL_PATH,
         key='EMPTY',
         openai_api_base=API_BASE,
         retry=3,
         run_cfg=dict(num_gpus=0),
         meta_template=api_meta_template,
         timeout=10800,
         max_workers=1024,
         pred_postprocessor=dict(type=extract_non_reasoning_content))
]

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
datasets = list(ruler_512k_datasets)

judge_cfg = dict(
    type=OpenAISDK,
    abbr=f'{JUDGE_MODEL_NAME}',
    path=JUDGE_MODEL_NAME,
    key='EMPTY',
    openai_api_base=JUDGE_API_BASE,
    meta_template=dict(round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]),
    query_per_second=16,
    batch_size=1024,
    temperature=0.001,
    tokenizer_path=JUDGE_MODEL_PATH,
    verbose=True,
    max_out_len=8192,
    max_seq_len=32768,
    mode='mid',
)

for item in datasets:
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg
    if 'llm_evaluator' in item['eval_cfg']['evaluator'].keys(
    ) and 'judge_cfg' in item['eval_cfg']['evaluator']['llm_evaluator']:
        item['eval_cfg']['evaluator']['llm_evaluator']['judge_cfg'] = judge_cfg

#######################################################################
#                       PART 2  Dataset Summarizer                    #
#######################################################################

summarizer = dict(
    dataset_abbrs=[
        ['ruler_512k', 'naive_average'],
    ],
    summary_groups=ruler_summary_groups,
)

for item in datasets:
    if 'max_out_len' in item['infer_cfg']['inferencer']:
        del item['infer_cfg']['inferencer']['max_out_len']

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        retry=0,
        task=dict(type=OpenICLInferConcurrentTask),
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner, max_num_workers=64, task=dict(type=OpenICLEvalTask)),
)
