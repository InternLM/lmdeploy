# flake8: noqa

from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

# Dataset Configurations
with read_base():
    # Datasets
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_academic import aime2025_datasets
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_academic import gpqa_datasets
    from opencompass.configs.datasets.HLE.hle_llmverify_academic import hle_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_v6_academic import LCBCodeGeneration_dataset
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_nocot_genericllmeval_gen_08c1de import mmlu_pro_datasets
    # Summary Groups
    from opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups

# <dataset_replace_tag>
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), []) + [LCBCodeGeneration_dataset]
# </dataset_replace_tag>

TASK_TAG = ''
API_SERVER_ADDR = 'http://<API_SERVER>'
MODEL_PATH = ''

models = [
    dict(
        abbr=TASK_TAG,
        key='dummy',
        openai_api_base=f'{API_SERVER_ADDR}/v1',
        type=OpenAISDK,
        path=MODEL_PATH,
        temperature=0.6,
        meta_template=dict(round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ], ),
        query_per_second=10,
        max_out_len=32768,
        max_seq_len=32768,
        batch_size=8,
        retry=10,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
        verbose=False,
    )
]

JUDGER_ADDR = 'http://<JUDGER_SERVER>'
JUDGER_MODEL_PATH = ''
judge_cfg = dict(
    abbr='CompassVerifier',
    type=OpenAISDK,
    path=JUDGER_MODEL_PATH,
    key='YOUR_API_KEY',
    openai_api_base=f'{JUDGER_ADDR}/v1',
    meta_template=dict(round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]),
    query_per_second=8,
    batch_size=32,
    temperature=0.001,
    max_out_len=8192,
    max_seq_len=64000,
    mode='mid',
)

for item in datasets:
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg
    if 'llm_evaluator' in item['eval_cfg']['evaluator'].keys(
    ) and 'judge_cfg' in item['eval_cfg']['evaluator']['llm_evaluator']:
        item['eval_cfg']['evaluator']['llm_evaluator']['judge_cfg'] = judge_cfg

#######################################################################
#                         Dataset Summarizer                          #
#######################################################################

core_summary_groups = [
    {
        'name':
        'core_average',
        'subsets': [
            ['IFEval', 'Prompt-level-strict-accuracy'],
            ['hle_llmjudge', 'accuracy'],
            ['aime2025_repeat_32', 'accuracy (32 runs average)'],
            ['GPQA_diamond_repeat_4', 'accuracy (4 runs average)'],
            ['mmlu_pro', 'naive_average'],
            ['lcb_code_generation_repeat_6', 'pass@1 (6 runs average)'],
        ],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['core_average', 'naive_average'],
        '',
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        '',
        'General Reasoning',
        ['hle_llmjudge', 'accuracy'],
        ['GPQA_diamond_repeat_4', 'accuracy (4 runs average)'],
        '',
        'Math Calculation',
        ['aime2025_repeat_32', 'accuracy (32 runs average)'],
        '',
        'Knowledge',
        ['mmlu_pro', 'naive_average'],
        '',
        'Code',
        ['lcb_code_generation_repeat_6', 'pass@1 (6 runs average)'],
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

#######################################################################
#                   Inference/Evaluation Configuration                #
#######################################################################

# infer with local runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0,  # Modify if needed
        task=dict(type=OpenICLInferTask),
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLEvalTask)),
)
