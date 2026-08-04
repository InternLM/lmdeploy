# flake8: noqa
"""Release-gate chat eval (no long-context).

Aligned with opencompass ``examples/eval_release_gate.py``:
  MMLU-Pro (computer science + physics)
  AIME2025 x4, GPQA-diamond x2, IFEval
  LiveCodeBench v6 x4, MATH-500 (cascade, n=1)
  sanitized MBPP (code exec)
"""

import copy
import os

from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferConcurrentTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

#######################################################################
#                     Gate parameters                                 #
#######################################################################

AIME_N = 4
GPQA_N = 2
LCB_N = 4

MMLU_PRO_KEEP = {
    'mmlu_pro_computer_science',
    'mmlu_pro_physics',
}

#######################################################################
#                          Import base configs                        #
#######################################################################

with read_base():
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_academic import \
        aime2025_datasets
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_academic import \
        gpqa_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import \
        ifeval_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_v6_academic import \
        LCBCodeGeneration_dataset
    from opencompass.configs.datasets.math.math_500_cascade_eval_gen_6ff468 import \
        math_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import \
        mmlu_pro_datasets
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_rawprompt_gen_30c1e5 import \
        sanitized_mbpp_datasets

#######################################################################
#                     Apply gate overrides                            #
#######################################################################

mmlu_pro_datasets = [
    ds for ds in mmlu_pro_datasets if ds['abbr'] in MMLU_PRO_KEEP
]

aime2025_datasets = [copy.deepcopy(ds) for ds in aime2025_datasets]
for ds in aime2025_datasets:
    ds['n'] = AIME_N
    ds['abbr'] = f'aime2025_repeat_{AIME_N}'

gpqa_datasets = [copy.deepcopy(ds) for ds in gpqa_datasets]
for ds in gpqa_datasets:
    split = ds['abbr'].split('_repeat_')[0]
    ds['n'] = GPQA_N
    ds['abbr'] = f'{split}_repeat_{GPQA_N}'

LCBCodeGeneration_dataset = copy.deepcopy(LCBCodeGeneration_dataset)
LCBCodeGeneration_dataset['n'] = LCB_N
LCBCodeGeneration_dataset['abbr'] = f'lcb_code_generation_repeat_{LCB_N}'

math_datasets = [copy.deepcopy(ds) for ds in math_datasets]
for ds in math_datasets:
    ds['n'] = 1

sanitized_mbpp_datasets = [
    copy.deepcopy(ds) for ds in sanitized_mbpp_datasets
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

models = [
    dict(type=OpenAISDK,
         abbr=f'{MODEL_NAME}',
         path=MODEL_PATH,
         key='EMPTY',
         openai_api_base=API_BASE,
         retry=3,
         run_cfg=dict(num_gpus=0),
         meta_template=api_meta_template,
         timeout=40800,
         max_workers=256,
         pred_postprocessor=dict(type=extract_non_reasoning_content))
]

#######################################################################
#                          Datasets + judge                           #
#######################################################################

datasets = (
    mmlu_pro_datasets + aime2025_datasets + gpqa_datasets + ifeval_datasets +
    math_datasets + sanitized_mbpp_datasets + [LCBCodeGeneration_dataset]
)

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
    batch_size=256,
    temperature=0.001,
    tokenizer_path=JUDGE_MODEL_PATH,
    verbose=True,
    max_out_len=8192,
    max_seq_len=32768,
    mode='mid',
)

for item in datasets:
    evaluator = item.get('eval_cfg', {}).get('evaluator', {})
    if isinstance(evaluator, dict):
        if 'judge_cfg' in evaluator:
            evaluator['judge_cfg'] = judge_cfg
        if ('llm_evaluator' in evaluator and isinstance(
                evaluator['llm_evaluator'], dict) and
                'judge_cfg' in evaluator['llm_evaluator']):
            evaluator['llm_evaluator']['judge_cfg'] = judge_cfg

#######################################################################
#                            Summarizer                               #
#######################################################################

mmlu_pro_summary_groups = [
    {
        'name': 'mmlu_pro_gate',
        'subsets': sorted(MMLU_PRO_KEEP),
    },
]

gate_summary_groups = [
    {
        'name': 'release_gate_average',
        'subsets': [
            ['IFEval', 'Prompt-level-strict-accuracy'],
            [f'aime2025_repeat_{AIME_N}', f'accuracy ({AIME_N} runs average)'],
            [f'GPQA_diamond_repeat_{GPQA_N}',
             f'accuracy ({GPQA_N} runs average)'],
            ['mmlu_pro_gate', 'naive_average'],
            [f'lcb_code_generation_repeat_{LCB_N}',
             f'pass@1 ({LCB_N} runs average)'],
            ['math_prm800k_500', 'accuracy'],
            ['sanitized_mbpp', 'score'],
        ],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['release_gate_average', 'naive_average'],
        '',
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        '',
        'General Reasoning',
        [f'GPQA_diamond_repeat_{GPQA_N}',
         f'accuracy ({GPQA_N} runs average)'],
        '',
        'Math',
        [f'aime2025_repeat_{AIME_N}', f'accuracy ({AIME_N} runs average)'],
        ['math_prm800k_500', 'accuracy'],
        '',
        'Knowledge (MMLU-Pro subset)',
        ['mmlu_pro_gate', 'naive_average'],
        'mmlu_pro_computer_science',
        'mmlu_pro_physics',
        '',
        'Code',
        [f'lcb_code_generation_repeat_{LCB_N}',
         f'pass@1 ({LCB_N} runs average)'],
        'sanitized_mbpp',
    ],
    summary_groups=gate_summary_groups + mmlu_pro_summary_groups,
)

for item in datasets:
    if 'max_out_len' in item['infer_cfg']['inferencer']:
        del item['infer_cfg']['inferencer']['max_out_len']

# NumWorkerPartitioner dataset-size cache; CHAT_TYPE separates chat / longtext suites.
_dataset_size_root = os.environ.get('REPORT_DIR', '.').rstrip('/')
_dataset_type = os.environ.get('CHAT_TYPE', 'default').rstrip('/')
dataset_size_path = f'{_dataset_size_root}/dataset_size_{_dataset_type}.json'

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=1,
        dataset_size_path=dataset_size_path,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        retry=0,
        task=dict(type=OpenICLInferConcurrentTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner, max_num_workers=64, task=dict(type=OpenICLEvalTask)),
)
