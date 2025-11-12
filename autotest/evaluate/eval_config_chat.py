# flake8: noqa

from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_academic import aime2025_datasets
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_academic import gpqa_datasets
    from opencompass.configs.datasets.HLE.hle_llmverify_academic import hle_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_v6_academic import LCBCodeGeneration_dataset
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets
    # Summary Groups
    from opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups

#######################################################################
#                         Model Configuration                         #
#######################################################################

MODEL_NAME = ''
MODEL_PATH = ''
API_BASE = ''
JUDGE_MODEL_PATH = ''
JUDGE_API_BASE = ''

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# Use OpenAISDK to configure LMDeploy OpenAI interface
models = [
    dict(type=OpenAISDK,
         abbr=f'{MODEL_NAME}-lmdeploy-api',
         path=MODEL_PATH,
         key='EMPTY',
         openai_api_base=API_BASE,
         retry=3,
         run_cfg=dict(num_gpus=0),
         meta_template=api_meta_template,
         pred_postprocessor=dict(type=extract_non_reasoning_content))
]

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation
mmlu_pro_datasets = [x for x in mmlu_pro_datasets if 'math' in x['abbr'] or 'other' in x['abbr']]

# Modify datasets list to exclude hle_datasets and LCBCodeGeneration_dataset
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), []) + [LCBCodeGeneration_dataset]

# LLM judge config: using LLM to evaluate predictions
judge_cfg = dict(
    type=OpenAISDK,
    path=JUDGE_MODEL_PATH,
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
            'mmlu_pro_math',
            'mmlu_pro_other',
            ['lcb_code_generation_repeat_6', 'pass@1 (6 runs average)'],
        ],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['core_average', 'naive_average'],
        ['IFEval', 'Prompt-level-strict-accuracy'],
        ['hle_llmjudge', 'accuracy'],
        ['GPQA_diamond_repeat_4', 'accuracy (4 runs average)'],
        ['aime2025_repeat_32', 'accuracy (32 runs average)'],
        ['mmlu_pro', 'naive_average'],
        'mmlu_pro_math',
        'mmlu_pro_other',
        ['lcb_code_generation_repeat_6', 'pass@1 (6 runs average)'],
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []) + core_summary_groups,
)

for item in datasets:
    if 'max_out_len' in item['infer_cfg']['inferencer']:
        del item['infer_cfg']['inferencer']['max_out_len']

#######################################################################
#                 PART 4  Inference/Evaluation Configuration          #
#######################################################################

# infer with local runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0,
        task=dict(type=OpenICLInferTask),
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLEvalTask)),
)
