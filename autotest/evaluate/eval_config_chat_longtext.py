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
    # Datasets
    from opencompass.configs.datasets.needlebench.needlebench_8k.needlebench_8k import (
        needlebench_datasets as needlebench_8k_datasets,
    )
    from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import (
        needlebench_datasets as needlebench_32k_datasets,
    )
    from opencompass.configs.datasets.needlebench.needlebench_128k.needlebench_128k import (
        needlebench_datasets as needlebench_128k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_8k_gen import (
        ruler_datasets as ruler_8k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_32k_gen import (
        ruler_datasets as ruler_32k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_64k_gen import (
        ruler_datasets as ruler_64k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_128k_gen import (
        ruler_datasets as ruler_128k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_256k_gen import (
        ruler_datasets as ruler_256k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_0k_gen import (
        babiLong_0k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_4k_gen import (
        babiLong_4k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_16k_gen import (
        babiLong_16k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_32k_gen import (
        babiLong_32k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_128k_gen import (
        babiLong_128k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_256k_gen import (
        babiLong_256k_datasets,
    )
    # Summary Groups
    from opencompass.configs.summarizers.groups.babilong import (
        babilong_summary_groups,
    )
    from opencompass.configs.summarizers.groups.ruler import (
        ruler_summary_groups,
    )
    from opencompass.configs.summarizers.needlebench import (
        needlebench_8k_summarizer,
        needlebench_32k_summarizer,
        needlebench_128k_summarizer,
    )

ruler_summary_groups = [
    g for g in ruler_summary_groups if g.get('name') != 'ruler_512k'
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
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

needlebench_8k_summary_groups = needlebench_8k_summarizer["summary_groups"]
needlebench_32k_summary_groups = needlebench_32k_summarizer["summary_groups"]
needlebench_128k_summary_groups = needlebench_128k_summarizer["summary_groups"]

# LLM judge config: using LLM to evaluate predictions
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
        ["ruler_8k", "naive_average"],
        ["ruler_32k", "naive_average"],
        ["ruler_64k", "naive_average"],
        ["ruler_128k", "naive_average"],
        ["ruler_256k", "naive_average"],
        ["NeedleBench-Overall-Score-8K", "weighted_average"],
        ["NeedleBench-Overall-Score-32K", "weighted_average"],
        ["NeedleBench-Overall-Score-128K", "weighted_average"],
        ['babilong_0k', 'naive_average'],
        ['babilong_4k', 'naive_average'],
        ['babilong_16k', 'naive_average'],
        ['babilong_32k', 'naive_average'],
        ['babilong_128k', 'naive_average'],
        ['babilong_256k', 'naive_average'],
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
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
