# flake8: noqa
"""Long-context release-gate eval (RULER NIAH 32k + 512k).

Aligned with opencompass ``examples/eval_release_gate_longcontext.py``:
  - ruler_niah_single_2 + ruler_niah_multivalue
  - 20 samples per dataset (80 gens total)
"""

import os

from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferConcurrentTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

#######################################################################
#                     Long-context parameters                         #
#######################################################################

NUM_SAMPLES = 20
TOKENIZER_MODEL = os.environ.get('TOKENIZER_MODEL', '')
MAX_SEQ_LENS = [1024 * 32, 1024 * 512]
ABBR_SUFFIXS = ['32k', '512k']
KEEP_ABBR = {
    'ruler_niah_single_2',
    'ruler_niah_multivalue',
}

#######################################################################
#                          Import base configs                        #
#######################################################################

with read_base():
    from opencompass.configs.datasets.ruler.ruler_niah_gen import niah_datasets

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
#                          Build datasets                             #
#######################################################################

_base = [ds for ds in niah_datasets if ds['abbr'] in KEEP_ABBR]

datasets = []
for max_seq_len, abbr_suffix in zip(MAX_SEQ_LENS, ABBR_SUFFIXS):
    for dataset in _base:
        tmp = dataset.deepcopy()
        tmp['abbr'] = f"{tmp['abbr']}_{abbr_suffix}"
        tmp['num_samples'] = NUM_SAMPLES
        tmp['max_seq_length'] = max_seq_len
        tmp['tokenizer_model'] = TOKENIZER_MODEL
        datasets.append(tmp)

#######################################################################
#                            Summarizer                               #
#######################################################################

_ruler_abbrs = [ds['abbr'] for ds in datasets]

longcontext_summary_groups = [
    {
        'name': 'ruler_niah_gate',
        'subsets': [[abbr, 'score'] for abbr in _ruler_abbrs],
    },
    {
        'name': 'ruler_niah_gate_32k',
        'subsets': [[abbr, 'score'] for abbr in _ruler_abbrs if abbr.endswith('_32k')],
    },
    {
        'name': 'ruler_niah_gate_512k',
        'subsets': [[abbr, 'score'] for abbr in _ruler_abbrs if abbr.endswith('_512k')],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['ruler_niah_gate', 'naive_average'],
        ['ruler_niah_gate_32k', 'naive_average'],
        ['ruler_niah_gate_512k', 'naive_average'],
        '',
        '32k',
        'ruler_niah_single_2_32k',
        'ruler_niah_multivalue_32k',
        '',
        '512k',
        'ruler_niah_single_2_512k',
        'ruler_niah_multivalue_512k',
    ],
    summary_groups=longcontext_summary_groups,
)

for item in datasets:
    if 'max_out_len' in item['infer_cfg']['inferencer']:
        del item['infer_cfg']['inferencer']['max_out_len']

# NumWorkerPartitioner dataset-size cache; CHAT_TYPE separates chat / longtext suites.
_dataset_size_root = os.environ.get('REPORT_DIR', '.').rstrip('/')
_dataset_type = os.environ.get('CHAT_TYPE', 'longtext-512k').rstrip('/')
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
