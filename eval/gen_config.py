# flake8: noqa

import argparse

imports_config = """# flake8: noqa

from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.models import OpenAISDK
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
"""

summarizer_config = """
#######################################################################
#                       PART 2  Dataset Summarizer                     #
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
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
"""

runner_config = """
#######################################################################
#                 PART 4  Inference/Evaluation Configuration          #
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
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLEvalTask)),
)
"""


def config_datasets(datasets: list[str]) -> str:
    config_dataset = """
# Dataset Configurations
with read_base():
    # Datasets
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_academic import aime2025_datasets
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_academic import gpqa_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_v6_academic import LCBCodeGeneration_dataset
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets
    from opencompass.configs.datasets.HLE.hle_llmverify_academic import hle_datasets

    # Summary Groups
    from opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups
"""
    if 'all' in datasets:
        selected_datasets = """
# select the datasets
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')),
               []) + [LCBCodeGeneration_dataset]"""
    else:
        selected_datasets = []
        if 'code' in datasets:
            selected_datasets.append('[LCBCodeGeneration_dataset]')
            datasets.remove('code')
        for d in datasets:
            selected_datasets.append(f'{d}_datasets')
        selected_datasets = ' + '.join(selected_datasets)
        selected_datasets = f'datasets = {selected_datasets}'
    return '\n'.join([config_dataset, selected_datasets])


def get_model_name_from_server(server: str) -> str:
    from openai import OpenAI
    try:
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{server}/v1')
        model_name = client.models.list().data[0].id
        return model_name
    except Exception as e:
        raise RuntimeError(f'Failed to get model name from server {server}: {e}')


def config_api_server(task_name: str, mode: str, api_server: str):
    if mode not in ['infer', 'e2e']:
        raise ValueError(f'In mode {mode}, api_server should not be configured')
    if not api_server:
        raise ValueError('api_server must be provided')

    api_server_config = f"""
models = [
    dict(
        abbr='{task_name}',
        key="dummy",
        openai_api_base='{api_server}/v1',
        type=OpenAISDK,
        path='{get_model_name_from_server(api_server)}',
        temperature=0.6,
        meta_template=dict(
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ],
        ),
        query_per_second=10,
        max_out_len=32768,
        max_seq_len=32768,
        batch_size=8,
        retry=10,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
        verbose=False,
    )
]"""
    return api_server_config


def config_judger(task_name: str, mode: str, judger_server: str):
    if mode not in ['eval', 'e2e']:
        raise ValueError(f'In mode {mode}, judger_server should not be configured')
    if not judger_server:
        raise ValueError('judger_server must be provided')
    model_name = get_model_name_from_server(judger_server)
    judger_config = f"""
judge_cfg = dict(
    abbr='CompassVerifier',
    type=OpenAISDK,
    path='{model_name}',
    key='YOUR_API_KEY',
    openai_api_base='{judger_server}/v1',
    meta_template=dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]),
    query_per_second=8,
    batch_size=8,
    temperature=0.001,
    max_out_len=8192,
    max_seq_len=32768,
    mode='mid',
)

for item in datasets:
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg
    if 'llm_evaluator' in item['eval_cfg']['evaluator'].keys() and 'judge_cfg' in item['eval_cfg']['evaluator']['llm_evaluator']:
        item['eval_cfg']['evaluator']['llm_evaluator']['judge_cfg'] = judge_cfg
"""

    if mode == 'eval':
        model_config = f"""
models = [
    dict(abbr='{task_name}'),
]"""
        return '\n'.join([model_config, judger_config])
    else:  # e2e
        return judger_config


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation config')
    parser.add_argument('task_name', type=str, help='Task name for evaluation')
    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        choices=['e2e', 'infer', 'eval'],
                        default='e2e',
                        help='Evaluation mode')
    parser.add_argument('-a', '--api-server', type=str, default='', help='API server address for inference')
    parser.add_argument('-j', '--judger-server', type=str, default='', help='Judger server address for evaluation')
    dataset_choices = ['aime2025', 'gpqa', 'ifeval', 'code', 'mmlu_pro', 'hle', 'all']
    parser.add_argument('-d',
                        '--datasets',
                        type=str,
                        default='all',
                        help=f"Comma-separated list of datasets. Available options: {', '.join(dataset_choices)}. "
                        'Use "all" to include all datasets.')
    parser.add_argument('-o', '--output-file', type=str, default='', help='Output file path (optional)')

    args = parser.parse_args()
    mode = args.mode
    task_name = args.task_name
    api_server = args.api_server
    judger_server = args.judger_server
    datasets = args.datasets
    output_file = args.output_file

    # validate inputs
    if mode in ['e2e', 'infer'] and not api_server:
        parser.error(f'In mode {mode}, --api-server must be provided')
    if mode in ['e2e', 'eval'] and not judger_server:
        parser.error(f'In mode {mode}, --judger-server must be provided')

    # Process server addresses
    if api_server and not api_server.startswith('http'):
        api_server = f'http://{api_server}'
    if judger_server and not judger_server.startswith('http'):
        judger_server = f'http://{judger_server}'

    # Process datasets
    if args.datasets == 'all':
        datasets = ['all']
    else:
        datasets = [d.strip().lower() for d in args.datasets.split(',')]

    candidates = ['aime2025', 'gpqa', 'ifeval', 'code', 'mmlu_pro', 'hle', 'all']
    for d in datasets:
        if d not in candidates:
            parser.error(f'Unknown dataset {d}, allowed values are {candidates} or "all"')

    # Generate config
    datasets_config = config_datasets(datasets)

    if mode == 'e2e':
        api_server_config = config_api_server(task_name, mode, api_server)
        judger_config = config_judger(task_name, mode, judger_server)
        final_config = '\n'.join(
            [imports_config, datasets_config, api_server_config, judger_config, summarizer_config, runner_config])
    elif mode == 'infer':
        api_server_config = config_api_server(task_name, mode, api_server)
        final_config = '\n'.join([imports_config, datasets_config, api_server_config, summarizer_config, runner_config])
    else:  # eval
        judger_config = config_judger(task_name, mode, judger_server)
        final_config = '\n'.join([imports_config, datasets_config, judger_config, summarizer_config, runner_config])

    # Output config
    if output_file:
        import os
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(final_config)
        print(f'Config written to {output_file}')
    else:
        print(final_config)


if __name__ == '__main__':
    main()
