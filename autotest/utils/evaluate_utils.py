import csv
import glob
import json
import os
import subprocess
import time

import allure
import pandas as pd
from mmengine.config import Config
from utils.common_utils import execute_command_with_logging
from utils.config_utils import get_case_str_by_config, get_cli_str, parse_config_by_case
from utils.constant import DEFAULT_PORT, DEFAULT_SERVER, EVAL_RUN_CONFIG


def write_to_summary(case_name, result, msg, metrics, result_dir):
    status = '✅ PASS' if result else f'❌ FAIL {msg}'

    config = parse_config_by_case(case_name)

    backend = config['backend']
    model = config['model']
    communicator = config['communicator']
    parallel_config_str = config['parallel_config']
    quant_policy = config['quant_policy']

    dataset_name = []
    dataset_metrics = []
    for key in sorted(metrics.keys()):
        dataset_name.append(key)
        dataset_metrics.append(metrics.get(key, ''))

    summary_dataset_name = ' | '.join(dataset_name)
    summary_dataset_metrics = ' | '.join(dataset_metrics)

    summary_file = os.environ.get('GITHUB_STEP_SUMMARY', '')
    md_summary_file = f'{result_dir}/summary_{case_name}.md'
    summary_line = f'| {model} | {quant_policy} | {backend} | {communicator} | {parallel_config_str} | {status} | {summary_dataset_metrics} |\n'  # noqa: E501

    write_header = not os.path.exists(md_summary_file) or os.path.getsize(md_summary_file) == 0
    with open(md_summary_file, 'a') as f:
        if write_header:
            dash_line = '-----|' * (len(metrics.keys()))
            f.write('## Model Evaluation Results\n')
            f.write(
                f'| Model | QuantPolicy | Backend | Communicator | Parallel config | Status | {summary_dataset_name} |\n'  # noqa
            )
            f.write(f'|-------|-------------|---------|--------------|----|--------|{dash_line}\n')
        f.write(summary_line)
    if summary_file:
        write_header = not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0
        with open(summary_file, 'a') as f:
            if write_header:
                dash_line = '-----|' * (len(metrics.keys()))
                f.write('## Model Evaluation Results\n')
                f.write(
                    f'| Model | QuantPolicy | Backend | Communicator | Parallel config | Status | {summary_dataset_name} |\n'  # noqa
                )
                f.write(f'|-------|-------------|---------|--------------|----|--------|{dash_line}\n')
            f.write(summary_line)
    else:
        print(
            f'Summary: {model} | {backend} | {communicator} | {parallel_config_str} | {status} | {summary_dataset_metrics}'  # noqa: E501
        )


def llm_summary(case_name, result, msg, work_dir, result_dir=None):
    metrics = {}

    if work_dir and os.path.exists(work_dir):
        try:
            summary_dirs = glob.glob(os.path.join(work_dir, '*', 'summary'))
            if not summary_dirs:
                raise FileNotFoundError('No summary directory found')

            summary_dir = summary_dirs[0]

            csv_files = glob.glob(os.path.join(summary_dir, 'summary_*.csv'))
            if not csv_files:
                raise FileNotFoundError('No CSV files found')

            csv_file = sorted(csv_files)[-1]
            if not os.path.exists(csv_file):
                raise FileNotFoundError('CSV file does not exist')

            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) < 5 or not row[4]:
                        continue

                    dataset = row[0]
                    metric_value = row[4]
                    try:
                        metrics[dataset] = f'{float(metric_value):.2f}'  # noqa: E231
                    except ValueError:
                        metrics[dataset] = metric_value

        except Exception as e:
            print(f'Error reading metrics: {str(e)}')
    if not result_dir:
        result_dir = work_dir
    write_to_summary(case_name, result, msg, metrics, result_dir)


def mllm_summary(case_name,
                 result,
                 msg,
                 work_dir,
                 result_dir=None,
                 dataset_list=['MMBench_V11_MINI', 'MMStar_MINI', 'AI2D_MINI', 'OCRBench_MINI']):

    metrics = {}
    pattern = os.path.join(work_dir, case_name, 'T*')
    t_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    if not t_dirs:
        return

    # 按修改时间排序
    t_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = t_dirs[0]

    for dataset in dataset_list:
        if dataset == 'OCRBench_MINI':
            score_file = f'{latest_dir}/{case_name}_{dataset}_score.json'
            cur_score = 0
            with open(score_file, 'r') as f:
                total_score = json.load(f)
                cur_score = total_score['Final Score Norm']
            metrics[dataset] = f'{cur_score:.2f}'  # noqa: E231
        else:
            score_file = f'{latest_dir}/{case_name}_{dataset}_acc.csv'
            df = pd.read_csv(score_file)
            cur_score = df['Overall'].iloc[0]
            if dataset == 'MMBench_V11_MINI':
                cur_score = df.loc[df['split'] == 'dev', 'Overall'].values
            cur_score = cur_score * 100
            metrics[dataset] = f'{cur_score.item():.2f}'  # noqa: E231
        if result_dir is None:
            result_dir = work_dir
    write_to_summary(case_name, result, msg, metrics, result_dir)


def eval_test(model_path, eval_path, case_name, port=DEFAULT_PORT, test_type='infer', extra_config={}, **kwargs):
    work_dir = None
    try:

        work_dir = os.path.join(eval_path, f'wk_{case_name}')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        eval_log = os.path.join(eval_path, f'log_{case_name}_{timestamp}.log')
        temp_config_path = os.path.join(eval_path, f'temp_{case_name}.py')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_file = os.path.join(parent_dir, 'evaluate/eval_config_chat.py')

        print(f'Starting OpenCompass evaluation for model: {model_path}')
        print(f'Model path: {model_path}')
        print(f'Case: {case_name}')
        print(f'Config file: {config_file}')

        original_cwd = os.getcwd()
        os.makedirs(work_dir, exist_ok=True)

        test_url = f'http://{DEFAULT_SERVER}:{port}/v1'

        try:
            if test_type == 'infer':
                if not os.path.exists(config_file):
                    return False, f'Config file {config_file} not found'

                cfg = Config.fromfile(config_file)

                cfg.MODEL_NAME = case_name
                cfg.MODEL_PATH = model_path
                cfg.API_BASE = test_url  # noqa: E231

                if cfg.models and len(cfg.models) > 0:
                    model_cfg = cfg.models[0]
                    model_cfg['abbr'] = case_name
                    model_cfg['path'] = case_name
                    model_cfg['openai_api_base'] = test_url
                    model_cfg['tokenizer_path'] = model_path

                    for key, value in kwargs.items():
                        model_cfg[key] = value

                cfg.dump(temp_config_path)
                print(f'Modified config saved to: {temp_config_path}')
            elif test_type == 'eval':
                if not os.path.exists(temp_config_path):
                    error_msg = f'Temp config file {temp_config_path} not found for eval stage'
                    llm_summary(case_name, False, error_msg, work_dir, eval_path)
                    return False, error_msg

                cfg = Config.fromfile(temp_config_path)
                print(f'Using existing temp config file: {temp_config_path}')
                eval_run_config = EVAL_RUN_CONFIG
                eval_case_name = get_case_str_by_config(eval_run_config)
                cfg.JUDGE_API_BASE = test_url
                cfg.JUDGE_MODEL_PATH = model_path
                cfg.JUDGE_MODEL_NAME = eval_case_name

                if hasattr(cfg, 'judge_cfg'):
                    cfg.judge_cfg['path'] = eval_case_name
                    cfg.judge_cfg['abbr'] = eval_case_name
                    cfg.judge_cfg['openai_api_base'] = test_url
                    cfg.judge_cfg['tokenizer_path'] = model_path

                if hasattr(cfg, 'datasets') and cfg.datasets:
                    for dataset in cfg.datasets:
                        if 'eval_cfg' in dataset and 'evaluator' in dataset['eval_cfg']:
                            evaluator = dataset['eval_cfg']['evaluator']

                            if 'judge_cfg' in evaluator:
                                evaluator['judge_cfg']['abbr'] = cfg.JUDGE_MODEL_NAME
                                evaluator['judge_cfg']['path'] = cfg.JUDGE_MODEL_NAME
                                evaluator['judge_cfg']['openai_api_base'] = cfg.JUDGE_API_BASE
                                evaluator['judge_cfg']['tokenizer_path'] = cfg.JUDGE_MODEL_PATH

                            if 'llm_evaluator' in evaluator and 'judge_cfg' in evaluator['llm_evaluator']:
                                evaluator['llm_evaluator']['judge_cfg']['abbr'] = cfg.JUDGE_MODEL_NAME
                                evaluator['llm_evaluator']['judge_cfg']['path'] = cfg.JUDGE_MODEL_NAME
                                evaluator['llm_evaluator']['judge_cfg']['openai_api_base'] = cfg.JUDGE_API_BASE
                                evaluator['llm_evaluator']['judge_cfg']['tokenizer_path'] = cfg.JUDGE_MODEL_PATH

                cfg.dump(temp_config_path)
                print(f'Modified config for eval stage saved to: {temp_config_path}')

            extra_config_str = get_cli_str(extra_config)
            cmd = f'opencompass {temp_config_path} --reuse -w {work_dir} -m {test_type} --dump-res-length {extra_config_str}'  # noqa
            print(f'Running command: {cmd}')
            print(f'Work directory: {work_dir}')

            result, stderr = execute_command_with_logging(cmd, eval_log, timeout=259200)

            allure.attach.file(eval_log, name=eval_log, attachment_type=allure.attachment_type.TEXT)

            if test_type == 'eval':
                llm_summary(case_name, result, stderr, work_dir, eval_path)

            return result, stderr
        except Exception as e:
            print(f'Error occurred: {e}')
            return False, f'Error occurred: {e}'
        finally:
            os.chdir(original_cwd)
            print(f'Returned to directory: {original_cwd}')

    except subprocess.TimeoutExpired:
        timeout_msg = (f'Evaluation timed out for {model_path} '
                       f'after 259200 seconds')
        if work_dir and test_type == 'eval':
            llm_summary(case_name, False, timeout_msg, work_dir, eval_path)
        return False, timeout_msg
    except Exception as e:
        error_msg = f'Error during evaluation for {model_path}: {str(e)}'
        if work_dir and test_type == 'eval':
            llm_summary(case_name, False, error_msg, work_dir, eval_path)
        return False, error_msg


def mllm_eval_test(model_path, eval_path, case_name, port=DEFAULT_PORT, test_type='infer', extra_config={}):
    work_dir = os.path.join(eval_path, f'wk_{case_name}')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    eval_log = os.path.join(eval_path, f'log_{case_name}_{timestamp}.log')

    print(f'Starting VLMEvalKit evaluation for model: {model_path}')
    print(f'Model path: {model_path}')
    print(f'Case: {case_name}')
    print(f'Work directory: {work_dir}')

    os.makedirs(work_dir, exist_ok=True)

    extra_config_str = get_cli_str(extra_config)

    if test_type == 'infer':
        cmd = f'python run.py --data MMBench_V11_MINI MMStar_MINI AI2D_MINI OCRBench_MINI --model {case_name} --base-url http://{DEFAULT_SERVER}:{port}/v1 --reuse --work-dir {work_dir} --mode infer {extra_config_str}'  # noqa
    elif test_type == 'eval':
        cmd = f'python run.py --data MMBench_V11_MINI MMStar_MINI AI2D_MINI OCRBench_MINI --model {case_name} --base-url http://{DEFAULT_SERVER}:empty/v1 --reuse --work-dir {work_dir} --api-nproc 32 --mode eval --judge turbomind_Qwen2.5-32B-Instruct_nccl_tp2_0 --judge-base-url http://{DEFAULT_SERVER}:{port}/v1'  # noqa

    result, msg = execute_command_with_logging(cmd, eval_log)

    allure.attach.file(eval_log, name=eval_log, attachment_type=allure.attachment_type.TEXT)

    if test_type == 'eval':
        mllm_summary(case_name,
                     result,
                     msg,
                     work_dir,
                     eval_path,
                     dataset_list=['MMBench_V11_MINI', 'MMStar_MINI', 'AI2D_MINI', 'OCRBench_MINI'])
    return result, msg
