# Copyright (c) OpenMMLab. All rights reserved.
import glob
import json
import logging
import os
import shutil
import subprocess
import time
from collections import OrderedDict
from typing import List

import fire
import pandas as pd
from mmengine.config import Config


def run_cmd(cmd_lines: List[str], log_path: str, cwd: str = None):
    """
    Args:
        cmd_lines: (list[str]): A command in multiple line style.
        log_path (str): Path to log file.
        cwd (str): Path to the current working directory.

    Returns:
        int: error code.
    """
    import platform

    system = platform.system().lower()

    if system == 'windows':
        sep = r'`'
    else:  # 'Linux', 'Darwin'
        sep = '\\'
    cmd_for_run = ' '.join(cmd_lines)
    cmd_for_log = f' {sep}\n'.join(cmd_lines) + '\n'
    with open(log_path, 'w', encoding='utf-8') as file_handler:
        file_handler.write(f'Command:\n{cmd_for_log}\n')
        file_handler.flush()
        process_res = subprocess.Popen(cmd_for_run,
                                       shell=True,
                                       cwd=cwd,
                                       stdout=file_handler,
                                       stderr=file_handler)
        process_res.wait()
        return_code = process_res.returncode

    if return_code != 0:
        logging.error(f'Got shell abnormal return code={return_code}')
        with open(log_path, 'r') as f:
            content = f.read()
            logging.error(f'Log error message\n{content}')
    return return_code


def _append_summary(content):
    summary_file = os.environ['GITHUB_STEP_SUMMARY']
    with open(summary_file, 'a') as f:
        f.write(content + '\n')


def add_summary(csv_path: str):
    """Add csv file to github step summary.

    Args:
        csv_path (str): Input csv file.
    """
    with open(csv_path, 'r') as fr:
        lines = fr.readlines()
        header = lines[0].strip().split(',')
        n_col = len(header)
        header = '|' + '|'.join(header) + '|'
        aligner = '|' + '|'.join([':-:'] * n_col) + '|'
        _append_summary(header)
        _append_summary(aligner)
        for line in lines[1:]:
            line = '|' + line.strip().replace(',', '|') + '|'
            _append_summary(line)
        _append_summary('\n')


def evaluate(models: List[str],
             datasets: List[str],
             workspace: str,
             evaluate_type: str,
             is_smoke: bool = False):
    """Evaluate models from lmdeploy using opencompass.

    Args:
        models: Input models.
        workspace: Working directory.
    """
    os.makedirs(workspace, exist_ok=True)
    output_csv = os.path.join(workspace, f'results_{evaluate_type}.csv')
    if os.path.exists(output_csv):
        os.remove(output_csv)
    num_model = len(models)
    for idx, ori_model in enumerate(models):
        print()
        print(50 * '==')
        print(f'Start evaluating {idx+1}/{num_model} {ori_model} ...')
        model = ori_model.lower()

        opencompass_dir = os.path.abspath(os.environ['OPENCOMPASS_DIR'])
        lmdeploy_dir = os.path.abspath(os.environ['LMDEPLOY_DIR'])
        config_path = os.path.join(
            lmdeploy_dir, f'.github/scripts/eval_{evaluate_type}_config.py')
        config_path_new = os.path.join(opencompass_dir, 'configs',
                                       'eval_lmdeploy.py')
        if os.path.exists(config_path_new):
            os.remove(config_path_new)
        shutil.copy(config_path, config_path_new)

        cfg = Config.fromfile(config_path_new)
        if not hasattr(cfg, model):
            logging.error(f'Model {model} not found in configuration file')
            continue

        model_cfg = cfg[model]
        logging.info(f'Start evaluating {model} ...\\nn{model_cfg}\n\n')

        with open(config_path_new, 'a') as f:
            f.write(f'\ndatasets = {datasets}\n')
            if is_smoke:
                f.write('\nfor d in datasets:\n')
                f.write("    if d['reader_cfg'] is not None:\n")
                f.write("        d['reader_cfg']['test_range'] = '[0:50]'\n")
            if model.startswith('hf'):
                f.write(f'\nmodels = [ *{model} ]\n')
            else:
                f.write(f'\nmodels = [ {model} ]\n')

        work_dir = os.path.join(workspace, model)
        cmd_eval = [
            f'python3 {opencompass_dir}/run.py {config_path_new} -w {work_dir} --reuse --max-num-workers 8 --dump-eval-details'  # noqa: E501
        ]
        eval_log = os.path.join(workspace, f'eval.{ori_model}.txt')
        start_time = time.time()
        ret = run_cmd(cmd_eval, log_path=eval_log, cwd=lmdeploy_dir)
        end_time = time.time()
        task_duration_seconds = round(end_time - start_time, 2)
        logging.info(f'task_duration_seconds: {task_duration_seconds}\n')
        if ret != 0:
            continue
        csv_files = glob.glob(f'{work_dir}/*/summary/summary_*.csv')

        if len(csv_files) < 1:
            logging.error(f'Did not find summary csv file {csv_files}')
            continue
        else:
            csv_file = max(csv_files, key=os.path.getctime)
        # print csv_txt to screen
        csv_txt = csv_file.replace('.csv', '.txt')
        if os.path.exists(csv_txt):
            with open(csv_txt, 'r') as f:
                print(f.read())

        # parse evaluation results from csv file
        model_results = OrderedDict()
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                row = line.strip().split(',')
                row = [_.strip() for _ in row]
                if row[-1] != '-':
                    model_results[row[0]] = row[-1]
        crows_pairs_json = glob.glob(os.path.join(
            work_dir, '*/results/*/crows_pairs.json'),
                                     recursive=True)
        if len(crows_pairs_json) == 1:
            with open(crows_pairs_json[0], 'r') as f:
                acc = json.load(f)['accuracy']
                acc = f'{float(acc):.2f}'
                model_results['crows_pairs'] = acc
        logging.info(f'\n{model}\n{model_results}')
        dataset_names = list(model_results.keys())

        row = ','.join([model, str(task_duration_seconds)] +
                       [model_results[_] for _ in dataset_names])

        if not os.path.exists(output_csv):
            with open(output_csv, 'w') as f:
                header = ','.join(['Model', 'task_duration_secs'] +
                                  dataset_names)
                f.write(header + '\n')
                f.write(row + '\n')
        else:
            with open(output_csv, 'a') as f:
                f.write(row + '\n')

    # write to github action summary
    _append_summary('## Evaluation Results')
    if os.path.exists(output_csv):
        add_summary(output_csv)


def create_model_links(src_dir: str, dst_dir: str):
    """Create softlinks for models."""
    paths = glob.glob(os.path.join(src_dir, '*'))
    model_paths = [os.path.abspath(p) for p in paths if os.path.isdir(p)]
    os.makedirs(dst_dir, exist_ok=True)
    for src in model_paths:
        _, model_name = os.path.split(src)
        dst = os.path.join(dst_dir, model_name)
        if not os.path.exists(dst):
            os.symlink(src, dst)
        else:
            logging.warning(f'Model_path exists: {dst}')


def generate_benchmark_report(report_path: str):
    # write to github action summary
    _append_summary('## Benchmark Results Start')
    subfolders = [f.path for f in os.scandir(report_path) if f.is_dir()]
    for dir_path in subfolders:
        second_subfolders = [
            f.path for f in sorted(os.scandir(dir_path), key=lambda x: x.name)
            if f.is_dir()
        ]
        for sec_dir_path in second_subfolders:
            model = sec_dir_path.replace(report_path + '/', '')
            print('-' * 25, model, '-' * 25)
            _append_summary('-' * 25 + model + '-' * 25 + '\n')

            benchmark_subfolders = [
                f.path
                for f in sorted(os.scandir(sec_dir_path), key=lambda x: x.name)
                if f.is_dir()
            ]
            for backend_subfolder in benchmark_subfolders:
                benchmark_type = backend_subfolder.replace(
                    sec_dir_path + '/', '')
                print('*' * 10, benchmark_type, '*' * 10)
                _append_summary('-' * 10 + benchmark_type + '-' * 10 + '\n')
                merged_csv_path = os.path.join(backend_subfolder,
                                               'summary.csv')
                csv_files = glob.glob(os.path.join(backend_subfolder, '*.csv'))
                average_csv_path = os.path.join(backend_subfolder,
                                                'average.csv')
                if merged_csv_path in csv_files:
                    csv_files.remove(merged_csv_path)
                if average_csv_path in csv_files:
                    csv_files.remove(average_csv_path)
                merged_df = pd.DataFrame()

                if len(csv_files) > 0:
                    for f in csv_files:
                        df = pd.read_csv(f)
                        merged_df = pd.concat([merged_df, df],
                                              ignore_index=True)

                    merged_df = merged_df.sort_values(by=merged_df.columns[0])

                    grouped_df = merged_df.groupby(merged_df.columns[0])
                    if 'generation' not in backend_subfolder:
                        average_values = grouped_df.pipe(
                            (lambda group: {
                                'mean': group.mean().round(decimals=3)
                            }))['mean']
                        average_values.to_csv(average_csv_path, index=True)
                        avg_df = pd.read_csv(average_csv_path)
                        merged_df = pd.concat([merged_df, avg_df],
                                              ignore_index=True)
                        add_summary(average_csv_path)
                    merged_df.to_csv(merged_csv_path, index=False)
                    if 'generation' in backend_subfolder:
                        add_summary(merged_csv_path)

    _append_summary('## Benchmark Results End')


if __name__ == '__main__':
    fire.Fire()
