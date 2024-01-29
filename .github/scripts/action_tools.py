# Copyright (c) OpenMMLab. All rights reserved.
import glob
import json
import logging
import os
import shutil
import subprocess
from typing import List

import fire


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


def evaluate(models: List[str], workspace: str):
    """Evaluate models from lmdeploy using opencompass.

    Args:
        models: Input models.
        workspace: Working directory.
    """
    os.makedirs(workspace, exist_ok=True)
    output_csv = os.path.join(workspace, 'results.csv')
    for ori_model in models:
        model = ori_model.lower()
        model_, precision = model.rsplit('_', 1)
        do_lite = precision in ['w4a16', 'w4kv8', 'w8a8']
        if do_lite:
            model = model_
        engine_type, model = model.split('_', 1)
        assert engine_type in ['tb', 'pt', 'hf']
        if engine_type == 'tb':
            pass
        elif engine_type == 'pt':
            raise RuntimeError('not support pytorch engine inference')

        opencompass_dir = os.path.abspath(os.environ['OPENCOMPASS_DIR'])
        lmdeploy_dir = os.path.abspath(os.environ['LMDEPLOY_DIR'])
        config_path = os.path.join(
            lmdeploy_dir, '.github/scripts/eval_opencompass_config.py')
        config_path_new = os.path.join(opencompass_dir, 'configs',
                                       'eval_lmdeploy.py')
        if os.path.exists(config_path_new):
            os.remove(config_path_new)
        shutil.copy(config_path, config_path_new)
        with open(config_path_new, 'a') as f:
            f.write(f'\nmodels = [ {ori_model} ]\n')

        work_dir = os.path.join(workspace, ori_model)
        cmd_eval = [
            f'python3 {opencompass_dir}/run.py {config_path_new} -w {work_dir}'
        ]
        eval_log = os.path.join(workspace, f'eval.{ori_model}.txt')
        ret = run_cmd(cmd_eval, log_path=eval_log, cwd=lmdeploy_dir)
        if ret != 0:
            continue
        csv_files = glob.glob(f'{work_dir}/*/summary/summary_*.csv')
        if len(csv_files) != 1:
            logging.error(f'Did not find summary csv file {csv_files}')
            continue
        csv_file = csv_files[0]
        # print csv_txt to screen
        csv_txt = csv_file.replace('.csv', '.txt')
        if os.path.exists(csv_txt):
            with open(csv_txt, 'r') as f:
                print(f.read())

        # parse evaluation results from csv file
        data = []
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                row = line.strip().split(',')
                row = [_.strip() for _ in row]
                if row[-1] != '-':
                    data.append((row[0], row[-1]))
        crows_pairs_json = glob.glob(os.path.join(
            work_dir, '*/results/*/crows_pairs.json'),
                                     recursive=True)
        if len(crows_pairs_json) == 1:
            with open(crows_pairs_json[0], 'r') as f:
                acc = json.load(f)['accuracy']
                acc = f'{float(acc):.2f}'
                data.append(('crows_pairs', acc))
        prec = '-' if precision == '' else precision
        row = ','.join([model, engine_type, prec] + [_[1] for _ in data])
        if not os.path.exists(output_csv):
            with open(output_csv, 'w') as f:
                header = ','.join(['Model', 'Engine', 'Precision'] +
                                  [_[0] for _ in data])
                f.write(header + '\n')
                f.write(row + '\n')
        else:
            with open(output_csv, 'a') as f:
                f.write(row + '\n')

    # write to github action summary
    _append_summary('## Evaluation Results')
    if os.path.exists(output_csv):
        add_summary(output_csv)


if __name__ == '__main__':
    fire.Fire()
