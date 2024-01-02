# Copyright (c) OpenMMLab. All rights reserved.
import glob
import json
import logging
import os
import re
import shutil
import subprocess
from typing import List

import fire

MODEL_CFGS = {
    'internlm_chat_7b': {
        'model-path': 'internlm-chat-7b',
        'model-name': 'internlm-chat-7b'
    },
    'internlm_chat_20b': {
        'model-path': 'internlm-chat-20b',
        'model-name': 'internlm-chat-20b'
    },
    'llama2_chat_7b': {
        'model-path': 'llama-2-7b-chat',
        'model-name': 'llama2'
    },
    'llama2_chat_13b': {
        'model-path': 'llama-2-13b-chat',
        'model-name': 'llama2'
    },
    'qwen_chat_7b': {
        'model-path': 'Qwen-7B-Chat',
        'model-name': 'qwen-7b'
    },
    'qwen_chat_14b': {
        'model-path': 'Qwen-14B-Chat',
        'model-name': 'qwen-14b'
    },
    'baichuan2_chat_7b': {
        'model-path': 'Baichuan2-7B-Chat',
        'model-name': 'baichuan2-7b'
    },
}


def run_cmd(cmd_lines: List[str], log_path: str):
    """
    Args:
        cmd_lines: (list[str]): A command in multiple line style.
        log_path (str): Path to log file.

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


def evaluate(models: List[str], model_root: str, workspace: str):
    """Evaluate models from lmdeploy using opencompass.

    Args:
        models: Input models.
        model_root: Root directory of HF models.
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
        if model not in MODEL_CFGS:
            logging.error(f'Model {model} is not in {MODEL_CFGS.keys()}')
            continue
        hf_model_path = os.path.join(model_root,
                                     MODEL_CFGS[model]['model-path'])
        model_name = MODEL_CFGS[model]['model-name']
        if engine_type == 'tb':
            extra_kwargs = {}
            if do_lite:
                tmp_hf_model = './hf_calibrate'
                cmd_calibrate = [
                    f'lmdeploy lite calibrate --model {hf_model_path}',
                    '--calib-dataset c4', '--calib-samples 128',
                    '--calib-seqlen 2048', f'--work-dir {tmp_hf_model}'
                ]
                calibrate_log = os.path.join(workspace,
                                             f'calibrate.{ori_model}.txt')
                ret = run_cmd(cmd_calibrate, calibrate_log)
                if ret != 0:
                    continue
                cmd_awq = [
                    f'lmdeploy lite auto_awq --model {hf_model_path}',
                    '--w-bits 4', '--w-group-size 128',
                    f'--work-dir {tmp_hf_model}'
                ]
                awq_log = os.path.join(workspace, f'awq.{ori_model}.txt')
                ret = run_cmd(cmd_awq, awq_log)
                if ret != 0:
                    continue
                hf_model_path = tmp_hf_model
                extra_kwargs['model-formart'] = 'awq'
                extra_kwargs['group-size'] = 128

            target_model = './turbomind'
            # convert
            cmd_convert = [
                f'lmdeploy convert --model-path {hf_model_path}',
                f'--model-name {model_name}', f'--dst-path {target_model}'
            ]
            # for lite models
            cmd_convert += [f'--{k} {v}' for k, v in extra_kwargs.items()]
            convert_log = os.path.join(workspace, f'convert.{ori_model}.txt')
            ret = run_cmd(cmd_convert, log_path=convert_log)
            if ret != 0:
                logging.error(f'Convert failed for model {ori_model}')
                continue

            if do_lite and 'kv8' in precision:
                cmd_kvint8 = [
                    'lmdeploy lite kv_qparams', f'--work_dir {tmp_hf_model}',
                    f'--turbomind_dir {target_model}/triton_models/weights',
                    '--kv_sym False', '--num_tp 1'
                ]
                kvint8_log = os.path.join(workspace, f'kvint8.{ori_model}.txt')
                ret = run_cmd(cmd_kvint8, kvint8_log)
                if ret != 0:
                    logging.error(f'Failed to run kvint8 for {ori_model}')
                    continue
                config_ini = os.path.join(target_model,
                                          'triton_models/weights/config.ini')
                # update config.ini
                with open(config_ini, 'r+') as f:
                    content = f.read()
                    content = re.sub(r'quant_policy = [0-9]',
                                     'quant_policy = 4', content)
                    f.seek(0)
                    f.write(content)

        elif engine_type == 'pt':
            pass
        opencompass_dir = os.path.abspath(os.environ['OPENCOMPASS_DIR'])
        lmdeploy_dir = os.path.abspath(os.environ['LMDEPLOY_DIR'])
        config_path = os.path.join(
            lmdeploy_dir, '.github/scripts/eval_opencompass_config.py')
        config_path_new = os.path.join(opencompass_dir, 'configs',
                                       'eval_lmdeploy.py')
        shutil.copy(config_path, config_path_new)
        with open(config_path_new, 'a') as f:
            f.write(f'\nmodels = [ {ori_model} ]\n')

        work_dir = os.path.join(workspace, ori_model)
        cmd_eval = [
            f'python3 {opencompass_dir}/run.py {config_path_new} -w {work_dir}'
        ]
        eval_log = os.path.join(workspace, f'eval.{ori_model}.txt')
        ret = run_cmd(cmd_eval, log_path=eval_log)
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

        row = ','.join([ori_model] + [_[1] for _ in data])
        if not os.path.exists(output_csv):
            with open(output_csv, 'w') as f:
                header = ','.join(['Model'] + [_[0] for _ in data])
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
