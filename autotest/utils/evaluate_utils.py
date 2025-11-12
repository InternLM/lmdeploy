import csv
import glob
import os
import subprocess

from mmengine.config import Config

DEFAULT_PORT = 23333


def write_to_summary(model_name, tp_num, result, msg, worker_id, backend_type, communicator, work_dir=None):
    status = '✅ PASS' if result else '❌ FAIL'

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

    dataset_name = []
    dataset_metrics = []
    for key in sorted(metrics.keys()):
        dataset_name.append(key)
        dataset_metrics.append(metrics.get(key, ''))

    summary_dataset_name = ' | '.join(dataset_name)
    summary_dataset_metrics = ' | '.join(dataset_metrics)

    summary_file = os.environ.get('GITHUB_STEP_SUMMARY', None)
    summary_line = f'| {model_name} | {backend_type} | {communicator} | TP{tp_num} | {status} | {summary_dataset_metrics} |\n'  # noqa: E501
    if summary_file:
        write_header = not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0
        with open(summary_file, 'a') as f:
            if write_header:
                dash_line = '-----|' * (len(metrics.keys()))
                f.write('## Model Evaluation Results\n')
                f.write(f'| Model | Backend | Communicator | TP | Status | {summary_dataset_name} |\n')
                f.write(f'|-------|---------|--------------|----|--------|{dash_line}\n')
            f.write(summary_line)
    else:
        print(
            f'Summary: {model_name} | {backend_type} | {communicator} | TP{tp_num} | {status} | {summary_dataset_metrics}'  # noqa: E501
        )


def restful_test(config, run_id, prepare_environment, worker_id='gw0', port=DEFAULT_PORT, test_type='infer', **kwargs):
    work_dir = None
    try:
        model_name = prepare_environment['model']
        backend_type = prepare_environment['backend']
        tp_num = prepare_environment.get('tp_num', 1)
        communicator = prepare_environment.get('communicator', 'cuda-ipc')
        quant_policy = prepare_environment.get('quant_policy', 0)

        summary_model_name = model_name
        if quant_policy in [4, 8]:
            summary_model_name = f'{model_name}-kvint{quant_policy}'

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        config_file = os.path.join(parent_dir, 'evaluate/eval_config_chat.py')

        model_base_path = config.get('model_path', '/nvme/qa_test_models')
        model_path = os.path.join(model_base_path, model_name)

        print(f'Starting OpenCompass evaluation for model: {model_name}')
        print(f'Model path: {model_path}')
        print(f'Backend: {backend_type}')
        print(f'Config file: {config_file}')

        log_path = config.get('eval_log_path', '/nvme/qa_test_models/evaluation_report') + f'/{run_id}'
        os.makedirs(log_path, exist_ok=True)

        original_cwd = os.getcwd()
        work_dir = os.path.join(log_path,
                                f"wk_{backend_type}_{model_name.replace('/', '_')}_{communicator}_{quant_policy}")
        os.makedirs(work_dir, exist_ok=True)

        try:

            temp_config_file = f"temp_{backend_type}_{summary_model_name.replace('/', '_')}_{communicator}.py"
            temp_config_path = os.path.join(log_path, temp_config_file)

            if test_type == 'infer':
                if not os.path.exists(config_file):
                    return False, f'Config file {config_file} not found'

                cfg = Config.fromfile(config_file)

                cfg.MODEL_NAME = summary_model_name
                cfg.MODEL_PATH = model_path
                cfg.API_BASE = f'http://127.0.0.1:{port}/v1'  # noqa: E231

                if cfg.models and len(cfg.models) > 0:
                    model_cfg = cfg.models[0]
                    model_cfg['abbr'] = f'{summary_model_name}-lmdeploy-api'
                    model_cfg['openai_api_base'] = f'http://127.0.0.1:{port}/v1'  # noqa: E231
                    model_cfg['path'] = model_path

                    for key, value in kwargs.items():
                        model_cfg[key] = value

                cfg.dump(temp_config_path)
                print(f'Modified config saved to: {temp_config_path}')
            elif test_type == 'eval':
                if not os.path.exists(temp_config_path):
                    error_msg = f'Temp config file {temp_config_path} not found for eval stage'
                    write_to_summary(summary_model_name, tp_num, False, error_msg, worker_id, backend_type,
                                     communicator, work_dir)
                    return False, error_msg

                cfg = Config.fromfile(temp_config_path)
                print(f'Using existing temp config file: {temp_config_path}')

                cfg.JUDGE_API_BASE = f'http://127.0.0.1:{port}/v1'
                cfg.JUDGE_MODEL_PATH = os.path.join(model_base_path, 'Qwen/Qwen2.5-32B-Instruct')

                if hasattr(cfg, 'judge_cfg'):
                    cfg.judge_cfg['path'] = cfg.JUDGE_MODEL_PATH
                    cfg.judge_cfg['openai_api_base'] = cfg.JUDGE_API_BASE
                    cfg.judge_cfg['tokenizer_path'] = cfg.JUDGE_MODEL_PATH

                if hasattr(cfg, 'datasets') and cfg.datasets:
                    for dataset in cfg.datasets:
                        if 'eval_cfg' in dataset and 'evaluator' in dataset['eval_cfg']:
                            evaluator = dataset['eval_cfg']['evaluator']

                            if 'judge_cfg' in evaluator:
                                evaluator['judge_cfg']['path'] = cfg.JUDGE_MODEL_PATH
                                evaluator['judge_cfg']['openai_api_base'] = cfg.JUDGE_API_BASE
                                evaluator['judge_cfg']['tokenizer_path'] = cfg.JUDGE_MODEL_PATH

                            if 'llm_evaluator' in evaluator and 'judge_cfg' in evaluator['llm_evaluator']:
                                evaluator['llm_evaluator']['judge_cfg']['path'] = cfg.JUDGE_MODEL_PATH
                                evaluator['llm_evaluator']['judge_cfg']['openai_api_base'] = cfg.JUDGE_API_BASE
                                evaluator['llm_evaluator']['judge_cfg']['tokenizer_path'] = cfg.JUDGE_MODEL_PATH

                cfg.dump(temp_config_path)
                print(f'Modified config for eval stage saved to: {temp_config_path}')

            cmd = [
                'opencompass', temp_config_path, '--reuse', '--max-num-workers', '16', '-w', work_dir, '-m', test_type
            ]
            print(f"Running command: {' '.join(cmd)}")
            print(f'Work directory: {work_dir}')

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=259200)

            stdout_output = result.stdout
            stderr_output = result.stderr

            log_filename = (f'{test_type}_{backend_type}_'
                            f"{model_name.replace('/', '_')}_"
                            f'{communicator}_'
                            f'{worker_id}_'
                            f'{quant_policy}.log')
            log_file = os.path.join(log_path, log_filename)
            cmd_command = ' '.join(cmd)

            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f'Model: {model_name}\n')
                f.write(f'Config file: {temp_config_file}\n')
                f.write(f'Backend: {backend_type}\n')
                f.write(f'TP Num: {tp_num}\n')
                f.write(f'Command: {cmd_command}\n')
                f.write(f'Work directory: {work_dir}\n')
                f.write(f'STDOUT: \n{stdout_output}\n')
                if stderr_output:
                    f.write(f'STDERR: \n{stderr_output}\n')
                f.write(f'Return code: {result.returncode}\n')

            print(f'STDOUT: \n{stdout_output}')
            if stderr_output:
                print(f'STDERR: \n{stderr_output}')
            print(f'Return code: {result.returncode}')

            evaluation_failed = False
            error_keywords = ['ERROR -', 'fail, see', 'task .* fail']
            for line in stdout_output.split('\n'):
                if any(keyword in line for keyword in error_keywords):
                    evaluation_failed = True
                    break

            if result.returncode == 0 and not evaluation_failed:
                final_result = True
                final_msg = f'Evaluation completed successfully for {model_name}'
            else:
                final_result = False
                final_msg = f'Evaluation failed for {model_name}'
                if result.returncode != 0:
                    final_msg += f'with return code {result.returncode}'
                elif evaluation_failed:
                    final_msg += 'with internal errors detected in logs'

                if stderr_output:
                    final_msg += f'\nSTDERR: {stderr_output}'
                else:
                    error_lines = []
                    for line in stdout_output.split('\n'):
                        if any(keyword in line for keyword in error_keywords):
                            error_lines.append(line)
                    if error_lines:
                        error_lines = ' | '.join(error_lines[:3])
                        final_msg += f'\nLog errors: {error_lines}'

            if test_type == 'eval':
                write_to_summary(summary_model_name, tp_num, final_result, final_msg, worker_id, backend_type,
                                 communicator, work_dir)

            return final_result, final_msg

        finally:
            os.chdir(original_cwd)
            print(f'Returned to directory: {original_cwd}')

    except subprocess.TimeoutExpired:
        timeout_msg = (f'Evaluation timed out for {model_name} '
                       f'after 259200 seconds')
        if work_dir and test_type == 'eval':
            write_to_summary(summary_model_name, tp_num, False, timeout_msg, worker_id, backend_type, communicator,
                             work_dir)
        return False, timeout_msg
    except Exception as e:
        error_msg = f'Error during evaluation for {model_name}: {str(e)}'
        if work_dir and test_type == 'eval':
            write_to_summary(summary_model_name, tp_num, False, error_msg, worker_id, backend_type, communicator,
                             work_dir)
        return False, error_msg
