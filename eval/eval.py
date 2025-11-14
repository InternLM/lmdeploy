import argparse
import os
import subprocess
from datetime import datetime


def read_config():
    """Get configuration content from config file in script directory.

    Returns:
        str: Configuration file content, returns None if reading fails
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.py')

    # Read config file content
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        return config_content
    except FileNotFoundError:
        print(f'Error: Config file not found at {config_path}')
        return None
    except Exception as e:
        print(f'Error reading config file: {e}')
        return None


def update_datasets(config, datasets):
    """Update datasets part in config according to datasets list.

    Args:
        config (str): Original configuration content
        datasets (list[str]): List of dataset names to include
    Returns:
        str: Updated configuration content
    """
    if 'all' in datasets:
        # datasets part of the config file specifies all datasets, no need to update
        return config

    selected_datasets = []
    if 'code' in datasets:
        selected_datasets.append('[LCBCodeGeneration_dataset]')
        datasets.remove('code')
    for d in datasets:
        selected_datasets.append(f'{d}_datasets')
    selected_datasets = ' + '.join(selected_datasets)
    selected_datasets = f'datasets = {selected_datasets}'

    # replace datasets part in config
    start_tag = '# <dataset_replace_tag>'
    end_tag = '# </dataset_replace_tag>'

    start_index = config.find(start_tag)
    end_index = config.find(end_tag)

    if start_index == -1 or end_index == -1:
        raise ValueError('replace tag not found in config file')

    end_index += len(end_tag)
    replacement = f'{start_tag}\n{selected_datasets}\n{end_tag}'
    result = config[:start_index] + replacement + config[end_index:]
    return result


def get_model_name_from_server(server: str, tag: str) -> str:
    from openai import OpenAI
    try:
        client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{server}/v1')
        model_name = client.models.list().data[0].id
        return model_name
    except Exception as e:
        raise RuntimeError(f'Failed to get model name from {tag}_server {server}: {e}')


def save_config(work_dir: str, config: str):
    """Save configuration content to a file in the specified directory.

    Args:
        work_dir (str): Directory to save the configuration file
        config (str): Configuration content to save
    """
    if not work_dir:
        return
    os.makedirs(work_dir, exist_ok=True)
    output_file = os.path.join(work_dir, 'config.py')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(config)
    print(f'Config written to {output_file}')


def perform_evaluation(config, api_server, judger_server, mode, work_dir, reuse):
    """Perform model evaluation by opencompass.

    Args:
        config (str): Configuration content
        api_server (str): API server address for inference
        judger_server (str): Judger server address for evaluation
        mode (str): Running mode selection, options: infer, eval, all, config
        work_dir (str): Output directory for evaluation results. If not specified,
            config will not be saved and execution will not be performed.
        reuse (str): Whether to reuse existing results
    """
    if mode in ['infer', 'all']:
        served_model_name = get_model_name_from_server(api_server, 'api')
        config = config.replace("MODEL_PATH = ''", f"MODEL_PATH = '{served_model_name}'")
    if mode in ['eval', 'all']:
        judger_model_name = get_model_name_from_server(judger_server, 'judger')
        config = config.replace("JUDGER_MODEL_PATH = ''", f"JUDGER_MODEL_PATH = '{judger_model_name}'")

    # write updated config to work_dir
    if work_dir:
        save_config(work_dir, config)
    else:
        print(config)
        return

    # execute opencompass command
    cmd = ['opencompass', f'{work_dir}/config.py', '-m', mode, '-w', work_dir]
    if reuse:
        # reuse previous outputs & results. If reuse is a string, it indicates a specific timestamp.
        try:
            datetime.strptime(reuse, '%Y%m%d_%H%M%S')
            cmd.extend(['-r', str(reuse)])
        except ValueError as e:
            print(e)
            raise ValueError(f'Invalid reuse timestamp format: {reuse}. Expected format: YYYYMMDD_HHMMSS') from e
    try:
        print(f'Executing command: {" ".join(cmd)}')
        result = subprocess.run(cmd, text=True, check=True)
        return result
    except Exception as e:
        print(f'命令执行失败！错误信息: {e}')
        return


def main():
    parser = argparse.ArgumentParser(description='Perform model evaluation')
    parser.add_argument('task_name', type=str, help='The name of an evaluation task')
    parser.add_argument('-a', '--api-server', type=str, default='', help='API server address for inference')
    parser.add_argument('-j', '--judger-server', type=str, default='', help='Judger server address for evaluation')
    dataset_choices = ['aime2025', 'gpqa', 'ifeval', 'code', 'mmlu_pro', 'hle', 'all']
    parser.add_argument('-d',
                        '--datasets',
                        nargs='+',
                        choices=dataset_choices,
                        default=['all'],
                        help=f"List of datasets. Available options: {', '.join(dataset_choices)}. "
                        'Use "all" to include all datasets.')
    parser.add_argument('-w',
                        '--work-dir',
                        type=str,
                        default='',
                        help='Output directory of evaluation. If not specified, outputs will not be saved.')
    parser.add_argument('-r',
                        '--reuse',
                        nargs='?',
                        type=str,
                        const='latest',
                        help='Reuse previous outputs & results, and run any missing jobs presented in the config. '
                        'If its argument is not specified, the latest results in the work_dir will be reused. '
                        'The argument should also be a specific timestamp, e.g. 20230516_144254')
    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        help='Running mode selection. '
                        'all: complete pipeline including both inference and evaluation (default). '
                        'infer: only perform model inference to generate results. '
                        'eval: only evaluate previously generated results. '
                        'config: generate configuration files without execution.',
                        choices=['all', 'infer', 'eval', 'config'],
                        default='all')
    args = parser.parse_args()
    task_name = args.task_name
    api_server = args.api_server
    judger_server = args.judger_server
    datasets = args.datasets
    mode = args.mode
    work_dir = args.work_dir

    # Process server addresses
    if api_server and not api_server.startswith('http'):
        api_server = f'http://{api_server}'
    if judger_server and not judger_server.startswith('http'):
        judger_server = f'http://{judger_server}'

    # read config file
    config = read_config()

    # update task name in config
    config = config.replace("TASK_TAG = ''", f"TASK_TAG = '{task_name}'")

    # update datasets part of config according to args.datasets
    config = update_datasets(config, datasets)

    # update api_server part of config according to args.api_server
    if api_server:
        config = config.replace("API_SERVER_ADDR = 'http://<API_SERVER>'", f"API_SERVER_ADDR = '{api_server}'")
    if judger_server:
        # update judger_server part of config according to args.judger_server
        config = config.replace("JUDGER_ADDR = 'http://<JUDGER_SERVER>'", f"JUDGER_ADDR = '{judger_server}'")

    if mode == 'config':
        try:
            # if api_server is accessible, retrieve /v1/models to get model_name
            served_model_name = get_model_name_from_server(api_server, 'api')
            config = config.replace("MODEL_PATH = ''", f"MODEL_PATH = '{served_model_name}'")
        finally:
            pass
        try:
            # if judger_server is accessible, retrieve /v1/models to get model_name
            judger_model_name = get_model_name_from_server(judger_server, 'judger')
            config = config.replace("JUDGER_MODEL_PATH = ''", f"JUDGER_MODEL_PATH = '{judger_model_name}'")
        finally:
            pass
        # write updated config to work_dir
        if work_dir:
            save_config(work_dir, config)
        else:
            print(config)
            return

    # perform evaluation
    perform_evaluation(config, api_server, judger_server, mode, work_dir, args.reuse)


if __name__ == '__main__':
    main()
