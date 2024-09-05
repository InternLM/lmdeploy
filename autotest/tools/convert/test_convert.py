import os
import subprocess
from subprocess import PIPE

import allure
import pytest
from utils.config_utils import (get_cuda_prefix_by_workerid,
                                get_turbomind_model_list)
from utils.get_run_config import get_command_with_extra, get_model_name


@pytest.mark.order(5)
@pytest.mark.convert
@pytest.mark.parametrize('model',
                         get_turbomind_model_list(model_type='chat_model') +
                         get_turbomind_model_list(model_type='base_model'))
def test_convert(config, model, worker_id):
    convert(config, model, get_cuda_prefix_by_workerid(worker_id))


@pytest.mark.order(5)
@pytest.mark.convert
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('model', [
    'internlm/internlm2_5-20b-chat',
    'internlm/internlm2_5-20b-chat-inner-4bits'
])
def test_convert_pr(config, model):
    convert(config, model, 'CUDA_VISIBLE_DEVICES=5')


def convert(config, model_case, cuda_prefix):
    origin_model_path = config.get('model_path') + '/' + model_case
    dst_path = config.get('dst_path') + '/workspace_' + model_case
    log_path = config.get('log_path')

    model_name = get_model_name(model_case)

    if 'w4' in model_case or ('4bits' in model_case
                              or 'awq' in model_case.lower()):
        cmd = get_command_with_extra(' '.join([
            'lmdeploy convert', model_name, origin_model_path, '--dst-path',
            dst_path, '--model-format awq --group-size 128'
        ]),
                                     config,
                                     model_case,
                                     True,
                                     cuda_prefix=cuda_prefix)
    elif 'gptq' in model_case.lower():
        cmd = get_command_with_extra(' '.join([
            'lmdeploy convert', model_name, origin_model_path, '--dst-path',
            dst_path, '--model-format gptq --group-size 128'
        ]),
                                     config,
                                     model_case,
                                     True,
                                     cuda_prefix=cuda_prefix)
    else:
        cmd = get_command_with_extra(' '.join([
            'lmdeploy convert', model_name, origin_model_path, '--dst-path',
            dst_path
        ]),
                                     config,
                                     model_case,
                                     True,
                                     cuda_prefix=cuda_prefix)

    convert_log = os.path.join(log_path,
                               'convert_' + model_case.split('/')[1] + '.log')
    print('reproduce command convert: ' + cmd + '\n')
    with open(convert_log, 'w') as f:
        # remove existing workspace
        subprocess.run([' '.join(['rm -rf', dst_path])],
                       stdout=f,
                       stderr=f,
                       shell=True,
                       text=True,
                       encoding='utf-8')

        f.writelines('reproduce command convert: ' + cmd + '\n')

        # convert
        convertRes = subprocess.run([cmd],
                                    stdout=f,
                                    stderr=PIPE,
                                    shell=True,
                                    text=True,
                                    encoding='utf-8')
        f.writelines(convertRes.stderr)
        # check result
        result = convertRes.returncode == 0

    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)

    assert result, convertRes.stderr
