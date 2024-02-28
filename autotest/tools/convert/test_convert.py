import os
import subprocess
from subprocess import PIPE

import allure
import pytest
from utils.config_utils import get_turbomind_model_list
from utils.get_run_config import get_command_with_extra, get_model_name


@pytest.mark.order(5)
@pytest.mark.convert
@pytest.mark.parametrize('model', get_turbomind_model_list())
def test_convert(config, model):
    convert(config, model)


@pytest.mark.order(5)
@pytest.mark.convert
@pytest.mark.pr_test
@pytest.mark.parametrize(
    'model', ['internlm2-chat-20b', 'internlm2-chat-20b-inner-w4a16'])
def test_convert_pr(config, model):
    convert(config, model)


def convert(config, model_case):
    origin_model_path = config.get('model_path') + '/' + model_case
    dst_path = config.get('dst_path') + '/workspace_' + model_case
    log_path = config.get('log_path')

    model_name = get_model_name(model_case)

    if 'w4' in model_case:
        cmd = get_command_with_extra(
            ' '.join([
                'lmdeploy convert', model_name, origin_model_path,
                '--dst-path', dst_path, '--model-format awq --group-size 128'
            ]), config, model_name, True)
    else:
        cmd = get_command_with_extra(
            ' '.join([
                'lmdeploy convert', model_name, origin_model_path,
                '--dst-path', dst_path
            ]), config, model_name, True)

    convert_log = os.path.join(log_path, 'convert_' + model_case + '.log')
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
