import os
import subprocess
from time import sleep, time

import allure
import pytest
from pytest import assume
from utils.config_utils import (get_all_model_list,
                                get_cuda_prefix_by_workerid, get_workerid)
from utils.get_run_config import get_command_with_extra
from utils.run_client_chat import command_line_test
from utils.run_restful_chat import (get_model, health_check, interactive_test,
                                    open_chat_test)

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    model_path = config.get('model_path')
    log_path = config.get('log_path')

    param = request.param
    model = param['model']
    cuda_prefix = param['cuda_prefix']
    tp_num = param['tp_num']

    if cuda_prefix is None:
        cuda_prefix = get_cuda_prefix_by_workerid(worker_id, tp_num=tp_num)

    worker_num = get_workerid(worker_id)
    if worker_num is None:
        port = DEFAULT_PORT
    else:
        port = DEFAULT_PORT + worker_num

    cmd = ['lmdeploy serve api_server ' + model_path + '/' + model]

    cmd = get_command_with_extra('lmdeploy serve api_server ' + model_path +
                                 '/' + model + ' --server-port ' + str(port),
                                 config,
                                 model,
                                 need_tp=True,
                                 cuda_prefix=cuda_prefix)

    if 'kvint8' in model:
        cmd += ' --quant-policy 4'
        if 'w4' in model or '4bits' in model:
            cmd += ' --model-format awq'
        else:
            cmd += ' --model-format hf'
    if 'w4' in model or '4bits' in model:
        cmd += ' --model-format awq'

    start_log = os.path.join(log_path,
                             'start_restful_' + model.split('/')[1] + '.log')

    print('reproduce command restful: ' + cmd)

    with open(start_log, 'w') as f:
        f.writelines('reproduce command restful: ' + cmd + '\n')

        # convert
        convertRes = subprocess.Popen([cmd],
                                      stdout=f,
                                      stderr=f,
                                      shell=True,
                                      text=True,
                                      encoding='utf-8')
        pid = convertRes.pid
    allure.attach.file(start_log, attachment_type=allure.attachment_type.TEXT)

    http_url = BASE_HTTP_URL + ':' + str(port)
    start_time = int(time())
    sleep(5)
    for i in range(120):
        sleep(1)
        end_time = int(time())
        total_time = end_time - start_time
        result = health_check(http_url)
        if result or total_time >= 120:
            break
    yield
    if pid > 0:
        kill_log = os.path.join(log_path,
                                'kill_' + model.split('/')[1] + '.log')

        with open(kill_log, 'w') as f:
            convertRes.kill()

    allure.attach.file(kill_log, attachment_type=allure.attachment_type.TEXT)


def getModelList(tp_num):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num
    } for item in get_all_model_list(tp_num) if 'chat' in item.lower()]


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=1),
                         indirect=True)
def test_restful_chat_tp1(request, config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config,
                     common_case_config,
                     worker_id=worker_id,
                     port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=2),
                         indirect=True)
def test_restful_chat_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config,
                     common_case_config,
                     worker_id=worker_id,
                     port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'internlm/internlm2-chat-20b',
    'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
    'tp_num': 2
}, {
    'model': 'internlm/internlm2-chat-20b-inner-w4a16',
    'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
    'tp_num': 2
}],
                         indirect=True)
def test_restful_chat_pr(config, common_case_config):
    run_all_step(config, common_case_config)


def run_all_step(config,
                 cases_info,
                 worker_id: str = 'default',
                 port: int = DEFAULT_PORT):
    http_url = BASE_HTTP_URL + ':' + str(port)

    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'
    for case in cases_info.keys():
        if (case == 'memory_test'
                or case == 'emoji_case') and 'chat' not in model.lower():
            continue

        case_info = cases_info.get(case)

        with allure.step(case + ' step1 - command chat regression'):
            chat_result, chat_log, msg = command_line_test(
                config, case, case_info, model, 'api_client', http_url)
            allure.attach.file(chat_log,
                               attachment_type=allure.attachment_type.TEXT)
        with assume:
            assert chat_result, msg

        with allure.step(case + ' step2 - restful_test - openai chat'):
            restful_result, restful_log, msg = open_chat_test(
                config, case_info, model, http_url)
            allure.attach.file(restful_log,
                               attachment_type=allure.attachment_type.TEXT)
        with assume:
            assert restful_result, msg

        with allure.step(case + ' step3 - restful_test - interactive chat'):
            active_result, interactive_log, msg = interactive_test(
                config, case_info, model, http_url)
            allure.attach.file(interactive_log,
                               attachment_type=allure.attachment_type.TEXT)

        with assume:
            assert active_result, msg
