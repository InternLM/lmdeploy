import os
import subprocess
from time import sleep, time

import allure
import pytest
from openai import OpenAI
from utils.config_utils import (get_cuda_prefix_by_workerid, get_vl_model_list,
                                get_workerid)
from utils.get_run_config import get_command_with_extra
from utils.run_restful_chat import health_check

from lmdeploy.serve.openai.api_client import APIClient

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

    cmd = get_command_with_extra('lmdeploy serve api_server ' + model_path +
                                 '/' + model + ' --server-port ' + str(port),
                                 config,
                                 model,
                                 need_tp=True,
                                 cuda_prefix=cuda_prefix)

    start_log = os.path.join(log_path,
                             'start_restful_' + model.split('/')[1] + '.log')

    print('reproduce command restful: ' + cmd)

    with open(start_log, 'w') as f:
        f.writelines('reproduce command restful: ' + cmd + '\n')

        startRes = subprocess.Popen([cmd],
                                    stdout=f,
                                    stderr=f,
                                    shell=True,
                                    text=True,
                                    encoding='utf-8')
        pid = startRes.pid
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
            startRes.kill()

    allure.attach.file(kill_log, attachment_type=allure.attachment_type.TEXT)


def getModelList(tp_num):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num
    } for item in get_vl_model_list(tp_num) if 'chat' in item.lower()]


@pytest.mark.order(7)
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=1),
                         indirect=True)
def test_restful_chat_tp1(worker_id):
    if get_workerid(worker_id) is None:
        run_all_step()
    else:
        run_all_step(port=DEFAULT_PORT + get_workerid(worker_id))


PIC = 'https://raw.githubusercontent.com/' + \
    'open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'


def run_all_step(port: int = DEFAULT_PORT):
    http_url = BASE_HTTP_URL + ':' + str(port)

    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': 'Describe the image please',
            }, {
                'type': 'image_url',
                'image_url': {
                    'url': PIC,
                },
            }],
        }],
        temperature=0.8,
        top_p=0.8)
    assert 'tiger' in str(response).lower(), response

    api_client = APIClient(http_url)
    model_name = api_client.available_models[0]
    messages = [{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'Describe the image please',
        }, {
            'type': 'image_url',
            'image_url': {
                'url': PIC,
            },
        }]
    }]
    for item in api_client.chat_completions_v1(model=model_name,
                                               messages=messages):
        continue
    assert 'tiger' in str(item).lower(), item
