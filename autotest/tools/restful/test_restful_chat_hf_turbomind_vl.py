import os

import allure
import pytest
from openai import OpenAI
from utils.config_utils import get_vl_model_list, get_workerid
from utils.run_restful_chat import start_restful_api, stop_restful_api

from lmdeploy.serve.openai.api_client import APIClient

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    param = request.param
    model = param['model']
    model_path = config.get('model_path') + '/' + model

    pid, startRes = start_restful_api(config, param, model, model_path,
                                      'turbomind', worker_id)
    yield
    stop_restful_api(pid, startRes, param)


def getModelList(tp_num):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num
    } for item in get_vl_model_list(tp_num)]


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=1),
                         indirect=True)
def test_restful_chat_tp1(config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config)
    else:
        run_all_step(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=2),
                         indirect=True)
def test_restful_chat_tp2(config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config)
    else:
        run_all_step(config, port=DEFAULT_PORT + get_workerid(worker_id))


PIC = 'https://raw.githubusercontent.com/' + \
    'open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'


def run_all_step(config, port: int = DEFAULT_PORT):
    http_url = BASE_HTTP_URL + ':' + str(port)
    log_path = config.get('log_path')

    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id

    restful_log = os.path.join(log_path,
                               'restful_vl_' + model_name + str(port) + '.log')
    file = open(restful_log, 'w')

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
    file.writelines(str(response).lower() + '\n')
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
    file.writelines(str(item) + '\n')
    assert 'tiger' in str(item).lower(), item

    allure.attach.file(restful_log,
                       attachment_type=allure.attachment_type.TEXT)
