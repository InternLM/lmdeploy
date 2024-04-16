import pytest
from utils.config_utils import get_turbomind_model_list, get_workerid
from utils.run_restful_chat import (run_all_step, start_restful_api,
                                    stop_restful_api)

DEFAULT_PORT = 23333


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    param = request.param
    model = param['model']
    model_path = config.get('dst_path') + '/workspace_' + model

    pid, startRes = start_restful_api(config, param, model, model_path,
                                      'turbomind', worker_id)
    yield
    stop_restful_api(pid, startRes)


def getModelList(tp_num):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num
    } for item in get_turbomind_model_list(tp_num)]


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
    'model': 'internlm/internlm2-chat-20b-inner-4bits',
    'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
    'tp_num': 2
}],
                         indirect=True)
def test_restful_chat_pr(config, common_case_config):
    run_all_step(config, common_case_config)
