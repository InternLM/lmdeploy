import os
from multiprocessing import Process

import pytest
from utils.config_utils import get_cuda_id_by_workerid, get_torch_model_list
from utils.pipeline_chat import (assert_pipeline_chat_log,
                                 run_pipeline_chat_test)


def getModelList(tp_num):
    return [
        item for item in get_torch_model_list(tp_num)
        if 'falcon' not in item.lower() and 'chatglm2' not in item.lower()
    ]


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', getModelList(tp_num=1))
def test_pipeline_chat_pytorch_tp1(config, common_case_config, model,
                                   worker_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'pytorch'))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', getModelList(tp_num=2))
def test_pipeline_chat_pytorch_tp2(config, common_case_config, model,
                                   worker_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                 tp_num=2)
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'pytorch'))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@pytest.mark.parametrize('model', ['internlm2-chat-20b'])
def test_pipeline_chat_pytorch_pr(config, common_case_config, model):
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'pytorch'))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model)
