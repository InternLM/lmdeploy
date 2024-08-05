import os
from multiprocessing import Process

import pytest
from utils.config_utils import get_cuda_id_by_workerid, get_vl_model_list
from utils.pipeline_chat import (assert_pipeline_vl_chat_log,
                                 run_pipeline_vl_chat_test)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_vl_model_list(tp_num=1))
def test_pipeline_chat_tp1(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    p = Process(target=run_pipeline_vl_chat_test, args=(config, model))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_vl_model_list(tp_num=2))
def test_pipeline_chat_tp2(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=2)
    p = Process(target=run_pipeline_vl_chat_test, args=(config, model))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model)


@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.pr_test
@pytest.mark.parametrize('model', [
    'liuhaotian/llava-v1.6-vicuna-7b', 'OpenGVLab/InternVL2-4B',
    'OpenGVLab/InternVL2-8B', 'internlm/internlm-xcomposer2d5-7b'
])
def test_pipeline_pr_test(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(
            int(get_cuda_id_by_workerid(worker_id)) + 5)
    p = Process(target=run_pipeline_vl_chat_test, args=(config, model))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model)
