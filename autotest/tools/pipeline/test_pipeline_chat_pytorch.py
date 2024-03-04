from multiprocessing import Process

import pytest
from utils.config_utils import get_torch_model_list
from utils.pipeline_chat import (assert_pipeline_chat_log,
                                 run_pipeline_chat_test)


def getModelList():
    return [
        item for item in get_torch_model_list()
        if 'falcon' not in item.lower() and 'chatglm2' not in item.lower()
    ]


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', getModelList())
def test_pipeline_chat_pytorch(config, common_case_config, model):
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
