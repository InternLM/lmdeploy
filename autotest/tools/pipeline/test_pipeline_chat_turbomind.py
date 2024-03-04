from multiprocessing import Process

import pytest
from utils.config_utils import get_turbomind_model_list
from utils.pipeline_chat import (assert_pipeline_chat_log,
                                 run_pipeline_chat_test)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list())
def test_pipeline_chat(config, common_case_config, model):
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'turbomind'))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@pytest.mark.parametrize(
    'model', ['internlm2-chat-20b', 'internlm2-chat-20b-inner-w4a16'])
def test_pipeline_chat_pr(config, common_case_config, model):
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'turbomind'))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model)
