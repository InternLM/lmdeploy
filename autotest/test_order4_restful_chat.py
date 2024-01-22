import os
import subprocess
from time import sleep, time

import allure
import conftest
import pytest
from utils.run_client_chat import command_line_test
from utils.run_restful_chat import (health_check, interactive_test,
                                    open_chat_test)


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config):
    dst_path = config.get('dst_path')
    log_path = config.get('log_path')

    param = request.param
    model = param['model']
    port = param['port']

    cmd = ['lmdeploy serve api_server ' + dst_path + '/workspace_' + model]
    start_log = os.path.join(log_path, 'start_restful_' + model + '.log')

    with open(start_log, 'w') as f:
        subprocess.run(['pwd'],
                       stdout=f,
                       stderr=f,
                       shell=True,
                       text=True,
                       encoding='utf-8')
        f.writelines('commondLine: ' + ' '.join(cmd) + '\n')

        # convert
        convertRes = subprocess.Popen(cmd,
                                      stdout=f,
                                      stderr=f,
                                      shell=True,
                                      text=True,
                                      encoding='utf-8')
        pid = convertRes.pid
    allure.attach.file(start_log, attachment_type=allure.attachment_type.TEXT)

    http_url = 'http://localhost:' + str(port)
    start_time = int(time())
    for i in range(60):
        sleep(1)
        end_time = int(time())
        total_time = end_time - start_time
        result = health_check(http_url)
        if result or total_time >= 60:
            break
    yield
    if pid > 0:
        kill_log = os.path.join(log_path, 'kill_' + model + '.log')

        with open(kill_log, 'w') as f:
            convertRes.kill()

    allure.attach.file(kill_log, attachment_type=allure.attachment_type.TEXT)


conftest._init_common_case_list()
case_list = conftest.global_common_case_List


def getCaseList():
    return case_list


@pytest.mark.restful_api
@pytest.mark.timeout(120)
class Test_restful:

    @pytest.mark.internlm_chat_7b
    @allure.story('internlm-chat-7b')
    @pytest.mark.parametrize('prepare_environment', [{
        'model': 'internlm-chat-7b',
        'port': 23333
    }],
                             indirect=True)
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_restful_internlm_chat_7b(self, config, common_case_config,
                                      usercase):
        model = 'internlm-chat-7b'
        port = 23333

        run_all_step(config, usercase, common_case_config.get(usercase), model,
                     port)

    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b')
    @pytest.mark.parametrize('prepare_environment', [{
        'model': 'internlm2-chat-20b',
        'port': 23333
    }],
                             indirect=True)
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_restful_internlm2_chat_20b(self, config, common_case_config,
                                        usercase):
        model = 'internlm2-chat-20b'
        port = 23333

        run_all_step(config, usercase, common_case_config.get(usercase), model,
                     port)


def run_all_step(config, case, case_info, model, port):
    result = True

    msg = ''
    http_url = 'http://localhost:' + str(port)

    with allure.step('step1 - command chat regression'):
        chat_result, chat_log, commondmsg = command_line_test(
            config, case, case_info, model, 'api_client', http_url)
        result = result & chat_result
        msg += commondmsg
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)

    with allure.step('step2 - restful_test - openai chat'):
        restful_result, restful_log, restfulOpenAiMsg = open_chat_test(
            config, case_info, model, http_url)
        result = result & restful_result
        msg += restfulOpenAiMsg
        allure.attach.file(restful_log,
                           attachment_type=allure.attachment_type.TEXT)

    with allure.step('step3 - restful_test - interactive chat'):
        active_result, interactive_log, restfulActiveMsg = interactive_test(
            config, case_info, model, http_url)
        result = result & active_result
        msg += restfulActiveMsg
        allure.attach.file(interactive_log,
                           attachment_type=allure.attachment_type.TEXT)

    assert result, msg
