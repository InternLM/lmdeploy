import os
import subprocess
from time import sleep

import allure
import conftest
import pytest
from utils.run_client_chat import commandLineTest
from utils.run_restful_chat import interactiveTest, openAiChatTest


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config):
    dst_path = config.get('dst_path')
    log_path = config.get('log_path')

    param = request.param
    model = param['model']
    port = param['port']

    cmd = [
        'lmdeploy serve api_server ' + dst_path + '/workspace_' + model +
        ' --server_name 0.0.0.0 --server_port ' + str(port) +
        ' --instance_num 32 --tp 1'
    ]
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
    sleep(20)
    yield
    if pid > 0:
        kill_log = os.path.join(log_path, 'kill_' + model + '.log')

        with open(kill_log, 'w') as f:
            convertRes.kill()

    allure.attach.file(kill_log, attachment_type=allure.attachment_type.TEXT)


conftest._init_restful_case_list()
case_list = conftest.global_restful_case_List


def getCaseList():
    return case_list


@pytest.mark.restful_api
class Test_restful:

    @pytest.mark.internlm_chat_7b
    @allure.story('internlm-chat-7b')
    @pytest.mark.parametrize('prepare_environment', [{
        'model': 'internlm-chat-7b',
        'port': 60006
    }],
                             indirect=True)
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_restful_internlm_chat_7b(self, config, restful_case_config,
                                      usercase):
        model = 'internlm-chat-7b'
        port = 60006

        run_all_step(config, usercase, restful_case_config.get(usercase),
                     model, port)


def run_all_step(config, case, case_info, model, port):
    result = True

    msg = ''
    http_url = 'http://localhost:' + str(port)

    with allure.step('step1 - command chat regression'):
        chat_result, chat_log, commondmsg = commandLineTest(
            config, case, case_info, model, 'api_client', http_url)
        result = result & chat_result
        msg += commondmsg

    with allure.step('step2 - restful_test - openai chat'):
        restful_result, restful_log, restfulOpenAiMsg = openAiChatTest(
            config, case_info, model, http_url)
        result = result & restful_result
        msg += restfulOpenAiMsg

    with allure.step('step3 - restful_test - interactive chat'):
        active_result, interactive_log, restfulActiveMsg = interactiveTest(
            config, case_info, model, http_url)
        result = result & active_result
        msg += restfulActiveMsg

    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    allure.attach.file(restful_log,
                       attachment_type=allure.attachment_type.TEXT)
    allure.attach.file(interactive_log,
                       attachment_type=allure.attachment_type.TEXT)

    assert result, msg
