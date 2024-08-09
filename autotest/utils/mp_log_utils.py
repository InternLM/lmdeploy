import os

import allure
from pytest import assume


def write_log(config,
              result,
              msg,
              is_new: bool = True,
              case_path_tag: str = 'default'):
    try:
        log_path = os.path.join(config.get('log_path'), case_path_tag)

        if is_new:
            file = open(log_path, 'w')
        else:
            file = open(log_path, 'a')

        file.writelines('result:' + result + ', reason:' + msg + '\n')
        file.close()
    except Exception as e:
        return False, None, f'Unknown error: {e}'


def assert_log(config, case_path_tag: str = 'default'):
    log_path = os.path.join(config.get('log_path'), case_path_tag)

    with open(log_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if 'result:False, reason:' in line:
                result = False
                msg = line
                break
            if 'result:True, reason:' in line and not result:
                result = True

    allure.attach.file(log_path, attachment_type=allure.attachment_type.TEXT)
    with assume:
        assert result, msg
