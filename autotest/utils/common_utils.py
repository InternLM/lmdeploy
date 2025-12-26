import os
import subprocess
import sys
from typing import Tuple

import psutil


def execute_command_with_logging(cmd, log_file_path: str) -> Tuple[bool, str]:
    if os.path.isfile(log_file_path):
        write_type = 'a'
    else:
        write_type = 'w'
    try:
        with open(log_file_path, write_type, encoding='utf-8') as log_file:
            start_msg = f'execute command: {cmd}\n'
            print(start_msg, end='')
            log_file.write(start_msg)
            log_file.flush()

            process = subprocess.run(cmd,
                                     shell=True,
                                     text=True,
                                     encoding='utf-8',
                                     errors='replace',
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     bufsize=1)

            if process.stdout:
                print(process.stdout, end='')
                log_file.write(process.stdout)

            if process.returncode == 0:
                result = True
                result_msg = f'success: {process.returncode}\n'
            else:
                result = False
                result_msg = f'fail: {process.returncode}\n'

            print(result_msg, end='')
            log_file.write(result_msg)

            return result, result_msg.strip()

    except Exception as e:
        error_msg = f'exec fail: {str(e)}\n'
        print(error_msg, file=sys.stderr, end='')

        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(error_msg)

        return False, error_msg.strip()


def kill_process(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()
