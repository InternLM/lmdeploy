import os
import subprocess
import sys
from typing import Tuple


def execute_command_with_logging(cmd,
                                 log_file_path: str,
                                 timeout: int = 3600,
                                 env=None,
                                 should_print=True) -> Tuple[bool, str]:
    if env is None:
        env = os.environ.copy()

    if os.path.isfile(log_file_path):
        write_type = 'a'
    else:
        write_type = 'w'
    try:
        result = True
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
                                     env=env,
                                     bufsize=1,
                                     timeout=timeout,
                                     start_new_session=True)

            if process.stdout:
                if should_print:
                    print(process.stdout, end='')
                log_file.write(process.stdout)

            if process.returncode == 0:
                result_msg = 'execute command success!\n'
            else:
                result = False
                result_msg = f'execute command fail: {process.returncode}\n'

            log_file.write(result_msg)

        return result, result_msg.strip()

    except Exception as e:
        error_msg = f'execute command fail exception: {str(e)}\n'
        print(error_msg, file=sys.stderr, end='')

        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(error_msg)

        return False, error_msg.strip()
