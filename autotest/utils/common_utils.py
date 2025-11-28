import os
from subprocess import PIPE, Popen

import psutil


def run_cmd(cmd, benchmark_log):
    if os.path.isfile(benchmark_log):
        write_type = 'a'
    else:
        write_type = 'w'
    with open(benchmark_log, write_type) as f:
        f.writelines('reproduce command: ' + cmd + '\n')
        print('reproduce command: ' + cmd)
        with Popen([cmd], stdin=PIPE, stdout=f, stderr=PIPE, shell=True, text=True, encoding='utf-8') as process:
            try:
                stdout, stderr = process.communicate(None)
            except Exception:
                kill_process(process.pid)
                raise
            except:  # noqa: E722
                kill_process(process.pid)
                raise
            retcode = process.poll()
    return retcode, stderr


def kill_process(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()
