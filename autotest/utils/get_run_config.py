import random
from time import sleep

import torch


def get_conda_allcate_prefix(config, model):
    tp_config = config.get('tp_config')
    cuda_prefix = ''
    if tp_config is None:
        return cuda_prefix

    if model in tp_config.keys():
        cuda_num = tp_config.get(model)
    else:
        cuda_num = 1

    sleep(random.uniform(0, 5))
    available_cuda = _get_available_cude()
    if len(available_cuda) < cuda_num:
        raise torch.cuda.OutOfMemoryError

    cuda_prefix = 'CUDA_VISIBLE_DEVICES=' + ','.join(
        random.sample(available_cuda, cuda_num))

    torch.cuda.empty_cache()
    return cuda_prefix


def get_tp_config(config, model, need_tp):
    tp_config = config.get('tp_config')
    tp_info = ''
    if tp_config is None or need_tp is False:
        return tp_info
    if model in tp_config.keys() and need_tp:
        tp_info = '--tp ' + str(tp_config.get(model))
    return tp_info


def get_tp_num(config, model):
    tp_config = config.get('tp_config')
    tp_num = 1
    if tp_config is None:
        return tp_num
    if model in tp_config.keys():
        tp_num = tp_config.get(model)
    return tp_num


def get_command_with_extra(cmd,
                           config,
                           model,
                           need_tp: bool = False,
                           cuda_prefix: str = None):
    if cuda_prefix is None:
        cuda_prefix = get_conda_allcate_prefix(config, model)
    tp_config = get_tp_config(config, model, need_tp)

    if cuda_prefix is not None and len(cuda_prefix) > 0:
        cmd = ' '.join([cuda_prefix, cmd])
    if tp_config is not None and len(tp_config) > 0:
        cmd = ' '.join([cmd, tp_config])

    torch.cuda.empty_cache()
    return cmd


def _get_available_cude():
    devices = torch.cuda.device_count()

    available_cuda = []
    for i in range(devices):
        if (torch.cuda.utilization(i) > 30):
            continue
        if ('no processes are running'
                not in torch.cuda.list_gpu_processes(i)):
            continue

        available_cuda.append(str(i))

    return available_cuda


if __name__ == '__main__':
    print(_get_available_cude())
