import random

import torch


def get_conda_allcate_prefix(config, model):
    cuda_allocate = config.get('quantization_cuda_allocate')
    tp_config = config.get('tp_config')
    cuda_prefix = ''
    if cuda_allocate is None or tp_config is None:
        return cuda_prefix

    if model in tp_config.keys():
        cuda_num = tp_config.get(model)
    else:
        cuda_num = 1

    available_cuda = _get_available_cude()
    if len(available_cuda) < cuda_num:
        raise torch.cuda.OutOfMemoryError

    cuda_prefix = 'CUDA_VISIBLE_DEVICES=' + ','.join(
        random.sample(available_cuda, cuda_num))
    return cuda_prefix


def get_tp_config(config, model):
    tp_config = config.get('tp_config')
    tp_info = ''
    if tp_config is None:
        return tp_info
    if model in tp_config.keys():
        tp_info = '--tp ' + str(tp_config.get(model))
    return tp_info


def get_command_with_extra(cmd, config, model):
    cuda_prefix = get_conda_allcate_prefix(config, model)
    tp_config = get_tp_config(config, model)

    if cuda_prefix is not None and len(cuda_prefix) > 0:
        cmd = ' '.join([cuda_prefix, cmd])
    if tp_config is not None and len(tp_config) > 0:
        cmd = ' '.join([cmd, tp_config])
    return cmd


def _get_available_cude():
    devices = torch.cuda.device_count()

    available_cuda = []
    for i in range(devices):
        if (torch.cuda.utilization(i) > 30):
            continue
        mem_info = torch.cuda.mem_get_info(i)
        if mem_info[0] / mem_info[1] < 0.95:
            continue
        available_cuda.append(str(i))

    return available_cuda
