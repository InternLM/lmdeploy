import random
from time import sleep

import torch

from lmdeploy.model import MODELS


def get_conda_allcate_prefix(config, model):
    cuda_prefix = ''
    tp_num = get_tp_num(config, model)
    if tp_num is None:
        return cuda_prefix
    available_cuda = _get_available_cude()
    if len(available_cuda) < tp_num:
        raise torch.cuda.OutOfMemoryError

    cuda_prefix = 'CUDA_VISIBLE_DEVICES=' + ','.join(
        random.sample(available_cuda, tp_num))

    torch.cuda.empty_cache()
    return cuda_prefix


def get_tp_config(config, model, need_tp):
    tp_num = str(get_tp_num(config, model))
    tp_info = ''
    if need_tp and tp_num is not None:
        tp_info = '--tp ' + str(get_tp_num(config, model))
    return tp_info


def get_tp_num(config, model):
    tp_config = config.get('tp_config')
    tp_num = 1
    if tp_config is None:
        return None
    model_name = _simple_model_name(model)
    if model_name in tp_config.keys():
        tp_num = tp_config.get(model_name)
    return tp_num


def get_command_with_extra(cmd,
                           config,
                           model,
                           need_tp: bool = False,
                           cuda_prefix: str = None,
                           need_sleep: bool = True,
                           extra: str = None):
    if need_sleep:
        sleep(random.uniform(0, 5))
    if cuda_prefix is None:
        cuda_prefix = get_conda_allcate_prefix(config, model)
    tp_config = get_tp_config(config, model, need_tp)

    if cuda_prefix is not None and len(cuda_prefix) > 0:
        cmd = ' '.join([cuda_prefix, cmd])
    if tp_config is not None and len(tp_config) > 0:
        cmd = ' '.join([cmd, tp_config])
    if extra is not None and len(extra) > 0:
        cmd = ' '.join([cmd, extra])

    torch.cuda.empty_cache()
    return cmd


def get_model_name(model):
    model_names = [
        'llama', 'llama2', 'internlm', 'internlm2', 'baichuan2', 'chatglm2',
        'falcon', 'yi', 'qwen'
    ]
    model_names += list(MODELS.module_dict.keys())
    model_names.sort()
    model_name = _simple_model_name(model)
    model_name = model_name.lower()

    if model_name in model_names:
        return model_name
    model_name = model_name.replace('-chat', '')
    model_name = model_name.replace('-v0.1', '')
    if model_name in model_names:
        return model_name
    if ('llama-2' in model_name):
        return 'llama2'
    if ('llava' in model_name):
        return 'vicuna'
    if ('yi-vl' in model_name):
        return 'yi-vl'
    return model_name.split('-')[0]


def _get_available_cude():
    devices = torch.cuda.device_count()

    available_cuda = []
    for i in range(devices):
        if (torch.cuda.utilization(i) > 5):
            continue
        if ('no processes are running'
                not in torch.cuda.list_gpu_processes(i)):
            continue

        available_cuda.append(str(i))

    return available_cuda


def _simple_model_name(model):
    if '/' in model:
        model_name = model.split('/')[1]
    else:
        model_name = model
    model_name = model_name.replace('-inner-w4a16', '')
    return model_name


if __name__ == '__main__':
    print(_simple_model_name('baichuan-inc/Baichuan2-7B-Chat-inner-w4a16'))
