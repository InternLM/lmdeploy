import os
import random
import re
import subprocess
from time import sleep

import torch

from lmdeploy.model import MODELS


def get_conda_allcate_prefix(config, model):
    device = os.environ.get('DEVICE', 'cuda')  # Default to cuda if not set
    handler = _get_device_handler(device)
    return handler.get_device_prefix(config, model)


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

    _clear_device_cache()
    return cmd


def get_model_name(model):
    model_names = ['llama', 'llama2', 'llama3', 'internlm', 'internlm2', 'baichuan2', 'chatglm2', 'yi', 'qwen']
    model_names += list(MODELS.module_dict.keys())
    model_names.sort()
    model_name = _simple_model_name(model)
    model_name = model_name.lower()

    if model_name in model_names:
        return model_name
    if model_name in model_names:
        return model_name
    if ('llama-2' in model_name):
        return 'llama2'
    if ('llama-3-1' in model_name):
        return 'llama3_1'
    if ('llama-3' in model_name):
        return 'llama3'
    if 'vicuna' in model_name and 'llava' not in model_name:
        return 'vicuna'
    if 'llava' in model_name and 'v1' in model_name and 'v1.6-34b' not in model_name and 'mistral' not in model_name:
        return 'llava-v1'
    if 'llava' in model_name and 'v1.6-34b' in model_name:
        return 'llava-chatml'
    if 'internvl-chat' in model_name and 'v1-2' in model_name:
        return 'internvl-zh-hermes2'
    elif 'llava-1.5' in model_name:
        return 'llava-v1'
    if ('yi-vl' in model_name):
        return 'yi-vl'
    if ('qwen' in model_name):
        return 'qwen'
    if ('internvl') in model_name:
        return 'internvl-internlm2'
    if ('internlm2') in model_name:
        return 'internlm2'
    if ('internlm-xcomposer2d5') in model_name:
        return 'internlm-xcomposer2d5'
    if ('internlm-xcomposer2') in model_name:
        return 'internlm-xcomposer2'
    if ('glm-4') in model_name:
        return 'glm4'
    if len(model_name.split('-')) > 2 and '-'.join(model_name.split('-')[0:2]) in model_names:
        return '-'.join(model_name.split('-')[0:2])
    return model_name.split('-')[0]


def _simple_model_name(model):
    if '/' in model:
        model_name = model.split('/')[1]
    else:
        model_name = model
    model_name = model_name.replace('-inner-4bits', '')
    model_name = model_name.replace('-inner-w8a8', '')
    model_name = model_name.replace('-4bits', '')
    return model_name


def close_pipeline(pipe):
    pipe.close()
    import gc
    gc.collect()
    _clear_device_cache()


def _clear_device_cache():
    """Clear cache based on the current device type."""
    device = os.environ.get('DEVICE', 'cuda')
    handler = _get_device_handler(device)
    handler.clear_cache()


def _get_device_handler(device):
    """Get the appropriate device handler based on device type."""
    handlers = {
        'cuda': CudaDeviceHandler(),
        'ascend': AscendDeviceHandler(),
    }

    # Return the specific handler if available, otherwise return default cuda handler
    return handlers.get(device, handlers['cuda'])


class DeviceHandler:
    """Base class for device handlers."""

    def get_device_prefix(self, config, model):
        """Get device-specific prefix for command execution."""
        return ''

    def clear_cache(self):
        """Clear device-specific cache."""
        pass

    def get_available_devices(self):
        """Get list of available devices."""
        return []


class CudaDeviceHandler(DeviceHandler):
    """Handler for CUDA devices."""

    def get_device_prefix(self, config, model):
        cuda_prefix = ''
        tp_num = get_tp_num(config, model)
        if tp_num is None or tp_num == 8:
            return cuda_prefix
        available_cuda = self.get_available_devices()
        if len(available_cuda) < tp_num:
            raise torch.cuda.OutOfMemoryError

        cuda_prefix = 'CUDA_VISIBLE_DEVICES=' + ','.join(random.sample(available_cuda, tp_num))
        self.clear_cache()
        return cuda_prefix

    def clear_cache(self):
        torch.cuda.empty_cache()

    def get_available_devices(self):
        devices = torch.cuda.device_count()
        available_cuda = []
        for i in range(devices):
            if (torch.cuda.utilization(i) > 5):
                continue
            if ('no processes are running' not in torch.cuda.list_gpu_processes(i)):
                continue
            available_cuda.append(str(i))
        return available_cuda


class AscendDeviceHandler(DeviceHandler):
    """Handler for Ascend devices."""

    def get_device_prefix(self, config, model):
        ascend_prefix = ''
        tp_num = get_tp_num(config, model)
        if tp_num is None or tp_num == 8:
            return ascend_prefix
        available_ascend = self.get_available_devices()
        if len(available_ascend) < tp_num:
            raise RuntimeError('Not enough Ascend devices available')

        selected_devices = sorted(random.sample(available_ascend, tp_num), key=int)
        ascend_prefix = 'ASCEND_RT_VISIBLE_DEVICES=' + ','.join(selected_devices)
        self.clear_cache()
        return ascend_prefix

    def clear_cache(self):
        try:
            import torch_npu
            torch_npu.npu.empty_cache()
        except ImportError:
            pass  # torch_npu not available

    def get_available_devices(self):
        """Get list of available Ascend devices by checking AICPU usage
        rate."""
        available_ascend = []
        try:
            # Get the number of NPU devices
            result = subprocess.run(['npu-smi', 'info', '-l'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return available_ascend

            # Parse the output to get device count
            # Looking for lines like "Device Count : X"
            device_count = 0
            for line in result.stdout.split('\n'):
                if 'Total Count' in line:
                    match = re.search(r'Total Count\s*:\s*(\d+)', line)
                    if match:
                        device_count = int(match.group(1))
                        break

            # Check each device's AICPU usage
            for i in range(device_count):
                try:
                    result = subprocess.run(
                        ['npu-smi', 'info', '-t', 'usages', '-i', str(i)], capture_output=True, text=True, timeout=10)
                    if result.returncode != 0:
                        continue

                    # Parse the output to get AICPU Usage Rate
                    # Looking for lines like "Aicpu Usage Rate(%) : X"
                    aicpu_usage = 100  # Default to 100% (busy)
                    for line in result.stdout.split('\n'):
                        if 'Aicpu Usage Rate(%)' in line:
                            match = re.search(r'Aicpu Usage Rate\(%\)\s*:\s*(\d+)', line)
                            if match:
                                aicpu_usage = int(match.group(1))
                                break

                    # If AICPU usage is 0, consider the device available
                    if aicpu_usage == 0:
                        available_ascend.append(str(i))
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    continue

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            # npu-smi command not found or other error
            pass
        return available_ascend
