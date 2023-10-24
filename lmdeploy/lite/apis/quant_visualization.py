# Copyright (c) OpenMMLab. All rights reserved.

import os
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from calibrate import calibrate
from matplotlib import rc
from transformers import AutoModelForCausalLM

from lmdeploy.lite.utils import collect_target_modules
from lmdeploy.pytorch.model import LoadWoInit

rc('mathtext', default='regular')

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'BaiChuanForCausalLM': 'DecoderLayer',
    'LlamaForCausalLM': 'LlamaDecoderLayer',
}
NORM_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMRMSNorm',
    'QWenLMHeadModel': 'RMSNorm',
    'BaiChuanForCausalLM': 'RMSNorm',
    'LlamaForCausalLM': 'LlamaRMSNorm',
}


def get_layer_idx(name):
    """Extract layer index from layer name."""
    return int(name.split('.')[2])


def draw1(linear_name,
          use_input=True,
          key='absmax',
          work_dir='work_dir',
          layers=None):
    """Draw the first type of plot which shows the absmax/absmean/max/mean/min
    value of a linear layer at different layers.

    Args:
        linear_name (str): Name of the linear layer.
        use_input (bool, optional): Use the input or output of a linear layer.
            Defaults to use the input of a linear layer.
        key (str, optional): The specific feature to plot. Can be 'absmax',
            'absmean', 'max', 'mean' or 'min'. Defaults to 'absmax'.
        work_dir (str, optional): Working directory where intermediate files
            are saved. Defaults to 'work_dir'.
        layers (list, optional): List of layers to draw. If None,
            all layers will be drawn. Defaults to None.
    """

    def draw_one(tensor, idx):
        tensor = tensor.to('cpu')
        dim = tensor.shape[0]
        x = list(range(dim))
        y = list(tensor.numpy())
        var = tensor.var().item()
        plt.plot(x, y)
        plt.xlabel('dim')
        plt.ylabel(key)
        plt.title(f'layer{idx}  variance = {round(var, 4)}')
        plt.savefig(f'{work_dir}/case_1/layer_{idx}.png')
        plt.clf()

    work_dir = Path(work_dir)
    tmp_dir = work_dir / 'case_1'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if use_input:
        stats = torch.load(f'{work_dir}/inputs_stats.pth')
    else:
        stats = torch.load(f'{work_dir}/outputs_stats.pth')
    assert key in stats
    stats = stats[key]
    tensor_list = {
        get_layer_idx(k): v
        for k, v in stats.items() if linear_name in k
    }
    if layers is None:
        for i, tensor in enumerate(tensor_list):
            draw_one(tensor, i)
    else:
        for layer_idx in layers:
            draw_one(tensor_list[layer_idx], layer_idx)


def draw2(linear_name,
          model_path,
          key='absmax',
          work_dir='work_dir',
          layers=None):
    """Draw the second type of plot which shows the relationship between
    activations and weights.

    Args:
        linear_name (str): Name of the linear layer.
        model_path (str): The name or path of the model to be loaded.
        key (str, optional): The specific feature to plot. Can be 'absmax',
            'absmean', 'max', 'mean' or 'min'. Defaults to 'absmax'.
        work_dir (str, optional): Working directory where intermediate
            files are saved. Defaults to 'work_dir'.
        layers (list, optional): List of layers to draw. If None,
            all layers will be drawn. Defaults to None.
    """

    def draw_one(act, weight, idx, topk=100):
        act = act.to('cpu').float()
        weight = weight.to('cpu').float()
        assert act.shape[0] == weight.shape[
            0] and act.ndim == 1 and weight.ndim == 1

        y1, x1 = act.topk(topk)
        y2, x2 = weight.topk(topk)
        x1, y1 = list(x1.numpy()), list(y1.numpy())
        x2, y2 = list(x2.numpy()), list(y2.numpy())
        tuple1 = [(x, y) for x, y in zip(x1, y1)]
        tuple1.sort(key=lambda x: x[0])
        x1, y1 = [x[0] for x in tuple1], [x[1] for x in tuple1]
        tuple2 = [(x, y) for x, y in zip(x2, y2)]
        tuple2.sort(key=lambda x: x[0])
        x2, y2 = [x[0] for x in tuple2], [x[1] for x in tuple2]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1 = ax.plot(x1, y1, '-', label='act')
        ax2 = ax.twinx()
        line2 = ax2.plot(x2, y2, '-r', label='weight')
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc=0)
        ax.grid()
        ax.set_xlabel('dim')
        ax.set_ylabel(f'Activation_{key}')
        ax2.set_ylabel(f'Weight_{key}')
        plt.title(f'layer{idx}  topk = {topk}')
        plt.savefig(f'{work_dir}/case_2/layer_{idx}.png')
        plt.clf()

    work_dir = Path(work_dir)
    tmp_dir = work_dir / 'case_2'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(f'{work_dir}/inputs_stats.pth')
    assert key in stats
    stats = stats[key]
    act_list = {
        get_layer_idx(k): v
        for k, v in stats.items() if linear_name in k
    }

    with init_empty_weights():
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)
        model.config.use_cache = False
    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    decoder_layers = collect_target_modules(model, layer_type)
    device_map = infer_auto_device_map(model,
                                       no_split_module_classes=[layer_type])
    for name in device_map.keys():
        if name in decoder_layers or 'lm_head' in name:
            device_map[name] = 'cpu'
        else:
            device_map[name] = 0
    with LoadWoInit():
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)
        model.config.use_cache = False

    weight_list = {}
    for k, v in model.state_dict().items():
        if linear_name in k:
            if key == 'absmax':
                v = v.abs().max(dim=0)[0]
            elif key == 'absmean':
                v = v.abs().mean(dim=0)[0]
            weight_list[get_layer_idx(k)] = v

    if layers is None:
        for i, (act, weight) in enumerate(zip(act_list, weight_list)):
            draw_one(act, weight, i)
    else:
        for layer_idx in layers:
            draw_one(act_list[layer_idx], weight_list[layer_idx], layer_idx)


def draw3(linear_name, use_input=True, key='absmax', work_dir='work_dir'):
    """Draw the third type of plot which is a boxplot showing the
    absmax/absmean/max/mean/min value of the input or output of a linear layer
    at different layers.

    Args:
        linear_name (str): Name of the linear layer.
        use_input (bool, optional): Use the input or output of a linear layer.
            Defaults to use the input of a linear layer.
        key (str, optional): The specific feature to plot. Can be 'absmax',
            'absmean', 'max', 'mean' or 'min'. Defaults to 'absmax'.
        work_dir (str, optional): Working directory where intermediate files
            are saved. Defaults to 'work_dir'.
    """

    def draw_one(tensor_list):
        all_data = []
        for val in tensor_list.values():
            all_data.append(val.to('cpu').numpy())
        plt.boxplot(all_data, None, None, None, 20)  # codespell-ignore
        plt.xlabel('layer')
        plt.ylabel(key)
        plt.title(f'linear_name {linear_name}')
        plt.savefig(f'{work_dir}/case_3.png')

    if use_input:
        stats = torch.load(f'{work_dir}/inputs_stats.pth')
    else:
        stats = torch.load(f'{work_dir}/outputs_stats.pth')
    assert key in stats
    stats = stats[key]
    tensor_list = {
        get_layer_idx(k): v
        for k, v in stats.items() if linear_name in k
    }
    draw_one(tensor_list)


def draw4(linear_name, use_input=True, work_dir='work_dir', layers=None):
    """Draw the fourth type of plot which shows the relationship between
    maximum and minimum values of activations.

    Args:
        linear_name (str): Name of the linear layer.
        use_input (bool, optional): Use the input or output of a linear layer.
            Defaults to use the input of a linear layer.
        work_dir (str, optional): Working directory where intermediate files
            are saved. Defaults to 'work_dir'.
        layers (list, optional): List of layers to draw. If None,
            all layers will be drawn. Defaults to None.
    """

    def draw_one(tensor_max, tensor_min, idx):
        tensor_max = tensor_max.to('cpu').numpy()
        tensor_min = tensor_min.to('cpu').numpy()
        plt.scatter(tensor_max, tensor_min)
        plt.xlabel('max')
        plt.ylabel('min')
        plt.title(f'layer{idx}')
        plt.savefig(f'{work_dir}/case_4/layer_{idx}.png')
        plt.clf()

    work_dir = Path(work_dir)
    tmp_dir = work_dir / 'case_4'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if use_input:
        stats = torch.load(f'{work_dir}/inputs_stats.pth')
    else:
        stats = torch.load(f'{work_dir}/outputs_stats.pth')
    stats_max = stats['max']
    stats_min = stats['min']
    max_list = {
        get_layer_idx(k): v
        for k, v in stats_max.items() if linear_name in k
    }
    min_list = {
        get_layer_idx(k): v
        for k, v in stats_min.items() if linear_name in k
    }
    if layers is None:
        for i, (mmax, mmin) in enumerate(zip(max_list, min_list)):
            draw_one(mmax, mmin, i)
    else:
        for layer_idx in layers:
            draw_one(max_list[layer_idx], min_list[layer_idx], layer_idx)


def draw(mode,
         pretrained_model_name_or_path=None,
         work_dir='work_dir',
         use_input=True,
         key='absmax',
         linear_name=None,
         layers=None,
         force_calibrate=False):
    """Draw specific type of plots based on the given mode.

    Args:
        mode (int): Plot mode. Can be 1, 2, 3 or 4.
        pretrained_model_name_or_path (str, optional): Path or name of the
            pretrained model. Required if `force_calibrate` is True or
            no calibration data exists in `work_dir`. Defaults to None.
        work_dir (str, optional): Working directory where intermediate files
            are saved. Defaults to 'work_dir'.
        use_input (bool, optional): Use the input or output of a linear layer.
            Defaults to use the input of a linear layer.
        key (str, optional): The specific feature to plot. Can be 'absmax',
            'absmean', 'max', 'mean' or 'min'. Defaults to 'absmax'.
        linear_name (str, optional): Name of the linear layer. Defaults to None
        layers (list, optional): List of layers to draw. If None, all layers
            will be drawn. Defaults to None.
        force_calibrate (bool, optional): Whether to force recalibration
            even if calibration data exists. Defaults to False.
    """
    if not os.path.exists(work_dir) or force_calibrate:
        assert pretrained_model_name_or_path is not None
        calibrate(pretrained_model_name_or_path, work_dir=work_dir)
    if mode == 1:
        draw1(linear_name, use_input, key, work_dir, layers=layers)
    elif mode == 2:
        assert pretrained_model_name_or_path
        draw2(linear_name,
              pretrained_model_name_or_path,
              key,
              work_dir,
              layers=layers)
    elif mode == 3:
        draw3(linear_name, use_input, key, work_dir)
    elif mode == 4:
        draw4(linear_name, use_input, work_dir, layers=layers)


if __name__ == '__main__':
    fire.Fire(draw)
