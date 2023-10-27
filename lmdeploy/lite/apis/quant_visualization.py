# Copyright (c) OpenMMLab. All rights reserved.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib import rc

from lmdeploy.lite.apis.calibrate import calibrate
from lmdeploy.lite.utils import collect_target_modules, load_hf_from_pretrained

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


def get_linear_input_or_output(linear_name, stats):
    """Retrieves the input or output data for a specific linear layer from a
    statistics dictionary (stats).

    Args:
        linear_name (str, optional): Name of the linear layer for which
            the visualization will be created. If not provided, it will
            process all available layers.

        stats (dict): A dictionary containing the statistics for each layer.
            Each key is the layer's name and its value is a dictionary that
            maps the layer's index to its corresponding statistics.
    """

    if linear_name is None:
        acts = {}
        for k, v in stats.items():
            linear_name = k.split('.')[-1]
            if linear_name in acts:
                acts[linear_name].update({get_layer_idx(k): v})
            else:
                acts[linear_name] = {get_layer_idx(k): v}
    else:
        acts = {
            linear_name: {
                get_layer_idx(k): v
                for k, v in stats.items() if linear_name in k
            }
        }

    return acts


def get_linear_weights(linear_name, model, key):
    """Fetches specific statistics (based on the provided key) for the weights
    of a specified linear layer in a given model. If `linear_name` is None, the
    function retrieves these statistics for all linear layers.

    The function supports the following keys: 'absmax', 'absmean', 'max',
    'mean', 'min'.

    Args:
        linear_name (str, optional): Name of the linear layer for which
            the visualization will be created. If not provided, it will
            process all available layers.
        model (torch.nn.Module): CasualLM model from which the weights
            will be retrieved.
        key (str): The specific feature to plot. Can be 'absmax',
            'absmean', 'max', 'mean' or 'min'. Defaults to 'absmax'.
    """
    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    name2layer = collect_target_modules(model, layer_type)
    name2fc = {}
    for l_name, layer in name2layer.items():
        name2fc.update(collect_target_modules(layer, nn.Linear, prefix=l_name))
    weights = {}
    for name, module in name2fc.items():
        if linear_name is None or linear_name in name:
            l_name = linear_name or name.split('.')[-1]
            if key == 'absmax':
                v = module.weight.detach().abs().max(dim=0)[0]
            elif key == 'absmean':
                v = module.weight.detach().abs().mean(dim=0)[0]
            elif key == 'max':
                v = module.weight.detach().max(dim=0)[0]
            elif key == 'mean':
                v = module.weight.detach().mean(dim=0)[0]
            elif key == 'min':
                v = module.weight.detach().min(dim=0)[0]
            else:
                raise NotImplementedError(
                    "Support key in ['absmax', 'absmean', 'max', 'mean', "
                    f"'min'], but got key = {key}")
            if l_name in weights:
                weights[l_name].update({get_layer_idx(name): v})
            else:
                weights[l_name] = {get_layer_idx(name): v}
    return weights


def draw1(linear_name=None,
          use_input=True,
          key='absmax',
          work_dir='work_dir',
          layers=None):
    """Draw the first type of plot which shows the absmax/absmean/max/mean/min
    value of a linear layer at different layers.

    Args:
        linear_name (str, optional): Name of the linear layer for which
            the visualization will be created. If not provided, it will
            process all available layers.
        use_input (bool, optional): Use the input or output of a linear layer.
            Defaults to use the input of a linear layer.
        key (str, optional): The specific feature to plot. Can be 'absmax',
            'absmean', 'max', 'mean' or 'min'. Defaults to 'absmax'.
        work_dir (str, optional): Working directory where intermediate files
            are saved. Defaults to 'work_dir'.
        layers (list, optional): List of layers to draw. If None,
            all layers will be drawn. Defaults to None.
    """

    def draw_one(tensor, idx, linear_name):
        tensor = tensor.to('cpu')
        dim = tensor.shape[0]
        x = list(range(dim))
        y = list(tensor.numpy())
        var = tensor.var().item()
        plt.plot(x, y)
        plt.xlabel('dim')
        plt.ylabel(key)
        plt.title(f'layer{idx}  variance = {round(var, 4)}')
        (tmp_dir / linear_name).mkdir(parents=True, exist_ok=True)
        plt.savefig(tmp_dir / linear_name / f'layer_{idx}.png')
        plt.clf()

    work_dir = Path(work_dir)
    name = 'linear_input' if use_input else 'linear_output'
    tmp_dir = work_dir / 'visualization' / f'case_1_{name}_{key}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if use_input:
        stats = torch.load(f'{work_dir}/inputs_stats.pth')
    else:
        stats = torch.load(f'{work_dir}/outputs_stats.pth')
    assert key in stats
    stats = stats[key]

    acts = get_linear_input_or_output(linear_name, stats)
    for linear_name, act_dict in acts.items():
        if layers is None:
            for i, act in act_dict.items():
                draw_one(act, i, linear_name)
        else:
            for i in layers:
                draw_one(act_dict[i], i, linear_name)


def draw2(linear_name,
          model_path,
          key='absmax',
          work_dir='work_dir',
          layers=None):
    """Draw the second type of plot which shows the relationship between
    activations and weights.

    Args:
        linear_name (str, optional): Name of the linear layer for which
            the visualization will be created. If not provided, it will
            process all available layers.
        model_path (str): The name or path of the model to be loaded.
        key (str, optional): The specific feature to plot. Can be 'absmax',
            'absmean', 'max', 'mean' or 'min'. Defaults to 'absmax'.
        work_dir (str, optional): Working directory where intermediate
            files are saved. Defaults to 'work_dir'.
        layers (list, optional): List of layers to draw. If None,
            all layers will be drawn. Defaults to None.
    """

    def draw_one(act, weight, idx, linear_name, topk=100):
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
        (tmp_dir / linear_name).mkdir(parents=True, exist_ok=True)
        plt.savefig(tmp_dir / linear_name / f'layer_{idx}.png')
        plt.clf()

    work_dir = Path(work_dir)
    tmp_dir = work_dir / 'visualization' / f'case_2_linear_input_{key}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(f'{work_dir}/inputs_stats.pth')
    assert key in stats
    stats = stats[key]

    acts = get_linear_input_or_output(linear_name, stats)

    model = load_hf_from_pretrained(model_path,
                                    torch_dtype=torch.float16,
                                    trust_remote_code=True)

    weights = get_linear_weights(linear_name, model, key)

    for linear_name in acts.keys():
        act_dict, weights_dict = acts[linear_name], weights[linear_name]
        if layers is None:
            for i in range(len(act_dict)):
                draw_one(act_dict[i], weights_dict[i], i, linear_name)
        else:
            for i in layers:
                draw_one(act_dict[i], weights_dict[i], i, linear_name)


def draw3(linear_name, use_input=True, key='absmax', work_dir='work_dir'):
    """Draw the third type of plot which is a boxplot showing the
    absmax/absmean/max/mean/min value of the input or output of a linear layer
    at different layers.

    Args:
        linear_name (str, optional): Name of the linear layer for which
            the visualization will be created. If not provided, it will
            process all available layers.
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
        plt.boxplot(all_data, None, None, None, 20)
        plt.xlabel('layer')
        plt.ylabel(key)
        plt.title(f'linear_name {linear_name}')
        plt.savefig(tmp_dir / f'{linear_name}.png')

    work_dir = Path(work_dir)
    name = 'linear_input' if use_input else 'linear_output'
    tmp_dir = work_dir / 'visualization' / f'case_3_{name}_{key}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if use_input:
        stats = torch.load(f'{work_dir}/inputs_stats.pth')
    else:
        stats = torch.load(f'{work_dir}/outputs_stats.pth')
    assert key in stats
    stats = stats[key]

    acts = get_linear_input_or_output(linear_name, stats)
    for linear_name, act_dict in acts.items():
        draw_one(act_dict)


def draw4(linear_name, use_input=True, work_dir='work_dir', layers=None):
    """Draw the fourth type of plot which shows the relationship between
    maximum and minimum values of activations.

    Args:
        linear_name (str, optional): Name of the linear layer for which
            the visualization will be created. If not provided, it will
            process all available layers.
        use_input (bool, optional): Use the input or output of a linear layer.
            Defaults to use the input of a linear layer.
        work_dir (str, optional): Working directory where intermediate files
            are saved. Defaults to 'work_dir'.
        layers (list, optional): List of layers to draw. If None,
            all layers will be drawn. Defaults to None.
    """

    def draw_one(tensor_max, tensor_min, idx, linear_name):
        tensor_max = tensor_max.to('cpu').numpy()
        tensor_min = tensor_min.to('cpu').numpy()
        plt.scatter(tensor_max, tensor_min)
        plt.xlabel('max')
        plt.ylabel('min')
        plt.title(f'layer{idx}')
        (tmp_dir / linear_name).mkdir(parents=True, exist_ok=True)
        plt.savefig(tmp_dir / linear_name / f'layer_{idx}.png')
        plt.clf()

    work_dir = Path(work_dir)
    name = 'linear_input' if use_input else 'linear_output'
    tmp_dir = work_dir / 'visualization' / f'case_4_{name}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if use_input:
        stats = torch.load(f'{work_dir}/inputs_stats.pth')
    else:
        stats = torch.load(f'{work_dir}/outputs_stats.pth')
    stats_max = stats['max']
    stats_min = stats['min']
    acts_max = get_linear_input_or_output(linear_name, stats_max)
    acts_min = get_linear_input_or_output(linear_name, stats_min)
    for linear_name in acts_max.keys():
        acts_max_cur_linear = acts_max[linear_name]
        acts_min_cur_linear = acts_min[linear_name]
        if layers is None:
            for i in range(len(acts_max_cur_linear)):
                draw_one(acts_max_cur_linear[i], acts_min_cur_linear[i], i,
                         linear_name)
        else:
            for i in layers:
                draw_one(acts_max_cur_linear[i], acts_min_cur_linear[i], i,
                         linear_name)


def draw(modes=None,
         linear_name=None,
         pretrained_model_name_or_path=None,
         work_dir='work_dir',
         use_input=True,
         key='absmax',
         layers=None,
         force_calibrate=False):
    """This function is used to visualize data from a pre-trained transformer
    model. Depending on the mode, it will leverage different visualization
    functions (draw1, draw2, draw3, draw4).

    Args:
        mode (list of int, optional): Modes determine which visualization
            function(s) will be used. If not provided, all four visualization
            functions are executed. Available modes: [1, 2, 3, 4].
        linear_name (str, optional): Name of the linear layer for which
            the visualization will be created. If not provided, it will
            process all available layers.
        pretrained_model_name_or_path (str, optional): Path or name of the
            pretrained model. Required if `force_calibrate` is True or
            no calibration data exists in `work_dir`. Defaults to None.
        work_dir (str, optional): Working directory where intermediate files
            are saved. Defaults to 'work_dir'.
        use_input (bool, optional): Use the input or output of a linear layer.
            Defaults to use the input of a linear layer.
        key (str, optional): The specific feature to plot. Can be 'absmax',
            'absmean', 'max', 'mean' or 'min'. Defaults to 'absmax'.
        layers (list, optional): List of layers to draw. If None, all layers
            will be drawn. Defaults to None.
        force_calibrate (bool, optional): Whether to force recalibration
            even if calibration data exists. Defaults to False.
    """
    if not os.path.exists(work_dir) or force_calibrate:
        assert pretrained_model_name_or_path is not None
        calibrate(pretrained_model_name_or_path, work_dir=work_dir)
    if modes is None:
        modes = [1, 2, 3, 4]
    elif isinstance(modes, int):
        modes = [modes]
    for mode in modes:
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
    import fire
    fire.Fire(draw)
