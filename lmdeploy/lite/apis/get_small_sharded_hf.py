# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import json
import os
import shutil

import torch
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a hugging face model to the smallest sharded one')
    parser.add_argument('src_dir', help='the directory of the model')
    parser.add_argument('dst_dir', help='the directory to save the new model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mkdir_or_exist(args.dst_dir)

    all_files = os.listdir(args.src_dir)
    for name in all_files:
        if not name.startswith(('pytorch_model', '.')):
            src_path = os.path.join(args.src_dir, name)
            dst_path = os.path.join(args.dst_dir, name)
            shutil.copy(src_path, dst_path)

    with open(os.path.join(args.src_dir, 'pytorch_model.bin.index.json')) as f:
        index = json.load(f)

    n_shard = len(index['weight_map'])
    new_index = copy.deepcopy(index)
    new_index['weight_map'] = {}
    cnt = 1

    checkpoints = set(index['weight_map'].values())
    for ckpt in checkpoints:
        state_dict = torch.load(os.path.join(args.src_dir, ckpt),
                                map_location='cuda')
        keys = sorted(list(state_dict.keys()))
        for k in keys:
            new_state_dict_name = 'pytorch_model-{:05d}-of-{:05d}.bin'.format(
                cnt, n_shard)
            new_index['weight_map'][k] = new_state_dict_name
            new_state_dict = {k: state_dict[k]}
            torch.save(new_state_dict,
                       os.path.join(args.dst_dir, new_state_dict_name))
            cnt += 1
        del state_dict
        torch.cuda.empty_cache()
    with open(os.path.join(args.dst_dir, 'pytorch_model.bin.index.json'),
              'w') as f:
        json.dump(new_index, f)
    assert new_index['weight_map'].keys() == index['weight_map'].keys(
    ), 'Mismatch on `weight_map`!'


if __name__ == '__main__':
    main()
