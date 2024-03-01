# Copyright (c) OpenMMLab. All rights reserved.

import torch
import os
import fire
import logging

class MedusaConverter():
    def __init__(self, medusa_pt_path: str, medusa_output_path: str,
                 medusa_num_heads: int, medusa_num_layers: int,
                 medusa_weight_type: str, tp: int):
        logging.basicConfig(level=logging.INFO)
        if not os.path.isfile(medusa_pt_path):
            logging.error(f'{medusa_pt_path} not exist')
            os._exit(os.EX_IOERR)
        self.medusa_pt_path = medusa_pt_path
        self.medusa_output_path = medusa_output_path
        if not os.path.exists(self.medusa_output_path):
            os.makedirs(self.medusa_output_path)

        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers

        self.medusa_weight_type = medusa_weight_type
        self.tp = tp

        self.medusa_weights = torch.load(medusa_pt_path)

    def _tp_split(self, tensor: torch.Tensor, tp: int, dim: int) -> torch.Tensor:
        split_size = tensor.shape[dim] // tp
        split_tensors = torch.split(tensor, split_size, dim=dim)
        return split_tensors

    def _export(self, tensor: torch.Tensor, save_name: str):
        if tensor.dtype == torch.bfloat16:
            if self.medusa_weight_type == "fp16":
                tensor = tensor.to(torch.float16)
            elif self.medusa_weight_type == "bf16":
                # numpy workaround
                tensor = tensor.view(torch.float16)
            else:
                logging.error(f'{self.medusa_weight_type} not support')
                os._exit(os.EX_CONFIG)

        tensor.contiguous().cpu().numpy().tofile(
            os.path.join(self.medusa_output_path, save_name))
        logging.info(f'saved to {os.path.join(self.medusa_output_path, save_name)}')

    def _convert_head(self, medusa_head: int, tp: int):
        for medusa_layer in range(self.medusa_num_layers):
            w_name = f"{medusa_head}.{medusa_layer}.linear.weight"
            b_name = f"{medusa_head}.{medusa_layer}.linear.bias"

            tensor_w = self.medusa_weights[w_name]
            tensor_b = self.medusa_weights[b_name]

            tensor_w = tensor_w.t()
            split_tensors_w = self._tp_split(tensor_w, tp, -1)
            split_tensors_b = self._tp_split(tensor_b, tp, -1)

            for rank, split_tensor in enumerate(split_tensors_w):
                w_name_after = f"medusa.{medusa_head}.{medusa_layer}.linear.{rank}.weight"
                logging.info(f'{w_name}->{w_name_after}, shape:{self.medusa_weights[w_name].shape}->{split_tensor.shape}')
                self._export(split_tensor, w_name_after)

            for rank, split_tensor in enumerate(split_tensors_b):
                b_name_after = f"medusa.{medusa_head}.{medusa_layer}.linear.{rank}.bias"
                logging.info(f'{b_name}->{b_name_after}, shape:{self.medusa_weights[b_name].shape}->{split_tensor.shape}')
                self._export(split_tensor, b_name_after)

        w_name = f"{medusa_head}.{self.medusa_num_layers}.weight"
        tensor_w = self.medusa_weights[w_name]

        tensor_w = tensor_w.t()
        split_tensors_w = self._tp_split(tensor_w, tp, 0)

        for rank, split_tensor in enumerate(split_tensors_w):
            w_name_after = f"medusa.{medusa_head}.{self.medusa_num_layers}.{rank}.weight"
            logging.info(f'{w_name}->{w_name_after}, shape:{self.medusa_weights[w_name].shape}->{split_tensor.shape}')
            self._export(split_tensor, w_name_after)

    def convert(self):
        for i in range(self.medusa_num_heads):
            self._convert_head(medusa_head=i, tp=self.tp)


def main(medusa_pt_path='/workdir/medusa-vicuna-13b-v1.3/medusa_lm_head.pt',
         medusa_output_path='/workdir/medusa_output/fp16/tp1',
         medusa_num_heads=5,
         medusa_num_layers=1,
         medusa_weight_type='fp16',
         tp=1):
    converter = MedusaConverter(medusa_pt_path, medusa_output_path,
                                medusa_num_heads, medusa_num_layers,
                                medusa_weight_type, tp)
    converter.convert()

if __name__ == '__main__':
    fire.Fire(main)
