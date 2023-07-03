# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
import sys
import os.path as osp
import torch
import numpy as np
import lmdeploy
from lmdeploy.model import MODELS
from .tokenizer import Tokenizer, Preprocessor, Postprocessor
from torch.nn.utils.rnn import pad_sequence

# TODO: find another way import _turbomind
lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
sys.path.append(osp.join(lmdeploy_dir, 'lib'))
import _turbomind as _tm

def _stop_words(stop_words: List[int]):
    if stop_words is None:
        return None
    assert isinstance(stop_words, List) and \
            all(isinstance(elem, int) for elem in stop_words), \
            f'stop_words must be a list but got {type(stop_words)}'
    # each id in stop_words represents a stop word
    # refer to https://github.com/fauxpilot/fauxpilot/discussions/165 for
    # detailed explanation about fastertransformer's stop_words
    stop_word_offsets = range(1, len(stop_words) + 1)
    stop_words = np.array([[stop_words,
                            stop_word_offsets]]).astype(np.int32)
    return stop_words

def _np_dict_to_tm_dict(np_dict:dict):
    ret = _tm.TensorMap()
    for k, v in np_dict.items():
        ret[k] = _tm.from_dlpack(v)

    return ret
    
def _tm_dict_to_torch_dict(tm_dict:_tm.TensorMap):
    ret = dict()
    for k, v in tm_dict.items():
        if v.type == _tm.DataType.TYPE_UINT32:
            v = v.view(_tm.DataType.TYPE_INT32)
        ret[k] = torch.from_dlpack(v)

    return ret

class TurboMind:
    def __init__(self, model_path:str, eos_id:int=2, stop_words: List[int]=None):
        self.eos_id = eos_id

        # create model instance
        self.node_id = 0
        self.node_num = 1
        self.gpu_count = 1
        self.device_id = 0
        self.world_size = self.node_num * self.gpu_count
        self.rank = self.node_id * self.gpu_count + self.device_id
        self.stream = 0
        model = _tm.AbstractTransformerModel.create_llama_model(model_path)
        model.create_shared_weights(self.device_id, self.rank)

        nccl_params = model.create_nccl_params(self.node_id)
        custom_comms = model.create_custom_comms(self.world_size)
        instance_comm = model.create_instance_comm(self.gpu_count)

        model_inst = model.create_model_instance(self.device_id,
            self.rank,
            self.stream,
            nccl_params,
            custom_comms[0])

        self.model_inst = model_inst
        self.instance_comm = instance_comm
        self.stop_words = _stop_words(stop_words)

    def stream_infer(self,
                    session_id,
                    input_ids,
                    request_output_len: int = 512,
                    sequence_start: bool = True,
                    sequence_end: bool = False,
                    step=1,
                    stop=False,
                    top_p=0.8,
                    top_k=40,
                    temperature=0.8,
                    repetition_penalty=1.05,
                    ignore_eos=False,
                    random_seed=None):
        device = f'cuda:{self.device_id}'
        np_one_ui = np.ones((1, ), dtype=np.uint32)
        np_one_ui64 = np.ones((1, ), dtype=np.uint64)
        np_one_i = np.ones((1, ), dtype=np.int32)
        np_one_f = np.ones((1, ), dtype=np.float32)
        
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
        input_ids = [torch.IntTensor(ids) for ids in input_ids]
        input_lengths = torch.IntTensor([len(ids) for ids in input_ids])
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.eos_id)
        input_lengths = input_lengths.detach().cpu().numpy()

        inputs = dict(
            input_ids=input_ids.to(device),
            input_lengths = input_lengths,
            request_output_len=np.full(input_lengths.shape, request_output_len, dtype=np.uint32),
            runtime_top_k=top_k * np_one_ui,
            runtime_top_p=top_p * np_one_f,
            temperature=temperature * np_one_f,
            repetition_penalty=repetition_penalty * np_one_f,
            step=step * np_one_i,

            # session input
            START=(1 if sequence_start else 0) * np_one_i,
            END=(1 if sequence_end else 0) * np_one_i,
            CORRID=session_id * np_one_ui64,
            STOP=(1 if stop else 0) * np_one_i
        )

        if ignore_eos:
            stop_words = None
            bad_words = torch.tensor([[[self.eos_id], [1]]], dtype=torch.int32).to(device)
        else:
            stop_words = self.stop_words
            bad_words = None

        if stop_words is not None:
            inputs['stop_words_list'] = stop_words
        if bad_words is not None:
            inputs['bad_words_list'] = bad_words

        if random_seed is not None:
            inputs['random_seed'] = random_seed * np_one_ui64
        tm_inputs = _np_dict_to_tm_dict(inputs)
        tm_outputs = self.model_inst.forward(tm_inputs, self.instance_comm)

        outputs = _tm_dict_to_torch_dict(tm_outputs)

        output_ids = outputs['output_ids'][0]
        sequence_length = outputs['sequence_length'].long()[0]
        return [(output[:l], l.item()) for output, l in zip(output_ids, sequence_length)]

