# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys
from queue import Queue
from threading import Thread
from typing import Iterable, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import lmdeploy

# TODO: find another way import _turbomind
lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
sys.path.append(osp.join(lmdeploy_dir, 'lib'))
import _turbomind as _tm  # noqa: E402


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
    stop_words = np.array([[stop_words, stop_word_offsets]]).astype(np.int32)
    return stop_words


def _np_dict_to_tm_dict(np_dict: dict):
    ret = _tm.TensorMap()
    for k, v in np_dict.items():
        ret[k] = _tm.from_dlpack(v)

    return ret


def _tm_dict_to_torch_dict(tm_dict: _tm.TensorMap):
    ret = dict()
    for k, v in tm_dict.items():
        if v.type == _tm.DataType.TYPE_UINT32:
            v = v.view(_tm.DataType.TYPE_INT32)
        ret[k] = torch.from_dlpack(v)

    return ret


class TurboMind:

    def __init__(self,
                 model_path: str,
                 data_type: str = 'fp16',
                 session_len: int = 2048,
                 eos_id: int = 2,
                 stop_words: List[int] = None,
                 device_id: int = 0,
                 node_id: int = 0,
                 device_num: int = 1,
                 node_num: int = 1):
        self.eos_id = eos_id

        # create model instance
        self.node_id = node_id
        self.node_num = node_num
        self.gpu_count = device_num
        self.device_id = device_id
        self.world_size = self.node_num * self.gpu_count
        self.rank = self.node_id * self.gpu_count + self.device_id
        self.session_len = session_len

        weight_dir = osp.join(model_path, 'triton_models', 'weights')
        model = _tm.AbstractTransformerModel.create_llama_model(
            weight_dir, tensor_para_size=self.gpu_count, data_type=data_type)
        model.create_shared_weights(self.device_id, self.rank)
        self.model = model
        self.stop_words = _stop_words(stop_words)

    def create_instance(self, cuda_stream_id=0):
        return TurboMindInstance(self, cuda_stream_id)


class TurboMindInstance:

    def __init__(self, tm_model, cuda_stream_id=0):
        self.tm_model = tm_model

        self.device_id = tm_model.device_id
        self.rank = tm_model.rank
        self.stop_words = tm_model.stop_words
        self.eos_id = tm_model.eos_id
        self.session_len = tm_model.session_len
        self.cuda_stream_id = cuda_stream_id

        # create instance
        model = tm_model.model
        nccl_params = model.create_nccl_params(tm_model.node_id)
        custom_comms = model.create_custom_comms(tm_model.world_size)
        instance_comm = model.create_instance_comm(tm_model.gpu_count)

        model_inst = model.create_model_instance(self.device_id, self.rank,
                                                 self.cuda_stream_id,
                                                 nccl_params, custom_comms[0])
        # model_inst.register_callback(self._forward_callback)
        self.model_inst = model_inst
        self.instance_comm = instance_comm
        self.que = Queue()
        self.thread = None

    def _forward_callback(self, result, ctx):
        self.que.put((False, result))

    def _forward_thread(self, inputs):

        def _func():
            output = self.model_inst.forward(inputs, self.instance_comm)
            self.que.put((True, output))

        self.thread = Thread(target=_func)
        self.thread.start()

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
                     random_seed=None,
                     stream_output=False):

        if stream_output:
            self.model_inst.register_callback(self._forward_callback)

        if len(input_ids) == 0:
            input_ids = []
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]

        batch_size = len(input_ids)

        def _broadcast_np(data, dtype, shape=(batch_size, )):
            if isinstance(data, Iterable):
                assert len(data) == batch_size
                return data

            return np.full(shape, data, dtype=dtype)

        input_ids = [torch.IntTensor(ids) for ids in input_ids]
        input_lengths = torch.IntTensor([len(ids) for ids in input_ids])
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.eos_id)

        if isinstance(session_id, int):
            session_id = [session_id]
        assert len(session_id) == batch_size

        step = _broadcast_np(step, np.int32)

        inputs = dict(
            input_ids=input_ids,
            input_lengths=input_lengths,
            request_output_len=np.full(
                input_lengths.shape, request_output_len, dtype=np.uint32),
            runtime_top_k=_broadcast_np(top_k, np.uint32),
            runtime_top_p=_broadcast_np(top_p, np.float32),
            temperature=_broadcast_np(temperature, np.float32),
            repetition_penalty=_broadcast_np(repetition_penalty, np.float32),
            step=step,

            # session input
            session_len=self.session_len *
            np.ones([
                batch_size,
            ], dtype=np.uint32),
            START=_broadcast_np((1 if sequence_start else 0), np.int32),
            END=_broadcast_np((1 if sequence_end else 0), np.int32),
            CORRID=np.array(session_id, dtype=np.uint64),
            STOP=_broadcast_np((1 if stop else 0), np.int32))

        if ignore_eos:
            stop_words = None
            bad_words = torch.tensor([[[self.eos_id], [1]]], dtype=torch.int32)
        else:
            stop_words = self.stop_words
            bad_words = None

        if stop_words is not None:
            inputs['stop_words_list'] = stop_words
        if bad_words is not None:
            inputs['bad_words_list'] = bad_words

        if random_seed is not None:
            inputs['random_seed'] = _broadcast_np(random_seed, np.uint64)
        tm_inputs = _np_dict_to_tm_dict(inputs)

        # start forward thread
        self._forward_thread(tm_inputs)

        seq_start = input_lengths + input_lengths.new_tensor(step)

        # generator
        while True:
            while self.que.qsize() > 1:
                self.que.get()

            finish, tm_outputs = self.que.get()

            outputs = _tm_dict_to_torch_dict(tm_outputs)

            output_ids = outputs['output_ids'][:, 0, :]
            sequence_length = outputs['sequence_length'].long()[:, 0].cpu()
            output_ids = [
                output_id[s:l] for output_id, s, l in zip(
                    output_ids, seq_start, sequence_length)
            ]
            sequence_length -= seq_start.to(sequence_length.device)
            yield [(output, l.item())
                   for output, l in zip(output_ids, sequence_length)]

            if finish:
                while self.que.qsize() > 0:
                    self.que.get()
                self.thread.join()
                break

        if stream_output:
            self.model_inst.unregister_callback()
