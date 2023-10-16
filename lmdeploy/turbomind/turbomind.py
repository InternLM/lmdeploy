# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os.path as osp
import sys
from configparser import ConfigParser
from contextlib import contextmanager
from queue import Queue
from threading import Thread
from typing import Iterable, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import lmdeploy
from lmdeploy.model import MODELS
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import get_logger

# TODO: find another way import _turbomind
lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
sys.path.append(osp.join(lmdeploy_dir, 'lib'))
import _turbomind as _tm  # noqa: E402


def _stop_words(stop_words: List[str], tokenizer: Tokenizer):
    """return list of stop-words to numpy.ndarray."""
    if stop_words is None:
        return None
    assert isinstance(stop_words, List) and \
           all(isinstance(elem, str) for elem in stop_words), \
           f'stop_words must be a list but got {type(stop_words)}'
    stop_words = [tokenizer.encode(stop_word)[-1] for stop_word in stop_words]
    assert isinstance(stop_words, List) and all(
        isinstance(elem, int) for elem in stop_words), 'invalid stop_words'
    # each id in stop_words represents a stop word
    # refer to https://github.com/fauxpilot/fauxpilot/discussions/165 for
    # detailed explanation about fastertransformer's stop_words
    stop_word_offsets = range(1, len(stop_words) + 1)
    stop_words = np.array([[stop_words, stop_word_offsets]]).astype(np.int32)
    return stop_words


def _np_dict_to_tm_dict(np_dict: dict):
    """map numpy.ndarray to turbomind's tensor."""
    ret = _tm.TensorMap()
    for k, v in np_dict.items():
        ret[k] = _tm.from_dlpack(v)

    return ret


def _tm_dict_to_torch_dict(tm_dict: _tm.TensorMap):
    """map turbomind's tensor to torch's tensor."""
    ret = dict()
    for k, v in tm_dict.items():
        if v.type == _tm.DataType.TYPE_UINT32:
            v = v.view(_tm.DataType.TYPE_INT32)
        ret[k] = torch.from_dlpack(v)

    return ret


@contextmanager
def cuda_ctx(device_id):
    old_device = torch.cuda.current_device()
    torch.cuda.set_device(device_id)
    yield
    torch.cuda.set_device(old_device)


class TurboMind:
    """LMDeploy's inference engine.

    Args:
        model_path (str): the path of turbomind's model
        eos_id (int): eos token id
        tp (int): tensor parallel
    """

    def __init__(self, model_path: str, eos_id: int = 2, tp: int = 1):
        self.eos_id = eos_id

        # TODO: support mpi
        node_id = 0
        node_num = 1

        # read meta from model path
        assert ((tp & (tp - 1) == 0) and tp != 0), 'tp should be 2^n'
        self.gpu_count = tp
        self.session_len = 2048
        data_type = 'fp16'
        ini_path = osp.join(model_path, 'triton_models/weights/config.ini')
        with open(ini_path, 'r') as f:
            parser = ConfigParser()
            parser.read_file(f)
            section_name = ''
            if 'turbomind' in parser:
                section_name = 'turbomind'
            elif 'llama' in parser:
                section_name = 'llama'

            if len(section_name) > 0:
                tp_cfg = parser.getint(section_name, 'tensor_para_size')
                self.session_len = parser.getint(section_name, 'session_len')
                if tp_cfg != 1 and tp_cfg != tp:
                    get_logger('turbomind').info(
                        f'found tp={tp_cfg} in config.ini.')
                    self.gpu_count = tp_cfg
            self.model_name = parser.get(section_name, 'model_name')
            data_type = parser.get(section_name, 'weight_type')
        model = MODELS.get(self.model_name)()
        tokenizer_model_path = osp.join(model_path, 'triton_models',
                                        'tokenizer')
        tokenizer = Tokenizer(tokenizer_model_path)
        self.stop_words = _stop_words(model.stop_words, tokenizer)

        # params
        self.node_id = node_id
        self.node_num = node_num
        self.world_size = self.node_num * self.gpu_count

        # create model
        weight_dir = osp.join(model_path, 'triton_models', 'weights')
        model = _tm.AbstractTransformerModel.create_llama_model(
            weight_dir, tensor_para_size=self.gpu_count, data_type=data_type)
        self.model = model
        self.nccl_params = model.create_nccl_params(self.node_id)
        torch.cuda.synchronize()

        # create weight
        def _create_weight(device_id):
            with cuda_ctx(device_id):
                rank = self.node_id * self.gpu_count + device_id
                model.create_shared_weights(device_id, rank)

        threads = []
        for device_id in range(self.gpu_count):
            t = Thread(target=_create_weight, args=(device_id, ))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def create_instance(self, cuda_stream_id=0):
        """Create a turbomind instance.

        Args:
            cuda_stream_id(int): identity of a cuda stream
        Returns:
            TurboMindInstance: an instance of turbomind
        """
        return TurboMindInstance(self, cuda_stream_id)


class TurboMindInstance:
    """Instance of TurboMind.

    Args:
        tm_model (str): turbomind's model path
        cuda_stream_id(int): identity of a cuda stream
    """

    def __init__(self, tm_model, cuda_stream_id=0):
        self.tm_model = tm_model
        self.cuda_stream_id = cuda_stream_id

        self.node_id = tm_model.node_id
        self.gpu_count = tm_model.gpu_count

        self.stop_words = tm_model.stop_words
        self.stop_tokens = [] if self.stop_words is None else \
            self.stop_words.flatten().tolist()
        self.eos_id = tm_model.eos_id
        self.session_len = tm_model.session_len

        self.nccl_params = tm_model.nccl_params
        self.instance_comm = tm_model.model.create_instance_comm(
            self.gpu_count)

        # create model instances
        model_insts = [None] * self.gpu_count
        threads = []
        for device_id in range(self.gpu_count):
            t = Thread(target=self._create_model_instance,
                       args=(device_id, model_insts))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        self.model_insts = model_insts
        self.que = Queue()
        self.threads = [None] * self.gpu_count

    def _create_model_instance(self, device_id, model_insts):
        with cuda_ctx(device_id):
            rank = self.node_id * self.gpu_count + device_id
            model_inst = self.tm_model.model.create_model_instance(
                device_id, rank, self.cuda_stream_id, self.nccl_params)
            model_insts[device_id] = model_inst

    def _forward_callback(self, result, ctx):
        self.que.put((False, result))

    def _forward_thread(self, inputs):

        def _func(device_id, enque_output):
            with cuda_ctx(device_id):
                output = self.model_insts[device_id].forward(
                    inputs, self.instance_comm)
                if enque_output:
                    self.que.put((True, output))

        for device_id in range(self.gpu_count):
            t = Thread(target=_func, args=(device_id, device_id == 0))
            t.start()
            self.threads[device_id] = t

    async def async_stream_infer(self, *args, **kwargs):
        """Async wrapper of self.stream_infer."""
        for output in self.stream_infer(*args, **kwargs):
            # Allow the pipeline add new requests into the queue.
            await asyncio.sleep(0)
            yield output

    def stream_infer(self,
                     session_id,
                     input_ids,
                     request_output_len: int = 512,
                     sequence_start: bool = True,
                     sequence_end: bool = False,
                     step=0,
                     stop=False,
                     top_p=0.8,
                     top_k=40,
                     temperature=0.8,
                     repetition_penalty=1.0,
                     ignore_eos=False,
                     random_seed=None,
                     stream_output=False):
        """Perform model inference.

        Args:
            session_id (int): the id of a session
            input_ids (numpy.ndarray): the token ids of a prompt
            request_output_len (int): the max number of to-be-generated tokens
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            step (int): the offset of the k/v cache
            stop (bool): indicator for cancelling the session
            top_p (float): If set to float < 1, only the smallest set of most
              probable tokens with probabilities that add up to top_p or higher
            are kept for generation.
            top_k (int): The number of the highest probability vocabulary
              tokens to keep for top-k-filtering
            temperature (float): to modulate the next token probability
            repetition_penalty (float): The parameter for repetition penalty.
              1.0 means no penalty
            ignore_eos (bool): indicator for ignoring eos
            random_seed (int): seed used by sampling
            stream_output (bool): indicator for stream output
        """
        if stream_output:
            self.model_insts[0].register_callback(self._forward_callback)

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
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.eos_id)

        if isinstance(session_id, int):
            session_id = [session_id]
        assert len(session_id) == batch_size

        step = _broadcast_np(step, np.int32)

        inputs = dict(
            input_ids=input_ids,
            input_lengths=input_lengths,
            request_output_len=np.full(input_lengths.shape,
                                       request_output_len,
                                       dtype=np.uint32),
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

            outputs = []
            for output, len_ in zip(output_ids, sequence_length):
                output, len_ = output, len_.item()
                if len(output) > 0 and output[-1].item() == self.eos_id:
                    outputs.append((output[:-1], len_ - 1))
                elif len(output) > 0 and output[-1].item() in self.stop_tokens:
                    outputs.append((output[:-1], len_))
                else:
                    outputs.append((output, len_))

            yield outputs

            if finish:
                for t in self.threads:
                    t.join()
                while self.que.qsize() > 0:
                    self.que.get()
                break

        if stream_output:
            self.model_insts[0].unregister_callback()

    def decode(self, input_ids):
        """Perform context decode on input tokens.

        Args:
            input_ids (numpy.ndarray): the batch of input token ids
        """

        if len(input_ids) == 0:
            input_ids = []
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]

        # append an extra token since input_len-1 tokens will be
        # decoded by context decoder
        for inputs in input_ids:
            inputs.append(0)

        batch_size = len(input_ids)

        def _broadcast_np(data, dtype, shape=(batch_size, )):
            if isinstance(data, Iterable):
                assert len(data) == batch_size
                return data

            return np.full(shape, data, dtype=dtype)

        input_ids = [torch.IntTensor(ids) for ids in input_ids]
        input_lengths = torch.IntTensor([len(ids) for ids in input_ids])
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.eos_id)

        inputs = dict(input_ids=input_ids,
                      input_lengths=input_lengths,
                      request_output_len=_broadcast_np(0, dtype=np.uint32),
                      is_return_logits=_broadcast_np(1, np.uint32))

        tm_inputs = _np_dict_to_tm_dict(inputs)

        # start forward thread
        self._forward_thread(tm_inputs)

        _, tm_outputs = self.que.get()

        outputs = _tm_dict_to_torch_dict(tm_outputs)
        logits = outputs['logits']

        return logits[:, :-1, :]
