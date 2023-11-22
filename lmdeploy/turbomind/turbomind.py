# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
import io
import json
import logging
import os.path as osp
import sys
from configparser import ConfigParser
from contextlib import contextmanager
from queue import Queue
from threading import Thread
from typing import Iterable, List, Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.nn.utils.rnn import pad_sequence

import lmdeploy
from lmdeploy.model import MODELS, BaseModel
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import get_logger

from .deploy.converter import get_model_format, supported_formats
from .deploy.source_model.base import INPUT_MODELS
from .deploy.target_model.base import OUTPUT_MODELS, TurbomindModelConfig
from .utils import (ModelSource, check_tm_model_input, create_hf_download_args,
                    get_hf_config_content, get_model_source)

# TODO: find another way import _turbomind
lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
sys.path.append(osp.join(lmdeploy_dir, 'lib'))
import _turbomind as _tm  # noqa: E402

logger = logging.getLogger(__name__)


def _stop_words(stop_words: List[str], tokenizer: Tokenizer):
    """return list of stop-words to numpy.ndarray."""
    if stop_words is None:
        return None
    assert isinstance(stop_words, List) and \
        all(isinstance(elem, str) for elem in stop_words), \
        f'stop_words must be a list but got {type(stop_words)}'
    stop_words = [
        tokenizer.encode(stop_word, False)[-1] for stop_word in stop_words
    ]
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
        model_source (int): model source
        model_name (str): needed when model_path is a hf model and not
            managed by lmdeploy
        model_format (str): needed when model_path is a hf model and not
            managed by lmdeploy
        group_size (int): needed when model_path is a hf model and not
            managed by lmdeploy
        tp (int): tensor parallel
    """

    def __init__(self,
                 model_path: str,
                 model_source: ModelSource = ModelSource.WORKSPACE,
                 model_name: Optional[str] = None,
                 model_format: Optional[str] = None,
                 group_size: Optional[int] = None,
                 tp: Optional[int] = None,
                 **kwargs):
        if tp is not None:
            assert ((tp & (tp - 1) == 0) and tp != 0), 'tp should be 2^n'
        self.gpu_count = tp if tp is not None else 1

        if model_source == ModelSource.WORKSPACE:
            tokenizer_model_path = osp.join(model_path, 'triton_models',
                                            'tokenizer')
            self.tokenizer = Tokenizer(tokenizer_model_path)
            self.model_comm = self._from_workspace(model_path)
        else:
            self.tokenizer = Tokenizer(model_path)
            self.model_comm = self._from_hf(model_source=model_source,
                                            model_path=model_path,
                                            model_name=model_name,
                                            model_format=model_format,
                                            group_size=group_size,
                                            tp=tp,
                                            **kwargs)

        self.eos_id = self.tokenizer.eos_token_id
        self.model: BaseModel = MODELS.get(self.model_name)(**kwargs)
        self.session_len = self.model.session_len
        self.stop_words = _stop_words(self.model.stop_words, self.tokenizer)

    def _create_weight(self, model_comm):
        """Allocate weight buffer, load params if from_workspace."""

        # TODO: support mpi
        self.node_id = 0
        self.node_num = 1
        self.nccl_params = model_comm.create_nccl_params(self.node_id)
        torch.cuda.synchronize()

        # create weight
        def _create_weight_func(device_id):
            with cuda_ctx(device_id):
                rank = self.node_id * self.gpu_count + device_id
                model_comm.create_shared_weights(device_id, rank)

        threads = []
        for device_id in range(self.gpu_count):
            t = Thread(target=_create_weight_func, args=(device_id, ))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def _load_kv_qparams(self, model_path, tm_params, **kwargs):
        """Load kv qparams when loading from hf."""
        if self.config.quant_policy:
            logger.warning('loading kv_cache quant scale')
            from lmdeploy.lite.apis.kv_qparams import main as kv_loader
            kv_sym = kwargs.get('kv_sym', False)
            kv_bits = kwargs.get('kv_bits', 8)
            tp = self.config.tensor_para_size
            kv_loader(model_path, model_path, kv_bits, kv_sym, tp, tm_params)
        else:
            for key in list(tm_params.keys()):
                if 'past_kv_scale' in key:
                    tm_params.pop(key)

    def _get_model_params(self, model_comm, tm_params):
        """Get turbomind model params when loading from hf."""

        def _get_params(device_id, que):
            with cuda_ctx(device_id):
                rank = self.node_id * self.gpu_count + device_id
                out = model_comm.get_params(device_id, rank)
                que.put(out)

        que = Queue()
        threads = []
        for device_id in range(self.gpu_count):
            t = Thread(target=_get_params, args=(device_id, que))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        for _ in range(self.gpu_count):
            tensor_map = que.get()
            for k, v in tensor_map.items():
                if k not in tm_params:
                    tm_params[k] = []
                tm_params[k].append(v)

    def _from_hf(self,
                 model_source: ModelSource,
                 model_path: str,
                 model_name: Optional[str] = None,
                 model_format: Optional[str] = None,
                 group_size: Optional[int] = None,
                 tp: Optional[int] = None,
                 **kwargs):
        """Load model which is in hf format."""
        # get model_name, group_size if is lmdeploy managed.
        if model_source == ModelSource.HF_LMDEPLOY:
            config = get_hf_config_content(model_path, local_files_only=True)
            tm_config = config['turbomind']
            tm_config.update(kwargs)
            var_shoud_be_none = dict(model_name=model_name,
                                     model_format=model_format,
                                     group_size=group_size)
            for key, value in var_shoud_be_none.items():
                assert value is None, f'{key} should be None when model is '\
                    f'from {model_source}'
            model_name = tm_config['model_name']
            group_size = tm_config['group_size']
            if tm_config['weight_type'] == 'int4':
                model_format = 'awq'
        else:
            assert model_name is not None, 'please supply model_name when ' \
                f'model is form {model_source}'
            if osp.exists(osp.join(model_path, 'outputs_stats.pth')):
                model_format = 'awq' if model_format is None else model_format
                group_size = 128 if group_size is None else group_size
            tm_config = kwargs

        assert model_name in MODELS.module_dict.keys(), \
            f"'{model_name}' is not supported. " \
            f'The supported models are: {MODELS.module_dict.keys()}'
        assert model_format in supported_formats, 'the model format ' \
            f'should be in {supported_formats}'

        data_type = 'fp16'
        output_format = 'fp16'
        inferred_model_format = get_model_format(model_name, model_format)
        cfg = TurbomindModelConfig.from_dict(tm_config, allow_none=True)

        # overwrite with input params
        cfg.model_name = model_name
        cfg.tensor_para_size = 1 if tp is None else tp
        cfg.rotary_embedding = cfg.size_per_head
        cfg.group_size = group_size
        if inferred_model_format.find('awq') != -1:
            cfg.weight_type = 'int4'
            output_format = 'w4'
            data_type = 'int4'
            assert group_size > 0, f'group_size: {group_size} should > 0'

        self.config = cfg
        self.model_name = model_name
        self.data_type = data_type

        input_model = INPUT_MODELS.get(inferred_model_format)(
            model_path=model_path, tokenizer_path=model_path, ckpt_path=None)

        output_model = OUTPUT_MODELS.get(output_format)(
            input_model=input_model, cfg=cfg, to_file=False, out_dir='')

        config = copy.deepcopy(output_model.cfg.__dict__)
        logger.warning(f'model_config:\n{json.dumps(config, indent=2)}')
        parser = ConfigParser()
        parser['llama'] = config
        with io.StringIO() as ss:
            parser.write(ss)
            ss.seek(0)
            config = ss.read()

        model_comm = _tm.AbstractTransformerModel.create_llama_model(
            model_dir='',
            config=config,
            tensor_para_size=self.gpu_count,
            data_type=data_type)

        # create empty weight
        self._create_weight(model_comm)

        # copy hf model weight to turbomind weight
        tm_params = output_model.tm_params
        self._get_model_params(model_comm, tm_params)
        logger.warning(f'get {len(tm_params)} model params')
        output_model.export()

        # load kv qparams
        self._load_kv_qparams(model_path, tm_params, **kwargs)
        assert len(tm_params) == 0, f'missing {tm_params.keys()}'

        return model_comm

    def _from_workspace(self, model_path: str):
        """Load model which is converted by `lmdeploy convert`"""
        ini_path = osp.join(model_path, 'triton_models', 'weights',
                            'config.ini')
        with open(ini_path, 'r') as f:
            parser = ConfigParser()
            parser.read_file(f)
            section_name = 'llama'
            tp_cfg = parser.getint(section_name, 'tensor_para_size')

            if tp_cfg != 1 and tp_cfg != self.gpu_count:
                get_logger('turbomind').info(
                    f'found tp={tp_cfg} in config.ini.')
                self.gpu_count = tp_cfg
            self.model_name = parser.get(section_name, 'model_name')
            self.data_type = parser.get(section_name, 'weight_type')
            cfg = parser._sections[section_name]
            cfg = TurbomindModelConfig.from_dict(cfg)
            self.config = cfg

        # create model
        weight_dir = osp.join(model_path, 'triton_models', 'weights')
        model_comm = _tm.AbstractTransformerModel.create_llama_model(
            weight_dir,
            tensor_para_size=self.gpu_count,
            data_type=self.data_type)

        # create weight and load params
        self._create_weight(model_comm)
        return model_comm

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        model_name: Optional[str] = None,
                        model_format: Optional[str] = None,
                        group_size: Optional[int] = None,
                        tp: Optional[int] = None,
                        **kwargs):
        """LMDeploy's turbomind inference engine.

        Args:
            pretrained_model_name_or_path (str):
                It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                      converted by `lmdeploy convert` command or download from
                      ii) and iii)
                    - ii) The model_id of a lmdeploy-quantized model hosted
                      inside a model repo on huggingface.co, such as
                      "InternLM/internlm-chat-20b-4bit",
                      "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                      on huggingface.co, such as "InternLM/internlm-chat-7b",
                      "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                      and so on.
            model_name (str): needed when pretrained_model_name_or_path is c)
            model_format (str): model format
            group_size (int): group size
            tp (int): tensor parallel size
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update configuration when initialize the engine.
        """
        model_source = get_model_source(pretrained_model_name_or_path)
        if model_source == ModelSource.WORKSPACE:
            local_path = pretrained_model_name_or_path
        else:
            check_tm_model_input(pretrained_model_name_or_path,
                                 model_name=model_name,
                                 **kwargs)
            if not osp.exists(pretrained_model_name_or_path):
                download_kwargs = create_hf_download_args(**kwargs)
                local_path = snapshot_download(pretrained_model_name_or_path,
                                               **download_kwargs)
            else:
                local_path = pretrained_model_name_or_path

        logger.warning(f'model_source: {model_source}')
        return cls(model_source=model_source,
                   model_path=local_path,
                   model_name=model_name,
                   model_format=model_format,
                   group_size=group_size,
                   tp=tp,
                   **kwargs)

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

    def __init__(self, tm_model: TurboMind, cuda_stream_id: int = 0):
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
            model_inst = self.tm_model.model_comm.create_model_instance(
                device_id, rank, self.cuda_stream_id, self.nccl_params)
            model_insts[device_id] = model_inst

    def _forward_callback(self, result, ctx):
        self.que.put((False, result))

    def _forward_thread(self, inputs):
        instance_comm = self.tm_model.model_comm.create_instance_comm(
            self.gpu_count)

        def _func(device_id, enque_output):
            with cuda_ctx(device_id):
                output = self.model_insts[device_id].forward(
                    inputs, instance_comm)
                if enque_output:
                    self.que.put((True, output))

        for device_id in range(self.gpu_count):
            t = Thread(target=_func,
                       args=(device_id, device_id == 0),
                       daemon=True)
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
        if stream_output and not stop:
            self.model_insts[0].register_callback(self._forward_callback)

        if len(input_ids) == 0:
            input_ids = [[]]
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
        self.que = Queue()
        self._forward_thread(tm_inputs)

        seq_start = input_lengths + input_lengths.new_tensor(step)

        # generator
        while True:
            while self.que.qsize() > 1:
                self.que.get()

            finish, tm_outputs = self.que.get()

            outputs = _tm_dict_to_torch_dict(tm_outputs)

            output_ids = outputs['output_ids'][:, 0, :]
            sequence_length = outputs['sequence_length'].long()[:, 0]
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

        if stream_output and not stop:
            self.model_insts[0].unregister_callback()

    def decode(self, input_ids):
        """Perform context decode on input tokens.

        Args:
            input_ids (numpy.ndarray): the batch of input token ids
        """

        if len(input_ids) == 0:
            input_ids = [[]]
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
