# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
import logging
import os.path as osp
import sys
from configparser import ConfigParser
from contextlib import contextmanager
from queue import Queue
from threading import Thread
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.nn.utils.rnn import pad_sequence

import lmdeploy
from lmdeploy.messages import (EngineGenerationConfig, ResponseType,
                               TurbomindEngineConfig)
from lmdeploy.model import (MODELS, BaseModel, ChatTemplateConfig,
                            best_match_model)
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import _stop_words, get_logger

from .deploy.converter import (get_model_format, supported_formats,
                               update_config_weight_type, update_output_format)
from .deploy.source_model.base import INPUT_MODELS
from .deploy.target_model.base import OUTPUT_MODELS, TurbomindModelConfig
from .utils import ModelSource, create_hf_download_args, get_model_source

# TODO: find another way import _turbomind
lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
sys.path.append(osp.join(lmdeploy_dir, 'lib'))
import _turbomind as _tm  # noqa: E402

logger = logging.getLogger(__name__)


def _construct_stop_or_bad_words(words: List[int] = None):
    if words is None or len(words) == 0:
        return None
    offsets = range(1, len(words) + 1)
    combined = np.array([[words, offsets]]).astype(np.int32)
    return combined


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


def _update_engine_config(config: TurbomindEngineConfig, **kwargs):
    if config is None:
        config = TurbomindEngineConfig()
    for k, v in kwargs.items():
        if v and hasattr(config, k):
            setattr(config, k, v)
            get_logger('turbomind').warning(
                f'kwargs {k} is deprecated to initialize model, '
                'use TurbomindEngineConfig instead.')
    return config


def _update_tm_config(dst: TurbomindModelConfig, src: TurbomindEngineConfig):
    dst_dict = copy.deepcopy(dst.__dict__)
    src_dict = copy.deepcopy(src.__dict__)
    src_dict['tensor_para_size'] = src_dict['tp']
    for k, v in src_dict.items():
        if v is not None and k in dst_dict:
            dst_dict[k] = v
    return TurbomindModelConfig.from_dict(dst_dict)


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
                 engine_config: TurbomindEngineConfig = None,
                 model_source: ModelSource = ModelSource.WORKSPACE,
                 model_name: Optional[str] = None,
                 model_format: Optional[str] = None,
                 group_size: Optional[int] = None,
                 tp: Optional[int] = None,
                 chat_template_config: Optional[ChatTemplateConfig] = None,
                 **kwargs):

        # check model_name equal in engine_config and passed in
        if engine_config is not None and engine_config.model_name is not None:
            if model_name is not None:
                assert model_name == engine_config.model_name, 'Got different'
                f' model names. model_name: {model_name}, engine_config model '
                f'name: {engine_config.model_name}'

        # if loading from workspace and engine_config is None, use config.ini
        # and ignore passed args like model_format, tp, etc.
        if model_source == ModelSource.WORKSPACE and engine_config is None:

            def _catch_args(**kwargs):
                args = []
                for k, v in kwargs.items():
                    if v and hasattr(TurbomindEngineConfig, k):
                        args.append(k)
                return args

            args = _catch_args(**kwargs,
                               model_name=model_name,
                               model_format=model_format,
                               tp=tp)
            if len(args) > 0:
                get_logger('turbomind').warning(
                    f'loading from workspace, ignore args {args} '
                    'please use TurbomindEngineConfig or modify config.ini')

        else:
            engine_config = _update_engine_config(engine_config,
                                                  model_name=model_name,
                                                  model_format=model_format,
                                                  group_size=group_size,
                                                  tp=tp,
                                                  **kwargs)

        # match model name
        if model_source == ModelSource.HF_MODEL and \
                engine_config.model_name is None:
            potential_names = best_match_model(model_path)
            if potential_names is None:
                logger.warning(f'Please input a model_name for {model_source}')
            else:
                engine_config.model_name = potential_names
                logger.warning('Best matched chat template name: '
                               f'{engine_config.model_name}')

        tp = engine_config.tp if engine_config is not None else 1
        assert ((tp & (tp - 1) == 0) and tp != 0), 'tp should be 2^n'
        self.gpu_count = tp

        if model_source == ModelSource.WORKSPACE:
            tokenizer_model_path = osp.join(model_path, 'triton_models',
                                            'tokenizer')
            self.tokenizer = Tokenizer(tokenizer_model_path)
            self.model_comm = self._from_workspace(model_path=model_path,
                                                   engine_config=engine_config)
        else:
            self.tokenizer = Tokenizer(model_path)
            self.model_comm = self._from_hf(model_source=model_source,
                                            model_path=model_path,
                                            engine_config=engine_config)

        if chat_template_config:
            if chat_template_config.model_name is None:
                chat_template_config.model_name = self.model_name
                logger.warning(f'Input chat template with model_name is None. '
                               f'Forcing to use {self.model_name}')
            self.model = chat_template_config.chat_template
        else:
            self.model: BaseModel = MODELS.get(self.model_name)(**kwargs)
        self.session_len = self.config.session_len
        self.eos_id = self.tokenizer.eos_token_id
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

    def _from_hf(self, model_source: ModelSource, model_path: str,
                 engine_config: TurbomindEngineConfig):
        """Load model which is in hf format."""
        assert model_source == ModelSource.HF_MODEL, \
            f'{model_source} is not supported'
        assert engine_config.model_name in MODELS.module_dict.keys(), \
            f"'{engine_config.model_name}' is not supported. " \
            f'The supported models are: {MODELS.module_dict.keys()}'
        assert engine_config.model_format in supported_formats, \
            f'The model format should be in {supported_formats}'

        # update model_format if not supplied and outputs_stats.pth exists
        if osp.exists(osp.join(model_path, 'outputs_stats.pth')) and \
                engine_config.model_format is None:
            engine_config.model_format = 'awq'

        data_type = 'fp16'
        output_format = 'fp16'
        inferred_model_format = get_model_format(engine_config.model_name,
                                                 engine_config.model_format)
        cfg = TurbomindModelConfig.from_engine_config(engine_config)
        if inferred_model_format.find('awq') != -1:
            cfg.weight_type = 'int4'
            output_format = 'w4'
            data_type = 'int4'
            cfg.group_size = 128
        else:
            output_format = update_output_format(engine_config.model_name,
                                                 inferred_model_format,
                                                 model_path, output_format)
            data_type = output_format
            update_config_weight_type(output_format, cfg)

        input_model = INPUT_MODELS.get(inferred_model_format)(
            model_path=model_path, tokenizer_path=model_path, ckpt_path=None)

        output_model = OUTPUT_MODELS.get(output_format)(
            input_model=input_model, cfg=cfg, to_file=False, out_dir='')

        cfg = output_model.cfg
        if engine_config.session_len is not None:
            cfg.session_len = engine_config.session_len

        self.config = cfg
        self.model_name = engine_config.model_name
        self.data_type = data_type

        logger.warning(f'model_config:\n\n{cfg.toini()}')

        model_comm = _tm.AbstractTransformerModel.create_llama_model(
            model_dir='',
            config=cfg.toini(),
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
        self._load_kv_qparams(model_path, tm_params, kv_sym=False, kv_bits=8)
        assert len(tm_params) == 0, f'missing {tm_params.keys()}'

        return model_comm

    def _from_workspace(self, model_path: str,
                        engine_config: TurbomindEngineConfig):
        """Load model which is converted by `lmdeploy convert`"""
        ini_path = osp.join(model_path, 'triton_models', 'weights',
                            'config.ini')
        # load cfg
        with open(ini_path, 'r') as f:
            parser = ConfigParser()
            parser.read_file(f)
        section_name = 'llama'
        _cfg = parser._sections[section_name]
        cfg = TurbomindModelConfig.from_dict(_cfg)

        # check whether input tp is valid
        if cfg.tensor_para_size != 1 and \
                self.gpu_count != cfg.tensor_para_size:
            get_logger('turbomind').info(
                f'found tp={cfg.tensor_para_size} in config.ini.')
            self.gpu_count = cfg.tensor_para_size

        # update cfg
        if engine_config is not None:
            engine_config.tp = cfg.tensor_para_size
            cfg = _update_tm_config(cfg, engine_config)
            if engine_config.session_len is not None:
                cfg.session_len = engine_config.session_len

        # update cls
        self.config = cfg
        self.model_name = cfg.model_name
        self.data_type = cfg.weight_type

        # create model
        logger.warning(f'model_config:\n\n{cfg.toini()}')
        weight_dir = osp.join(model_path, 'triton_models', 'weights')
        model_comm = _tm.AbstractTransformerModel.create_llama_model(
            model_dir=weight_dir,
            config=cfg.toini(),
            tensor_para_size=self.gpu_count,
            data_type=self.data_type)

        # create weight and load params
        self._create_weight(model_comm)
        return model_comm

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            engine_config: TurbomindEngineConfig = None,
            model_name: Optional[str] = None,
            model_format: Optional[str] = None,
            group_size: Optional[int] = None,
            tp: Optional[int] = None,
            chat_template_config: Optional[ChatTemplateConfig] = None,
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
                      on huggingface.co, such as "internlm/internlm-chat-7b",
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
            if model_name is None and (engine_config is None
                                       or engine_config.model_name is None):
                # huggingface repo id will be changed to local path in .cache
                # have to match name in ahead.
                model_name = best_match_model(pretrained_model_name_or_path)
            if not osp.exists(pretrained_model_name_or_path):
                download_kwargs = create_hf_download_args(**kwargs)
                local_path = snapshot_download(pretrained_model_name_or_path,
                                               **download_kwargs)
            else:
                local_path = pretrained_model_name_or_path

        logger.warning(f'model_source: {model_source}')
        return cls(model_path=local_path,
                   engine_config=engine_config,
                   model_source=model_source,
                   model_name=model_name,
                   model_format=model_format,
                   group_size=group_size,
                   tp=tp,
                   chat_template_config=chat_template_config,
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

    def _update_generation_config(self, config: EngineGenerationConfig,
                                  **kwargs: dict):
        if config is None:
            config = EngineGenerationConfig()
        # backward compatibility
        # if doesn't supply stop words, use default
        if config.stop_words is None and self.stop_words is not None:
            config.stop_words = self.stop_words[0][0].tolist()

        deprecated_kwargs = []
        for k, v in kwargs.items():
            if k in config.__dict__:
                config.__dict__[k] = v
                deprecated_kwargs.append(k)
        if kwargs.get('request_output_len'):
            config.max_new_tokens = kwargs['request_output_len']
            deprecated_kwargs.append('request_output_len')
        for k in deprecated_kwargs:
            get_logger('turbomind').warning(
                f'kwargs {k} is deprecated for inference, '
                'use GenerationConfig instead.')
        return config

    def end(self, session_id: int):
        """End the given session."""
        input_ids = [self.tm_model.tokenizer.eos_token_id]
        end_generator = self.tm_model.create_instance()
        for outputs in end_generator.stream_infer(session_id,
                                                  input_ids,
                                                  request_output_len=0,
                                                  sequence_start=False,
                                                  sequence_end=True):
            pass

    def cancel(self, session_id: int):
        """Stop current streaming inference."""
        input_ids = [self.tm_model.tokenizer.eos_token_id]
        stop_generator = self.tm_model.create_instance()
        for outputs in stop_generator.stream_infer(session_id,
                                                   input_ids,
                                                   request_output_len=0,
                                                   sequence_start=False,
                                                   sequence_end=False,
                                                   stop=True):
            pass

    def prepare_inputs(self,
                       session_id,
                       input_ids,
                       gen_config: EngineGenerationConfig,
                       input_embeddings=None,
                       input_embedding_ranges=None,
                       sequence_start: bool = True,
                       sequence_end: bool = False,
                       step=0,
                       stop=False):
        """Convert inputs format."""
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
                                       gen_config.max_new_tokens,
                                       dtype=np.uint32),
            runtime_top_k=_broadcast_np(gen_config.top_k, np.uint32),
            runtime_top_p=_broadcast_np(gen_config.top_p, np.float32),
            temperature=_broadcast_np(gen_config.temperature, np.float32),
            repetition_penalty=_broadcast_np(gen_config.repetition_penalty,
                                             np.float32),
            step=step,

            # session input
            START=_broadcast_np((1 if sequence_start else 0), np.int32),
            END=_broadcast_np((1 if sequence_end else 0), np.int32),
            CORRID=np.array(session_id, dtype=np.uint64),
            STOP=_broadcast_np((1 if stop else 0), np.int32))

        if input_embeddings is not None:
            assert len(input_embeddings) == len(input_embedding_ranges)
            if isinstance(input_embeddings[0], np.ndarray):
                input_embeddings = [input_embeddings]
                input_embedding_ranges = [input_embedding_ranges]
            # convert to lookup table type
            if self.tm_model.config.weight_type == 'fp32':
                input_embeddings = [[x.astype(np.float32) for x in y]
                                    for y in input_embeddings]
            elif self.tm_model.config.weight_type == 'bf16':
                input_embeddings = [[
                    torch.from_numpy(x).bfloat16().view(torch.half).numpy()
                    for x in y
                ] for y in input_embeddings]
            else:
                input_embeddings = [[x.astype(np.float16) for x in y]
                                    for y in input_embeddings]

            input_embeddings = [[torch.from_numpy(x).squeeze() for x in y]
                                for y in input_embeddings]
            input_embeddings = [torch.cat(x) for x in input_embeddings]
            input_embeddings = pad_sequence(input_embeddings, batch_first=True)
            input_embeddings = input_embeddings.reshape(
                input_embeddings.shape[0], -1).view(torch.int8)

            _input_embedding_ranges = []
            for x in input_embedding_ranges:
                if x is not None and len(x) != 0:
                    _input_embedding_ranges.append(torch.IntTensor(x))
                else:
                    _input_embedding_ranges.append(torch.IntTensor(size=(0,
                                                                         2)))
            input_embedding_ranges = pad_sequence(_input_embedding_ranges,
                                                  batch_first=True,
                                                  padding_value=-1)
            inputs['input_embeddings'] = input_embeddings
            inputs['input_embedding_ranges'] = input_embedding_ranges

        bad_words = []
        if gen_config.bad_words is not None:
            bad_words.extend(gen_config.bad_words)
        if gen_config.ignore_eos:
            stop_words = None
            bad_words.append(self.eos_id)
        else:
            stop_words = gen_config.stop_words
        stop_words = _construct_stop_or_bad_words(stop_words)
        bad_words = _construct_stop_or_bad_words(bad_words)

        if stop_words is not None:
            inputs['stop_words_list'] = stop_words
        if bad_words is not None:
            inputs['bad_words_list'] = bad_words

        if gen_config.random_seed is not None:
            inputs['random_seed'] = _broadcast_np(gen_config.random_seed,
                                                  np.uint64)
        return inputs, input_lengths

    async def async_stream_infer(self,
                                 session_id,
                                 input_ids,
                                 input_embeddings=None,
                                 input_embedding_ranges=None,
                                 sequence_start: bool = True,
                                 sequence_end: bool = False,
                                 step=0,
                                 stop=False,
                                 gen_config: EngineGenerationConfig = None,
                                 stream_output=False,
                                 **kwargs):
        """Perform model inference.

        Args:
            session_id (int): the id of a session
            input_ids (numpy.ndarray): the token ids of a prompt
            input_embeddings (List[numpy.ndarray]): embeddings features
            input_embedding_ranges (List[Tuple[int,int]]): the begin/end
              offsets of input_embeddings to input_ids
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            step (int): the offset of the k/v cache
            stop (bool): indicator for cancelling the session
            gen_config (EngineGenerationConfig): generation config
            stream_output (bool): indicator for stream output
            kwargs (dict): kwargs for backward compatibility
        """
        if stream_output and not stop:
            self.model_insts[0].register_callback(self._forward_callback)

        gen_config = self._update_generation_config(gen_config, **kwargs)
        inputs, input_lengths = self.prepare_inputs(
            session_id=session_id,
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            input_embedding_ranges=input_embedding_ranges,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            step=step,
            stop=stop,
            gen_config=gen_config)

        tm_inputs = _np_dict_to_tm_dict(inputs)
        # start forward thread
        self.que = Queue()
        self._forward_thread(tm_inputs)

        seq_start = input_lengths + input_lengths.new_tensor(step)

        # generator
        while True:
            # Thanks for https://github.com/frankxyy and his issue
            # https://github.com/InternLM/lmdeploy/issues/832
            while self.que.qsize() == 0:
                await asyncio.sleep(0.002)  # sleep(0) makes server unstable

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
            status = ResponseType.FINISH if finish else ResponseType.SUCCESS
            for output, len_ in zip(output_ids, sequence_length):
                output, len_ = output, len_.item()
                if len(output) > 0 and output[-1].item() == self.eos_id \
                        and not gen_config.ignore_eos:
                    outputs = (status, output[:-1].tolist(), len_ - 1)
                elif len(output) > 0 and \
                    gen_config.stop_words is not None and \
                        output[-1].item() in gen_config.stop_words:
                    outputs = (status, output[:-1].tolist(), len_)
                else:
                    outputs = (status, output.tolist(), len_)
            yield outputs

            if finish:
                for t in self.threads:
                    t.join()
                while self.que.qsize() > 0:
                    self.que.get()
                break

        if stream_output and not stop:
            self.model_insts[0].unregister_callback()

    def stream_infer(self,
                     session_id,
                     input_ids,
                     input_embeddings=None,
                     input_embedding_ranges=None,
                     sequence_start: bool = True,
                     sequence_end: bool = False,
                     step=0,
                     stop=False,
                     gen_config: EngineGenerationConfig = None,
                     stream_output=False,
                     **kwargs):
        """Perform model inference.

        Args:
            session_id (int): the id of a session
            input_ids (numpy.ndarray): the token ids of a prompt
            input_embeddings (List[numpy.ndarray]): embeddings features
            input_embedding_ranges (List[Tuple[int,int]]): the begin/end
              offsets of input_embeddings to input_ids
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            step (int): the offset of the k/v cache
            stop (bool): indicator for cancelling the session
            gen_config (EngineGenerationConfig): generation config
            stream_output (bool): indicator for stream output
            kwargs (dict): kwargs for backward compatibility
        """
        if stream_output and not stop:
            self.model_insts[0].register_callback(self._forward_callback)

        gen_config = self._update_generation_config(gen_config, **kwargs)
        inputs, input_lengths = self.prepare_inputs(
            session_id=session_id,
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            input_embedding_ranges=input_embedding_ranges,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            step=step,
            stop=stop,
            gen_config=gen_config)

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
            status = ResponseType.FINISH if finish else ResponseType.SUCCESS
            for output, len_ in zip(output_ids, sequence_length):
                output, len_ = output, len_.item()
                if len(output) > 0 and output[-1].item() == self.eos_id \
                        and not gen_config.ignore_eos:
                    outputs = (status, output[:-1].tolist(), len_ - 1)
                elif len(output) > 0 and \
                    gen_config.stop_words is not None and \
                        output[-1].item() in gen_config.stop_words:
                    outputs = (status, output[:-1].tolist(), len_)
                else:
                    outputs = (status, output.tolist(), len_)
            yield outputs

            if finish:
                for t in self.threads:
                    t.join()
                while self.que.qsize() > 0:
                    self.que.get()
                break

        if stream_output and not stop:
            self.model_insts[0].unregister_callback()

    def decode(self,
               input_ids,
               steps: List[int] = None,
               sequence_start: bool = True,
               sequence_end: bool = True):
        """Perform context decode on input tokens.

        Args:
            input_ids (numpy.ndarray): the batch of input token ids
            steps (List[int]): the offset of the k/v cache
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
        """

        if len(input_ids) == 0:
            input_ids = [[]]
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
        if steps is None:
            steps = [0] * len(input_ids)
        assert isinstance(steps, List) and len(steps) == len(input_ids)

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
        steps = torch.IntTensor([step for step in steps])

        inputs = dict(input_ids=input_ids,
                      input_lengths=input_lengths,
                      request_output_len=_broadcast_np(0, dtype=np.uint32),
                      is_return_logits=_broadcast_np(1, np.uint32),
                      START=_broadcast_np((1 if sequence_start else 0),
                                          np.int32),
                      END=_broadcast_np((1 if sequence_end else 0), np.int32),
                      step=steps)

        tm_inputs = _np_dict_to_tm_dict(inputs)

        # start forward thread
        self._forward_thread(tm_inputs)

        _, tm_outputs = self.que.get()

        outputs = _tm_dict_to_torch_dict(tm_outputs)
        logits = outputs['logits']

        return logits[:, :-1, :]

    def get_ppl(self, input_ids: Union[List[int], List[List[int]]]):
        """Get perplexity scores given a list of input tokens.

        Args:
            input_ids (Union[List[int], List[List[int]]]): the batch of input token ids
        """  # noqa 501

        if len(input_ids) == 0:
            input_ids = [[]]
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]

        max_input_len = 16 * 1024
        # max_input_len = 16
        n_max_iter = np.ceil(
            max([len(input_id)
                 for input_id in input_ids]) / max_input_len).astype(int)

        device = 'cpu' if n_max_iter > 0 else 'cuda'

        index_range_starts = []
        index_range_ends = []
        for input_id in input_ids:
            index_range_start = np.array(
                [i * max_input_len for i in range(n_max_iter)])
            index_range_end = index_range_start + max_input_len
            index_range_start[index_range_start >= len(input_id)] = len(
                input_id)
            index_range_end[index_range_end >= len(input_id)] = len(input_id)
            index_range_starts.append(index_range_start)
            index_range_ends.append(index_range_end)

        logits = []
        for i in range(n_max_iter):
            steps = [start[i] for start in index_range_starts]
            _input_ids = [
                input_id[start[i]:end[i]] for input_id, start, end in zip(
                    input_ids, index_range_starts, index_range_ends)
            ]
            _logits = self.decode(_input_ids,
                                  steps,
                                  sequence_start=(i == 0),
                                  sequence_end=(i == n_max_iter - 1))
            _logits = _logits.to(device=device)
            logits.append(_logits)

        # concat logits. Shape is [bsz, seq_len, vocab_size]
        logits = torch.cat(logits, dim=1)

        # get target ids
        padding_token_id = -100
        target_ids = [(_input_ids + [padding_token_id])[1:]
                      for _input_ids in input_ids]
        target_ids = [
            torch.Tensor(torch.LongTensor(_target_ids))
            for _target_ids in target_ids
        ]
        target_ids = pad_sequence(target_ids,
                                  batch_first=True,
                                  padding_value=padding_token_id)
        target_ids = target_ids.to(logits.device)
        target_mask = target_ids != padding_token_id
        target_count = torch.sum(target_mask, dim=-1)

        # compute cross entropy loss
        bsz, seq_len, vocab_size = logits.shape
        flat_logits = logits.contiguous().view(-1, vocab_size)
        flat_target_ids = target_ids.contiguous().view(-1)
        flat_loss_matrix = torch.nn.functional.cross_entropy(
            flat_logits,
            flat_target_ids,
            reduction='none',
            ignore_index=padding_token_id)

        loss_matrix = flat_loss_matrix.view(bsz, seq_len)
        loss_sum = torch.sum(loss_matrix * target_mask, dim=1)
        loss_avg = loss_sum / target_count
        loss_avg = loss_avg.cpu().numpy()
        return loss_avg
