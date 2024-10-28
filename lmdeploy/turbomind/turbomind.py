# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
import json
import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import repeat
from queue import LifoQueue, Queue
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

import lmdeploy
from lmdeploy.messages import (EngineOutput, GenerationConfig, ResponseType,
                               TurbomindEngineConfig)
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import get_logger, get_max_batch_size, get_model

from .deploy.config import TurbomindModelConfig
from .supported_models import is_supported
from .utils import ModelSource, get_model_source

# TODO: find another way import _turbomind
lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
sys.path.append(osp.join(lmdeploy_dir, 'lib'))
import _turbomind as _tm  # noqa: E402

logger = get_logger('lmdeploy')

MAX_LOGPROBS = 1024


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


class TurboMind:
    """LMDeploy's inference engine.

    Args:
        model_path (str): the path of turbomind's model
        mode_name (str): the name of the served model
        chat_template_name (str): the name of the chat template, which is
            supposed to be a builtin chat template defined in
            `lmdeploy/model.py`
        engine_config (TurbomindEngineConfig): the config of the inference
            engine
        model_source (int): the source of the model, which is either
            turbomind model, or a transformers model
    """

    def __init__(self,
                 model_path: str,
                 model_name: str = None,
                 chat_template_name: str = None,
                 engine_config: TurbomindEngineConfig = None,
                 model_source: ModelSource = ModelSource.WORKSPACE,
                 **kwargs):
        self.model_name = model_name
        self.chat_template_name = chat_template_name

        _engine_config = copy.deepcopy(engine_config)
        if _engine_config is None:
            _engine_config = TurbomindEngineConfig()
        if _engine_config.max_batch_size is None:
            _engine_config.max_batch_size = get_max_batch_size('cuda')
        assert _engine_config.max_batch_size > 0, 'max_batch_size should be' \
            f' greater than 0, but got {_engine_config.max_batch_size}'

        self.gpu_count = _engine_config.tp

        if model_source == ModelSource.WORKSPACE:
            tokenizer_model_path = osp.join(model_path, 'triton_models',
                                            'tokenizer')
            self.tokenizer = Tokenizer(tokenizer_model_path)
            self.model_comm = self._from_workspace(
                model_path=model_path, engine_config=_engine_config)
        else:
            if not osp.exists(model_path):
                model_path = get_model(model_path, _engine_config.download_dir,
                                       _engine_config.revision)
            self.tokenizer = Tokenizer(model_path)
            self.model_comm = self._from_hf(model_source=model_source,
                                            model_path=model_path,
                                            engine_config=_engine_config)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as e:
            ranks = [
                self.node_id * self.gpu_count + device_id
                for device_id in range(self.gpu_count)
            ]
            for _ in e.map(self.model_comm.process_weight,
                           range(self.gpu_count), ranks):
                pass
            # implicit synchronization
            for _ in e.map(self.model_comm.create_engine,
                           range(self.gpu_count), ranks,
                           repeat(self.nccl_params)):
                pass

        self.session_len = self.config.session_len
        self.eos_id = self.tokenizer.eos_token_id

    def _create_weight(self, model_comm):
        """Allocate weight buffer, load params if from_workspace."""

        # TODO: support mpi
        self.node_id = 0
        self.node_num = 1
        self.nccl_params = model_comm.create_nccl_params(self.node_id)
        torch.cuda.synchronize()

        # create weight
        def _create_weight_func(device_id):
            rank = self.node_id * self.gpu_count + device_id
            model_comm.create_shared_weights(device_id, rank)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            for device_id in range(self.gpu_count):
                futures.append(executor.submit(_create_weight_func, device_id))
            for future in futures:
                future.result()

    def _get_model_params(self, model_comm, tm_params):
        """Get turbomind model params when loading from hf."""

        def _get_params(device_id, que):
            rank = self.node_id * self.gpu_count + device_id
            out = model_comm.get_params(device_id, rank)
            que.put(out)

        que = Queue()
        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            for device_id in range(self.gpu_count):
                futures.append(executor.submit(_get_params, device_id, que))
            for future in futures:
                future.result()

        for _ in range(self.gpu_count):
            tensor_map = que.get()
            for k, v in tensor_map.items():
                if k not in tm_params:
                    tm_params[k] = []
                tm_params[k].append(v)

    def _postprocess_config(self, tm_config, engine_config):
        """postprocess turbomind config by."""
        import copy
        self.config = copy.deepcopy(tm_config)
        # Update the attribute values in `self.config` with the valid values
        # from the corresponding attributes in `engine_config`, such as
        # `session_len`, `quant_policy`, `rope_scaling_factor`, etc.
        self.config.update_from_engine_config(engine_config)

        # update some attributes of `engine_config` which depends on
        # `session_len`
        self.engine_config = engine_config
        if engine_config.max_prefill_token_num is not None \
                and engine_config.num_tokens_per_iter == 0:
            self.engine_config.num_tokens_per_iter = \
                engine_config.max_prefill_token_num
            self.engine_config.max_prefill_iters = (
                self.config.session_len + engine_config.max_prefill_token_num -
                1) // engine_config.max_prefill_token_num

        # pack `self.config` and `self.engine_config` into a dict
        self.config_dict = self.config.to_dict()
        self.config_dict.update(dict(engine_config=asdict(self.engine_config)))
        logger.info(f'turbomind model config:\n\n'
                    f'{json.dumps(self.config_dict, indent=2)}')

    def _from_hf(self, model_source: ModelSource, model_path: str,
                 engine_config: TurbomindEngineConfig):
        """Load model which is in hf format."""
        assert model_source == ModelSource.HF_MODEL, \
            f'{model_source} is not supported'
        assert is_supported(model_path), (
            f'turbomind does not support {model_path}. '
            'Plz try pytorch engine instead.')

        # convert transformers model into turbomind model
        from .deploy.converter import get_tm_model
        tm_model = get_tm_model(model_path, self.model_name,
                                self.chat_template_name, engine_config)

        self._postprocess_config(tm_model.tm_config, engine_config)

        model_comm = _tm.AbstractTransformerModel.create_llama_model(
            model_dir='',
            config=yaml.safe_dump(self.config_dict),
            tensor_para_size=self.gpu_count,
            data_type=self.config.model_config.weight_type)

        # create empty weight
        self._create_weight(model_comm)

        # copy hf model weight to turbomind weight
        tm_params = tm_model.tm_params
        self._get_model_params(model_comm, tm_params)
        logger.warning(f'get {len(tm_params)} model params')
        tm_model.export()
        # there should be no left turbomind params.
        if len(tm_params) > 0:
            uninitialized = list(tm_params.keys())
            logger.warning(
                'the model may not be loaded successfully '
                f'with {len(tm_params)} uninitialized params:\n{uninitialized}'
            )
        return model_comm

    def _from_workspace(self, model_path: str,
                        engine_config: TurbomindEngineConfig):
        """Load model which is converted by `lmdeploy convert`"""
        config_path = osp.join(model_path, 'triton_models', 'weights',
                               'config.yaml')
        # load TurbomindModelConfig from config file
        with open(config_path, 'r') as f:
            _cfg = yaml.safe_load(f)
        cfg = TurbomindModelConfig.from_dict(_cfg)

        # always use tp in converted model (config.yaml)
        if cfg.tensor_para_size != engine_config.tp:
            logger.warning(
                'tp in engine_config is different from in config.yaml'
                f'({config_path}), {engine_config.tp} vs '
                f'{cfg.tensor_para_size}, using tp={cfg.tensor_para_size}')
        self.gpu_count = cfg.tensor_para_size
        engine_config.tp = self.gpu_count

        self._postprocess_config(cfg, engine_config)

        weight_dir = osp.join(model_path, 'triton_models', 'weights')
        model_comm = _tm.AbstractTransformerModel.create_llama_model(
            model_dir=weight_dir,
            config=yaml.safe_dump(self.config_dict),
            tensor_para_size=self.gpu_count,
            data_type=self.config.weight_type)

        # create weight and load params
        self._create_weight(model_comm)
        return model_comm

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        model_name: str = None,
                        chat_template_name: str = None,
                        engine_config: TurbomindEngineConfig = None,
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
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update configuration when initialize the engine.
        """
        model_source = get_model_source(pretrained_model_name_or_path)
        logger.info(f'model_source: {model_source}')
        return cls(model_path=pretrained_model_name_or_path,
                   model_name=model_name,
                   chat_template_name=chat_template_name,
                   engine_config=engine_config,
                   model_source=model_source,
                   **kwargs)

    def create_instance(self, cuda_stream_id=0):
        """Create a turbomind instance.

        Args:
            cuda_stream_id(int): identity of a cuda stream
        Returns:
            TurboMindInstance: an instance of turbomind
        """
        return TurboMindInstance(self, self.config, cuda_stream_id)


class TurboMindInstance:
    """Instance of TurboMind.

    Args:
        tm_model (str): turbomind's model path
        cuda_stream_id(int): identity of a cuda stream
    """

    def __init__(self,
                 tm_model: TurboMind,
                 config: TurbomindModelConfig,
                 cuda_stream_id: int = 0):
        self.tm_model = tm_model
        self.cuda_stream_id = cuda_stream_id

        self.node_id = tm_model.node_id
        self.gpu_count = tm_model.gpu_count

        self.eos_id = tm_model.eos_id
        self.session_len = tm_model.session_len

        self.nccl_params = tm_model.nccl_params

        # create model instances
        self.model_inst = self._create_model_instance(0)

        self.que = Queue()
        self.executor: ThreadPoolExecutor = None
        self.future = None
        self.config = config

    def _create_model_instance(self, device_id):
        rank = self.node_id * self.gpu_count + device_id
        model_inst = self.tm_model.model_comm.create_model_instance(
            device_id, rank, self.cuda_stream_id, self.nccl_params)
        return model_inst

    def _forward_callback(self, result, ctx):
        self.que.put((False, result))

    def _forward_thread(self, inputs):
        instance_comm = self.tm_model.model_comm.create_instance_comm(
            self.gpu_count)

        def _func():
            try:
                output = self.model_inst.forward(inputs, instance_comm)
            except Exception as e:
                logger.error(f'unhandled exception: {e}')
                self.que.put((-1, None))
                return
            self.que.put((True, output))

        self.executor = ThreadPoolExecutor(1)
        self.future = self.executor.submit(_func)

    def _async_forward_callback(self, result, ctx, que: LifoQueue):
        que.put((False, result))

    def _async_forward_thread(self, inputs, que: LifoQueue):
        instance_comm = self.tm_model.model_comm.create_instance_comm(
            self.gpu_count)

        def _func():
            try:
                output = self.model_inst.forward(inputs, instance_comm)
            except Exception as e:
                logger.error(f'unhandled exception: {e}')
                que.put((-1, None))
                return
            que.put((True, output))

        self.executor = ThreadPoolExecutor(1)
        self.future = self.executor.submit(_func)

    def _get_logprobs(self,
                      logprob_vals: torch.Tensor,
                      logprob_indexes: torch.Tensor,
                      logprob_nums: torch.Tensor,
                      output_ids: torch.Tensor,
                      logprobs: int = None,
                      length: int = None,
                      out_logprobs: List[Dict[int, float]] = None,
                      session_id: int = None):
        if logprobs is None:
            return None
        if out_logprobs is None:
            out_logprobs = []
        if len(output_ids) <= len(out_logprobs):
            return out_logprobs
        offset = len(out_logprobs)
        for (token_id, idx, val, n) in zip(output_ids[offset:length],
                                           logprob_indexes[offset:length],
                                           logprob_vals[offset:length],
                                           logprob_nums[offset:length]):
            topn = min(n.item(), logprobs)
            tok_res = {idx[i].item(): val[i].item() for i in range(topn)}
            if token_id.item() not in tok_res:
                valid_n = n.item()
                tok_res[token_id.item()] = \
                    val[:valid_n][idx[:valid_n] == token_id].item()
            ids = list(tok_res.keys())
            for k in ids:
                if tok_res[k] == float('-inf'):
                    tok_res.pop(k)
            out_logprobs.append(tok_res)
        return out_logprobs

    def end(self, session_id: int):
        """End the given session."""
        input_ids = [self.tm_model.tokenizer.eos_token_id]
        end_generator = self.tm_model.create_instance()
        for outputs in end_generator.stream_infer(
                session_id,
                input_ids,
                sequence_start=False,
                sequence_end=True,
                gen_config=GenerationConfig(max_new_tokens=0)):
            pass

    async def async_end(self, session_id: int):
        """End the given session."""
        self.end(session_id)

    def cancel(self, session_id: int):
        """Stop current streaming inference."""
        input_ids = [self.tm_model.tokenizer.eos_token_id]
        stop_generator = self.tm_model.create_instance()
        for outputs in stop_generator.stream_infer(
                session_id,
                input_ids,
                sequence_start=False,
                sequence_end=False,
                stop=True,
                gen_config=GenerationConfig(max_new_tokens=0)):
            pass

    async def async_cancel(self, session_id: int):
        """End the given session."""
        self.cancel(session_id)

    def prepare_embeddings(self,
                           input_embeddings=None,
                           input_embedding_ranges=None):
        """Convert embeddings."""
        if input_embeddings is None:
            return None, None

        assert len(input_embeddings) == len(input_embedding_ranges)
        if not isinstance(input_embeddings[0], (list, type(None))):
            input_embeddings = [input_embeddings]
            input_embedding_ranges = [input_embedding_ranges]

        if all([isinstance(x, type(None)) for x in input_embeddings]):
            return None, None

        hidden_dim = None
        for embeddings in input_embeddings:
            if embeddings is not None:
                hidden_dim = embeddings[0].squeeze().shape[-1]
                break
        assert hidden_dim is not None

        # construct input_embeddings
        for i in range(len(input_embeddings)):
            item = input_embeddings[i] or []
            # convert to torch.Tensor if input is np.ndarray
            if item and isinstance(item[0], np.ndarray):
                item = [torch.from_numpy(x).squeeze() for x in item]
            # convert to lookup table type
            _MAP = dict(float=torch.float,
                        bfloat16=torch.bfloat16,
                        float16=torch.float16)
            dtype = _MAP.get(self.tm_model.config.weight_type, torch.float16)
            item = [x.to(dtype=dtype) for x in item]
            item = item or [torch.zeros(0, hidden_dim, dtype=dtype)]
            input_embeddings[i] = item
        input_embeddings = [torch.cat(x) for x in input_embeddings]
        input_embeddings = pad_sequence(input_embeddings, batch_first=True)
        input_embeddings = input_embeddings.reshape(input_embeddings.shape[0],
                                                    -1).view(torch.int8)
        # construct input_embedding_ranges
        for i in range(len(input_embedding_ranges)):
            item = input_embedding_ranges[i] or []
            item = torch.IntTensor(item).reshape(-1, 2)
            input_embedding_ranges[i] = item
        input_embedding_ranges = pad_sequence(input_embedding_ranges,
                                              batch_first=True,
                                              padding_value=-1)

        return input_embeddings, input_embedding_ranges

    def prepare_inputs(self,
                       session_id,
                       input_ids,
                       gen_config: GenerationConfig,
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
            runtime_min_p=_broadcast_np(gen_config.min_p, np.float32),
            temperature=_broadcast_np(gen_config.temperature, np.float32),
            repetition_penalty=_broadcast_np(gen_config.repetition_penalty,
                                             np.float32),
            step=step,

            # session input
            START=_broadcast_np((1 if sequence_start else 0), np.int32),
            END=_broadcast_np((1 if sequence_end else 0), np.int32),
            CORRID=np.array(session_id, dtype=np.uint64),
            STOP=_broadcast_np((1 if stop else 0), np.int32))

        input_embeddings, input_embedding_ranges = self.prepare_embeddings(
            input_embeddings, input_embedding_ranges)
        if input_embeddings is not None:
            inputs['input_embeddings'] = input_embeddings
            inputs['input_embedding_ranges'] = input_embedding_ranges

        if gen_config.min_new_tokens is not None:
            inputs['min_length'] = _broadcast_np(gen_config.min_new_tokens,
                                                 np.int32)

        if gen_config.logprobs is not None and gen_config.logprobs > 0:
            if gen_config.logprobs > MAX_LOGPROBS:
                gen_config.logprobs = MAX_LOGPROBS
                logger.warning('logprobs shoudd be in range [1, 1024]'
                               f'update logprobs={gen_config.logprobs}')
            inputs['logprobs'] = _broadcast_np(gen_config.logprobs, np.int32)

        bad_words = []
        if gen_config.bad_token_ids is not None:
            bad_words.extend(gen_config.bad_token_ids)
        if gen_config.ignore_eos:
            stop_words = None
            bad_words.append(self.eos_id)
        else:
            stop_words = gen_config.stop_token_ids or []
            if self.eos_id not in stop_words:
                stop_words.append(self.eos_id)
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
                                 gen_config: GenerationConfig = None,
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
            gen_config (GenerationConfig): generation config
            stream_output (bool): indicator for stream output
            kwargs (dict): kwargs for backward compatibility
        """
        # start forward thread
        que = LifoQueue()
        from functools import partial
        _forward_callback = partial(self._async_forward_callback, que=que)
        _forward_thread = partial(self._async_forward_thread, que=que)
        if stream_output and not stop:
            logger.info(f'Register stream callback for {session_id}')
            self.model_inst.register_callback(_forward_callback)

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
        _forward_thread(tm_inputs)

        seq_start = input_lengths + input_lengths.new_tensor(step)

        out_logprobs = None
        prev_len = 0
        # generator
        while True:
            while que.qsize() == 0:  # let other requests in
                await asyncio.sleep(0.002)

            finish, tm_outputs = que.get()
            if finish < 0:
                yield EngineOutput(status=ResponseType.INTERNAL_ENGINE_ERROR,
                                   token_ids=[],
                                   num_token=0)
                self.executor.shutdown()
                break

            outputs = _tm_dict_to_torch_dict(tm_outputs)

            output_ids = outputs['output_ids'][:, 0, :]
            sequence_length = outputs['sequence_length'].long()[:, 0]
            output_ids = [
                output_id[s:l] for output_id, s, l in zip(
                    output_ids, seq_start, sequence_length)
            ]
            sequence_length -= seq_start.to(sequence_length.device)

            if 'logprob_vals' in outputs:
                logprob_vals = outputs['logprob_vals'][0, 0]
                logprob_indexes = outputs['logprob_indexes'][0, 0]
                logprob_nums = outputs['logprob_nums'][0, 0]
                out_logprobs = self._get_logprobs(logprob_vals,
                                                  logprob_indexes,
                                                  logprob_nums, output_ids[0],
                                                  gen_config.logprobs,
                                                  sequence_length.cpu().item(),
                                                  out_logprobs, session_id)

            outputs = []
            status = ResponseType.FINISH if finish else ResponseType.SUCCESS
            for output, len_ in zip(output_ids, sequence_length):
                output, len_ = output, len_.item()
                if len(output) > 0 and output[-1].item() == self.eos_id \
                        and not gen_config.ignore_eos:
                    outputs = EngineOutput(status, output[:-1].tolist(),
                                           len_ - 1)
                elif len(output) > 0 and \
                    gen_config.stop_token_ids is not None and \
                        output[-1].item() in gen_config.stop_token_ids:
                    outputs = EngineOutput(status, output[:-1].tolist(), len_)
                else:
                    outputs = EngineOutput(status, output.tolist(), len_)
            if outputs.num_token < prev_len and not finish:
                continue
            else:
                prev_len = outputs.num_token

            if out_logprobs:
                output_token_len = len(outputs.token_ids)
                outputs.logprobs = out_logprobs[:output_token_len]

            yield outputs

            if finish:
                self.future.result()
                self.executor.shutdown()
                break

        if stream_output and not stop:
            logger.info(f'UN-register stream callback for {session_id}')
            self.model_inst.unregister_callback()

    def stream_infer(self,
                     session_id,
                     input_ids,
                     input_embeddings=None,
                     input_embedding_ranges=None,
                     sequence_start: bool = True,
                     sequence_end: bool = False,
                     step=0,
                     stop=False,
                     gen_config: GenerationConfig = None,
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
            gen_config (GenerationConfig): generation config
            stream_output (bool): indicator for stream output
            kwargs (dict): kwargs for backward compatibility
        """
        if stream_output and not stop:
            logger.info(f'Register stream callback for {session_id}')
            self.model_inst.register_callback(self._forward_callback)

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
        out_logprobs = None

        # generator
        while True:
            while self.que.qsize() > 1:
                self.que.get()

            finish, tm_outputs = self.que.get()
            if finish < 0:
                yield EngineOutput(status=ResponseType.INTERNAL_ENGINE_ERROR,
                                   token_ids=[],
                                   num_token=0)
                self.executor.shutdown()
                break

            outputs = _tm_dict_to_torch_dict(tm_outputs)

            output_ids = outputs['output_ids'][:, 0, :]
            sequence_length = outputs['sequence_length'].long()[:, 0]
            output_ids = [
                output_id[s:l] for output_id, s, l in zip(
                    output_ids, seq_start, sequence_length)
            ]
            sequence_length -= seq_start.to(sequence_length.device)

            if 'logprob_vals' in outputs:
                logprob_vals = outputs['logprob_vals'][0, 0]
                logprob_indexes = outputs['logprob_indexes'][0, 0]
                logprob_nums = outputs['logprob_nums'][0, 0]
                out_logprobs = self._get_logprobs(logprob_vals,
                                                  logprob_indexes,
                                                  logprob_nums, output_ids[0],
                                                  gen_config.logprobs,
                                                  sequence_length.cpu().item(),
                                                  out_logprobs, session_id)

            outputs = []
            status = ResponseType.FINISH if finish else ResponseType.SUCCESS
            for output, len_ in zip(output_ids, sequence_length):
                output, len_ = output, len_.item()
                if len(output) > 0 and output[-1].item() == self.eos_id \
                        and not gen_config.ignore_eos:
                    outputs = EngineOutput(status, output[:-1].tolist(),
                                           len_ - 1, out_logprobs)
                elif len(output) > 0 and \
                    gen_config.stop_token_ids is not None and \
                        output[-1].item() in gen_config.stop_token_ids:
                    outputs = EngineOutput(status, output[:-1].tolist(), len_,
                                           out_logprobs)
                else:
                    outputs = EngineOutput(status, output.tolist(), len_,
                                           out_logprobs)

            if out_logprobs:
                output_token_len = len(outputs.token_ids)
                outputs.logprobs = out_logprobs[:output_token_len]

            yield outputs

            if finish:
                self.future.result()
                self.executor.shutdown()
                while self.que.qsize() > 0:
                    self.que.get()
                break

        if stream_output and not stop:
            logger.info(f'UN-register stream callback for {session_id}')
            self.model_inst.unregister_callback()

    def decode(self,
               input_ids,
               steps: List[int] = None,
               input_embeddings=None,
               input_embedding_ranges=None,
               sequence_start: bool = True,
               sequence_end: bool = True):
        """Perform context decode on input tokens.

        Args:
            input_ids (numpy.ndarray): the batch of input token ids
            steps (List[int]): the offset of the k/v cache
            input_embeddings (List[List[Union[torch.Tensor, np.ndarray]]]):
                embeddings features
            input_embedding_ranges: (List[List[Tuple[int, int]]]):
                the begin/end offsets of input_embeddings to input_ids
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
        input_ids = [x[:] for x in input_ids]
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

        input_embeddings, input_embedding_ranges = self.prepare_embeddings(
            input_embeddings, input_embedding_ranges)
        if input_embeddings is not None:
            inputs['input_embeddings'] = input_embeddings
            inputs['input_embedding_ranges'] = input_embedding_ranges

        tm_inputs = _np_dict_to_tm_dict(inputs)

        # start forward thread
        self._forward_thread(tm_inputs)

        res, tm_outputs = self.que.get()
        if res < 0:
            return None

        outputs = _tm_dict_to_torch_dict(tm_outputs)
        logits = outputs['logits']

        return logits[:, :-1, :]
