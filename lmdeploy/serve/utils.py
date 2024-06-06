# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

InputIdsType = List[int]
InputEmbsType = Union[None, List[Union[torch.Tensor, np.ndarray]]]
InputEmbRngsType = Union[None, List[Tuple[int, int]]]
PromptType = Union[str, List[Dict]]


def _get_event_loop():
    """get event loop."""
    try:
        event_loop = asyncio.get_event_loop()
    except Exception:
        logger.warning('Can not found event loop in current thread.'
                       ' Create a new event loop.')
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
    return event_loop


class LogitsMixin:
    """Helper class to calculate logits and ppl."""

    def prepare_inputs(self,
                       prompts: Union[PromptType, List[PromptType]],
                       do_preprocess: bool = True):
        if hasattr(self, '_convert_prompts'):
            prompts = self._convert_prompts(prompts)
        need_list_wrap = isinstance(prompts, str) or isinstance(
            prompts[0], Dict)
        prompts = [prompts] if need_list_wrap else prompts

        decorated = []
        input_ids = []
        input_embeddings = []
        input_embedding_ranges = []
        for prompt in prompts:
            out = _get_event_loop().run_until_complete(
                self._get_prompt_input(prompt,
                                       do_preprocess=do_preprocess,
                                       sequence_start=True,
                                       adapter_name=None))
            decorated.append(out['prompt'])
            input_ids.append(out['input_ids'])
            input_embeddings.append(out.get('input_embeddings', None))
            input_embedding_ranges.append(
                out.get('input_embedding_ranges', None))

        outputs = dict(prompts=decorated, input_ids=input_ids)
        if not any(input_embeddings):
            input_embeddings = None
            input_embedding_ranges = None
        outputs['input_embeddings'] = input_embeddings
        outputs['input_embedding_ranges'] = input_embedding_ranges

        return outputs

    def get_logits(
        self,
        input_ids: Union[InputIdsType, List[InputIdsType]],
        input_embeddings: Union[InputEmbsType, List[InputEmbsType]] = None,
        input_embedding_ranges: Union[InputEmbRngsType,
                                      List[InputEmbRngsType]] = None):
        """Get logits given a list of input tokens.

        Args:
            input_ids (Union[List[int], List[List[int]]]): the batch of
                input token ids
        """
        assert len(input_ids) > 0
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
        for input_id in input_ids:
            assert len(input_id) > 0

        max_input_len = self.backend_config.max_prefill_token_num
        n_max_iter = np.ceil(
            max([len(input_id)
                 for input_id in input_ids]) / max_input_len).astype(int)

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

        def _split_embeddings(input_ids, niter, iter_len, embeddings,
                              embedding_ranges):
            embs = [None] * niter
            ranges = [None] * niter

            if embeddings is None:
                return embs, ranges

            for i in range(niter):
                iembs = []
                iranges = []
                for emb, (begin, end) in zip(embeddings, embedding_ranges):
                    assert end <= len(input_ids)
                    if begin >= (i + 1) * iter_len or end <= i * iter_len:
                        continue
                    if isinstance(emb, np.ndarray):
                        emb = torch.from_numpy(emb)
                    emb = emb.squeeze()
                    offx = max(iter_len * i - begin, 0)
                    offy = max(end - iter_len * (i + 1), 0)
                    emb = emb[offx:emb.shape[0] - offy]
                    off = max(begin - iter_len * i, 0)
                    rng = [off, off + emb.shape[0]]
                    iembs.append(emb)
                    iranges.append(rng)

                iembs = iembs or None
                iranges = iranges or None
                embs[i] = iembs
                ranges[i] = iranges

            return embs, ranges

        if input_embeddings is not None:
            if not isinstance(input_embeddings[0], list):
                input_embeddings = [input_embeddings]
                input_embedding_ranges = [input_embedding_ranges]
            _input_embeddings = []
            _input_embedding_ranges = []
            for i in range(len(input_ids)):
                embeddings, ranges = _split_embeddings(
                    input_ids[i], n_max_iter, max_input_len,
                    input_embeddings[i], input_embedding_ranges[i])
                _input_embeddings.append(embeddings)
                _input_embedding_ranges.append(ranges)
            input_embeddings = _input_embeddings
            input_embedding_ranges = _input_embedding_ranges

        logits = []
        generator = self.engine.create_instance()
        for i in range(n_max_iter):
            steps = [start[i] for start in index_range_starts]
            _input_ids = [
                input_id[start[i]:end[i]] for input_id, start, end in zip(
                    input_ids, index_range_starts, index_range_ends)
            ]
            embeddings = None
            ranges = None
            if input_embeddings is not None:
                embeddings = [x[i] for x in input_embeddings]
                ranges = [x[i] for x in input_embedding_ranges]

            _logits = generator.decode(_input_ids,
                                       steps,
                                       input_embeddings=embeddings,
                                       input_embedding_ranges=ranges,
                                       sequence_start=(i == 0),
                                       sequence_end=(i == n_max_iter - 1))
            _logits = _logits.cpu()
            logits.append(_logits)

        # concat logits. Shape is [bsz, seq_len, vocab_size]
        logits = torch.cat(logits, dim=1)
        return logits

    def get_ppl(self, input_ids: Union[List[int], List[List[int]]]):
        """Get perplexity scores given a list of input tokens.

        Args:
            input_ids (Union[List[int], List[List[int]]]): the batch of
                input token ids
        """
        assert len(input_ids) > 0
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
        for input_id in input_ids:
            assert len(input_id) > 1

        max_input_len = self.backend_config.max_prefill_token_num
        n_max_iter = np.ceil(
            max([len(input_id)
                 for input_id in input_ids]) / max_input_len).astype(int)

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

        generator = self.engine.create_instance()
        all_loss_matrix = []
        all_target_mask = []
        for i in range(n_max_iter):
            steps = [start[i] for start in index_range_starts]
            _input_ids = [
                input_id[start[i]:end[i]] for input_id, start, end in zip(
                    input_ids, index_range_starts, index_range_ends)
            ]
            _logits = generator.decode(_input_ids,
                                       steps,
                                       sequence_start=(i == 0),
                                       sequence_end=(i == n_max_iter - 1))
            _logits = _logits.cpu()
            padding_token_id = -100
            target_ids = [(x + [padding_token_id])[1:] for x in _input_ids]
            target_ids = [
                torch.Tensor(torch.LongTensor(_target_ids))
                for _target_ids in target_ids
            ]
            target_ids = pad_sequence(target_ids,
                                      batch_first=True,
                                      padding_value=padding_token_id)
            target_ids = target_ids.to(_logits.device)
            target_mask = target_ids != padding_token_id
            target_count = torch.sum(target_mask, dim=-1)
            # compute cross entropy loss
            bsz, seq_len, vocab_size = _logits.shape
            flat_logits = _logits.contiguous().view(-1, vocab_size)
            flat_target_ids = target_ids.contiguous().view(-1)
            flat_loss_matrix = torch.nn.functional.cross_entropy(
                flat_logits,
                flat_target_ids,
                reduction='none',
                ignore_index=padding_token_id)

            all_loss_matrix.append(flat_loss_matrix.view(bsz, seq_len))
            all_target_mask.append(target_mask)

        all_loss_matrix = torch.cat(all_loss_matrix, dim=1)
        all_target_mask = torch.cat(all_target_mask, dim=1)
        target_count = torch.sum(all_target_mask, dim=-1)
        loss_sum = torch.sum(all_loss_matrix * all_target_mask, dim=1)
        loss_avg = loss_sum / target_count
        loss_avg = loss_avg.cpu().numpy()
        return loss_avg
