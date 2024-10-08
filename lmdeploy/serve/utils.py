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

    def prepare_inputs(self, prompts: Union[PromptType, List[PromptType]]):
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
                                       do_preprocess=True,
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

        bs = len(input_ids)
        # TODO: a better way to determine `max_input_len`, at most allocate
        # 2G mem for logits with shape [bs, max_input_len, vocab_size]
        vocab_size = self.hf_tm_cfg.vocab_size
        max_input_len = 2 * 1024**3 // (bs * vocab_size * 4)

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
                                       steps=steps,
                                       input_embeddings=embeddings,
                                       input_embedding_ranges=ranges,
                                       sequence_start=(i == 0),
                                       sequence_end=(i == n_max_iter - 1))
            _logits = _logits.cpu()
            logits.append(_logits)

        # concat logits. Shape is [bsz, seq_len, vocab_size]
        logits = torch.cat(logits, dim=1)
        return logits

    def get_ppl(self, input_ids: Union[List[int],
                                       List[List[int]]]) -> List[float]:
        """Get perplexity scores given a list of input tokens that have to be
        of the same length.

        Args:
            input_ids (Union[List[int], List[List[int]]]): the batch of
                input token ids

        Returns:
            Union[float, List[float]]: A list of perplexity scores.
        """
        assert isinstance(input_ids, List)
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]

        generator = self.engine.create_instance()

        # TODO: a better way to determine `max_input_len`, at most allocate
        # 2G mem for logits with shape [bs, max_input_len, vocab_size]
        vocab_size = self.hf_tm_cfg.vocab_size
        max_input_len = 2 * 1024**3 // (vocab_size * 4)
        sizes = [len(_) for _ in input_ids]
        losses = []
        target_counts = []
        sorted_index_values = sorted(list(enumerate(sizes)),
                                     key=lambda x: x[1],
                                     reverse=True)
        sizes = [value for index, value in sorted_index_values]
        indices = [index for index, value in sorted_index_values]
        logger.info(f'sorted sizes: {sizes}')
        logger.info(f'sorted indices: {indices}')
        for (start, end) in self._batch_iterator(sizes, max_input_len):
            logger.info(f'start: {start}, end: {end}')
            _input_ids = [input_ids[indices[i]] for i in range(start, end)]
            if start == end:
                loss, target_count = self._get_long_text_ppl(
                    generator=generator,
                    input_ids=_input_ids,
                    max_input_len=max_input_len)
                losses.append(loss)
                target_counts.append(target_count)
            else:
                loss, target_count = self._get_ppl(
                    generator=generator,
                    input_ids=_input_ids,
                    max_input_len=max_input_len,
                )
                losses.append(loss)
                target_counts.append(target_count)
        loss = torch.concatenate(losses)
        target_count = torch.concatenate(target_counts)
        loss_avg = loss / target_count
        loss_avg = loss_avg.numpy().tolist()
        result = list(range(len(loss_avg)))
        for index, sorted_index in enumerate(indices):
            result[sorted_index] = loss_avg[index]
        return result

    def _batch_iterator(self, sizes, max_value):
        """Return an iterator that calculates intervals (start, end) of a
        descend-order list, in which the sum of values in the range is the
        maximum number not less than max_value. By "the sum of values",

        here it means $$len(sizes[start:end]) * sizes[start]$$
        """
        i = 0
        while i < len(sizes):
            current_sum = 0
            start_index = i

            while i < len(
                    sizes) and current_sum + sizes[start_index] <= max_value:
                current_sum += sizes[start_index]
                i += 1

            yield (start_index, i)
            if i > start_index:
                continue
            else:
                i += 1

    def _get_long_text_ppl(self, generator, input_ids, max_input_len):
        assert isinstance(input_ids, List) and len(input_ids) == 1
        seq_len = len(input_ids[0])
        assert seq_len > max_input_len
        logger.info(f'get long text ppl: seq_len {seq_len}')

        losses = []
        target_counts = []
        for i in range(0, seq_len, max_input_len):
            token_ids = input_ids[:, i:i + max_input_len]
            step = [i]
            # shift token_ids by 1 to the left
            target_ids = input_ids[:, i + 1:i + 1 + max_input_len]

            loss, target_count = self._get_ppl(
                generator=generator,
                input_ids=token_ids,
                max_input_len=max_input_len,
                target_ids=target_ids,
                steps=step,
                sequence_start=(i == 0),
                sequence_end=(i + max_input_len >= seq_len))
            losses.append(loss)
            target_counts.append(target_count)
        loss_sum = torch.concatenate(losses).sum().unsqueeze(0)
        target_count = torch.concatenate(target_counts).sum().unsqueeze(0)
        return loss_sum, target_count

    def _get_ppl(self,
                 generator,
                 input_ids,
                 max_input_len,
                 target_ids=None,
                 steps=None,
                 sequence_start: bool = True,
                 sequence_end: bool = True):
        assert isinstance(input_ids, List)
        assert all(isinstance(_, List) for _ in input_ids)
        if target_ids:
            assert all(isinstance(_, List) for _ in target_ids)

        lens = [len(_) for _ in input_ids]
        total_len = sum(lens)
        assert sum(lens) <= max_input_len

        logger.info(f'get_ppl: bs: {len(input_ids)}, lens: {lens}, '
                    f'total_len: {total_len}')
        torch.cuda.empty_cache()
        logits = generator.decode(input_ids=input_ids,
                                  steps=steps,
                                  sequence_start=sequence_start,
                                  sequence_end=sequence_end)
        bsz, seq_len, vocab_size = logits.shape
        logits = logits.float()
        padding_token_id = -100
        if target_ids is None:
            # shift token_ids by 1 to the left
            target_ids = [x[1:] + [padding_token_id] for x in input_ids]
        else:
            target_ids = [
                target_ids[i] + [padding_token_id]
                if len(target_ids[i]) < len(input_ids[i]) else target_ids[i]
                for i in range(bsz)
            ]
        target_ids = [
            torch.Tensor(torch.LongTensor(_target_ids))
            for _target_ids in target_ids
        ]
        target_ids = pad_sequence(target_ids,
                                  batch_first=True,
                                  padding_value=padding_token_id)
        target_ids = target_ids.to(logits.device)
        target_mask = target_ids != padding_token_id

        # compute cross entropy loss
        flat_logits = logits.contiguous().view(-1, vocab_size)
        flat_target_ids = target_ids.contiguous().view(-1)
        flat_loss_matrix = torch.nn.functional.cross_entropy(
            flat_logits,
            flat_target_ids,
            reduction='none',
            ignore_index=padding_token_id)
        flat_loss_matrix = flat_loss_matrix.view(bsz, seq_len)
        loss = flat_loss_matrix.sum(dim=-1).cpu()
        target_count = target_mask.sum(dim=-1).cpu()
        return loss, target_count
