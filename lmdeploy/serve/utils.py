# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from lmdeploy.messages import GenerationConfig
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

InputIdsType = List[int]
InputEmbsType = Union[None, List[Union[torch.Tensor, np.ndarray]]]
InputEmbRngsType = Union[None, List[Tuple[int, int]]]
PromptType = Union[str, List[Dict]]


class LogitsMixin:
    """Helper class to calculate ppl."""

    async def _async_get_logits(
            self,
            input_ids,
            input_embeddings=None,
            input_embedding_ranges=None,
            steps: List[int] = None,
            sequence_start: bool = True,
            sequence_end: bool = True,
            adapter_name: str = None) -> List[torch.Tensor]:
        assert input_ids and all(isinstance(_, List) for _ in input_ids)
        assert (input_embeddings is None or isinstance(input_embeddings, List)
                and len(input_ids) == len(input_embeddings))
        assert (input_embedding_ranges is None
                or isinstance(input_embedding_ranges, List)
                and len(input_ids) == len(input_embedding_ranges))
        logits = []
        for i in range(len(input_ids)):
            session_id = next(self._session_id)
            history_len = (0 if session_id not in self.id2step else
                           self.id2step[session_id])
            token_ids = input_ids[i].copy()
            embeddings = input_embeddings[i] if input_embeddings else None
            ranges = (input_embedding_ranges[i]
                      if input_embedding_ranges else None)
            # `max_new_tokens=0` means we don't need engine to generate tokens
            # `output_logits=True` requests engine to output logits
            gen_config = GenerationConfig(max_new_tokens=0, output_logits=True)
            async with self.model_inst(session_id) as inst:
                async with self.safe_run(inst,
                                         session_id=session_id,
                                         input_ids=token_ids,
                                         input_embeddings=embeddings,
                                         input_embedding_ranges=ranges,
                                         gen_config=gen_config,
                                         adapter_name=adapter_name,
                                         stream_output=False,
                                         sequence_start=sequence_start,
                                         sequence_end=sequence_end,
                                         step=history_len) as gen:
                    async for outputs in gen:
                        # We only need to process the final `outputs` since
                        # stream_output is False
                        pass
                    logits.append(outputs.logits)
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

        # TODO: a better way to determine `max_input_len`, at most allocate
        # 2G mem for logits with shape [bs, max_input_len, vocab_size]
        vocab_size = self.hf_tm_cfg.vocab_size
        max_input_len = 2 * 1024**3 // (vocab_size * 4)
        sizes = [len(_) for _ in input_ids]
        result = []
        sorted_index_values = sorted(list(enumerate(sizes)),
                                     key=lambda x: x[1],
                                     reverse=True)
        sizes = [value for index, value in sorted_index_values]
        indices = [index for index, value in sorted_index_values]
        logger.info(f'sorted sizes: {sizes}')
        logger.info(f'sorted indices: {indices}')
        for (start, end) in self._batch_iterator(sizes, max_input_len):
            logger.info(f'start: {start}, end: {end}')
            if start == end:
                _input_ids = input_ids[indices[start]]
                res = self._get_long_text_ppl(input_ids=_input_ids,
                                              max_input_len=max_input_len)
                result.append(res)
            else:
                _input_ids = [input_ids[indices[i]] for i in range(start, end)]
                res = self._get_ppl(
                    input_ids=_input_ids,
                    max_input_len=max_input_len,
                )
                result.extend(res)
        output = list(range(len(result)))
        for index, sorted_index in enumerate(indices):
            output[sorted_index] = result[index]
        return output

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

    def _get_long_text_ppl(self, input_ids, max_input_len):
        assert all(isinstance(_, int) for _ in input_ids)
        seq_len = len(input_ids)
        assert seq_len > max_input_len
        logger.info(f'get long text ppl: seq_len {seq_len}')

        losses = []
        target_counts = []
        for i in range(0, seq_len, max_input_len):
            token_ids = input_ids[i:i + max_input_len]
            step = [i]
            # shift token_ids by 1 to the left
            target_ids = input_ids[i + 1:i + 1 + max_input_len]

            loss, target_count = self._get_ppl(
                input_ids=[token_ids],
                max_input_len=max_input_len,
                target_ids=[target_ids],
                steps=step,
                sequence_start=(i == 0),
                sequence_end=(i + max_input_len >= seq_len))
            losses.extend(loss)
            target_counts.extend(target_count)
        loss_sum = sum(losses)
        target_count = sum(target_counts)
        return loss_sum / target_count

    def _get_ppl(self,
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
        logits = self._run(
            coro=self._async_get_logits(input_ids=input_ids,
                                        steps=steps,
                                        sequence_start=sequence_start,
                                        sequence_end=sequence_end)).result()
        result = []
        for _logits in logits:
            _logits = _logits.float()
            seq_len, vocab_size = _logits.shape
            padding_token_id = -100
            if target_ids is None:
                # shift token_ids by 1 to the left
                target_ids = [x[1:] + [padding_token_id] for x in input_ids]
            else:
                target_ids = [
                    target_ids[i] + [padding_token_id] if
                    len(target_ids[i]) < len(input_ids[i]) else target_ids[i]
                    for i in range(len(input_ids))
                ]
            target_ids = [
                torch.Tensor(torch.LongTensor(_target_ids))
                for _target_ids in target_ids
            ]
            target_ids = pad_sequence(target_ids,
                                      batch_first=True,
                                      padding_value=padding_token_id)
            target_ids = target_ids.to(_logits.device)
            target_mask = target_ids != padding_token_id
            # compute cross entropy loss
            flat_logits = _logits.contiguous().view(-1, vocab_size)
            flat_target_ids = target_ids.contiguous().view(-1)
            flat_loss_matrix = torch.nn.functional.cross_entropy(
                flat_logits,
                flat_target_ids,
                reduction='none',
                ignore_index=padding_token_id)
            loss = flat_loss_matrix.sum()
            target_count = target_mask.sum()
            result.append(loss.item() / target_count.item())
        return result
