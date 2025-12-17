# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from lmdeploy.messages import GenerationConfig
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

InputIdsType = List[int]
InputEmbsType = Union[None, List[Union[torch.Tensor, np.ndarray]]]
InputEmbRngsType = Union[None, List[Tuple[int, int]]]
PromptType = Union[str, List[Dict]]


class LogitsMixin:
    """Helper class to get logits, reward score and calculate ppl."""

    def get_reward_score(self, input_ids: List) -> List[float]:
        """
        Args:
            input_ids(List): a list of token_id or a list of token_id list or a tensor containing
                token_ids
        Return:
            reward score in a list. If the input_ids is a list of token_id, the return value
            is still a list with length 1.
        """
        supported_reward_models = ['InternLM2ForRewardModel', 'Qwen2ForRewardModel']
        if self.arch not in supported_reward_models:
            raise ValueError(f'{self.arch} is not in reward model list: {supported_reward_models}')
        assert isinstance(input_ids, List)
        assert all(isinstance(x, int) for x in input_ids) or all(isinstance(x, List) for x in input_ids)
        # Make input_ids a list of token_id list
        input_ids = [input_ids] if isinstance(input_ids[0], int) else input_ids
        logits = self._run(coro=self._async_get_logits(input_ids=input_ids)).result()
        logits = [x.squeeze() for x in logits]
        scores = [x[-1].cpu().item() for x in logits]
        return scores

    async def _async_get_reward_score(self, input_ids: List) -> List[float]:
        """Async version of get_reward_score."""
        supported_reward_models = ['InternLM2ForRewardModel', 'Qwen2ForRewardModel']
        if self.arch not in supported_reward_models:
            raise ValueError(f'{self.arch} is not in reward model list: {supported_reward_models}')
        assert isinstance(input_ids, List)
        assert all(isinstance(x, int) for x in input_ids) or all(isinstance(x, List) for x in input_ids)
        # Make input_ids a list of token_id list
        input_ids = [input_ids] if isinstance(input_ids[0], int) else input_ids

        logits = await self._async_get_logits(input_ids=input_ids)

        logits = [x.squeeze() for x in logits]
        scores = [x[-1].cpu().item() for x in logits]
        return scores

    async def _async_get_logits(self,
                                input_ids,
                                steps: List[int] = None,
                                sequence_start: bool = True,
                                sequence_end: bool = True) -> List[torch.Tensor]:
        assert input_ids and all(isinstance(_, List) for _ in input_ids)
        assert steps is None or (len(steps) == len(input_ids))

        logits = [None] * len(input_ids)

        async def _proc(i):
            async with self.model_inst(session_id=i) as inst:
                input_len = len(input_ids[i])
                # TODO(lvhan): Fix the ugly code later on
                max_new_tokens = 1 if self.backend == 'turbomind' else 0
                # The reason to set `top_k=1` is that pt engine crashes at top_k sampling stage
                # when perform inference on a reward model.
                gen_config = GenerationConfig(max_new_tokens=max_new_tokens, output_logits='all', top_k=1)
                async with self.safe_run(inst,
                                         session_id=i,
                                         input_ids=input_ids[i],
                                         gen_config=gen_config,
                                         stream_output=False,
                                         sequence_start=sequence_start,
                                         sequence_end=sequence_end,
                                         step=steps[i] if steps else 0) as gen:
                    async for outputs in gen:
                        pass
                    logits[i] = outputs.logits[:input_len, :]

        session_ids = list(range(len(input_ids)))
        tasks = [_proc(i) for i in range(len(input_ids))]
        await asyncio.gather(*tasks)
        if sequence_end and self.backend == 'pytorch':
            for session_id in session_ids:
                await self.end_session(session_id)
        return logits

    def get_ppl(self, input_ids: Union[List[int], List[List[int]]]) -> List[float]:
        """Get perplexity scores given a list of input tokens that have to be
        of the same length.

        Args:
            input_ids (Union[List[int], List[List[int]]]): the batch of
                input token ids

        Returns:
            List[float]: A list of perplexity scores.
        """
        assert isinstance(input_ids, List)
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
        assert all(len(_) > 1 for _ in input_ids)

        # TODO: a better way to determine `max_input_len`, at most allocate
        # 2G mem for logits with shape [bs, max_input_len, vocab_size]
        vocab_size = self.hf_cfg.vocab_size
        max_input_len = 2 * 1024**3 // (vocab_size * 4)
        sizes = [len(_) for _ in input_ids]
        result = []
        sorted_index_values = sorted(list(enumerate(sizes)), key=lambda x: x[1], reverse=True)
        sizes = [value for index, value in sorted_index_values]
        indices = [index for index, value in sorted_index_values]
        logger.info(f'sorted sizes: {sizes}')
        logger.info(f'sorted indices: {indices}')
        for (start, end) in self._batch_iterator(sizes, max_input_len):
            logger.info(f'start: {start}, end: {end}')
            if start == end:
                _input_ids = input_ids[indices[start]]
                res = self._get_long_text_ppl(input_ids=_input_ids, max_input_len=max_input_len)
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

            while i < len(sizes) and current_sum + sizes[start_index] <= max_value:
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
            loss = self._get_ppl(input_ids=[token_ids],
                                 max_input_len=len(token_ids),
                                 target_ids=[target_ids],
                                 steps=step,
                                 sequence_start=(i == 0),
                                 sequence_end=False)
            losses.extend(loss)
            target_counts.append(len(target_ids))
        losses = [loss * target_count for loss, target_count in zip(losses, target_counts)]
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
        assert (isinstance(input_ids, List) and all(isinstance(_, List) for _ in input_ids))
        assert steps is None or len(steps) == len(input_ids)
        assert target_ids is None or len(target_ids) == len(input_ids)

        lens = [len(_) for _ in input_ids]
        total_len = sum(lens)
        assert sum(lens) <= max_input_len

        logger.info(f'get_ppl: bs: {len(input_ids)}, lens: {lens}, '
                    f'total_len: {total_len}, steps: {steps}')
        torch.cuda.empty_cache()

        logits = self._run(coro=self._async_get_logits(
            input_ids=input_ids, steps=steps, sequence_start=sequence_start, sequence_end=sequence_end)).result()
        padding_token_id = -100
        if target_ids is None:
            target_ids = [x[1:] + [padding_token_id] for x in input_ids]
        else:
            target_ids = [
                target_ids[i] + [padding_token_id] if len(target_ids[i]) < len(input_ids[i]) else target_ids[i]
                for i in range(len(input_ids))
            ]
        target_ids = [torch.Tensor(torch.LongTensor(_target_ids)) for _target_ids in target_ids]

        result = []
        for _logits, _target_ids in zip(logits, target_ids):
            _logits = _logits.float()
            vocab_size = _logits.shape[-1]
            _target_ids = _target_ids.to(_logits.device)
            target_mask = _target_ids != padding_token_id
            # compute cross entropy loss
            flat_logits = _logits.contiguous().view(-1, vocab_size)
            flat_target_ids = _target_ids.contiguous().view(-1)
            flat_loss_matrix = torch.nn.functional.cross_entropy(flat_logits,
                                                                 flat_target_ids,
                                                                 reduction='none',
                                                                 ignore_index=padding_token_id)
            loss = flat_loss_matrix.sum()
            target_count = target_mask.sum()
            result.append(loss.item() / target_count.item())
        logger.info(f'ppl result: {result}')
        return result
