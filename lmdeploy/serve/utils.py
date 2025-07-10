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
                                max_input_len: int = None,
                                sequence_start: bool = True,
                                sequence_end: bool = True) -> List[torch.Tensor]:
        assert input_ids and all(isinstance(_, List) for _ in input_ids)
        assert steps is None or (len(steps) == len(input_ids))

        steps = steps or [0] * len(input_ids)
        max_input_len = max_input_len or max([len(x) for x in input_ids])

        if self.backend == 'turbomind':
            logits = await self._async_get_logits_by_turbomind(input_ids, steps, max_input_len)
        else:
            logits = await self._async_get_logits_by_pytorch(input_ids, steps, max_input_len, sequence_start,
                                                             sequence_end)
        return logits

    async def _async_get_logits_by_turbomind(self, input_ids, steps, max_input_len):
        assert len(input_ids) == len(steps)

        if any(s != 0 for s in steps):
            assert self.backend_config.enable_prefix_caching, 'please enable prefix caching'
        assert all(s % self.backend_config.cache_block_seq_len == 0 for s in steps)

        logits = [None] * len(input_ids)
        gen_config = GenerationConfig(max_new_tokens=1, output_logits='all', do_sample=False)

        async def _proc(i):
            session_id = next(self._session_id)
            async with self.model_inst(session_id=session_id) as inst:
                token_ids = input_ids[i][:steps[i] + max_input_len]
                input_len = len(token_ids)
                async with self.safe_run(inst,
                                         session_id=session_id,
                                         input_ids=token_ids,
                                         gen_config=gen_config,
                                         stream_output=False,
                                         step=steps[i]) as gen:
                    async for outputs in gen:
                        pass
                    logits[i] = outputs.logits[:input_len - steps[i], :]

        tasks = [_proc(i) for i in range(len(input_ids))]
        await asyncio.gather(*tasks)

        return logits

    async def _async_get_logits_by_pytorch(self,
                                           input_ids: List[List[int]],
                                           steps: List[int],
                                           max_input_len: int,
                                           sequence_start: bool = True,
                                           sequence_end: bool = True):
        logits = [None] * len(input_ids)

        async def _proc(i):
            session_id = next(self._session_id)
            async with self.model_inst(session_id=session_id) as inst:
                token_ids = input_ids[i][steps[i]:steps[i] + max_input_len]
                input_len = len(token_ids)
                # The reason to set `top_k=1` is that pt engine crashes at top_k sampling stage
                # when perform inference on a reward model.
                gen_config = GenerationConfig(max_new_tokens=0, output_logits='all', top_k=1)
                async with self.safe_run(inst,
                                         session_id=session_id,
                                         input_ids=token_ids,
                                         gen_config=gen_config,
                                         stream_output=False,
                                         sequence_start=sequence_start,
                                         sequence_end=sequence_end,
                                         step=steps[i]) as gen:
                    async for outputs in gen:
                        pass
                    logits[i] = outputs.logits[:input_len, :]

        session_ids = list(range(len(input_ids)))
        tasks = [_proc(i) for i in range(len(input_ids))]
        await asyncio.gather(*tasks)
        if sequence_end:
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

        max_input_len = self.backend_config.max_prefill_token_num
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
                steps = [0] * len(_input_ids)
                res = self._get_ppl(input_ids=_input_ids, steps=steps, max_input_len=max_input_len)
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
            loss, target_count = self._get_ppl(input_ids=[input_ids],
                                               steps=[i],
                                               max_input_len=max_input_len,
                                               sequence_start=(i == 0),
                                               sequence_end=False)
            losses.extend(loss)
            target_counts.extend(target_count)
        losses = [loss * target_count for loss, target_count in zip(losses, target_counts)]
        loss_sum = sum(losses)
        target_count = sum(target_counts)
        return loss_sum / target_count

    def _get_ppl(self, input_ids, steps, max_input_len, sequence_start: bool = True, sequence_end: bool = True):
        assert isinstance(steps, List) and len(steps) == len(input_ids)

        torch.cuda.empty_cache()

        logits = self._run(coro=self._async_get_logits(input_ids=input_ids,
                                                       steps=steps,
                                                       max_input_len=max_input_len,
                                                       sequence_start=sequence_start,
                                                       sequence_end=sequence_end)).result()
        padding_token_id = -100
        # shift token_ids by 1 to the left
        target_ids = [s[steps[i] + 1:steps[i] + 1 + max_input_len] for i, s in enumerate(input_ids)]
        target_ids = [t + [padding_token_id] if len(t) < max_input_len else t for t in target_ids]
        target_ids = [torch.Tensor(torch.LongTensor(t)) for t in target_ids]

        result = []
        target_counts = []
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
            target_counts.append(target_count)
        logger.info(f'ppl result: {result}')
        return result, target_counts
