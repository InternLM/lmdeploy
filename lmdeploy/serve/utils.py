# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class LogitsMixin:
    """Helper class to calculate logits and ppl."""

    def get_logits(self, input_ids: Union[List[int], List[List[int]]]):
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

        logits = []
        generator = self.engine.create_instance()
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

        logits = self.get_logits(input_ids)
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
