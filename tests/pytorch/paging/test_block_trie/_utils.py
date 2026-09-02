import numpy as np
import torch

from lmdeploy.multimodal.constants import Modality
from lmdeploy.pytorch.multimodal.data_type import MultiModalData


class BlockTrieTestMixin:

    def _image_multimodals(self,
                           start: int,
                           end: int,
                           value: float,
                           image_token_id: int = 99,
                           content_hash: str | None = None):
        data = torch.full((2, 2), value, dtype=torch.float32)
        return dict(image=[MultiModalData(data=data,
                                          start=start,
                                          end=end,
                                          meta=dict(image_token_id=image_token_id),
                                          content_hash=content_hash)])

    def _modal_data(self, start: int, end: int, value: float, modality: Modality):
        data = torch.full((2, 2), value, dtype=torch.float32)
        return MultiModalData(data=data,
                              start=start,
                              end=end,
                              modality=modality,
                              meta=dict(token_id=int(value)))

    def _multi_image_multimodals(self, spans: list[tuple[int, int, float]]):
        return dict(image=[
            MultiModalData(data=torch.full((2, 2), value, dtype=torch.float32),
                           start=start,
                           end=end,
                           modality=Modality.IMAGE,
                           meta=dict(image_token_id=99)) for start, end, value in spans
        ])

    def _routed_experts(self, num_tokens: int, offset: int = 0):
        values = np.arange(offset, offset + num_tokens * 2, dtype=np.uint16)
        return values.reshape(num_tokens, 2, 1)

    def _add_published_ssm_checkpoint(self, scheduler, token_ids):
        seq = scheduler.add_session(len(scheduler.sessions)).add_sequence(token_ids)
        scheduler.block_manager.allocate(seq)
        scheduler.block_trie.allocate(seq)
        state_idx = scheduler.block_trie.state_checkpoints.reserve_save(seq)
        assert state_idx >= 0
        assert scheduler.block_trie.state_checkpoints.publish_save(seq)
        return seq, seq.prefix_cache.trie_cursor, state_idx
