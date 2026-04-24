# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
from enum import Enum

IMAGE_TOKEN = '<IMAGE_TOKEN>'


class Modality(Enum):
    IMAGE = 'image'
    VIDEO = 'video'
    AUDIO = 'audio'
    TIME_SERIES = 'time_series'

    def __eq__(self, other):
        if isinstance(other, Modality):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(self.value)


@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: str | list[str] | None = None
    video_token: str | list[str] | None = None
    audio_token: str | list[str] | None = None
    ts_token: str | list[str] | None = None

    image_token_id: int | None = None
    video_token_id: int | None = None
    audio_token_id: int | None = None
    ts_token_id: int | None = None

    def get_token_id_by_modality(self, modality: Modality) -> int | None:
        """Get token ID for a given modality."""
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
            Modality.TIME_SERIES: self.ts_token_id,
        }.get(modality)
