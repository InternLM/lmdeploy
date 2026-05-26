# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.serve.openai.protocol import DeltaMessage


def first_stream_delta(
    deltas: list[tuple[DeltaMessage | None, bool]],
) -> tuple[DeltaMessage | None, bool]:
    """Return the first delta from ``ResponseParser.stream_chunk``."""
    if not deltas:
        return None, False
    return deltas[0]
