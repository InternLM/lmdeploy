# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.serve.openai.protocol import DeltaMessage


def first_stream_delta(
    deltas: list[tuple[DeltaMessage, bool]],
) -> tuple[DeltaMessage | None, bool]:
    """Return the first delta from ``ResponseParser.stream_chunk``.

    Returns ``(None, False)`` when ``deltas`` is empty.
    """
    if not deltas:
        return None, False
    return deltas[0]
