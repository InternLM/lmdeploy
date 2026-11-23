# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from . import ops as ext_ops


def lightning_indexer(
    query: Tensor,
    key: Tensor,
    weights: Tensor,
    actual_seq_lengths_query: Tensor,
    actual_seq_lengths_key: Tensor,
    block_table: Tensor,
    sparse_count: int,
) -> Tensor:
    """Run the LMDeploy Lightning Indexer kernel."""
    return ext_ops.lightning_indexer(
        query,
        key,
        weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        sparse_count=sparse_count,
    )
