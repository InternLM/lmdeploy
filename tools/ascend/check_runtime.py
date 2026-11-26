"""Fail-fast A3 runtime check.

Run inside a clean environment without dlinfer.
"""

import argparse
import importlib
import importlib.abc
import importlib.util
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-kernels', action='store_true', help='Also check GDN/Triton dependencies')
    args = parser.parse_args()
    if importlib.util.find_spec('dlinfer') is not None:
        raise RuntimeError('Run validation in a clean environment without the dlinfer package.')

    class BlockDlinfer(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname.split('.')[0] == 'dlinfer':
                raise RuntimeError('Native Ascend path attempted to import ' + fullname)

    sys.meta_path.insert(0, BlockDlinfer())
    import torch
    import torch_npu

    from lmdeploy.pytorch.backends import get_backend
    from lmdeploy.pytorch.backends.ascend.runtime import initialize

    if not torch.npu.is_available():
        raise RuntimeError('A working Ascend NPU is required; hardware checks cannot be skipped.')
    initialize()
    soc = torch.npu.get_soc_version()
    if not 250 <= soc <= 255:
        raise RuntimeError(f"Expected A3 SoC 250..255, got {soc}.")
    required = (
        'npu_fused_infer_attention_score',
        'npu_fused_infer_attention_score_v2',
        'npu_interleave_rope',
        'npu_lightning_indexer',
        'npu_sparse_flash_attention',
    )
    for name in required:
        if not callable(getattr(torch_npu, name, None)):
            raise RuntimeError(f"torch_npu.{name} is unavailable")
    for name in ('npu_grouped_matmul', 'npu_moe_distribute_dispatch_v2', 'npu_moe_distribute_combine_v2'):
        getattr(torch.ops.npu, name)
    getattr(torch.ops.atb, '_npu_reshape_and_cache')
    versions = dict(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        soc=soc,
        backend=get_backend('ascend').__module__,
        devices=torch.npu.device_count(),
    )
    if args.require_kernels:
        for name in (
            'causal_conv1d',
            'rms_norm_gated',
            'fla.chunk',
            'fla.fused_recurrent',
            'fla.sigmoid_gating',
            'reject_sample',
        ):
            importlib.import_module('lmdeploy.pytorch.kernels.ascend.triton_ops.' + name)
        import triton

        from lmdeploy.pytorch.kernels.ascend.triton_ops.reject_sample import _TRITON_AVAILABLE

        if not _TRITON_AVAILABLE:
            raise RuntimeError('Triton Ascend get_element extension is required for Triton rejection sampling.')
        versions['triton'] = triton.__version__
    print(json.dumps(versions, indent=2))


if __name__ == '__main__':
    main()
