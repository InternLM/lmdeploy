# Copyright (c) OpenMMLab. All rights reserved.
"""A same-named child module must never become a backend's kernel callable."""

import subprocess
import sys

import pytest

pytest.importorskip('torch_npu', reason='Real Ascend kernel imports require the vendor runtime')


@pytest.mark.parametrize('preload', [False, True])
def test_kernel_callables_survive_module_import_order(preload):
    code = f'''
import importlib
bindings = {{
    'moe': ['fused_moe', 'fused_moe_w8a8', 'moe_gating_topk_softmax'],
    'norm': ['rms_norm'],
    'linear': ['linear'],
    'apply_rotary_emb': ['apply_rotary_pos_emb', 'apply_rotary_pos_emb_interleaved'],
    'nsa': ['fill_kv_cache', 'lightning_indexer'],
    'awq_modules': ['awq_linear'],
}}
if {preload!r}:
    for name in ['fused_moe', 'rms_norm', 'linear', 'apply_rotary_pos_emb',
                 'fill_kv_cache', 'lightning_indexer', 'moe_gating_topk_softmax']:
        importlib.import_module('lmdeploy.pytorch.kernels.ascend.' + name)
for module, names in bindings.items():
    backend = importlib.import_module('lmdeploy.pytorch.backends.ascend.' + module)
    for name in names:
        assert callable(getattr(backend, name)), (module, name, getattr(backend, name))
'''
    subprocess.run([sys.executable, '-c', code], check=True)
