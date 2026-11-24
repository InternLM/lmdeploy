# Ascend GDN prefill kernels

These MIT-licensed kernels are imported from
[Ascend/triton-ascend-kernels](https://gitcode.com/Ascend/triton-ascend-kernels),
commit `8e70a44afa537fcbcdc6d59e6d2196a38f1e7b6f`, using the frozen A3 validation
snapshot. Copyright notices and the complete [MIT license](LICENSE) are retained.

The imported slice contains `attention/fla/*.py`, `norm/l2norm.py`, and the device
query helpers from `utils.py` (here named `device.py`). Only module locations and
repository formatting change. The snapshot already contained two compatibility
edits used by the pre-migration and native validation environments:

- `cumsum.py` stores in the destination tensor's dtype.
- `solve_tril.py` imports slice helpers from `triton.language.extra.cann.extension`.

The recurrence, tiling, launch options, and arithmetic remain those of that
validated snapshot. This removes the build-time checkout of a floating GitCode
branch and its conflicting legacy Torch/Triton dependency pins. Triton-Ascend
and the CANN/TorchNPU runtime remain external dependencies.

Provenance hashes, comparison tests and the CI failure analysis are recorded in
the [A3 validation reports](https://github.com/silencelamb/lmdeploy/tree/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation).
