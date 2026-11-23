# Ascend test selection

The common `pytest tests` command runs the CPU contract tests in this directory
and excludes `hardware/` before importing its test modules. These contracts also
run in the existing CUDA CI because they exercise shared configuration, model
interfaces, and import isolation without executing NPU kernels. The kernel
import-order test additionally requires `torch_npu` and skips when it is absent.

```bash
python -m pytest -q tests/pytorch/ascend
```

Hardware validation is opt-in. Run it in a separate Ascend runner with matching
PyTorch, torch_npu, CANN/NNAL, and Triton Ascend dependencies, without dlinfer:

```bash
python tools/ascend/check_runtime.py --require-kernels
python -m pytest -q tests/pytorch/ascend/hardware --run-ascend-hardware
```

The opt-in suite requires a working NPU runtime. It includes real NPU attention,
RoPE, sparse attention, and cache-copy checks. The convolution ring tests use CPU
reference calculations but import the Ascend Triton module; they are included in
this environment-specific suite and do not prove NPU convolution correctness.

No Ascend GitHub Actions runner is configured by this change. An Ascend job can
run the commands above once the repository has an appropriate runner and image.
End-to-end model equivalence and performance experiments are separate from these
unit and operator tests.

## Test provenance

The six `test_ascend_*.py` modules were ported from dlinfer commit
`3b9096341a41e9d067104a39d914e47ca2d7ec50`, retaining their reference calculations
and assertions while updating imports and graph interfaces. The lifecycle and
sparse graph-metadata tests now run with CPU tensors and mocked runtime calls.

`test_native_integration.py` adds migration-specific contracts. Its small,
mixed-dtype cache-allocation case adapts the intent of dlinfer's former
`test_lmdeploy_cache_allocator.py` to LMDeploy's current allocation interface.
`hardware/test_native_cache_copy.py` and `test_kernel_import_order.py` were added
for the native cache implementation and the import-order regression.
