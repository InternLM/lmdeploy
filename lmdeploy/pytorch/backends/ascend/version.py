# Copyright (c) 2026, DeepLink. All rights reserved.


import torch
import torch_npu
from packaging.version import InvalidVersion, Version

MIN_TORCH_VERSION = Version('2.8.0')
MIN_TORCH_NPU_VERSION = Version('2.8.0.post1')


def _parse_version(raw_version: str, package_name: str) -> Version:
    try:
        return Version(raw_version)
    except InvalidVersion as exc:
        raise RuntimeError(
            f"Invalid {package_name} version {raw_version!r}; LMDeploy Ascend requires "
            f"torch>={MIN_TORCH_VERSION} and torch-npu>={MIN_TORCH_NPU_VERSION}."
        ) from exc


def ensure_ascend_runtime(
    torch_version: str | None = None,
    torch_npu_version: str | None = None,
    check_graph_api: bool = True,
) -> tuple[Version, Version]:
    """Validate the only supported Ascend graph-update runtime path."""
    parsed_torch = _parse_version(torch.__version__ if torch_version is None else torch_version, 'torch')
    parsed_torch_npu = _parse_version(
        torch_npu.__version__ if torch_npu_version is None else torch_npu_version,
        'torch-npu',
    )

    if parsed_torch < MIN_TORCH_VERSION:
        raise RuntimeError(
            f"Unsupported torch version {parsed_torch}; LMDeploy Ascend requires "
            f"torch>={MIN_TORCH_VERSION}. The legacy ATB graph-task update path "
            'has been removed.'
        )
    if parsed_torch_npu < MIN_TORCH_NPU_VERSION:
        raise RuntimeError(
            f"Unsupported torch-npu version {parsed_torch_npu}; LMDeploy Ascend "
            f"requires torch-npu>={MIN_TORCH_NPU_VERSION}. The legacy ATB "
            'graph-task update path has been removed.'
        )
    if parsed_torch.release[:2] != parsed_torch_npu.release[:2]:
        raise RuntimeError(
            'torch and torch-npu must use the same major.minor release for the '
            f"Ascend backend, but got torch {parsed_torch} and torch-npu "
            f"{parsed_torch_npu}."
        )

    graph_cls = getattr(getattr(torch, 'npu', None), 'NPUGraph', None)
    if check_graph_api and not callable(getattr(graph_cls, 'update', None)):
        raise RuntimeError(
            'torch.npu.NPUGraph.update is unavailable. LMDeploy Ascend graph mode '
            f"requires torch>={MIN_TORCH_VERSION} and "
            f"torch-npu>={MIN_TORCH_NPU_VERSION}."
        )

    return parsed_torch, parsed_torch_npu
