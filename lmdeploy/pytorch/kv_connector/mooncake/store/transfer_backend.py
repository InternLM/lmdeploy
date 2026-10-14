# Copyright (c) OpenMMLab. All rights reserved.
"""Device-aware transfer backends for the Mooncake Store connector.

The Mooncake Store Python API (``batch_is_exist`` / ``batch_put_from_multi_buffers``
/ ``batch_get_into_multi_buffers``) is device-agnostic: it operates on raw buffer
addresses that have been registered with the store. What differs between CUDA and
Ascend is only *how* those buffers are registered and *how* the store is set up:

* CUDA (H200): GPU HBM is exposed to the RDMA NIC through GPUDirect RDMA. Buffers
  are registered directly with the store (``store.register_buffer``) and the store
  owns its internal TransferEngine.
* Ascend (A3): NPU HBM is registered with an external ``TransferEngine``
  (``engine.register_memory``) which is then injected into the store through
  ``setup(..., engine=...)``. The store reads and writes device HBM addresses
  directly, relying on Ascend's unified-memory RDMA path. No explicit D2H/H2D copy
  is needed on this path.

The registered addresses and the per-block scatter layout are identical across
backends, so the send/receive transfer threads (``worker_threads.py``) require no
device-specific changes. This mirrors vllm-ascend's ``ascend_store`` mooncake
backend, which transfers directly to/from NPU HBM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from lmdeploy.utils import get_logger

from .data import MooncakeStoreConfig, MooncakeStoreRegistration

if TYPE_CHECKING:
    pass

logger = get_logger('lmdeploy')


class TransferBackend:
    """Device-specific Mooncake Store setup, registration, and stream fencing.

    Subclasses encapsulate the three points where CUDA and Ascend differ:
    ``setup_store`` (store initialization arguments), ``register_buffer`` (which
    object registers a KV-cache region) and ``make_ready_event`` (the compute-stream
    fence event). The transfer data path itself is shared by the worker threads.
    """

    device_type: str = 'cuda'

    def setup_store(self, store: Any, config: MooncakeStoreConfig,
                    local_hostname: str) -> int:
        """Initialize the store and return its status code (0 on success)."""
        raise NotImplementedError

    def register_buffer(self, store: Any,
                        registration: MooncakeStoreRegistration) -> int:
        """Register one KV-cache region and return the status code (0 on success)."""
        raise NotImplementedError

    def make_ready_event(self) -> Any:
        """Return a fence event recorded on the compute stream."""
        raise NotImplementedError

    def close(self) -> None:
        """Release backend-owned resources (e.g. the Ascend TransferEngine)."""
        return None


class CudaTransferBackend(TransferBackend):
    """Direct GPU-HBM registration, matching the original H200 implementation.

    Buffers are registered with the store and the store owns its TransferEngine,
    exactly as before this abstraction was introduced. Kept as a distinct class so
    the CUDA path is explicit and never accidentally changes.
    """

    device_type = 'cuda'

    def setup_store(self, store: Any, config: MooncakeStoreConfig,
                    local_hostname: str) -> int:
        return store.setup(
            local_hostname,
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.device_name,
            config.master_server_address,
        )

    def register_buffer(self, store: Any,
                        registration: MooncakeStoreRegistration) -> int:
        return store.register_buffer(registration.address, registration.size)

    def make_ready_event(self) -> Any:
        return torch.cuda.Event()


class AscendTransferBackend(TransferBackend):
    """NPU-HBM registration via an external TransferEngine (Ascend A3).

    A process-local ``TransferEngine`` is initialized once, its engine is injected
    into the store through ``setup(..., engine=...)``, and each KV-cache HBM region
    is registered with the engine via ``register_memory``. The store then transfers
    directly to and from device HBM addresses, exactly as on CUDA. This is the
    proven store-path used by vllm-ascend; it needs no host staging or D2H/H2D copy.
    """

    device_type = 'npu'

    def __init__(self) -> None:
        # Fail fast if torch_npu is unavailable: this backend is only built on
        # Ascend, where torch_npu is a required dependency.
        import torch_npu  # noqa: F401  registers torch.npu
        self._transfer_engine: Any | None = None

    def _get_transfer_engine(self, local_hostname: str, device_name: str) -> Any:
        if self._transfer_engine is not None:
            return self._transfer_engine
        from mooncake.engine import TransferEngine
        engine = TransferEngine()
        # The engine uses the Ascend transport protocol and a P2P handshake; the
        # store itself is configured separately with the user-supplied metadata
        # server. Mirrors vllm-ascend GlobalTE.get_transfer_engine.
        ret = engine.initialize(local_hostname, 'P2PHANDSHAKE', 'ascend',
                                device_name or '')
        if ret != 0:
            raise RuntimeError(
                f'Ascend TransferEngine.initialize failed with return code {ret}')
        self._transfer_engine = engine
        return engine

    def setup_store(self, store: Any, config: MooncakeStoreConfig,
                    local_hostname: str) -> int:
        engine = self._get_transfer_engine(local_hostname, config.device_name)
        local_seg = f'{local_hostname}:{engine.get_rpc_port()}'
        return store.setup(
            local_hostname=local_seg,
            metadata_server=config.metadata_server,
            global_segment_size=config.global_segment_size,
            local_buffer_size=config.local_buffer_size,
            protocol=config.protocol,
            rdma_devices=config.device_name,
            master_server_addr=config.master_server_address,
            engine=engine.get_engine(),
        )

    def register_buffer(self, store: Any,
                        registration: MooncakeStoreRegistration) -> int:
        engine = self._transfer_engine
        if engine is None:
            raise RuntimeError(
                'Ascend TransferEngine must be initialized by setup_store before '
                'register_buffer is called')
        return engine.register_memory(registration.address, registration.size)

    def make_ready_event(self) -> Any:
        return torch.npu.Event()

    def close(self) -> None:
        # The TransferEngine has no explicit teardown in the mooncake API; dropping
        # the reference lets it be reclaimed when the worker shuts down.
        self._transfer_engine = None


def build_transfer_backend(device_backend: str) -> TransferBackend:
    """Construct the transfer backend for a resolved device name."""
    if device_backend == 'cuda':
        return CudaTransferBackend()
    if device_backend == 'ascend':
        return AscendTransferBackend()
    raise ValueError(f'Unsupported device_backend: {device_backend!r}')
