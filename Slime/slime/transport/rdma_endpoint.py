import asyncio
from typing import Dict, Any

from slime import _slime_c

class RDMAEndpoint:
    """Manages RDMA endpoint lifecycle including resource allocation and data operations.
    
    An RDMA endpoint represents a communication entity with:
    - Memory Region (MR) registration
    - Peer connection establishment
    - Queue Pair (QP) management
    - Completion Queue (CQ) handling
    """

    def __init__(
        self,
        device_name: str,
        ib_port: int = 1,
        link_type: str = "Ethernet",
    ):
        """Initialize an RDMA endpoint bound to specific hardware resources.
        
        Args:
            device_name: RDMA NIC device name (e.g. 'mlx5_0')
            ib_port: InfiniBand physical port number (1-based indexing)
            transport_type: Underlying transport ('Ethernet' or 'InfiniBand')
        """
        self._ctx = _slime_c.rdma_context()
        self.initialize_endpoint(device_name, ib_port, link_type)

    @property
    def local_endpoint_info(self) -> Dict[str, Any]:
        """Retrieve local endpoint parameters for peer connection setup.
        
        Returns:
            Dictionary containing:
            - 'gid': Global Identifier (IPv6 format for RoCE)
            - 'qp_num': Queue Pair number
            - 'lid': Local ID (InfiniBand only)
        """
        return self._ctx.local_info()

    def initialize_endpoint(
        self,
        device_name: str,
        ib_port: int,
        transport_type: str,
    ) -> int:
        """Configure the endpoint with hardware resources.
        
        Returns:
            0 on success, non-zero error code matching IBV_ERROR_* codes
        """
        return self._ctx.init_rdma_context(device_name, ib_port, transport_type)

    def connect_to(
        self,
        remote_endpoint_info: Dict[str, Any]
    ) -> None:
        """Establish RC (Reliable Connection) to a remote endpoint.
        
        Args:
            remote_endpoint_info: Dictionary from remote's local_endpoint_info()
        """
        self._ctx.connect(remote_endpoint_info)
        self._ctx.launch_cq_future()  # Start background CQ polling

    def stop(self):
        """
        Safely stops the endpoint by terminating
        all background activities and releasing resources.
        """
        self._ctx.stop_cq_future()

    def register_memory_region(
        self,
        mr_identifier: str,
        virtual_address: int,
        length_bytes: int,
    ) -> None:
        """Register a Memory Region (MR) for RDMA operations.
        
        Args:
            mr_identifier: Unique key to reference this MR
            virtual_address: Starting VA of the memory block
            length_bytes: Size of the region in bytes
        """
        self._ctx.register_memory_region(mr_identifier, virtual_address, length_bytes)
    
    def register_remote_memory_region(
        self,
        remote_mr_info: str
    ) -> None:
        """Register a Remote Memory Region (MR) for RDMA operations.
        
        Args:
            remote_mr_info:
                - key: mr_key
                - value: mr_info
        """
        self._ctx.register_remote_memory_region(remote_mr_info)
    
    async def send_async(
        self, mr_key, offset, length
    ) -> int:
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def _completion_handler(status: int):
            loop.call_soon_threadsafe(future.set_result, status)

        self._ctx.send_async(
            mr_key,
            offset,
            length,
            _completion_handler
        )

        return await future
    
    async def recv_async(
        self, mr_key, offset, length
    ) -> int:
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def _completion_handler(status: int):
            loop.call_soon_threadsafe(future.set_result, status)

        self._ctx.recv_async(
            mr_key,
            offset,
            length,
            _completion_handler
        )

        return await future

    async def read_batch_async(
        self,
        mr_key: str,
        target_offset: int,
        source_offset: int,
        length: int,
    ) -> int:
        """Perform batched read from remote MR to local buffer.
        
        Args:
            remote_mr_key: Target MR identifier registered at remote
            remote_offset: Offset in remote MR (bytes)
            local_buffer_addr: Local destination VA
            read_size: Data size in bytes
            
        Returns:
            ibv_wc_status code (0 = IBV_WC_SUCCESS)
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def _completion_handler(status: int):
            loop.call_soon_threadsafe(future.set_result, status)

        self._ctx.batch_r_rdma_async(
            mr_key,
            target_offset,
            source_offset,
            length,
            _completion_handler
        )

        return await future

    async def read_async(
        self,
        mr_key: str,
        target_offset: int,
        source_offset: int,
        length: int,
    ) -> int:
        """Read data from remote memory region.
        
        Args:
            remote_mr_key: Target MR identifier registered at remote
            remote_offset: Offset in remote MR (bytes)
            local_buffer_addr: Local destination VA
            read_size: Data size in bytes
            
        Returns:
            ibv_wc_status code (0 = IBV_WC_SUCCESS)
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def _completion_handler(status: int):
            loop.call_soon_threadsafe(future.set_result, status)

        self._ctx.r_rdma_async(
            mr_key,
            target_offset,
            source_offset,
            length,
            _completion_handler
        )

        return await future

