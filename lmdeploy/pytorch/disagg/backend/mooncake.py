# Copyright (c) OpenMMLab. All rights reserved.

import json
import os
import random
import re
import socket
import subprocess
from typing import Dict, Optional, Tuple

from lmdeploy.utils import get_logger
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.config import MigrationBackend, MigrationProtocol
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment
from lmdeploy.pytorch.disagg.request import DistServeConnectionRequest, DistServeInitRequest

logger = get_logger('lmdeploy')

def generate_qp_num() -> int:
    """Randomly generate a Queue Pair number"""
    return random.randint(1, 2**24-1)

def get_rdma_nics():
   """
   Get all available RDMA network interface cards on the current machine
   
   Returns:
       list: List of RDMA NICs, e.g. ['erdma_0', 'erdma_1']
   """
   rdma_nics = []
   
   # Try to get from /sys/class/infiniband/ directory
   if os.path.exists('/sys/class/infiniband/'):
       rdma_nics = os.listdir('/sys/class/infiniband/')
   
   # If the above method yields no results, try using ibv_devices command
   if not rdma_nics:
       try:
           result = subprocess.run(['ibv_devices'], stdout=subprocess.PIPE, text=True)
           if result.returncode == 0:
               # Parse ibv_devices output
               # Sample output:
               # device                 node GUID
               # ------              ----------------
               # erdma_0             02163efffe3fc264
               # erdma_1             02163efffe3fc317
               lines = result.stdout.strip().split('\n')
               for line in lines[2:]:  # Skip header lines
                   if line.strip():
                       device_name = line.split()[0].strip()
                       rdma_nics.append(device_name)
       except Exception as e:
           print(f"Error executing ibv_devices command: {e}")
   
   # Last try, using ibstat command
   if not rdma_nics:
       try:
           result = subprocess.run(['ibstat'], stdout=subprocess.PIPE, text=True)
           if result.returncode == 0:
               # Parse ibstat output
               # Sample output:
               # CA 'erdma_0'
               #     CA type: ...
               # ...
               # CA 'erdma_1'
               #     CA type: ...
               ca_pattern = re.compile(r"CA '([^']+)'")
               matches = ca_pattern.findall(result.stdout)
               if matches:
                   rdma_nics = matches
       except Exception as e:
           print(f"Error executing ibstat command: {e}")
   
   return rdma_nics

def get_local_ip_by_remote() -> str:
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        raise ValueError("Can not get local ip")

@MIGRATION_BACKENDS.register_module(MigrationBackend.Mooncake.name)
class MooncakeBackend(MigrationBackendImpl):

    def __init__(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "to run LMDeploy with MooncakeBackend."
            ) from e
        
        self.engine = TransferEngine()
        self.hostname = get_local_ip_by_remote()
        self.ibv_device = get_rdma_nics()
        # self.device_name = nics[self.rank % len(nics)]
        self.buffer_map: Dict[str, Dict] = {}
        self.remote_urls: Dict[str, str] = {}  # Maps remote engine ID to its URL, e.g. 192.168.0.147:16442
        self.endpoints: Dict[int, str] = {}  # Store all endpoints config

    def p2p_initialize(self, init_request: DistServeInitRequest):
        """initialize p2p connection"""
        
        # TODO: Support more types of metadata_server
        # e.g. "etcd://192.168.0.137:2379"
        metadata_server = "P2PHANDSHAKE" 
            
        # Default protocol (Currently only RDMA is supported)
        protocol = "rdma"  

        # Get the device name from request
        device_name = self.ibv_device[init_request.rank % len(self.ibv_device)]
        
        # Initialize the engine
        result = self.engine.initialize(self.hostname, metadata_server, protocol, device_name)
        if result != 0:
            raise RuntimeError(f"Failed to initialize Mooncake engine: {result}")
        
        logger.info(f"Mooncake engine initialized with hostname {self.hostname}, RPC port: {self.engine.get_rpc_port()}")

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        """ Register memory region """    
        # Transmit buffer address to int
        buffer_addr = register_mr_request.addr
        buffer_length = register_mr_request.length
        
        # Register memory region with the engine
        result = self.engine.register_memory(buffer_addr, buffer_length)
        if result != 0:
            raise RuntimeError(f"Failed to register memory region: {result}")
        
        self.buffer_map[register_mr_request.mr_key] = {
            'addr': buffer_addr,
            'length': buffer_length,
            'offset': register_mr_request.offset
        }
        
        logger.info(f"Registered memory region with key {register_mr_request.mr_key}, addr: {buffer_addr}, length: {buffer_length}")

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        """Get endpoint information"""      
        # FIXME: Need to return local endpoint information
        endpoint_info = self.endpoints[remote_engine_id] if remote_engine_id in self.endpoints else None
        
        return json.dumps(endpoint_info)

    def p2p_connect(self, connect_request: DistServeConnectionRequest):
        """ Connect to a remote engine """
        self.endpoints[connect_request.remote_engine_id] = connect_request.remote_endpoint_info

        # Convert RoCEv2 interface ID to IP address
        def extract_rocev2_ip(interface_id: int) -> str:
            ip_hex = (interface_id >> 32) & 0xFFFFFFFF
            ip_bytes = [(ip_hex >> (8 * i)) & 0xFF for i in reversed(range(4))]
            return ".".join(map(str, ip_bytes))

        # Parse the remote endpoint information from the request
        remote_endpoint_info = json.loads(connect_request.remote_endpoint_info)

        # Extract the remote endpoint information
        rdma_info = remote_endpoint_info["rdma_info"][0]  # Default to the first RDMA info
        interface_id = rdma_info["gid"]["interface_id"]
        ip = extract_rocev2_ip(interface_id)
        qpn = rdma_info["qpn"]

        port = f'{16000 + int(qpn)}'  # Example port calculation, adjust as needed
        endpoint_url = f"{ip}:{qpn}"  # mooncake_cake_id

        self.remote_urls[connect_request.remote_engine_id] = endpoint_url

        logger.info(f"Connecting to remote engine {connect_request.remote_engine_id} at endpoint {endpoint_url}")

    def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        """Migrate data between engines"""
        if assignment.remote_engine_id not in self.endpoints:
            raise RuntimeError(f"No connection to remote engine {assignment.remote_engine_id}")
        
        for task in assignment.batch:
            if task.mr_key not in self.buffer_map:
                raise RuntimeError(f"Memory region with key {task.mr_key} not registered")
            
            # Get local buffer information
            # TODO: check if the buffer addr and session id is correct
            buffer_info = self.buffer_map[task.mr_key]
            local_addr = buffer_info['addr'] + buffer_info['offset'] + task.source_offset
            mooncake_session_id = self.remote_urls[assignment.remote_engine_id]
            
            # Currently, only sync transfer is supported
            result = self.engine.transfer_sync(
                mooncake_session_id,
                local_addr,
                task.target_offset,
                task.length,
            )
            if result != 0:
                raise RuntimeError(f"Failed to perform sync transfer: {result}")

    def store(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    def load(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError
