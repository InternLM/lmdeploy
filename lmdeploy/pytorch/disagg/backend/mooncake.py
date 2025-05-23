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
        self.nics = get_rdma_nics()
        # self.device_name = nics[self.rank % len(nics)]
        self.buffer_map: Dict[str, Dict] = {}
        self.endpoints: Dict[int, Dict] = {}

    def p2p_initialize(self, init_request: DistServeInitRequest):
        """initialize p2p connection"""
        
        # TODO: Check metadata_server
        # e.g. "etcd://192.168.0.137:2379"
        metadata_server = "P2PHANDSHAKE" 
            
        # Default protocol (Currently only RDMA is supported)
        protocol = "rdma"  

        # Get the device name from request
        device_name = self.nics[init_request.rank % len(self.nics)]
        
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
        
        # Generate a session ID
        mooncake_session_id = f"{self.hostname}:{self.engine.get_rpc_port()}"

        # Setup mapping for the registered memory region
        self.buffer_map[register_mr_request.mr_key] = {
            'addr': buffer_addr,
            'length': buffer_length,
            'offset': register_mr_request.offset,
            'session_id': mooncake_session_id
        }
        
        logger.info(f"Registered memory region with key {register_mr_request.mr_key}, addr: {buffer_addr}, length: {buffer_length}")

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        """Get endpoint information"""      
        endpoint_info = self.endpoints.get(remote_engine_id)
        
        return json.dumps(endpoint_info)

    def p2p_connect(self, connect_request: DistServeConnectionRequest):
        """ Connect to a remote engine """
        remote_endpoint_info = json.loads(connect_request.remote_endpoint_info)

        # Store endpoint information
        gid = remote_endpoint_info.get('gid')
        lid = remote_endpoint_info.get('lid')
        if not gid:
            logger.info(f"No GID found in remote endpoint info for remote engine {connect_request.remote_engine_id}")
        if not lid:
            logger.info(f"No LID found in remote endpoint info for remote engine {connect_request.remote_engine_id}")
        
        qp_num = self._generate_qp_num()
        logger.info(f"Using QP number: {qp_num} for remote engine {connect_request.remote_engine_id}")

        endpoint_info = {
            'gid': gid,
            'lid': lid,
            'qp_num': qp_num,
        }

        self.endpoints[connect_request.remote_engine_id] = endpoint_info

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
            mooncake_session_id = buffer_info['session_id']
            
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
    
# TODO:
# 1. Implement the connect method to establish a connection with the remote engine.
# 2. Read the endpoint_info of dlslime and modify current endpoint_info to be compatible with it.
# 3. Check the correctness of the migration address.
# 3. Test the correctness of the code.