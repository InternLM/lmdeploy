# Copyright (c) OpenMMLab. All rights reserved.

import json
import os
import random
import struct
import re
import socket
import subprocess
from typing import Dict, Optional, Tuple, List

from lmdeploy.utils import get_logger
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.config import DistServeEngineConfig, MigrationBackend, MigrationProtocol
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment
from lmdeploy.pytorch.disagg.request import DistServeConnectionRequest, DistServeInitRequest

logger = get_logger('lmdeploy')

def get_all_rdma_info() -> Dict:
    """
    Get comprehensive RDMA information including device list and detailed device info
    
    Returns:
        dict: Dictionary containing:
            - 'devices': List of RDMA device names (e.g. ['erdma_0', 'erdma_1'])
            - 'device_details': Dict with device name as key and device info as value
    """
    rdma_info = {
        'devices': [],
        'device_details': {}
    }

    try:
        # First get list of all RDMA devices
        result = subprocess.run(['ibv_devices'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.warning(f"Failed to get RDMA devices: {result.stderr}")
            return rdma_info
            
        # Parse ibv_devices output to get device names
        # Sample output:
        # device                 node GUID
        # ------              ----------------
        # erdma_0             02163efffe3fc264
        # erdma_1             02163efffe3fc317
        lines = result.stdout.strip().split('\n')
        for line in lines[2:]:  # Skip header lines
            if line.strip():
                device_name = line.split()[0].strip()
                rdma_info['devices'].append(device_name)
        
        # Get detailed information for each device
        for device_name in rdma_info['devices']:
            device_info = get_rdma_device_detailed_info(device_name)
            if device_info:
                rdma_info['device_details'][device_name] = device_info
                
    except Exception as e:
        logger.error(f"Error getting RDMA information: {e}")
   
    return rdma_info

def get_rdma_device_detailed_info(device_name: str) -> Optional[Dict]:
    """Get detailed RDMA device information from the system"""
    try:
        # Get device attributes using ibv_devinfo
        result = subprocess.run(['ibv_devinfo', '-d', device_name], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.warning(f"Failed to get device info for {device_name}: {result.stderr}")
            return None
            
        device_info = {'port_info': {}}  # Initialize port_info
        lines = result.stdout.strip().split('\n')
        
        current_port = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'node_guid:' in line:
                device_info['node_guid'] = line.split(':', 1)[1].strip()
            elif 'sys_image_guid:' in line:
                device_info['sys_image_guid'] = line.split(':', 1)[1].strip()
            elif line.startswith('port:'):
                # Parse port line like "port:	1	state: PORT_ACTIVE (4)"
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        port_num = int(parts[1])
                        current_port = port_num
                        device_info['port_info'][port_num] = {}
                        
                        # Extract state if present in the same line
                        if 'state:' in line:
                            state_part = line.split('state:', 1)[1].strip()
                            # Extract state name, e.g., "PORT_ACTIVE (4)" -> "PORT_ACTIVE"
                            state = state_part.split('(')[0].strip()
                            device_info['port_info'][port_num]['state'] = state
                    except (ValueError, IndexError):
                        logger.warning(f"Failed to parse port line: {line}")
            elif current_port is not None and ':' in line:
                # Parse port attributes
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'state':
                    # Extract state name, e.g., "PORT_ACTIVE (4)" -> "PORT_ACTIVE"
                    state = value.split('(')[0].strip()
                    device_info['port_info'][current_port]['state'] = state
                elif key == 'max_mtu':
                    device_info['port_info'][current_port]['max_mtu'] = value.split('(')[0].strip()
                elif key == 'active_mtu':
                    device_info['port_info'][current_port]['active_mtu'] = value.split('(')[0].strip()
                elif key == 'sm_lid':
                    try:
                        device_info['port_info'][current_port]['lid'] = int(value)
                    except ValueError:
                        device_info['port_info'][current_port]['lid'] = 0
            elif 'GID[' in line and current_port is not None:
                # Parse GID lines
                if 'gids' not in device_info['port_info'][current_port]:
                    device_info['port_info'][current_port]['gids'] = []
                
                gid_match = re.search(r'GID\[\s*(\d+)\]:\s*([a-fA-F0-9:]+)', line)
                if gid_match:
                    gid_index = int(gid_match.group(1))
                    gid_value = gid_match.group(2)
                    device_info['port_info'][current_port]['gids'].append({
                        'index': gid_index,
                        'gid': gid_value
                    })
        
        # If no ports were found through parsing, try to get port info separately
        if not device_info['port_info']:
            # Default to port 1 and query it directly
            device_info['port_info'][1] = {}
            port_result = subprocess.run(['ibv_devinfo', '-d', device_name, '-i', '1'], 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if port_result.returncode == 0:
                port_lines = port_result.stdout.strip().split('\n')
                port_info = {'gids': []}
                
                for line in port_lines:
                    line = line.strip()
                    if 'state:' in line:
                        state = line.split(':', 1)[1].strip().split('(')[0].strip()
                        port_info['state'] = state
                    elif 'max_mtu:' in line:
                        mtu = line.split(':', 1)[1].strip().split('(')[0].strip()
                        port_info['max_mtu'] = mtu
                    elif 'active_mtu:' in line:
                        mtu = line.split(':', 1)[1].strip().split('(')[0].strip()
                        port_info['active_mtu'] = mtu
                    elif 'sm_lid:' in line:
                        try:
                            lid = int(line.split(':', 1)[1].strip())
                            port_info['lid'] = lid
                        except ValueError:
                            port_info['lid'] = 0
                    elif 'GID[' in line:
                        gid_match = re.search(r'GID\[\s*(\d+)\]:\s*([a-fA-F0-9:]+)', line)
                        if gid_match:
                            gid_index = int(gid_match.group(1))
                            gid_value = gid_match.group(2)
                            port_info['gids'].append({
                                'index': gid_index,
                                'gid': gid_value
                            })
                
                device_info['port_info'][1] = port_info
        
        return device_info
        
    except Exception as e:
        logger.error(f"Error getting RDMA device info for {device_name}: {e}")
        return None

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

def parse_gid_to_components(gid_str: str) -> Dict:
    """Parse GID string to subnet_prefix and interface_id"""
    try:
        # GID format: xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx
        gid_parts = gid_str.split(':')
        if len(gid_parts) != 8:
            raise ValueError(f"Invalid GID format: {gid_str}")
        
        # First 4 parts are subnet_prefix, last 4 are interface_id
        subnet_prefix_hex = ''.join(gid_parts[:4])
        interface_id_hex = ''.join(gid_parts[4:])
        
        subnet_prefix = int(subnet_prefix_hex, 16)
        interface_id = int(interface_id_hex, 16)
        
        return {
            'subnet_prefix': subnet_prefix,
            'interface_id': interface_id
        }
    except Exception as e:
        logger.error(f"Error parsing GID {gid_str}: {e}")
        # Return default values if parsing fails
        return {
            'subnet_prefix': 0,
            'interface_id': 10592651870457495552
        }

def mtu_string_to_enum(mtu_str: str) -> int:
    """Convert MTU string to enum value"""
    mtu_map = {
        '256': 1,
        '512': 2,
        '1024': 3,
        '2048': 4,
        '4096': 5
    }
    return mtu_map.get(mtu_str, 3)  # Default to 1024 (enum 3)

def get_qp_info_from_device(device_name: str, port_num: int, gid_index: int) -> Dict:
    """Get actual QP information from RDMA device"""
    try:
        # In a real implementation, you would query the actual QP information
        # This is a placeholder that should be replaced with actual RDMA library calls
        # For example, using rdma-core libraries or direct system calls
        
        # TODO: Replace with actual QP creation/query logic
        # For now, we'll use a deterministic approach based on device characteristics
        # rather than random values
        
        # Generate deterministic values based on device and port info
        base_qpn = hash(f"{device_name}_{port_num}_{gid_index}") % 0xFFFFFF + 1
        base_psn = hash(f"psn_{device_name}_{port_num}_{gid_index}") % 0xFFFFFF
        
        return {
            'qpn': base_qpn,
            'psn': base_psn
        }
    except Exception as e:
        logger.error(f"Error getting QP info for {device_name}: {e}")
        # Fallback to deterministic values
        return {
            'qpn': 1000 + port_num * 100 + gid_index,
            'psn': 2000 + port_num * 100 + gid_index
        }


class MooncakeMigrationManagement:
    """Manages migration for a single connection in Mooncake backend"""
    
    def __init__(self, init_request: DistServeInitRequest):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "to run LMDeploy with MooncakeBackend."
            ) from e
        
        self.rank = init_request.rank
        self.local_engine_config: DistServeEngineConfig = init_request.local_engine_config
        self.remote_engine_config: DistServeEngineConfig = init_request.remote_engine_config
        self.remote_engine_id = init_request.remote_engine_id
        
        self.engine = TransferEngine()
        self.hostname = get_local_ip_by_remote()
        
        # Get all RDMA information once during initialization
        self.rdma_info = get_all_rdma_info()
        self.ibv_devices = self.rdma_info['devices']  # List of device names
        self.device_details = self.rdma_info['device_details']  # Detailed device info
        
        self.buffer_map: Dict[str, Dict] = {}
        self.remote_url: str = ""  # Store remote URL for this connection
        self.remote_endpoint_info: Dict = {}  # Store remote endpoint info
        self.num_qps_per_gid = 1  # Default to 1 QP per GID
        
        # Initialize the p2p connection
        self._initialize_p2p(init_request)
    
    def _initialize_p2p(self, init_request: DistServeInitRequest):
        """Initialize p2p connection for this specific link"""
        # TODO: Support more types of metadata_server
        # e.g. "etcd://192.168.0.137:2379"
        metadata_server = "P2PHANDSHAKE" 
            
        # Default protocol (Currently only RDMA is supported)
        protocol = "rdma"  

        # Get the device name from request
        if not self.ibv_devices:
            raise RuntimeError("No RDMA devices available")
            
        device_name = self.ibv_devices[self.rank % len(self.ibv_devices)]
        
        # Initialize the engine
        result = self.engine.initialize(self.hostname, metadata_server, protocol, device_name)
        if result != 0:
            raise RuntimeError(f"Failed to initialize Mooncake engine: {result}")
        
        logger.info(f"Mooncake engine initialized for remote_engine_id {self.remote_engine_id} "
                   f"with hostname {self.hostname}, RPC port: {self.engine.get_rpc_port()}")

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        """Register memory region for this connection"""    
        # Transmit buffer address to int
        buffer_addr = register_mr_request.addr
        buffer_length = register_mr_request.length
        
        # Register memory region with the engine
        result = self.engine.register_memory(buffer_addr, buffer_length)
        if result != 0:
            raise RuntimeError(f"Failed to register memory region: {result}")
        
        self.buffer_map[register_mr_request.mr_key] = {
            'mr_key': register_mr_request.mr_key,
            'addr': buffer_addr,
            'length': buffer_length,
            'offset': register_mr_request.offset
        }
        
        logger.info(f"Registered memory region with key {register_mr_request.mr_key}, "
                   f"addr: {buffer_addr}, length: {buffer_length} for remote_engine_id {self.remote_engine_id}")

    @property
    def endpoint_info(self) -> Dict:
        """Get endpoint information for this connection"""      
        
        # Generate memory region information
        mr_info = {}
        rank = -1
        
        # Generate MR info for each registered buffer
        for rkey in self.buffer_map:
            rank = rank + 1
            mr_info[str(rank)] = {
                'addr': self.buffer_map[rkey]['addr'],
                'length': self.buffer_map[rkey]['length'],
                'rkey': rkey
            }
        
        # Generate RDMA connection information from cached device info
        rdma_info = []
        
        # Use cached RDMA device information
        if self.ibv_devices and self.device_details:
            device_name = self.ibv_devices[0]  # Use first available device
            device_info = self.device_details.get(device_name)
            
            if device_info and 'port_info' in device_info:
                # Iterate over all detected ports
                for port_num, port_data in device_info['port_info'].items():
                    if port_data.get('state') == 'PORT_ACTIVE':
                        # Iterate over all GIDs for this active port
                        if 'gids' in port_data and port_data['gids']:
                            for gid_entry in port_data['gids']:
                                # Consider only valid, non-zero GIDs
                                if gid_entry['gid'] != '0000:0000:0000:0000:0000:0000:0000:0000':
                                    active_gid = gid_entry['gid']
                                    gid_index = gid_entry['index']
                                    gid_components = parse_gid_to_components(active_gid)
                                    
                                    # Create multiple QPs per GID if needed
                                    for qp_idx in range(self.num_qps_per_gid):
                                        # Get actual QP information from the device
                                        qp_info = get_qp_info_from_device(device_name, port_num, gid_index + qp_idx)

                                        rdma_info.append({
                                            'gid': {
                                                'interface_id': gid_components['interface_id'],
                                                'subnet_prefix': gid_components['subnet_prefix']
                                            },
                                            'gidx': gid_index,
                                            'lid': port_data.get('lid', 0),
                                            'mtu': mtu_string_to_enum(port_data.get('active_mtu', '1024')),
                                            'psn': qp_info['psn'],
                                            'qpn': qp_info['qpn']
                                        })

        # Fallback if no real RDMA info could be obtained
        if not rdma_info:
            logger.warning("Could not get real RDMA device info, using fallback values.")
            num_fallback_entries = len(mr_info) if mr_info else 1
            for idx in range(num_fallback_entries):
                qp_info = get_qp_info_from_device("fallback", 1, idx)
                rdma_info.append({
                    'gid': {
                        'interface_id': 10592651870457495552,
                        'subnet_prefix': 0
                    },
                    'gidx': 1,
                    'lid': 0,
                    'mtu': 3,  # Default MTU (1024)
                    'psn': qp_info['psn'],
                    'qpn': qp_info['qpn']
                })
        
        # Create the endpoint information structure
        endpoint_info = {
            'mr_info': mr_info,
            'rdma_info': rdma_info
        }
        
        logger.info(f"Endpoint information for remote engine {self.remote_engine_id} generated successfully")
        
        return endpoint_info

    def connect(self, connect_request: DistServeConnectionRequest):
        """Connect to the remote engine"""
        self.remote_endpoint_info = json.loads(connect_request.remote_endpoint_info)

        # Convert RoCEv2 interface ID to IP address
        def extract_rocev2_ip(interface_id: int) -> str:
            ip_hex = (interface_id >> 32) & 0xFFFFFFFF
            ip_bytes = struct.pack('<I', ip_hex)
            return socket.inet_ntoa(ip_bytes)

        # Extract the remote endpoint information
        rdma_info = self.remote_endpoint_info["rdma_info"][0]  # Default to the first RDMA info
        interface_id = rdma_info["gid"]["interface_id"]
        ip = extract_rocev2_ip(interface_id)

        port = self.engine.get_rpc_port()
        endpoint_url = f"{ip}:{port}"  # mooncake_cake_id

        self.remote_url = endpoint_url

        logger.info(f"Connected to remote engine {self.remote_engine_id} at endpoint {endpoint_url}")

    def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        """Migrate data to the remote engine"""
        if not self.remote_url:
            raise RuntimeError(f"No connection established to remote engine {self.remote_engine_id}")
        
        for task in assignment.batch:
            if task.mr_key not in self.buffer_map:
                raise RuntimeError(f"Memory region with key {task.mr_key} not registered")
            
            # Get local buffer information
            # TODO: check if the buffer addr and session id is correct
            buffer_info = self.buffer_map[task.mr_key]
            local_addr = buffer_info['addr'] + buffer_info['offset'] + task.source_offset
            mooncake_session_id = self.remote_url
            
            # Currently, only sync transfer is supported
            # FIXME: sync transfer fail
            result = self.engine.transfer_sync_read(
                mooncake_session_id,
                local_addr,
                task.target_offset,
                task.length,
            )
            if result != 0:
                raise RuntimeError(f"Failed to perform sync transfer: {result}")


@MIGRATION_BACKENDS.register_module(MigrationBackend.Mooncake.name)
class MooncakeBackend(MigrationBackendImpl):
    """Mooncake backend that manages multiple migration connections"""

    def __init__(self):
        self.links: Dict[int, MooncakeMigrationManagement] = {}

    def p2p_initialize(self, init_request: DistServeInitRequest):
        """Initialize p2p connection for a specific remote engine"""
        self.links[init_request.remote_engine_id] = MooncakeMigrationManagement(init_request)

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        """Register memory region for a specific remote engine connection"""
        if register_mr_request.remote_engine_id not in self.links:
            raise RuntimeError(f"No connection initialized for remote engine {register_mr_request.remote_engine_id}")
        
        self.links[register_mr_request.remote_engine_id].register_memory_region(register_mr_request)

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        """Get endpoint information for a specific remote engine"""
        if remote_engine_id not in self.links:
            raise RuntimeError(f"No connection initialized for remote engine {remote_engine_id}")
        
        return self.links[remote_engine_id].endpoint_info

    def p2p_connect(self, connect_request: DistServeConnectionRequest):
        """Connect to a specific remote engine"""
        if connect_request.remote_engine_id not in self.links:
            raise RuntimeError(f"No connection initialized for remote engine {connect_request.remote_engine_id}")
        
        self.links[connect_request.remote_engine_id].connect(connect_request)

    def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        """Migrate data to a specific remote engine"""
        if assignment.remote_engine_id not in self.links:
            raise RuntimeError(f"No connection established to remote engine {assignment.remote_engine_id}")
        
        self.links[assignment.remote_engine_id].p2p_migrate(assignment, async_op=async_op)

    def store(self, assignment: MigrationAssignment, async_op: bool = False):
        """Store operation - not implemented for Mooncake"""
        raise NotImplementedError

    def load(self, assignment: MigrationAssignment, async_op: bool = False):
        """Load operation - not implemented for Mooncake"""
        raise NotImplementedError