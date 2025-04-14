import enum

from typing import List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor

import requests

from lmdeploy.logger import get_logger

from lmdeploy.disagg.messages import (
    DisaggEngineConfig,
    MigrationInitRequest,
    TCPInitRequest,
    RDMAInitRequest,
    NVLinkInitRequest,
    MigrationTransportProtocol,
    MigrationConnectionRequest
)

logger = get_logger("lmdeploy")


class PDConnectionStatus(enum.Enum):
    Disconnect = enum.auto()
    Connected = enum.auto()


class PDConnectionPool:
    def __init__(self):
        self.pool = {}

    def init_connection(self, p0: str, p1: str):
        self.pool[(p0, p1)] = PDConnectionStatus.Disconnect


def pd_consolidation(
    endpoint: Tuple[str, str],
    protocol: MigrationTransportProtocol=MigrationTransportProtocol.RDMA,
    *,
    rdma_link_type: str = None,
    available_nic: Optional[Tuple[List[str], List[str]]] = None
):
    """pd consolidation."""
    def get_engine_config(engine_endpoint):
        engine_config = requests.get(f"{engine_endpoint}/distserve/engine_info", timeout=5).json()
        return DisaggEngineConfig.model_validate_json(engine_config)

    def p2p_initialize(engine_endpoint, init_request: MigrationInitRequest):
        return requests.post(
            f"{engine_endpoint}/distserve/p2p_initialize",
            timeout=5,
            json=init_request.model_dump(mode="json")).json()
    
    def p2p_connect(engine_endpoint, conn_request: List[MigrationConnectionRequest]):
        return requests.post(
            f"{engine_endpoint}/distserve/p2p_connect",
            timeout=5,
            json=[req.model_dump(mode="json") for req in conn_request]
        ).json()

    # Step 1. Get Remote Engine Configuration
    prefill_engine_config = get_engine_config(endpoint[0])
    decode_engine_config = get_engine_config(endpoint[1])

    # Note: Only tp is supported by now
    assert prefill_engine_config.dp_size == None
    assert prefill_engine_config.pp_size == None

    assert decode_engine_config.dp_size == None
    assert decode_engine_config.pp_size == None

    # Note: Only Same Parallel Configurations are supported by now
    assert prefill_engine_config.tp_size == decode_engine_config.tp_size

    # Step 2. Construct Initialize Configuration
    prefill_init_req = MigrationInitRequest(
        protocol=protocol,
        remote_engine_id=endpoint[1],
        remote_engine_config=decode_engine_config,
    )
    decode_init_req = MigrationInitRequest(
        protocol=protocol,
        remote_engine_id=endpoint[0],
        remote_engine_config=prefill_engine_config,
    )

    if protocol == MigrationTransportProtocol.RDMA:
        prefill_init_req.rdma_init_request = RDMAInitRequest(
            device_name=None,
            ib_port=1,
            link_type="Ethernet"
        )
        decode_init_req.rdma_init_request = RDMAInitRequest(
            device_name=None,
            ib_port=1,
            link_type="Ethernet"
        )
    else:
        raise NotImplementedError

    prefill_endpoint_info = p2p_initialize(endpoint[0], prefill_init_req)
    decode_endpoint_info = p2p_initialize(endpoint[1], decode_init_req)

    # Step 3. Connection
    if protocol == MigrationTransportProtocol.RDMA:
        prefill_endpoint_conn_reqs = [
            MigrationConnectionRequest(
                protocol=protocol,
                remote_engine_id=endpoint[1],
                remote_endpoint_info=info
            )
            for info in decode_endpoint_info
        ]
        decode_endpoint_conn_reqs = [
            MigrationConnectionRequest(
                protocol=protocol,
                remote_engine_id=endpoint[0],
                remote_endpoint_info=info
            )
            for info in prefill_endpoint_info
        ]
        p2p_connect(endpoint[0], prefill_endpoint_conn_reqs)
        p2p_connect(endpoint[1], decode_endpoint_conn_reqs)
        logger.info(f"{endpoint} connected")
    
def pd_consolidation_multi_thread(
    p_endpoints: List[str],
    d_endpoints: List[str],
    protocol: MigrationTransportProtocol = MigrationTransportProtocol.RDMA,
    *,
    rdma_link_type: str = None,
    available_nic: Optional[Tuple[List[str], List[str]]] = None
):
    with ThreadPoolExecutor() as executor:
        for pid in p_endpoints:
            for did in d_endpoints:
                executor.submit(
                    pd_consolidation,
                    (pid, did),
                    protocol,
                    rdma_link_type=rdma_link_type,
                    available_nic=available_nic
                )
