# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

from lmdeploy.pytorch.disagg.conn.protocol import (DistServeInitRequest, DistServeKVTransferEndpointInfo,
                                                   MigrationProtocol)
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment


class MigrationBackendImpl:

    @abstractmethod
    def p2p_initialize(self, init_request: DistServeInitRequest):
        raise NotImplementedError

    @abstractmethod
    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        raise NotImplementedError

    @abstractmethod
    def endpoint_info(self, remote_engine_id: str, protocol: MigrationProtocol):
        return NotImplementedError

    @abstractmethod
    def p2p_connect(self, remote_engine_id: str, conn_req: DistServeKVTransferEndpointInfo):
        raise NotImplementedError

    @abstractmethod
    def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    @abstractmethod
    def store(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    @abstractmethod
    def load(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError
