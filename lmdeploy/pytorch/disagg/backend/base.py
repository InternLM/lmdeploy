# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

from lmdeploy.pytorch.disagg.config import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment
from lmdeploy.pytorch.disagg.request import DistServeConnectionRequest, DistServeInitRequest


class MigrationBackendImpl:

    @abstractmethod
    def p2p_initialize(self, init_request: DistServeInitRequest):
        raise NotImplementedError

    @abstractmethod
    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        raise NotImplementedError

    @abstractmethod
    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        return NotImplementedError

    @abstractmethod
    def p2p_connect(self, conn_req: DistServeConnectionRequest):
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
