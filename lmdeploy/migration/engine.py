# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from .context import RDMAContext


class TransferEngine:

    def __init__(self):
        self.links: Dict[int, RDMAContext] = {}

    def init_link(
        self,
        session_id: int,
        dev_name: str,
        ib_port=1,
        link_type: str = 'Ethernet',
    ) -> RDMAContext:
        if session_id in self.links:
            raise KeyError(f'session_id {session_id} already in links')
        self.links[session_id] = RDMAContext(
            dev_name=dev_name,
            ib_port=ib_port,
            link_type=link_type,
        )
        return self.links[session_id]

    def stop_link(self, session_id: int):
        if session_id not in self.links:
            raise KeyError(f'session_id {id} not in links')
        self.links[session_id].stop_link()
        del self.links[session_id]
