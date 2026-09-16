# Copyright (c) OpenMMLab. All rights reserved.
"""Wire protocol for Mooncake prefix-key lookups.

Requests use ZMQ multipart frames.  The first frame is a named message tag so the protocol can grow without overloading
a payload field.
"""

LOOKUP_MSG = b'lookup'
RESP_ERR = b'\x00'

__all__ = ['LOOKUP_MSG', 'RESP_ERR']
