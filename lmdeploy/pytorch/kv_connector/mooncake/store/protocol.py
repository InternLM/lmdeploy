# Copyright (c) OpenMMLab. All rights reserved.
"""Wire protocol for Mooncake prefix-key lookups.

Requests use ZMQ multipart frames.  The first frame is a named message tag so the protocol can grow without overloading
a payload field.

Lookup payload frames are token_len (u32), hash width (u16), packed block hashes, and recompute_blocks (u32). Integers
use big-endian encoding. The worker applies the rewind before selecting a hybrid state boundary.
"""

LOOKUP_MSG = b'lookup'
RESP_ERR = b'\x00'

__all__ = ['LOOKUP_MSG', 'RESP_ERR']
