# Copyright (c) OpenMMLab. All rights reserved.

import json

from fastapi.responses import StreamingResponse

from .utils import APIServerException


class ProxyStreamingResponse(StreamingResponse):
    """StreamingResponse that can handle exceptions thrown by the generator."""

    def __init__(self, content, **kwargs):
        super().__init__(content, **kwargs)

    async def stream_response(self, send) -> None:
        iterator = self.body_iterator.__aiter__()
        try:
            # get the first chunk(stream_generate's first yield)
            first_chunk = await iterator.__anext__()

        except APIServerException as e:
            headers = self._convert_headers_to_asgi(e.headers) if e.headers else self.raw_headers
            await send({'type': 'http.response.start', 'status': e.status_code, 'headers': headers})
            await send({
                'type': 'http.response.body',
                'body': e.body,
                'more_body': False,
            })
            return

        # normal case, send the header first
        await send({
            'type': 'http.response.start',
            'status': self.status_code,
            'headers': self.raw_headers,
        })

        # send body with the first chunk
        await send({
            'type': 'http.response.body',
            'body': first_chunk,
            'more_body': True,
        })

        # continue streaming output
        try:
            async for chunk in iterator:
                await send({
                    'type': 'http.response.body',
                    'body': chunk,
                    'more_body': True,
                })
        except Exception:
            error_data = {'error': True, 'status': 500, 'message': 'Internal streaming error'}
            await send({
                'type': 'http.response.body',
                'body': json.dumps(error_data).encode('utf-8'),
                'more_body': False,
            })
            return

        await send({
            'type': 'http.response.body',
            'body': b'',
            'more_body': False,
        })

    def _convert_headers_to_asgi(self, headers: dict) -> list[tuple[bytes, bytes]]:
        """Convert dict headers to ASGI raw header tuples."""
        return [(name.lower().encode('latin-1'), str(value).encode('latin-1')) for name, value in headers.items()]
