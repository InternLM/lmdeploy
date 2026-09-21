# Copyright (c) OpenMMLab. All rights reserved.
#!/usr/bin/env python3
"""Simple mock backend server for testing the router's transparent proxy
feature.

Usage:
    python mock_backend_server.py [--port PORT] [--host HOST]

Example:
    # Start server on port 8081
    python mock_backend_server.py --port 8081

    # Test with curl
    curl -X POST http://localhost:8081/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello world"}'
"""

import argparse
import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockBackendHandler(BaseHTTPRequestHandler):
    """Handler for mock backend server requests."""

    def log_request_info(self, method: str):
        """Log request information to stdout."""
        timestamp = datetime.now().isoformat()
        print(f"\n{'='*60}")
        print(f'[{timestamp}] {method} {self.path}')
        print(f"{'='*60}")
        print(f'Client: {self.client_address[0]}:{self.client_address[1]}')
        print(f'Path: {self.path}')
        print('Headers:')
        for header, value in self.headers.items():
            print(f'  {header}: {value}')

    def send_json_response(self, status_code: int, data: dict):
        """Send a JSON response."""
        response_body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def do_GET(self):
        """Handle GET requests."""
        self.log_request_info('GET')

        if self.path == '/health' or self.path == '/healthz':
            print('Body: (none)')
            print('\n[SUCCESS] Health check passed')
            self.send_json_response(200, {'status': 'healthy'})

        elif self.path == '/v1/models':
            print('Body: (none)')
            print('\n[SUCCESS] Models list requested')
            self.send_json_response(200, {
                'object': 'list',
                'data': [
                    {
                        'id': 'mock-model',
                        'object': 'model',
                        'owned_by': 'mock-server'
                    }
                ]
            })

        else:
            print('Body: (none)')
            print(f'\n[SUCCESS] GET request received at {self.path}')
            self.send_json_response(200, {
                'message': f'GET request received at {self.path}',
                'path': self.path
            })

    def do_POST(self):
        """Handle POST requests."""
        self.log_request_info('POST')

        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else ''

        # Try to parse as JSON for pretty printing
        try:
            body_json = json.loads(body) if body else {}
            print('Body (JSON):')
            print(json.dumps(body_json, indent=2))
        except json.JSONDecodeError:
            print(f"Body (raw): {body[:500]}{'...' if len(body) > 500 else ''}")
            body_json = {'raw': body}

        # Handle different endpoints
        if self.path == '/generate':
            print('\n[SUCCESS] /generate endpoint called')
            response = {
                'text': 'This is a mock response from the generate endpoint.',
                'prompt': body_json.get('prompt', ''),
                'model': body_json.get('model', 'mock-model'),
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 15,
                    'total_tokens': 25
                }
            }
            self.send_json_response(200, response)

        elif self.path == '/v1/chat/completions':
            print('\n[SUCCESS] /v1/chat/completions endpoint called')
            response = {
                'id': 'mock-chat-completion',
                'object': 'chat.completion',
                'model': body_json.get('model', 'mock-model'),
                'choices': [
                    {
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': 'This is a mock response from chat completions.'
                        },
                        'finish_reason': 'stop'
                    }
                ],
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 12,
                    'total_tokens': 22
                }
            }
            self.send_json_response(200, response)

        elif self.path == '/v1/completions':
            print('\n[SUCCESS] /v1/completions endpoint called')
            response = {
                'id': 'mock-completion',
                'object': 'text_completion',
                'model': body_json.get('model', 'mock-model'),
                'choices': [
                    {
                        'index': 0,
                        'text': 'This is a mock completion response.',
                        'finish_reason': 'stop'
                    }
                ],
                'usage': {
                    'prompt_tokens': 5,
                    'completion_tokens': 8,
                    'total_tokens': 13
                }
            }
            self.send_json_response(200, response)

        else:
            # Handle any other POST endpoint (transparent proxy test)
            print(f'\n[SUCCESS] Custom endpoint {self.path} called (transparent proxy)')
            response = {
                'message': f'POST request received at {self.path}',
                'path': self.path,
                'body_received': body_json,
                'server': 'mock-backend-server'
            }
            self.send_json_response(200, response)

    def log_message(self, format, *args):
        """Suppress default logging (we do our own)."""
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Mock backend server for testing router transparent proxy'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8081,
        help='Port to listen on (default: 8081)'
    )
    parser.add_argument(
        '--host', '-H',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    args = parser.parse_args()

    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, MockBackendHandler)

    print('Mock Backend Server')
    print('================')
    print(f'Listening on {args.host}:{args.port}')
    print('')
    print('Available endpoints:')
    print('  GET  /health           - Health check')
    print('  GET  /v1/models        - List models')
    print('  POST /generate         - Generate endpoint')
    print('  POST /v1/completions   - Completions endpoint')
    print('  POST /v1/chat/completions - Chat completions endpoint')
    print('  POST /*                - Any other path (transparent proxy test)')
    print('')
    print('Press Ctrl+C to stop')
    print('')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down...')
        httpd.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()
