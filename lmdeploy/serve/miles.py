"""Gated single-GPU server for Miles' synchronous PPO inference protocol.

Run ``python -m lmdeploy.serve.miles --help`` for launch options. The gate
delays model allocation until Miles has offloaded its training models.
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer


def wait_for_gate(host, port):
    if port is None:
        return

    class GateHandler(BaseHTTPRequestHandler):
        activated = False

        def do_POST(self):
            if self.path != '/gate/activate':
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
            GateHandler.activated = True

        def log_message(self, *args):
            pass

    with HTTPServer((host, port), GateHandler) as server:
        print(f'Miles launch gate waiting on {host}:{port}', flush=True)
        while not GateHandler.activated:
            server.handle_request()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=23333)
    parser.add_argument('--gate-port', type=int)
    parser.add_argument('--max-batch-size', type=int, default=64)
    parser.add_argument('--cache-fraction', type=float, default=0.8)
    parser.add_argument('--api-key')
    args = parser.parse_args()
    wait_for_gate(args.host, args.gate_port)

    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.serve.openai.api_server import serve
    config = PytorchEngineConfig(
        tp=1, dtype='bfloat16', max_batch_size=args.max_batch_size,
        cache_max_entry_count=args.cache_fraction,
        logprobs_mode='temperature_logprobs')
    serve(model_path=args.model, backend='pytorch', backend_config=config,
          server_name=args.host, server_port=args.port,
          api_keys=args.api_key, miles_compat=True,
          enable_abort_handling=True, generation_config='lmdeploy')


if __name__ == '__main__':
    main()
