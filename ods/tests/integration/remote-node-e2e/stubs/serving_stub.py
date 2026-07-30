"""Minimal OpenAI-shaped /v1/models endpoint for the node-agent to probe.

Stdlib only, so the container needs no pip install. The node-agent reaches
this over the bridge network exactly as it would reach a real vLLM or
llama-server on its own node.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_ID = "e2e-stub-model"
PORT = 8000


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib naming
        if self.path.rstrip("/") != "/v1/models":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "e2e"}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        """Silence per-request logging: the agent polls this constantly."""


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
