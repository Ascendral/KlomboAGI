"""
KlomboAGI HTTP Server — the conversation and control interface.

Runs as a lightweight HTTP server alongside the daemon.
Exposes the brain to any device on the network.

Endpoints:
  POST /hear          — Talk to Klombo (returns response)
  GET  /status        — Full system status
  GET  /hardware      — Hardware state
  GET  /health        — Quick health check (for monitoring)
  POST /mission       — Create a mission
  GET  /missions      — List missions
  POST /learn         — Teach Klombo something
  GET  /beliefs       — What Klombo believes (paginated)
  GET  /curiosity     — What Klombo wants to know
"""

from __future__ import annotations

import json
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klomboagi.core.genesis import Genesis


class KlomboHandler(BaseHTTPRequestHandler):
    """HTTP request handler for KlomboAGI."""

    server: "KlomboServer"

    def log_message(self, format, *args):
        """Override to use our own logging."""
        pass  # Silence default stderr logging

    def _send_json(self, data: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        genesis = self.server.genesis

        try:
            if self.path == "/health":
                self._send_json({
                    "status": "alive",
                    "uptime_seconds": round(time.time() - self.server.start_time, 1),
                    "total_turns": genesis.total_turns if genesis else 0,
                })

            elif self.path == "/status":
                if not genesis:
                    self._send_json({"error": "Brain not initialized"}, 503)
                    return
                self._send_json({
                    "status": genesis.status(),
                    "total_turns": genesis.total_turns,
                    "deep_thinks": genesis.deep_thinks,
                    "surprises": genesis.total_surprises,
                    "topic": genesis.context.current_topic or None,
                    "beliefs": len(genesis.base._beliefs),
                    "uptime_seconds": round(time.time() - self.server.start_time, 1),
                })

            elif self.path == "/hardware":
                if not genesis:
                    self._send_json({"error": "Brain not initialized"}, 503)
                    return
                hw = genesis.scan_hardware()
                self._send_json({
                    "summary": hw.summary(),
                    "cpu": {"model": hw.cpu.model, "cores": hw.cpu.cores_logical,
                            "usage": hw.cpu.usage_percent},
                    "ram": {"total_gb": round(hw.ram.total_gb, 1),
                            "available_gb": round(hw.ram.available_gb, 1),
                            "usage": hw.ram.usage_percent},
                    "gpu": {"model": hw.gpu.model, "cores": hw.gpu.cores,
                            "vram_gb": round(hw.gpu.vram_gb, 1)},
                })

            elif self.path == "/beliefs":
                if not genesis:
                    self._send_json({"error": "Brain not initialized"}, 503)
                    return
                beliefs = []
                for stmt, belief in list(genesis.base._beliefs.items())[:100]:
                    beliefs.append({
                        "statement": stmt,
                        "confidence": round(belief.truth.confidence, 3) if hasattr(belief.truth, 'confidence') else 0,
                        "subject": belief.subject if hasattr(belief, 'subject') else "",
                    })
                self._send_json({"count": len(genesis.base._beliefs), "beliefs": beliefs})

            elif self.path == "/curiosity":
                if not genesis:
                    self._send_json({"error": "Brain not initialized"}, 503)
                    return
                gaps = []
                for gap in genesis.base.curiosity.gaps[:20]:
                    gaps.append({
                        "concept": gap.concept,
                        "resolved": gap.resolved,
                        "context": getattr(gap, 'context', ''),
                    })
                self._send_json({"gaps": gaps})

            elif self.path == "/missions":
                self._send_json({"error": "No storage connected"}, 501)

            else:
                self._send_json({"error": f"Unknown path: {self.path}"}, 404)

        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def do_POST(self):
        genesis = self.server.genesis

        try:
            body = self._read_body()

            if self.path == "/hear":
                if not genesis:
                    self._send_json({"error": "Brain not initialized"}, 503)
                    return
                message = body.get("message", "")
                if not message:
                    self._send_json({"error": "Missing 'message' field"}, 400)
                    return
                start = time.time()
                response = genesis.hear(message)
                elapsed = time.time() - start
                self._send_json({
                    "response": response,
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "turn": genesis.total_turns,
                    "topic": genesis.context.current_topic or None,
                })

            elif self.path == "/learn":
                if not genesis:
                    self._send_json({"error": "Brain not initialized"}, 503)
                    return
                topic = body.get("topic", "")
                if not topic:
                    self._send_json({"error": "Missing 'topic' field"}, 400)
                    return
                response = genesis._active_learn(topic)
                self._send_json({"response": response})

            else:
                self._send_json({"error": f"Unknown path: {self.path}"}, 404)

        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)


class KlomboServer(HTTPServer):
    """HTTP server with Genesis brain attached."""

    def __init__(self, host: str = "0.0.0.0", port: int = 3141,
                 genesis: "Genesis | None" = None):
        self.genesis = genesis
        self.start_time = time.time()
        super().__init__((host, port), KlomboHandler)


def run_server(genesis: "Genesis | None" = None, host: str = "0.0.0.0",
               port: int = 3141, background: bool = False) -> KlomboServer:
    """Start the HTTP server.

    Args:
        genesis: Genesis brain instance
        host: Bind address (0.0.0.0 = all interfaces)
        port: Port number (3141 = pi * 1000, the mind's port)
        background: If True, run in a background thread

    Returns:
        The server instance
    """
    server = KlomboServer(host, port, genesis)

    if background:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    print(f"KlomboAGI listening on http://{host}:{port}")
    print(f"  POST /hear     — Talk to Klombo")
    print(f"  GET  /status   — System status")
    print(f"  GET  /hardware — Hardware info")
    print(f"  GET  /health   — Health check")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
    return server
