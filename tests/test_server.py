"""Tests for the KlomboAGI HTTP server."""

import json
import time
import urllib.request

from klomboagi.core.genesis import Genesis
from klomboagi.server import run_server


def _get(port, path):
    r = urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=5)
    return json.loads(r.read())


def _post(port, path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}{path}", data=body,
        headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=10)
    return json.loads(r.read())


class TestServer:
    """Integration tests for the HTTP server."""

    @classmethod
    def setup_class(cls):
        cls.genesis = Genesis()
        cls.server = run_server(cls.genesis, port=3199, background=True)
        time.sleep(0.5)

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()

    def test_health(self):
        data = _get(3199, "/health")
        assert data["status"] == "alive"
        assert "uptime_seconds" in data

    def test_hardware(self):
        data = _get(3199, "/hardware")
        assert "cpu" in data
        assert "ram" in data
        assert data["cpu"]["cores"] > 0

    def test_hear(self):
        data = _post(3199, "/hear", {"message": "what are you"})
        assert "response" in data
        assert len(data["response"]) > 0
        assert "elapsed_ms" in data

    def test_beliefs(self):
        data = _get(3199, "/beliefs")
        assert "count" in data
        assert "beliefs" in data

    def test_curiosity(self):
        data = _get(3199, "/curiosity")
        assert "gaps" in data

    def test_hear_empty_message_rejected(self):
        try:
            _post(3199, "/hear", {"message": ""})
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_unknown_path_404(self):
        try:
            _get(3199, "/nonexistent")
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 404
