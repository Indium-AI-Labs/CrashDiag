"""In-container HTTP control plane for the CrashDiag victim stack."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from crashdiag.sandbox_apps.mock import SandboxBackend
from crashdiag.sandbox_apps.target.runtime import VictimRuntime

RUNTIME = VictimRuntime()
_RPC_METHODS = frozenset(SandboxBackend.ACTIONS) | {
    "observe",
    "health_check",
    "trigger_oom_kill",
    "set_env_var",
    "set_dependency_version",
    "set_disk_usage",
    "set_proxy_target_port",
    "set_service_state",
    "set_expected_env_var",
    "set_required_dependency_version",
    "set_app_port",
    "set_disk_health_threshold",
    "execute_action",
}


class VictimHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._write(HTTPStatus.OK, RUNTIME.health_check())
            return
        if self.path == "/observe":
            self._write(HTTPStatus.OK, RUNTIME.observe())
            return
        self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/rpc":
            self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return
        op = payload.get("op")
        if op not in _RPC_METHODS:
            self._write(HTTPStatus.BAD_REQUEST, {"error": "unknown_op", "op": op})
            return
        args = payload.get("args") or []
        kwargs = payload.get("kwargs") or {}
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid_args"})
            return
        try:
            result = getattr(RUNTIME, str(op))(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface victim errors to the host
            self._write(
                HTTPStatus.BAD_REQUEST,
                {"error": "rpc_failed", "detail": f"{type(exc).__name__}: {exc}"},
            )
            return
        self._write(HTTPStatus.OK, {"ok": True, "result": result})

    def _write(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 8080), VictimHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RUNTIME.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
