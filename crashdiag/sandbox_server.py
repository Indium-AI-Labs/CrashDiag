"""Safe HTTP service for disposable CrashDiag simulation sessions.

This server deliberately exposes :class:`~crashdiag.sandbox_apps.mock.MockSandbox`
state, not the container host.  Fault injection changes in-memory fields only: it
never allocates memory to force an OOM, fills a filesystem, installs packages, or
edits a real reverse proxy.  That makes the service suitable for dataset generation
and training-loop integration before a separately isolated real-infrastructure
backend is implemented.

API (all request and response bodies are JSON):

* ``GET /healthz`` -- unauthenticated service liveness.
* ``POST /v1/sessions`` -- create an isolated session.
* ``DELETE /v1/sessions/{id}`` -- delete a session.
* ``POST /v1/sessions/{id}/reset`` -- replace its state with a fresh sandbox.
* ``GET /v1/sessions/{id}/observe`` and ``.../health`` -- inspect state.
* ``POST /v1/sessions/{id}/actions/{name}`` -- execute an allowlisted action.
* ``POST /v1/sessions/{id}/mutations/{name}`` -- apply an allowlisted mutation.
* ``POST /v1/sessions/{id}/faults/{name}`` -- inject a built-in fault by name.

When configured, the bearer token protects every endpoint except ``/healthz``.
Sessions are bounded by count and an idle TTL and are also isolated by per-session
locks, so concurrent training workers cannot interleave state changes in one step.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import math
import os
import secrets
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, ClassVar, Iterator, Mapping
from urllib.parse import unquote, urlsplit

from .faults.modules import ALL_FAULTS
from .sandbox_apps.mock import MockSandbox, SandboxBackend

LOGGER = logging.getLogger("crashdiag.sandbox_server")


class SessionNotFound(LookupError):
    """Raised when a session does not exist or its idle TTL elapsed."""


class SessionCapacityError(RuntimeError):
    """Raised when the configured live-session limit has been reached."""


class SessionOperationLimitError(RuntimeError):
    """Raised when a session has exhausted its bounded state-change budget."""


class APIError(Exception):
    """An expected client-facing HTTP error."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = message


@dataclass
class _Session:
    sandbox: SandboxBackend
    created_at: float
    last_access: float
    operation_count: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionStore:
    """Thread-safe collection of bounded, idle-expiring sandbox sessions."""

    def __init__(
        self,
        *,
        max_sessions: int = 128,
        session_ttl_seconds: float = 900.0,
        max_operations_per_session: int = 64,
        sandbox_factory: Callable[[], SandboxBackend] = MockSandbox,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int):
            raise TypeError("max_sessions must be an integer")
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if isinstance(session_ttl_seconds, bool) or not isinstance(
            session_ttl_seconds, (int, float)
        ):
            raise TypeError("session_ttl_seconds must be numeric")
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        if isinstance(max_operations_per_session, bool) or not isinstance(
            max_operations_per_session, int
        ):
            raise TypeError("max_operations_per_session must be an integer")
        if max_operations_per_session < 1:
            raise ValueError("max_operations_per_session must be at least 1")
        if not callable(sandbox_factory):
            raise TypeError("sandbox_factory must be callable")

        self.max_sessions = max_sessions
        self.session_ttl_seconds = float(session_ttl_seconds)
        self.max_operations_per_session = max_operations_per_session
        self.sandbox_factory = sandbox_factory
        self._clock = clock
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    def _purge_expired_locked(self, now: float) -> int:
        expired: list[str] = []
        for session_id, session in self._sessions.items():
            if now - session.last_access < self.session_ttl_seconds:
                continue
            # Do not expire a state transition already in progress.  Every access
            # updates last_access while holding the store lock before releasing it.
            if session.lock.acquire(blocking=False):
                session.lock.release()
                expired.append(session_id)
        for session_id in expired:
            del self._sessions[session_id]
        return len(expired)

    def purge_expired(self) -> int:
        """Remove and count sessions whose idle TTL elapsed."""

        with self._lock:
            return self._purge_expired_locked(self._clock())

    def create(self) -> tuple[str, SandboxBackend]:
        """Create a fresh isolated sandbox, enforcing the capacity bound."""

        # Construct before taking the collection lock so a custom factory cannot
        # stall unrelated requests.  The default factory is local and side-effect
        # free beyond allocating an in-memory object.
        sandbox = self.sandbox_factory()
        if not isinstance(sandbox, SandboxBackend):
            raise TypeError("sandbox_factory must return a SandboxBackend")
        now = self._clock()
        with self._lock:
            self._purge_expired_locked(now)
            if len(self._sessions) >= self.max_sessions:
                raise SessionCapacityError(
                    f"session capacity of {self.max_sessions} has been reached"
                )
            session_id = secrets.token_urlsafe(24)
            while session_id in self._sessions:  # astronomically unlikely, explicit
                session_id = secrets.token_urlsafe(24)
            self._sessions[session_id] = _Session(sandbox, now, now)
        return session_id, sandbox

    @contextmanager
    def lease(
        self,
        session_id: str,
        *,
        count_operation: bool = False,
    ) -> Iterator[_Session]:
        """Lock and yield a live session while refreshing its idle timer.

        State-growing actions and mutations set ``count_operation``.  The
        bounded counter prevents a single long-lived session from accumulating
        unbounded histories, logs, environment keys, or dependency keys.
        """

        if not isinstance(session_id, str) or not session_id:
            raise SessionNotFound("session not found")
        with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound("session not found")
            # Lock order is always store -> session.  Deletion follows the same
            # order, so it cannot remove a session during an active operation.
            session.lock.acquire()
            try:
                if count_operation:
                    if session.operation_count >= self.max_operations_per_session:
                        raise SessionOperationLimitError(
                            "session operation limit of "
                            f"{self.max_operations_per_session} has been reached"
                        )
                    session.operation_count += 1
                session.last_access = now
            except Exception:
                session.lock.release()
                raise
        try:
            yield session
        finally:
            session.lock.release()

    def reset(self, session_id: str) -> SandboxBackend:
        """Replace a live session's state with a new sandbox instance."""

        sandbox = self.sandbox_factory()
        if not isinstance(sandbox, SandboxBackend):
            raise TypeError("sandbox_factory must return a SandboxBackend")
        with self.lease(session_id) as session:
            session.sandbox = sandbox
            session.operation_count = 0
            return session.sandbox

    def delete(self, session_id: str) -> None:
        """Delete a live session, or raise :class:`SessionNotFound`."""

        with self._lock:
            self._purge_expired_locked(self._clock())
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound("session not found")
            session.lock.acquire()
            try:
                del self._sessions[session_id]
            finally:
                session.lock.release()

    def stats(self) -> dict[str, int]:
        """Return non-sensitive service capacity counters."""

        with self._lock:
            self._purge_expired_locked(self._clock())
            return {"active": len(self._sessions), "maximum": self.max_sessions}


_FAULTS = {fault.name: fault for fault in ALL_FAULTS}
_MUTATIONS = frozenset(
    {
        "trigger_oom_kill",
        "set_env_var",
        "set_dependency_version",
        "set_disk_usage",
        "set_proxy_target_port",
        "set_expected_env_var",
        "set_required_dependency_version",
        "set_app_port",
        "set_disk_health_threshold",
    }
)


class SandboxRequestHandler(BaseHTTPRequestHandler):
    """JSON request handler; instances are created by ``SandboxHTTPServer``."""

    server_version = "CrashDiagSandbox/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    max_request_bytes: ClassVar[int] = 64 * 1024
    max_parameter_count: ClassVar[int] = 8
    max_parameter_name_chars: ClassVar[int] = 64
    max_parameter_string_chars: ClassVar[int] = 1024

    @property
    def sandbox_server(self) -> "SandboxHTTPServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._handle("POST")

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._handle("DELETE")

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self._write_error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "method not allowed")

    def _handle(self, method: str) -> None:
        try:
            path = urlsplit(self.path).path
            if method == "GET" and path == "/healthz":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "crashdiag-sandbox",
                        "scenario_schema_versions": [1, 2],
                        "sessions": self.sandbox_server.sessions.stats(),
                    },
                )
                return
            self._require_authorization()
            segments = self._segments(path)
            self._dispatch(method, segments)
        except APIError as exc:
            headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
            self._write_error(exc.status, exc.code, exc.message, headers=headers)
        except SessionNotFound:
            self._write_error(HTTPStatus.NOT_FOUND, "session_not_found", "session not found")
        except SessionCapacityError as exc:
            self._write_error(HTTPStatus.TOO_MANY_REQUESTS, "session_capacity", str(exc))
        except SessionOperationLimitError as exc:
            self._write_error(
                HTTPStatus.TOO_MANY_REQUESTS,
                "session_operation_limit",
                str(exc),
            )
        except TimeoutError:
            self.close_connection = True
            try:
                self._write_error(
                    HTTPStatus.REQUEST_TIMEOUT,
                    "request_timeout",
                    "request timed out",
                )
            except OSError:
                return
        except (TypeError, ValueError) as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            LOGGER.exception("unexpected sandbox API error")
            self._write_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "internal server error",
            )

    @staticmethod
    def _segments(path: str) -> list[str]:
        try:
            return [unquote(segment, errors="strict") for segment in path.split("/") if segment]
        except UnicodeDecodeError as exc:
            raise APIError(400, "invalid_path", "path is not valid UTF-8") from exc

    def _require_authorization(self) -> None:
        token = self.sandbox_server.bearer_token
        if token is None:
            return
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {token}"):
            raise APIError(401, "unauthorized", "a valid bearer token is required")

    def _dispatch(self, method: str, segments: list[str]) -> None:
        if segments == ["v1", "sessions"]:
            if method != "POST":
                raise APIError(405, "method_not_allowed", "method not allowed")
            self._read_json_object()
            session_id, sandbox = self.sandbox_server.sessions.create()
            self._write_json(
                HTTPStatus.CREATED,
                {
                    "session_id": session_id,
                    "ttl_seconds": self.sandbox_server.sessions.session_ttl_seconds,
                    "observation": sandbox.observe(),
                },
            )
            return

        if len(segments) < 3 or segments[:2] != ["v1", "sessions"]:
            raise APIError(404, "not_found", "endpoint not found")
        session_id = segments[2]

        if len(segments) == 3 and method == "DELETE":
            self.sandbox_server.sessions.delete(session_id)
            self._write_json(HTTPStatus.OK, {"deleted": True, "session_id": session_id})
            return

        if len(segments) == 4 and segments[3] == "reset":
            if method != "POST":
                raise APIError(405, "method_not_allowed", "method not allowed")
            self._read_json_object()
            sandbox = self.sandbox_server.sessions.reset(session_id)
            self._write_json(
                HTTPStatus.OK,
                {"reset": True, "session_id": session_id, "observation": sandbox.observe()},
            )
            return

        if len(segments) == 4 and segments[3] in {"observe", "health"}:
            if method != "GET":
                raise APIError(405, "method_not_allowed", "method not allowed")
            with self.sandbox_server.sessions.lease(session_id) as session:
                payload = (
                    session.sandbox.observe()
                    if segments[3] == "observe"
                    else session.sandbox.health_check()
                )
            self._write_json(HTTPStatus.OK, payload)
            return

        if len(segments) == 5 and segments[3] == "actions":
            if method != "POST":
                raise APIError(405, "method_not_allowed", "method not allowed")
            action = segments[4]
            if action not in SandboxBackend.ACTIONS:
                raise APIError(404, "unknown_action", "action is not allowlisted")
            parameters = self._parameters()
            with self.sandbox_server.sessions.lease(
                session_id, count_operation=True
            ) as session:
                result = session.sandbox.execute_action(action, parameters)
            self._write_json(HTTPStatus.OK, result)
            return

        if len(segments) == 5 and segments[3] == "mutations":
            if method != "POST":
                raise APIError(405, "method_not_allowed", "method not allowed")
            mutation = segments[4]
            if mutation not in _MUTATIONS:
                raise APIError(404, "unknown_mutation", "mutation is not allowlisted")
            parameters = self._parameters()
            with self.sandbox_server.sessions.lease(
                session_id, count_operation=True
            ) as session:
                method_to_call = getattr(session.sandbox, mutation)
                method_to_call(**parameters)
                observation = session.sandbox.observe()
            self._write_json(
                HTTPStatus.OK,
                {"mutation": mutation, "observation": observation},
            )
            return

        if len(segments) == 5 and segments[3] in {"faults", "inject"}:
            if method != "POST":
                raise APIError(405, "method_not_allowed", "method not allowed")
            fault_name = segments[4]
            fault = _FAULTS.get(fault_name)
            if fault is None:
                raise APIError(404, "unknown_fault", "fault is not allowlisted")
            payload = self._read_json_object()
            if payload:
                raise APIError(400, "invalid_request", "fault injection takes no parameters")
            with self.sandbox_server.sessions.lease(
                session_id, count_operation=True
            ) as session:
                fault.inject(session.sandbox)
                observation = session.sandbox.observe()
            self._write_json(
                HTTPStatus.OK,
                {"fault": fault_name, "observation": observation},
            )
            return

        raise APIError(404, "not_found", "endpoint not found")

    def _parameters(self) -> dict[str, Any]:
        payload = self._read_json_object()
        if "parameters" in payload:
            if set(payload) != {"parameters"}:
                raise APIError(400, "invalid_request", "only 'parameters' is allowed")
            parameters = payload["parameters"]
            if not isinstance(parameters, Mapping):
                raise APIError(400, "invalid_request", "parameters must be a JSON object")
            normalized = dict(parameters)
            self._validate_parameters(normalized)
            return normalized
        # Accepting a direct object keeps the tiny API convenient for manual use;
        # the bundled client always sends the explicit wrapper.
        self._validate_parameters(payload)
        return payload

    def _validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        if len(parameters) > self.max_parameter_count:
            raise APIError(
                400,
                "invalid_request",
                f"at most {self.max_parameter_count} parameters are allowed",
            )
        for name, value in parameters.items():
            if not isinstance(name, str) or not name:
                raise APIError(
                    400,
                    "invalid_request",
                    "parameter names must be non-empty strings",
                )
            if len(name) > self.max_parameter_name_chars:
                raise APIError(
                    400,
                    "invalid_request",
                    "parameter name is too long",
                )
            if isinstance(value, str):
                if len(value) > self.max_parameter_string_chars:
                    raise APIError(
                        400,
                        "invalid_request",
                        "parameter string value is too long",
                    )
                continue
            if value is None or isinstance(value, (bool, int)):
                continue
            if isinstance(value, float) and math.isfinite(value):
                continue
            raise APIError(
                400,
                "invalid_request",
                "parameter values must be finite JSON scalars",
            )

    def _read_json_object(self) -> dict[str, Any]:
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding:
            raise APIError(400, "invalid_request", "chunked request bodies are unsupported")
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise APIError(400, "invalid_request", "invalid Content-Length") from exc
        if length < 0:
            raise APIError(400, "invalid_request", "invalid Content-Length")
        if length > self.max_request_bytes:
            raise APIError(413, "request_too_large", "request body is too large")
        body = self.rfile.read(length)
        if len(body) != length:
            self.close_connection = True
            raise APIError(
                HTTPStatus.REQUEST_TIMEOUT,
                "request_timeout",
                "request body was not received before the timeout",
            )
        if not body:
            return {}
        media_type = self.headers.get_content_type()
        if media_type != "application/json":
            raise APIError(415, "unsupported_media_type", "Content-Type must be application/json")
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIError(400, "invalid_json", "request body must be valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise APIError(400, "invalid_request", "request body must be a JSON object")
        return value

    def _write_error(
        self,
        status: int,
        code: str,
        message: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._write_json(
            status,
            {"error": {"code": code, "message": message}},
            headers=headers,
        )

    def _write_json(
        self,
        status: int,
        payload: Any,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if headers:
            for name, value in headers.items():
                self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError):
            self.close_connection = True


class SandboxHTTPServer(ThreadingHTTPServer):
    """Bounded threading server carrying session and auth configuration."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        bearer_token: str | None = None,
        token: str | None = None,
        max_sessions: int = 128,
        session_ttl_seconds: float = 900.0,
        max_operations_per_session: int = 64,
        max_workers: int = 64,
        request_timeout_seconds: float = 10.0,
        sandbox_factory: Callable[[], SandboxBackend] = MockSandbox,
    ) -> None:
        if bearer_token is not None and token is not None and bearer_token != token:
            raise ValueError("bearer_token and token disagree")
        selected_token = bearer_token if bearer_token is not None else token
        if selected_token == "":
            selected_token = None
        if selected_token is not None and not isinstance(selected_token, str):
            raise TypeError("bearer token must be a string or None")
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, (int, float))
            or not math.isfinite(float(request_timeout_seconds))
            or request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be a positive finite number")
        self.bearer_token = selected_token
        self.max_workers = max_workers
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._worker_slots = threading.BoundedSemaphore(max_workers)
        self._worker_count_lock = threading.Lock()
        self._active_workers = 0
        self.sessions = SessionStore(
            max_sessions=max_sessions,
            session_ttl_seconds=session_ttl_seconds,
            max_operations_per_session=max_operations_per_session,
            sandbox_factory=sandbox_factory,
        )
        super().__init__(server_address, SandboxRequestHandler)

    @property
    def active_workers(self) -> int:
        """Return the current bounded request-worker count."""

        with self._worker_count_lock:
            return self._active_workers

    def process_request(
        self,
        request: socket.socket,
        client_address: tuple[str, int],
    ) -> None:
        """Start a request worker only when a bounded slot is available."""

        if not self._worker_slots.acquire(blocking=False):
            self._reject_worker_capacity(request)
            return
        counted = False
        try:
            request.settimeout(self.request_timeout_seconds)
            with self._worker_count_lock:
                self._active_workers += 1
                counted = True
            super().process_request(request, client_address)
        except Exception:
            with self._worker_count_lock:
                self._worker_slots.release()
                if counted:
                    self._active_workers -= 1
            raise

    def process_request_thread(
        self,
        request: socket.socket,
        client_address: tuple[str, int],
    ) -> None:
        """Release the bounded worker slot after each connection finishes."""

        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._worker_count_lock:
                # Keep slot release and the observable worker count ordered.
                # A caller that sees zero can therefore acquire a worker slot.
                self._worker_slots.release()
                self._active_workers -= 1

    def _reject_worker_capacity(self, request: socket.socket) -> None:
        # Read only the bounded request head before closing. On Windows, closing
        # a socket with unread request bytes can turn the intended 503 into a
        # connection reset at the client. This path is reached only after the
        # worker pool is full, so keep the drain short and strictly bounded.
        try:
            request.settimeout(min(self.request_timeout_seconds, 0.25))
            request_head = bytearray()
            while b"\r\n\r\n" not in request_head and len(request_head) < 8192:
                chunk = request.recv(min(1024, 8192 - len(request_head)))
                if not chunk:
                    break
                request_head.extend(chunk)
        except OSError:
            pass
        body = b'{"error":{"code":"server_busy","message":"server is busy"}}'
        response = (
            b"HTTP/1.1 503 Service Unavailable\r\n"
            b"Content-Type: application/json; charset=utf-8\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"Cache-Control: no-store\r\n"
            b"Connection: close\r\n\r\n"
            + body
        )
        try:
            request.sendall(response)
        except OSError:
            pass
        finally:
            self.shutdown_request(request)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface used by the container entry point."""

    parser = argparse.ArgumentParser(description="Run the safe CrashDiag mock sandbox API")
    parser.add_argument(
        "--host",
        default=os.environ.get("CRASHDIAG_SANDBOX_HOST", "127.0.0.1"),
        help="listen address (default: CRASHDIAG_SANDBOX_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CRASHDIAG_SANDBOX_PORT", "8765")),
        help="listen port (default: CRASHDIAG_SANDBOX_PORT or 8765)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CRASHDIAG_SANDBOX_TOKEN") or None,
        help="optional bearer token (prefer CRASHDIAG_SANDBOX_TOKEN)",
    )
    parser.add_argument(
        "--max-sessions",
        type=_positive_int,
        default=_positive_int(os.environ.get("CRASHDIAG_MAX_SESSIONS", "128")),
    )
    parser.add_argument(
        "--session-ttl",
        type=_positive_float,
        default=_positive_float(os.environ.get("CRASHDIAG_SESSION_TTL_SECONDS", "900")),
        help="idle session TTL in seconds",
    )
    parser.add_argument(
        "--max-operations-per-session",
        type=_positive_int,
        default=_positive_int(
            os.environ.get("CRASHDIAG_MAX_OPERATIONS_PER_SESSION", "64")
        ),
        help="state-changing operation budget for each session",
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=_positive_int(os.environ.get("CRASHDIAG_MAX_WORKERS", "64")),
        help="maximum concurrent request workers",
    )
    parser.add_argument(
        "--request-timeout",
        type=_positive_float,
        default=_positive_float(
            os.environ.get("CRASHDIAG_REQUEST_TIMEOUT_SECONDS", "10")
        ),
        help="per-connection socket timeout in seconds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the HTTP service until interrupted."""

    args = build_argument_parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        raise SystemExit("port must be between 0 and 65535")
    logging.basicConfig(level=os.environ.get("CRASHDIAG_LOG_LEVEL", "INFO"))
    server = SandboxHTTPServer(
        (args.host, args.port),
        bearer_token=args.token,
        max_sessions=args.max_sessions,
        session_ttl_seconds=args.session_ttl,
        max_operations_per_session=args.max_operations_per_session,
        max_workers=args.max_workers,
        request_timeout_seconds=args.request_timeout,
    )
    host, port = server.server_address[:2]
    LOGGER.info("safe mock sandbox listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("sandbox server interrupted")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SandboxHTTPServer",
    "SandboxRequestHandler",
    "SessionCapacityError",
    "SessionNotFound",
    "SessionOperationLimitError",
    "SessionStore",
    "build_argument_parser",
    "main",
]
