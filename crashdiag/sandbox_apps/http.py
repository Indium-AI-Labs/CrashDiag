"""Standard-library client for the remote CrashDiag mock sandbox service."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from types import TracebackType
from typing import Any, ClassVar
from urllib import error, parse, request

from .mock import SandboxBackend


class SandboxTransportError(RuntimeError):
    """The sandbox service could not be reached or returned invalid JSON."""


class SandboxHTTPError(RuntimeError):
    """A structured non-success response from the sandbox service."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"sandbox API returned {status} {code}: {message}")
        self.status = int(status)
        self.code = code
        self.message = message
        self.payload = dict(payload or {})


_NO_BODY = object()


class HttpSandbox(SandboxBackend):
    """A session-scoped :class:`SandboxBackend` over HTTP.

    Construction creates a fresh remote session unless ``session_id`` is supplied.
    Use the client as a context manager, or call :meth:`close`, to delete sessions
    promptly.  An attached session is not deleted by default; set
    ``delete_on_close=True`` to transfer that ownership explicitly.

    ``base_url`` is the service root, for example ``http://127.0.0.1:8765``.
    The client uses only :mod:`urllib` and has no third-party dependencies.
    """

    max_response_bytes: ClassVar[int] = 4 * 1024 * 1024

    def __init__(
        self,
        base_url: str,
        api_token: str | None = None,
        timeout: float = 10.0,
        *,
        token: str | None = None,
        session_id: str | None = None,
        delete_on_close: bool | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        normalized_url = base_url.strip().rstrip("/")
        parts = parse.urlsplit(normalized_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("base_url must be an absolute http:// or https:// URL")
        if parts.query or parts.fragment:
            raise ValueError("base_url must not contain a query string or fragment")
        if api_token is not None and token is not None and api_token != token:
            raise ValueError("api_token and token disagree")
        selected_token = api_token if api_token is not None else token
        if selected_token is not None and not isinstance(selected_token, str):
            raise TypeError("api_token must be a string or None")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ValueError("session_id must be a non-empty string or None")
        if delete_on_close is not None and not isinstance(delete_on_close, bool):
            raise TypeError("delete_on_close must be a boolean or None")

        self.base_url = normalized_url
        self.api_token = selected_token
        self.timeout = float(timeout)
        self.session_id: str | None = session_id
        self._closed = False
        self._owns_session = session_id is None
        self._delete_on_close = (
            self._owns_session if delete_on_close is None else delete_on_close
        )

        if self.session_id is None:
            payload = self._request("POST", "/v1/sessions", {})
            created_id = payload.get("session_id")
            if not isinstance(created_id, str) or not created_id:
                raise SandboxTransportError("create-session response has no session_id")
            self.session_id = created_id

    def __enter__(self) -> "HttpSandbox":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        """Whether this client has been closed."""

        return self._closed

    def close(self) -> None:
        """Close the client and delete its remote session when owned.

        Closing is idempotent.  A 404 during cleanup means the TTL or another
        worker already removed the session and is therefore treated as success.
        Other cleanup failures are surfaced while retaining the open state and
        session ID, allowing the caller to retry :meth:`close` safely.
        """

        if self._closed:
            return
        session_id = self.session_id
        if not self._delete_on_close or session_id is None:
            self._closed = True
            self.session_id = None
            return
        try:
            self._request("DELETE", self._session_path(session_id))
        except SandboxHTTPError as exc:
            if exc.status != 404:
                raise
        # Commit the local lifecycle transition only after successful cleanup
        # (or an idempotent 404).  On transport/HTTP failure the session ID and
        # open state remain available so callers can retry instead of leaking a
        # live session until its TTL expires.
        self._closed = True
        self.session_id = None

    def reset(self) -> dict[str, Any]:
        """Reset the remote session and return its fresh observation."""

        result = self._request("POST", f"{self._current_session_path()}/reset", {})
        observation = result.get("observation")
        if not isinstance(observation, dict):
            raise SandboxTransportError("reset response has no observation object")
        return observation

    def observe(self) -> dict[str, Any]:
        """Return the remote session's current state snapshot."""

        return self._request("GET", f"{self._current_session_path()}/observe")

    def health_check(self) -> dict[str, Any]:
        """Return the mechanically computed health result from remote state."""

        return self._request("GET", f"{self._current_session_path()}/health")

    def restart_app(self) -> dict[str, Any]:
        return self._action("restart_app")

    def rollback_env_var(self, name: str | None = None) -> dict[str, Any]:
        parameters: dict[str, Any] = {} if name is None else {"name": name}
        return self._action("rollback_env_var", parameters)

    def fix_dependency(
        self,
        name: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        if name is not None:
            parameters["name"] = name
        if version is not None:
            parameters["version"] = version
        return self._action("fix_dependency", parameters)

    def clear_disk(self, target_percent: float = 40.0) -> dict[str, Any]:
        return self._action("clear_disk", {"target_percent": target_percent})

    def fix_port_config(self, target_port: int | None = None) -> dict[str, Any]:
        parameters: dict[str, Any] = (
            {} if target_port is None else {"target_port": target_port}
        )
        return self._action("fix_port_config", parameters)

    def wait_and_observe(self) -> dict[str, Any]:
        return self._action("wait_and_observe")

    def trigger_oom_kill(self) -> None:
        self._mutation("trigger_oom_kill")

    def set_env_var(self, name: str, value: str) -> None:
        self._mutation("set_env_var", {"name": name, "value": value})

    def set_dependency_version(self, name: str, version: str) -> None:
        self._mutation("set_dependency_version", {"name": name, "version": version})

    def set_disk_usage(self, percent: float) -> None:
        self._mutation("set_disk_usage", {"percent": percent})

    def set_proxy_target_port(self, port: int) -> None:
        self._mutation("set_proxy_target_port", {"port": port})

    def set_expected_env_var(self, name: str, value: str) -> None:
        self._mutation("set_expected_env_var", {"name": name, "value": value})

    def set_required_dependency_version(self, name: str, version: str) -> None:
        self._mutation(
            "set_required_dependency_version",
            {"name": name, "version": version},
        )

    def set_app_port(self, port: int) -> None:
        self._mutation("set_app_port", {"port": port})

    def set_disk_health_threshold(self, percent: float) -> None:
        self._mutation("set_disk_health_threshold", {"percent": percent})

    def inject_fault(self, fault_name: str) -> dict[str, Any]:
        """Inject one server-allowlisted built-in fault and return the observation."""

        if not isinstance(fault_name, str) or not fault_name:
            raise ValueError("fault_name must be a non-empty string")
        encoded = parse.quote(fault_name, safe="")
        result = self._request(
            "POST", f"{self._current_session_path()}/faults/{encoded}", {}
        )
        observation = result.get("observation")
        if not isinstance(observation, dict):
            raise SandboxTransportError("fault response has no observation object")
        return observation

    def prepare_hard_scenario(
        self,
        fault_name: str,
        sample_seed: int,
        scenario_profile: str,
    ) -> dict[str, Any]:
        """Prepare schema-v2 state in one authenticated setup request.

        The policy action and post-action verification remain separate live
        requests. This only collapses latency-bound setup mutations.
        """

        try:
            result = self._request(
                "POST",
                f"{self._current_session_path()}/scenarios/hard",
                {
                    "fault_name": fault_name,
                    "sample_seed": sample_seed,
                    "scenario_profile": scenario_profile,
                },
            )
        except SandboxHTTPError as exc:
            if exc.status == 404:
                raise NotImplementedError(
                    "remote sandbox does not support atomic hard scenarios"
                ) from exc
            raise
        observation = result.get("observation")
        health = result.get("health")
        if not isinstance(observation, dict) or not isinstance(health, dict):
            raise SandboxTransportError(
                "hard-scenario response has no observation/health objects"
            )
        return result

    def service_health(self) -> dict[str, Any]:
        """Query unauthenticated service liveness (independent of session health)."""

        self._ensure_open()
        return self._request("GET", "/healthz")

    def _action(
        self,
        action: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        encoded = parse.quote(action, safe="")
        return self._request(
            "POST",
            f"{self._current_session_path()}/actions/{encoded}",
            {"parameters": dict(parameters or {})},
        )

    def _mutation(
        self,
        mutation: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        encoded = parse.quote(mutation, safe="")
        self._request(
            "POST",
            f"{self._current_session_path()}/mutations/{encoded}",
            {"parameters": dict(parameters or {})},
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("HttpSandbox is closed")

    @staticmethod
    def _session_path(session_id: str) -> str:
        return f"/v1/sessions/{parse.quote(session_id, safe='')}"

    def _current_session_path(self) -> str:
        self._ensure_open()
        if self.session_id is None:
            raise RuntimeError("HttpSandbox has no session")
        return self._session_path(self.session_id)

    def _url(self, path: str) -> str:
        # Permit a reverse-proxy path prefix in base_url.  Also tolerate callers
        # supplying a base that already ends in /v1.
        if self.base_url.endswith("/v1") and path.startswith("/v1/"):
            path = path[3:]
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        payload: object = _NO_BODY,
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "CrashDiag-HttpSandbox/1.0",
        }
        data: bytes | None = None
        if payload is not _NO_BODY:
            try:
                data = json.dumps(payload, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError, RecursionError) as exc:
                raise ValueError("request payload must be JSON-serializable") from exc
            headers["Content-Type"] = "application/json"
        if self.api_token is not None:
            headers["Authorization"] = f"Bearer {self.api_token}"
        http_request = request.Request(
            self._url(path), data=data, headers=headers, method=method
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
                if len(raw) > self.max_response_bytes:
                    raise SandboxTransportError("sandbox response is too large")
        except error.HTTPError as exc:
            try:
                raw = exc.read(self.max_response_bytes + 1)
            finally:
                exc.close()
            parsed = self._decode_object(raw, allow_empty=True)
            error_payload = parsed.get("error", {}) if isinstance(parsed, dict) else {}
            code = (
                error_payload.get("code", "http_error")
                if isinstance(error_payload, Mapping)
                else "http_error"
            )
            message = (
                error_payload.get("message", exc.reason)
                if isinstance(error_payload, Mapping)
                else exc.reason
            )
            raise SandboxHTTPError(
                exc.code,
                str(code),
                str(message),
                parsed if isinstance(parsed, Mapping) else None,
            ) from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise SandboxTransportError(f"sandbox request failed: {exc}") from exc
        return self._decode_object(raw)

    @staticmethod
    def _decode_object(raw: bytes, *, allow_empty: bool = False) -> dict[str, Any]:
        if not raw and allow_empty:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxTransportError("sandbox returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise SandboxTransportError("sandbox response must be a JSON object")
        return value


# Conventional spelling retained as an alias for callers that prefer all-caps HTTP.
HTTPSandbox = HttpSandbox


__all__ = [
    "HTTPSandbox",
    "HttpSandbox",
    "SandboxHTTPError",
    "SandboxTransportError",
]
