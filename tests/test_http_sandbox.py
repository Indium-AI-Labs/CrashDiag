"""Offline end-to-end tests for the stdlib HTTP sandbox service."""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest
from contextlib import contextmanager
from typing import Iterator
from urllib import error, request

from crashdiag.faults import BadEnvVar, DiskFull
from crashdiag.sandbox_apps.http import (
    HttpSandbox,
    SandboxHTTPError,
    SandboxTransportError,
)
from crashdiag.sandbox_server import SandboxHTTPServer
from training.generate_dataset import sample_seed
from training.generate_dataset import prepare_scenario
from training.common import observation_messages
from training.grpo import configure_reward_backend, mechanical_reward


@contextmanager
def running_server(
    *,
    token: str | None = "integration-secret",
    max_sessions: int = 8,
    ttl: float = 60.0,
    max_operations_per_session: int = 64,
    max_workers: int = 16,
    request_timeout_seconds: float = 2.0,
) -> Iterator[tuple[SandboxHTTPServer, str]]:
    server = SandboxHTTPServer(
        ("127.0.0.1", 0),
        bearer_token=token,
        max_sessions=max_sessions,
        session_ttl_seconds=ttl,
        max_operations_per_session=max_operations_per_session,
        max_workers=max_workers,
        request_timeout_seconds=request_timeout_seconds,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield server, f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class HttpSandboxIntegrationTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_reward_backend(sandbox_url=None, api_token=None)

    def test_fault_module_round_trip_and_session_isolation(self) -> None:
        with running_server() as (server, base_url):
            first = HttpSandbox(base_url, api_token="integration-secret")
            second = HttpSandbox(base_url, api_token="integration-secret")
            first_id = first.session_id
            try:
                self.assertTrue(first.health_check()["healthy"])
                self.assertTrue(second.health_check()["healthy"])

                fault = BadEnvVar()
                fault.inject(first)
                self.assertFalse(fault.is_resolved(first))
                self.assertFalse(first.health_check()["checks"]["environment"])
                self.assertTrue(second.health_check()["healthy"])

                action_result = first.execute_action(
                    "rollback_env_var", {"name": "APP_ENV"}
                )
                self.assertEqual(action_result["action"], "rollback_env_var")
                self.assertTrue(fault.is_resolved(first))
                with self.assertRaisesRegex(ValueError, "unsupported sandbox action"):
                    first.execute_action("run_shell", {"command": "whoami"})
            finally:
                first.close()
                second.close()

            self.assertEqual(server.sessions.stats()["active"], 0)
            assert first_id is not None
            attached = HttpSandbox(
                base_url,
                api_token="integration-secret",
                session_id=first_id,
            )
            try:
                with self.assertRaises(SandboxHTTPError) as caught:
                    attached.observe()
                self.assertEqual(caught.exception.status, 404)
                self.assertEqual(caught.exception.code, "session_not_found")
            finally:
                attached.close()

    def test_all_safe_mutations_actions_fault_endpoint_and_reset(self) -> None:
        with running_server() as (_, base_url):
            with HttpSandbox(base_url, api_token="integration-secret") as sandbox:
                sandbox.trigger_oom_kill()
                self.assertEqual(sandbox.observe()["process"]["last_exit_reason"], "OOMKilled")
                sandbox.restart_app()

                sandbox.set_env_var("APP_ENV", "broken")
                sandbox.rollback_env_var("APP_ENV")

                sandbox.set_dependency_version("web-framework", "0.0.1")
                sandbox.fix_dependency("web-framework")

                sandbox.set_disk_usage(99.0)
                sandbox.clear_disk(40.0)

                sandbox.set_proxy_target_port(8081)
                sandbox.fix_port_config()
                sandbox.set_expected_env_var("APP_ENV", "canary")
                sandbox.set_required_dependency_version("web-framework", "2.1.1")
                sandbox.set_app_port(8443)
                sandbox.set_disk_health_threshold(80.0)
                configured = sandbox.observe()
                self.assertEqual(configured["environment"]["expected"]["APP_ENV"], "canary")
                self.assertEqual(
                    configured["dependencies"]["required"]["web-framework"],
                    "2.1.1",
                )
                self.assertEqual(configured["network"]["app_port"], 8443)
                self.assertEqual(configured["disk"]["healthy_below_percent"], 80.0)
                self.assertTrue(sandbox.health_check()["healthy"])

                observation = sandbox.inject_fault("disk_full")
                self.assertGreaterEqual(
                    observation["disk"]["used_percent"],
                    observation["disk"]["healthy_below_percent"],
                )
                fault = DiskFull()
                self.assertFalse(fault.is_resolved(sandbox))
                sandbox.clear_disk()
                self.assertTrue(fault.is_resolved(sandbox))

                sandbox.set_env_var("APP_ENV", "broken-again")
                reset_observation = sandbox.reset()
                self.assertTrue(reset_observation["healthy"])
                self.assertEqual(reset_observation["clock_ticks"], 0)

                with self.assertRaises(SandboxHTTPError) as unknown_fault:
                    sandbox.inject_fault("format_host_disk")
                self.assertEqual(unknown_fault.exception.status, 404)
                self.assertEqual(unknown_fault.exception.code, "unknown_fault")

    def test_authentication_and_public_liveness(self) -> None:
        with running_server() as (_, base_url):
            with request.urlopen(f"{base_url}/healthz", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["service"], "crashdiag-sandbox")
            self.assertIn(2, payload["scenario_schema_versions"])

            with self.assertRaises(SandboxHTTPError) as missing:
                HttpSandbox(base_url)
            self.assertEqual(missing.exception.status, 401)
            self.assertEqual(missing.exception.code, "unauthorized")

            with self.assertRaises(SandboxHTTPError) as wrong:
                HttpSandbox(base_url, api_token="wrong")
            self.assertEqual(wrong.exception.status, 401)

            with HttpSandbox(base_url, api_token="integration-secret") as sandbox:
                self.assertEqual(sandbox.service_health()["status"], "ok")

        # The direct CLI/server API remains dependency-free and optionally
        # unauthenticated for loopback-only local development.
        with running_server(token=None) as (_, base_url):
            with HttpSandbox(base_url) as sandbox:
                self.assertTrue(sandbox.health_check()["healthy"])

    def test_capacity_then_idle_ttl_reclaims_session(self) -> None:
        with running_server(max_sessions=1, ttl=0.05) as (server, base_url):
            first = HttpSandbox(base_url, api_token="integration-secret")
            try:
                with self.assertRaises(SandboxHTTPError) as full:
                    HttpSandbox(base_url, api_token="integration-secret")
                self.assertEqual(full.exception.status, 429)
                self.assertEqual(full.exception.code, "session_capacity")

                time.sleep(0.12)
                second = HttpSandbox(base_url, api_token="integration-secret")
                try:
                    self.assertEqual(server.sessions.stats()["active"], 1)
                    with self.assertRaises(SandboxHTTPError) as expired:
                        first.observe()
                    self.assertEqual(expired.exception.status, 404)
                    self.assertEqual(expired.exception.code, "session_not_found")
                finally:
                    second.close()
            finally:
                # Expired-session cleanup is intentionally idempotent.
                first.close()

    def test_non_allowlisted_endpoint_and_invalid_parameters_are_rejected(self) -> None:
        with running_server() as (_, base_url):
            with HttpSandbox(base_url, api_token="integration-secret") as sandbox:
                assert sandbox.session_id is not None
                endpoint = (
                    f"{base_url}/v1/sessions/{sandbox.session_id}/mutations/run_shell"
                )
                raw_request = request.Request(
                    endpoint,
                    data=b'{"parameters":{"command":"whoami"}}',
                    method="POST",
                    headers={
                        "Authorization": "Bearer integration-secret",
                        "Content-Type": "application/json",
                    },
                )
                with self.assertRaises(error.HTTPError) as rejected:
                    request.urlopen(raw_request, timeout=2)
                self.assertEqual(rejected.exception.code, 404)
                try:
                    body = json.loads(rejected.exception.read().decode("utf-8"))
                finally:
                    rejected.exception.close()
                self.assertEqual(body["error"]["code"], "unknown_mutation")

                with self.assertRaises(SandboxHTTPError) as bad_parameters:
                    sandbox._request(  # exercise remote schema validation directly
                        "POST",
                        f"/v1/sessions/{sandbox.session_id}/actions/restart_app",
                        {"parameters": {"command": "whoami"}},
                    )
                self.assertEqual(bad_parameters.exception.status, 400)
                self.assertTrue(sandbox.health_check()["healthy"])

    def test_parameter_shape_and_string_size_are_bounded(self) -> None:
        with running_server() as (_, base_url):
            with HttpSandbox(base_url, api_token="integration-secret") as sandbox:
                assert sandbox.session_id is not None
                action_path = (
                    f"/v1/sessions/{sandbox.session_id}/actions/restart_app"
                )
                with self.assertRaises(SandboxHTTPError) as too_many:
                    sandbox._request(
                        "POST",
                        action_path,
                        {"parameters": {f"field_{index}": index for index in range(9)}},
                    )
                self.assertEqual(too_many.exception.status, 400)

                mutation_path = (
                    f"/v1/sessions/{sandbox.session_id}/mutations/set_env_var"
                )
                with self.assertRaises(SandboxHTTPError) as too_long:
                    sandbox._request(
                        "POST",
                        mutation_path,
                        {"parameters": {"name": "APP_ENV", "value": "x" * 1025}},
                    )
                self.assertEqual(too_long.exception.status, 400)

                with self.assertRaises(SandboxHTTPError) as nested:
                    sandbox._request(
                        "POST",
                        mutation_path,
                        {"parameters": {"name": "APP_ENV", "value": {"nested": True}}},
                    )
                self.assertEqual(nested.exception.status, 400)
                self.assertTrue(sandbox.health_check()["healthy"])

    def test_session_operation_budget_bounds_state_and_reset_reopens_budget(self) -> None:
        with running_server(max_operations_per_session=2) as (_, base_url):
            with HttpSandbox(base_url, api_token="integration-secret") as sandbox:
                sandbox.wait_and_observe()
                sandbox.wait_and_observe()
                with self.assertRaises(SandboxHTTPError) as exhausted:
                    sandbox.wait_and_observe()
                self.assertEqual(exhausted.exception.status, 429)
                self.assertEqual(
                    exhausted.exception.code,
                    "session_operation_limit",
                )
                self.assertTrue(sandbox.health_check()["healthy"])

                sandbox.reset()
                sandbox.wait_and_observe()

    def test_worker_bound_rejects_excess_and_slow_connection_times_out(self) -> None:
        with running_server(
            max_workers=1,
            request_timeout_seconds=1.0,
        ) as (server, base_url):
            host, port = server.server_address[:2]
            slow = socket.create_connection((host, port), timeout=1)
            try:
                slow.sendall(
                    b"POST /v1/sessions HTTP/1.1\r\n"
                    b"Host: localhost\r\n"
                    b"Authorization: Bearer integration-secret\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: 10\r\n\r\n"
                    b"{"
                )
                deadline = time.monotonic() + 2.5
                while server.active_workers != 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(server.active_workers, 1)

                with self.assertRaises(error.HTTPError) as busy:
                    request.urlopen(f"{base_url}/healthz", timeout=1)
                self.assertEqual(busy.exception.code, 503)
                try:
                    payload = json.loads(busy.exception.read().decode("utf-8"))
                finally:
                    busy.exception.close()
                self.assertEqual(payload["error"]["code"], "server_busy")

                deadline = time.monotonic() + 2.5
                while server.active_workers and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(server.active_workers, 0)
                with request.urlopen(f"{base_url}/healthz", timeout=1) as response:
                    self.assertEqual(response.status, 200)
            finally:
                slow.close()

    def test_close_can_retry_after_transient_delete_failure(self) -> None:
        with running_server() as (server, base_url):
            sandbox = HttpSandbox(base_url, api_token="integration-secret")
            session_id = sandbox.session_id
            original_request = sandbox._request
            failed_once = False

            def flaky_request(method: str, path: str, *args: object, **kwargs: object):
                nonlocal failed_once
                if method == "DELETE" and not failed_once:
                    failed_once = True
                    raise SandboxTransportError("simulated transient failure")
                return original_request(method, path, *args, **kwargs)

            sandbox._request = flaky_request  # type: ignore[method-assign]
            with self.assertRaises(SandboxTransportError):
                sandbox.close()
            self.assertFalse(sandbox.closed)
            self.assertEqual(sandbox.session_id, session_id)
            self.assertEqual(server.sessions.stats()["active"], 1)

            sandbox.close()
            self.assertTrue(sandbox.closed)
            self.assertIsNone(sandbox.session_id)
            self.assertEqual(server.sessions.stats()["active"], 0)

    def test_grpo_reward_replays_exact_scenario_through_http(self) -> None:
        with running_server() as (_, base_url):
            configure_reward_backend(
                sandbox_url=base_url,
                api_token="integration-secret",
            )
            seed = sample_seed(81, "port_proxy_misconfig", 11)
            _, local_sandbox, _ = prepare_scenario("port_proxy_misconfig", seed)
            prompt = observation_messages(local_sandbox.observe())
            rewards = mechanical_reward(
                ['{"action":"fix_port_config","parameters":{"target_port":8080}}'],
                fault_name=["port_proxy_misconfig"],
                sample_seed=[seed],
                prompts=[prompt],
            )

        self.assertEqual(rewards, [1.0])


if __name__ == "__main__":
    unittest.main()
