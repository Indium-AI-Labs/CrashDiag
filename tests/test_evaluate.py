from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from crashdiag.faults.modules import ALL_FAULTS
from crashdiag.faults.workflows import WORKFLOWS
from crashdiag.sandbox_apps.mock import MockSandbox
from training.evaluate import (
    LocalTransformersAgent,
    build_parser,
    format_report,
    run_evaluation,
    sandbox_factory,
    save_report,
)


class _FaultAwareAgent:
    ACTIONS = {
        "process": "restart_app",
        "environment": "rollback_env_var",
        "dependencies": "fix_dependency",
        "disk": "clear_disk",
        "port_proxy": "fix_port_config",
        "cache": "clear_cache",
        "tls": "renew_tls_certificate",
        "permissions": "restore_file_permissions",
        "migration": "apply_database_migration",
        "db_pool": "reset_database_pool",
        "dns": "restore_dns_configuration",
        "rate_limit": "restore_rate_limit_configuration",
        "worker": "restart_worker",
        "container": "redeploy_container",
        "temp": "clear_temp_files",
        "logs": "rotate_logs",
        "lb_config": "restore_load_balancer_config",
        "network_config": "restore_network_config",
        "replica": "sync_replica",
        "db_config": "restore_database_config",
        "dead_letter": "flush_dead_letter_queue",
        "cache_config": "restore_cache_config",
        "circuit_breaker": "reset_circuit_breaker",
        "cron": "restore_cron_schedule",
        "search_index": "rebuild_index",
        "tls_config": "restore_tls_config",
    }

    def __init__(self) -> None:
        self.calls = 0

    def choose_action(self, observation: dict[str, Any], history: Any = None) -> dict[str, Any]:
        del history
        self.calls += 1
        failure = observation["health"]["failures"][0]
        return {"action": self.ACTIONS[failure], "parameters": {}}


class _WaitAgent:
    def choose_action(self, observation: Any, history: Any = None) -> dict[str, Any]:
        del observation, history
        return {"action": "wait_and_observe", "parameters": {}}


class _ClosableMockSandbox(MockSandbox):
    closed_count = 0

    def close(self) -> None:
        type(self).closed_count += 1


class _FakeBatch(dict[str, Any]):
    def to(self, device: Any) -> "_FakeBatch":
        self["moved_to"] = device
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.template_calls = 0
        self.decode_calls = 0

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.template_calls += 1
        self.messages = messages
        self.template_kwargs = kwargs
        return "rendered chat"

    def __call__(self, prompt: str, **kwargs: Any) -> _FakeBatch:
        self.prompt = prompt
        self.tokenizer_kwargs = kwargs
        return _FakeBatch(input_ids=[[10, 11]])

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        self.decode_calls += 1
        self.decoded_tokens = list(tokens)
        self.decode_kwargs = kwargs
        return self.completion


class _FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_calls += 1
        self.kwargs = kwargs
        return [[10, 11, 20, 21]]


class EvaluationTests(unittest.TestCase):
    def test_all_faults_are_scored_mechanically_for_each_episode(self) -> None:
        agent = _FaultAwareAgent()
        _ClosableMockSandbox.closed_count = 0

        report = run_evaluation(
            agent,
            episodes_per_fault=2,
            make_sandbox=_ClosableMockSandbox,
        )

        total_workflows = len(WORKFLOWS)
        self.assertEqual(report["summary"]["total_episodes"], total_workflows * 2)
        self.assertEqual(report["summary"]["resolved_episodes"], total_workflows * 2)
        self.assertEqual(report["summary"]["success_rate"], 1.0)
        self.assertEqual(_ClosableMockSandbox.closed_count, total_workflows * 2)
        self.assertEqual(set(report["per_fault"]), set(WORKFLOWS))
        for metrics in report["per_fault"].values():
            self.assertEqual(metrics["episodes"], 2)
            self.assertEqual(metrics["resolved"], 2)
            self.assertEqual(metrics["success_rate"], 1.0)
        for trajectory in report["trajectories"]:
            self.assertTrue(trajectory["resolved"])
            self.assertGreaterEqual(len(trajectory["steps"]), 1)
            self.assertGreaterEqual(trajectory["metadata"]["action_limit"], 1)
            self.assertTrue(trajectory["metadata"]["scenario_prepared"])
        for fault_name in WORKFLOWS:
            seeds = {
                item["metadata"]["sample_seed"]
                for item in report["trajectories"]
                if item["fault_name"] == fault_name
            }
            self.assertEqual(len(seeds), 2)

    def test_wait_policy_is_not_graded_as_success_and_report_round_trips(self) -> None:
        report = run_evaluation(_WaitAgent())
        total_workflows = len(WORKFLOWS)
        self.assertEqual(report["summary"]["total_episodes"], total_workflows)
        self.assertEqual(report["summary"]["resolved_episodes"], 0)
        self.assertEqual(report["summary"]["success_rate"], 0.0)
        self.assertTrue(all(not item["resolved"] for item in report["trajectories"]))

        with tempfile.TemporaryDirectory() as directory:
            output = save_report(report, Path(directory) / "nested" / "evaluation.json")
            loaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(loaded, report)
        self.assertIn(f"overall: 0/{total_workflows} (0.0%)", format_report(report))

    def test_local_agent_generates_once_and_defensively_parses_json(self) -> None:
        tokenizer = _FakeTokenizer('{"actions":[{"action":"clear_disk","parameters":{}}]}')
        model = _FakeModel()
        agent = LocalTransformersAgent(
            model,
            tokenizer,
            max_new_tokens=17,
            temperature=0.0,
        )

        action = agent.choose_action({"disk": {"used_percent": 99.0}})

        self.assertEqual(action, {"actions": [{"action": "clear_disk", "parameters": {}}]})
        self.assertEqual(model.generate_calls, 1)
        self.assertEqual(tokenizer.template_calls, 1)
        self.assertEqual(tokenizer.decode_calls, 1)
        self.assertEqual(tokenizer.decoded_tokens, [20, 21])
        self.assertEqual(model.kwargs["max_new_tokens"], 17)
        self.assertFalse(model.kwargs["do_sample"])
        self.assertNotIn("temperature", model.kwargs)

    def test_local_agent_falls_back_to_wait_on_malformed_output(self) -> None:
        tokenizer = _FakeTokenizer("I would restart the app")
        model = _FakeModel()

        action = LocalTransformersAgent(model, tokenizer).choose_action({})

        self.assertEqual(action, {"actions": [{"action": "wait_and_observe", "parameters": {}}]})
        self.assertEqual(model.generate_calls, 1)

    def test_parser_exposes_endpoint_and_remote_sandbox_options(self) -> None:
        args = build_parser().parse_args(
            [
                "--model",
                "served-model",
                "--base-url",
                "http://model:8000/v1",
                "--sandbox-url",
                "http://sandbox:8765",
                "--episodes-per-fault",
                "3",
            ]
        )
        self.assertEqual(args.model, "served-model")
        self.assertEqual(args.base_url, "http://model:8000/v1")
        self.assertEqual(args.sandbox_url, "http://sandbox:8765")
        self.assertEqual(args.episodes_per_fault, 3)

    def test_local_sandbox_factory_has_no_network_dependency(self) -> None:
        create = sandbox_factory(sandbox_url=None)
        first = create()
        second = create()
        self.assertIsInstance(first, MockSandbox)
        self.assertIsInstance(second, MockSandbox)
        self.assertIsNot(first, second)

    def test_invalid_episode_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_evaluation(_WaitAgent(), episodes_per_fault=0)


if __name__ == "__main__":
    unittest.main()
