"""Offline tests for deterministic trainer and evaluation reports."""

from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as element_tree
from pathlib import Path

from training.reporting import (
    ReportingError,
    generate_evaluation_report,
    generate_trainer_report,
)


def _strict_json(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


class TrainerReportTests(unittest.TestCase):
    def test_sft_report_preserves_zeroes_deduplicates_and_escapes_svg(self) -> None:
        state = {
            "log_history": [
                {
                    "step": 1,
                    "epoch": 0.1,
                    "loss": 1.25,
                    "learning_rate": 0.0002,
                    "grad_norm": 0.0,
                    "ignored_text": "not numeric",
                    "ignored_nan": float("nan"),
                },
                {"step": 1, "eval_loss": 1.1},
                {
                    "step": 2,
                    "epoch": 0.2,
                    "loss": 0.8,
                    "learning_rate": 0.0001,
                    "grad_norm": 0.4,
                },
                {"step": 2, "eval_loss": 0.75},
                {"step": 2, "eval_loss": 0.7},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trainer_state.json"
            source.write_text(json.dumps(state), encoding="utf-8")
            original = source.read_bytes()

            bundle = generate_trainer_report(
                source,
                root / "reports",
                kind="sft",
                title="SFT <metrics> & diagnostics",
            )

            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(
                {path.name for path in bundle.charts},
                {"loss.svg", "learning_rate.svg", "gradient_norm.svg"},
            )
            self.assertTrue(all(path.is_file() and path.stat().st_size for path in bundle.files))
            summary = _strict_json(bundle.summary_path)
            self.assertEqual(summary["metrics"]["grad_norm"]["first"], 0.0)
            self.assertEqual(summary["metrics"]["eval_loss"]["points"], 2)
            self.assertEqual(summary["metrics"]["eval_loss"]["last"], 0.7)
            self.assertNotIn("ignored_text", summary["metrics"])
            self.assertNotIn("ignored_nan", summary["metrics"])
            history = _strict_json(bundle.metrics_path)
            self.assertTrue(history["records"])
            for chart in bundle.charts:
                element_tree.parse(chart)
            loss_svg = (root / "reports" / "loss.svg").read_text(encoding="utf-8")
            self.assertIn("SFT &lt;metrics&gt;", loss_svg)
            self.assertNotIn("SFT <metrics>", loss_svg)

    def test_grpo_report_discovers_reward_and_policy_metric_aliases(self) -> None:
        state = {
            "log_history": [
                {
                    "step": 1,
                    "loss": 0.2,
                    "learning_rate": 1e-5,
                    "rewards/mechanical_reward/mean": 0.25,
                    "reward_std": 0.1,
                    "crashdiag/success_rate": 0.25,
                    "entropy": 0.8,
                    "completion/mean_length": 42.0,
                    "frac_reward_zero_std": 0.5,
                    "crashdiag_backend_error": [False, True],
                },
                {
                    "step": 2,
                    "loss": 0.15,
                    "learning_rate": 8e-6,
                    "rewards/mechanical_reward/mean": 0.5,
                    "reward_std": 0.2,
                    "crashdiag/success_rate": 0.5,
                    "entropy": 0.7,
                    "completion/mean_length": 39.0,
                    "frac_reward_zero_std": 0.25,
                    "kl": float("inf"),
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trainer_state.json"
            source.write_text(json.dumps(state), encoding="utf-8")
            bundle = generate_trainer_report(
                source,
                root / "reports",
                kind="grpo",
            )

            names = {path.name for path in bundle.charts}
            self.assertTrue(
                {"loss.svg", "learning_rate.svg", "reward.svg", "policy_diagnostics.svg"}
                .issubset(names)
            )
            summary = _strict_json(bundle.summary_path)
            self.assertEqual(
                summary["metrics"]["crashdiag/success_rate"]["last"], 0.5
            )
            self.assertEqual(
                summary["metrics"]["rewards/mechanical_reward/mean"]["last"],
                0.5,
            )
            self.assertNotIn("crashdiag_backend_error", summary["metrics"])
            self.assertNotIn("kl", summary["metrics"])
            self.assertEqual(summary["scoring"], "diagnostic_trainer_metrics_only")
            for chart in bundle.charts:
                element_tree.parse(chart)

    def test_malformed_or_empty_trainer_history_fails_without_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, payload in enumerate(({}, {"log_history": []}, [])):
                source = root / f"state-{index}.json"
                source.write_text(json.dumps(payload), encoding="utf-8")
                output = root / f"report-{index}"
                with self.subTest(payload=payload):
                    with self.assertRaises(ReportingError):
                        generate_trainer_report(source, output, kind="sft")
                    self.assertFalse(output.exists())


class EvaluationReportTests(unittest.TestCase):
    def test_evaluation_report_visualizes_mechanical_rates_without_trajectories(self) -> None:
        evaluation = {
            "summary": {
                "total_episodes": 4,
                "resolved_episodes": 3,
                "success_rate": 0.75,
            },
            "per_fault": {
                "bad_env_var": {
                    "difficulty": "easy",
                    "episodes": 2,
                    "resolved": 2,
                    "success_rate": 1.0,
                },
                "disk_<full>&": {
                    "difficulty": "medium",
                    "episodes": 2,
                    "resolved": 1,
                    "success_rate": 0.5,
                },
            },
            "trajectories": [{"large": "must not be copied into summaries"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evaluation.json"
            source.write_text(json.dumps(evaluation), encoding="utf-8")
            bundle = generate_evaluation_report(source, root / "report")

            self.assertEqual(bundle.report_type, "mechanical_evaluation")
            self.assertEqual(len(bundle.charts), 1)
            element_tree.parse(bundle.charts[0])
            summary = _strict_json(bundle.summary_path)
            self.assertEqual(summary["overall_success_rate"], 0.75)
            self.assertEqual(
                summary["per_fault"]["disk_<full>&"]["success_rate"], 0.5
            )
            self.assertNotIn("trajectories", summary)
            svg = bundle.charts[0].read_text(encoding="utf-8")
            self.assertIn("disk_&lt;full&gt;&amp;", svg)
            self.assertNotIn("disk_<full>&", svg)
            self.assertIn(
                "executable sandbox state",
                bundle.markdown_path.read_text(encoding="utf-8"),
            )

    def test_invalid_evaluation_rate_is_rejected(self) -> None:
        evaluation = {
            "summary": {"success_rate": 1.0},
            "per_fault": {
                "disk_full": {
                    "episodes": 1,
                    "resolved": 1,
                    "success_rate": 1.5,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evaluation.json"
            source.write_text(json.dumps(evaluation), encoding="utf-8")
            output = root / "report"
            with self.assertRaisesRegex(ReportingError, "disk_full"):
                generate_evaluation_report(source, output)
            self.assertFalse(output.exists())


class ReportIntegrationWiringTests(unittest.TestCase):
    def test_reports_are_created_before_each_artifact_stage_is_finalized(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sft = (root / "training" / "sft.py").read_text(encoding="utf-8")
        grpo = (root / "training" / "grpo.py").read_text(encoding="utf-8")
        evaluate = (root / "training" / "evaluate.py").read_text(encoding="utf-8")

        self.assertLess(
            sft.index("report_bundle = generate_trainer_report("),
            sft.index("uploader.upload_directory("),
        )
        self.assertLess(
            grpo.index("report_bundle = generate_trainer_report("),
            grpo.index("uploader.upload_directory("),
        )
        self.assertLess(
            evaluate.index("report_bundle = generate_evaluation_report("),
            evaluate.index("uploader.upload_files("),
        )
        self.assertIn("[output_path, *report_bundle.files]", evaluate)


if __name__ == "__main__":
    unittest.main()
