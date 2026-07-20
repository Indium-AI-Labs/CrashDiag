"""Static safety and independence checks for the Kaggle notebooks."""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "sft": ROOT / "notebooks" / "sft.ipynb",
    "grpo": ROOT / "notebooks" / "grpo.ipynb",
    "grpo_hard": ROOT / "notebooks" / "grpo_hard.ipynb",
    "eval": ROOT / "notebooks" / "eval.ipynb",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(notebook: dict[str, Any]) -> str:
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def _code_source(notebook: dict[str, Any]) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _literal_flags(
    source: str,
    *,
    call_name: str | None = None,
    list_name: str | None = None,
) -> set[str]:
    """Return literal CLI flags passed by a notebook workflow."""

    flags: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        values: list[ast.expr] | None = None
        if call_name and isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == call_name:
                if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                    values = list(node.args[0].elts)
        if list_name and isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == list_name
                for target in node.targets
            ) and isinstance(node.value, (ast.List, ast.Tuple)):
                values = list(node.value.elts)
        if list_name and isinstance(node, ast.Call):
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "extend"
                and isinstance(function.value, ast.Name)
                and function.value.id == list_name
                and node.args
                and isinstance(node.args[0], (ast.List, ast.Tuple))
            ):
                values = list(node.args[0].elts)
        for value in values or []:
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and value.value.startswith("--")
            ):
                flags.add(value.value)
    return flags


class NotebookStructureTests(unittest.TestCase):
    def test_notebooks_are_valid_clean_and_all_code_cells_compile(self) -> None:
        for workflow, path in NOTEBOOKS.items():
            with self.subTest(workflow=workflow):
                notebook = _load(path)
                self.assertEqual(notebook["nbformat"], 4)
                self.assertLessEqual(notebook["nbformat_minor"], 4)
                self.assertEqual(
                    notebook["metadata"]["crashdiag"]["workflow"], workflow
                )
                self.assertTrue(
                    notebook["metadata"]["crashdiag"][
                        "independent_kaggle_entrypoint"
                    ]
                )
                self.assertGreater(len(notebook["cells"]), 5)
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs"), [])
                    compile(
                        "".join(cell["source"]),
                        f"{path.name}:cell-{index}",
                        "exec",
                    )

    def test_notebooks_contain_no_embedded_credentials_or_secret_cli_flags(self) -> None:
        for workflow, path in NOTEBOOKS.items():
            with self.subTest(workflow=workflow):
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{12,}", text))
                self.assertNotIn("--hf-token", text.lower())
                self.assertNotIn("--sandbox-token", text.lower())
                self.assertNotIn("source .env", text.lower())
                self.assertNotIn(". ./.env", text.lower())


class NotebookWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sft_notebook = _load(NOTEBOOKS["sft"])
        grpo_notebook = _load(NOTEBOOKS["grpo"])
        eval_notebook = _load(NOTEBOOKS["eval"])
        hard_notebook = _load(NOTEBOOKS["grpo_hard"])
        cls.sft = _source(sft_notebook)
        cls.grpo = _source(grpo_notebook)
        cls.eval = _source(eval_notebook)
        cls.grpo_hard = _source(hard_notebook)
        cls.sft_code = _code_source(sft_notebook)
        cls.grpo_code = _code_source(grpo_notebook)
        cls.eval_code = _code_source(eval_notebook)
        cls.grpo_hard_code = _code_source(hard_notebook)

    def test_hard_grpo_notebook_enforces_calibration_smoke_and_promotion(self) -> None:
        required = (
            "PASTE_HARD_GRPO_RUN_ID_HERE",
            "PASTE_TRAINER_COMMIT_HERE",
            "training.generate_grpo_hard",
            'required_secret("HF_TOKEN")',
            'required_secret("CRASHDIAG_SANDBOX_TOKEN")',
            "download_stage(stage, target, include_paths=paths)",
            '"grpo_hard_train.jsonl"',
            '"grpo_hard_eval.jsonl"',
            '"parent_sft.json"',
            "read_parent_reference",
            'service.get("scenario_schema_versions"',
            'service.get("hard_scenario_batch") is not True',
            "calibrate_main",
            'CALIBRATION_STAGE = "calibration-wide-v1"',
            "uploader.stage_is_complete(CALIBRATION_STAGE)",
            '"1.8", "2.1", "2.4"',
            '"--top-p", "1.0"',
            '"--top-k", "0"',
            '"--num-generations", str(NUM_GENERATIONS)',
            '"--reward-workers", "8"',
            '"--require-nonzero-update"',
            '"--minimum-gate-steps", str(SMOKE_STEPS)',
            'smoke_gate.get("passed") is not True',
            '"--model", str(PARENT_SFT_DIR)',
            "evaluate_jsonl_main",
            '"hard-evaluation"',
            '"regression-evaluation"',
            "promotion_gate(",
            "require_passed(promotion",
            "uploader.complete_run",
            "display(SVG(filename=str(chart)))",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.grpo_hard)
        self.assertLess(self.grpo_hard.index("calibrate_main(["), self.grpo_hard.index("grpo_main(smoke_args)"))
        self.assertLess(self.grpo_hard.index("grpo_main(smoke_args)"), self.grpo_hard.index("grpo_main(full_args)"))
        self.assertLess(self.grpo_hard.index("promotion_gate("), self.grpo_hard.index("uploader.complete_run"))
        self.assertEqual(self.grpo_hard.count('"--model", str(PARENT_SFT_DIR)'), 3)

    def test_sft_notebook_downloads_verified_bucket_data_before_training(self) -> None:
        required = (
            "git\", \"clone",
            "pip\", \"install",
            "PASTE_DATASET_RUN_ID_HERE",
            "PASTE_DATASET_SOURCE_COMMIT_HERE",
            "required_secret(\"HF_TOKEN\")",
            "ArtifactUploader",
            "download_run(RUN_DIR, allow_incomplete=True)",
            "verify_local_run(RUN_DIR, require_complete=False)",
            "DATASET_DIR / \"manifest.json\"",
            "DATASET_DIR / \"_SUCCESS.json\"",
            "artifact_commit != CURRENT_COMMIT",
            "sft_main",
            '\"--dataset\", str(DATASET_DIR / \"sft_train.jsonl\")',
            '\"--eval-dataset\", str(DATASET_DIR / \"sft_eval.jsonl\")',
            "outputs/sft",
            "from IPython.display import SVG, display",
            "SFT_REPORT_DIR",
            "metrics_summary.json",
            "display(SVG(filename=str(chart)))",
            "expected_reports - manifest_paths",
            'functools.partial(trl.SFTConfig, loss_type="nll")',
            "sft/_SUCCESS.json",
            "crashdiag_handoff.txt",
            "SOURCE_COMMIT",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.sft)
        self.assertNotIn("CRASHDIAG_SANDBOX_TOKEN", self.sft)
        self.assertNotIn("generate_dataset_main", self.sft)
        self.assertNotIn("uploader.start_run", self.sft)
        self.assertNotIn('"data/sft_train.jsonl"', self.sft)
        self.assertLess(self.sft.index("download_run"), self.sft.index("sft_main"))
        self.assertLess(
            self.sft.index('functools.partial(trl.SFTConfig, loss_type="nll")'),
            self.sft.index("sft_main(["),
        )
        self.assertLess(
            self.sft.index("sft_main(["),
            self.sft.index("display(SVG(filename=str(chart)))"),
        )

    def test_grpo_notebook_restores_verified_sft_and_uses_real_sandbox(self) -> None:
        required = (
            "PASTE_SFT_RUN_ID_HERE",
            "PASTE_SFT_SOURCE_COMMIT_HERE",
            "https://sandbox.devaanshpathak.com",
            "required_secret(\"HF_TOKEN\")",
            "required_secret(\"CRASHDIAG_SANDBOX_TOKEN\")",
            "download_run(RUN_DIR, allow_incomplete=True)",
            "verify_local_run(RUN_DIR, require_complete=False)",
            "artifact_commit != CURRENT_COMMIT",
            '"--model", str(SFT_DIR)',
            "SFT_DIR / \"adapter_config.json\"",
            "SFT_DIR / \"_SUCCESS.json\"",
            "DATASET_DIR / \"grpo_train.jsonl\"",
            "HttpSandbox",
            "grpo_main",
            "from IPython.display import SVG, display",
            "GRPO_REPORT_DIR",
            "metrics_summary.json",
            "display(SVG(filename=str(chart)))",
            "expected - manifest_paths",
            "--eval-file\", \"\"",
            "GRPO_STAGE = \"grpo-smoke\" if MODE == \"smoke\" else \"grpo\"",
            "evaluate_main",
            "outputs/evaluation-report",
            "mechanical_evaluation_summary.json",
            "uploader.complete_run",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.grpo)
        self.assertLess(
            self.grpo.index("grpo_main(grpo_args)"),
            self.grpo.index("display_signed_report(\n    GRPO_REPORT_DIR"),
        )
        self.assertLess(
            self.grpo.index("evaluate_main(["),
            self.grpo.index('REPO_DIR / "outputs/evaluation-report"'),
        )

    def test_eval_notebook_scores_exact_answer_free_dataset_and_uploads(self) -> None:
        required = (
            'BUCKET_ID = "devaanshpa/CrashDiag"',
            'SANDBOX_URL = "https://sandbox.devaanshpathak.com"',
            'EVALUATION_STAGE = "sft-eval"',
            'DATASET_DIR / "grpo_eval.jsonl"',
            'SFT_DIR / "adapter_model.safetensors"',
            'required_secret("HF_TOKEN")',
            'required_secret("CRASHDIAG_SANDBOX_TOKEN")',
            'download_run(RUN_DIR, allow_incomplete=True)',
            'verify_local_run(RUN_DIR, require_complete=False)',
            'artifact_commit != CURRENT_COMMIT',
            'AutoPeftModelForCausalLM.from_pretrained(',
            'configure_reward_backend(',
            'mechanical_reward(',
            'prompts=[row["prompt"]]',
            'sample_seed=[row["sample_seed"]]',
            '"backend_error": bool(extra["crashdiag_backend_error"][0])',
            '"resolved": bool(extra["crashdiag_resolved"][0])',
            'generate_evaluation_report(',
            'uploader.upload_files(',
            'display(SVG(filename=str(chart)))',
            'manifest_sha256',
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.eval)
        self.assertNotIn("grpo_main", self.eval)
        self.assertNotIn("sft_main", self.eval)
        self.assertNotIn("uploader.complete_run", self.eval)
        self.assertLess(
            self.eval.index('EVAL_FILE = DATASET_DIR / "grpo_eval.jsonl"'),
            self.eval.index("mechanical_reward("),
        )
        self.assertLess(
            self.eval.index("mechanical_reward("),
            self.eval.index("uploader.upload_files("),
        )

    def test_inter_notebook_contract_is_bucket_and_run_id_not_local_state(self) -> None:
        for source in (self.sft, self.grpo, self.eval):
            self.assertIn('BUCKET_ID = "devaanshpa/CrashDiag"', source)
            self.assertIn('REPO_DIR = Path("/kaggle/working/CrashDiag")', source)
            self.assertIn("CRASHDIAG_ARTIFACT_UPLOAD_POLICY", source)
        self.assertIn("RUN_ID=", self.sft)
        self.assertIn("Set RUN_ID to the exact value", self.grpo)
        self.assertNotIn("outputs/sft\"),\n    \"--train-file", self.grpo)

    def test_notebooks_remove_unused_torchao_before_training_install(self) -> None:
        for workflow, source in (
            ("sft", self.sft),
            ("grpo", self.grpo),
            ("eval", self.eval),
        ):
            with self.subTest(workflow=workflow):
                probe = '[sys.executable, "-m", "pip", "show", "torchao"]'
                uninstall = (
                    '[sys.executable, "-m", "pip", "uninstall", "-y", "torchao"]'
                )
                install = (
                    '[sys.executable, "-m", "pip", "install", "-e", '
                    'f"{REPO_DIR}[train]"]'
                )
                self.assertIn("CrashDiag does not use it", source)
                self.assertIn(probe, source)
                self.assertIn(uninstall, source)
                self.assertIn(f"subprocess.run({uninstall}, check=True)", source)
                self.assertIn(install, source)
                self.assertLess(source.index(probe), source.index(uninstall))
                self.assertLess(source.index(uninstall), source.index(install))

    def test_notebook_cli_flags_match_current_backend_parsers(self) -> None:
        from training.calibrate_grpo import build_parser as calibration_parser
        from training.evaluate import build_parser as evaluation_parser
        from training.evaluate_jsonl import build_parser as jsonl_evaluation_parser
        from training.grpo import build_parser as grpo_parser
        from training.sft import build_parser as sft_parser

        workflows = (
            (
                "sft",
                _literal_flags(self.sft_code, call_name="sft_main"),
                sft_parser(),
            ),
            (
                "grpo",
                _literal_flags(self.grpo_code, list_name="grpo_args"),
                grpo_parser(),
            ),
            (
                "evaluation",
                _literal_flags(self.grpo_code, call_name="evaluate_main"),
                evaluation_parser(),
            ),
            (
                "hard_calibration",
                _literal_flags(self.grpo_hard_code, call_name="calibrate_main"),
                calibration_parser(),
            ),
            (
                "hard_smoke",
                _literal_flags(self.grpo_hard_code, list_name="smoke_args"),
                grpo_parser(),
            ),
            (
                "hard_full",
                _literal_flags(self.grpo_hard_code, list_name="full_args"),
                grpo_parser(),
            ),
            (
                "hard_jsonl_evaluation",
                _literal_flags(self.grpo_hard_code, call_name="evaluate_jsonl_main"),
                jsonl_evaluation_parser(),
            ),
        )
        for workflow, flags, parser in workflows:
            with self.subTest(workflow=workflow):
                self.assertTrue(flags)
                unknown = flags.difference(parser._option_string_actions)
                self.assertEqual(unknown, set())


if __name__ == "__main__":
    unittest.main()
