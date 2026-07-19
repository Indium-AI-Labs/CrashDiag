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
        cls.sft = _source(sft_notebook)
        cls.grpo = _source(grpo_notebook)
        cls.sft_code = _code_source(sft_notebook)
        cls.grpo_code = _code_source(grpo_notebook)

    def test_sft_notebook_is_a_complete_independent_producer(self) -> None:
        required = (
            "git\", \"clone",
            "pip\", \"install",
            "required_secret(\"HF_TOKEN\")",
            "ArtifactUploader",
            "uploader.start_run",
            "generate_dataset_main",
            "sft_main",
            "data/grpo_train.jsonl",
            "outputs/sft",
            "sft/_SUCCESS.json",
            "crashdiag_handoff.txt",
            "SOURCE_COMMIT",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.sft)
        self.assertNotIn("CRASHDIAG_SANDBOX_TOKEN", self.sft)

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
            "--eval-file\", \"\"",
            "GRPO_STAGE = \"grpo-smoke\" if MODE == \"smoke\" else \"grpo\"",
            "evaluate_main",
            "uploader.complete_run",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.grpo)

    def test_inter_notebook_contract_is_bucket_and_run_id_not_local_state(self) -> None:
        for source in (self.sft, self.grpo):
            self.assertIn('BUCKET_ID = "devaanshpa/CrashDiag"', source)
            self.assertIn('REPO_DIR = Path("/kaggle/working/CrashDiag")', source)
            self.assertIn("CRASHDIAG_ARTIFACT_UPLOAD_POLICY", source)
        self.assertIn("RUN_ID=", self.sft)
        self.assertIn("Set RUN_ID to the exact value", self.grpo)
        self.assertNotIn("outputs/sft\"),\n    \"--train-file", self.grpo)

    def test_notebook_cli_flags_match_current_backend_parsers(self) -> None:
        from training.evaluate import build_parser as evaluation_parser
        from training.generate_dataset import build_parser as dataset_parser
        from training.grpo import build_parser as grpo_parser
        from training.sft import build_parser as sft_parser

        workflows = (
            (
                "dataset",
                _literal_flags(self.sft_code, call_name="generate_dataset_main"),
                dataset_parser(),
            ),
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
        )
        for workflow, flags, parser in workflows:
            with self.subTest(workflow=workflow):
                self.assertTrue(flags)
                unknown = flags.difference(parser._option_string_actions)
                self.assertEqual(unknown, set())


if __name__ == "__main__":
    unittest.main()
