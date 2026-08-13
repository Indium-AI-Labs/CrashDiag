"""Static safety checks for the multi-model Qwen2.5 notebook workflow."""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
PER_MODEL = {
    "qwen2.5_14b",
    "qwen2.5_7b",
    "qwen2.5_3b",
    "qwen2.5_1.5b",
    "qwen2.5_0.5b",
}
NOTEBOOK_NAMES = {
    "sft.ipynb",
    "eval_sft.ipynb",
    "grpo.ipynb",
    "eval_grpo.ipynb",
}
TOP_LEVEL = {"eval_all_baselines.ipynb"}


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


class ModelNotebookTests(unittest.TestCase):
    def test_only_expected_workflow_dirs_present(self) -> None:
        entries = {path.name for path in NOTEBOOK_ROOT.iterdir()}
        self.assertEqual(entries, PER_MODEL | TOP_LEVEL)
        for slug in PER_MODEL:
            self.assertEqual(
                {path.name for path in (NOTEBOOK_ROOT / slug).iterdir()},
                NOTEBOOK_NAMES,
            )

    def test_notebooks_are_safe_valid_python(self) -> None:
        for path in NOTEBOOK_ROOT.rglob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                source = "".join(cell.get("source", []))
                self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{12,}", source))
                if cell.get("cell_type") == "code":
                    ast.parse(source, f"{path.name}:cell-{index}", "exec")

    def test_per_model_notebooks_use_their_model_and_bucket(self) -> None:
        for slug in PER_MODEL:
            model_dir = NOTEBOOK_ROOT / slug
            expected_base = {
                "qwen2.5_14b": "Qwen/Qwen2.5-14B-Instruct",
                "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
                "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
                "qwen2.5_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
                "qwen2.5_0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
            }[slug]
            for name in NOTEBOOK_NAMES:
                source = _source(model_dir / name)
                self.assertIn(f'BASE_MODEL = "{expected_base}"', source)
                self.assertIn(f'MODEL_SLUG = "{slug}"', source)
                self.assertIn('BUCKET_ID = "devaanshpa/CrashDiag"', source)
                self.assertIn("CRASHDIAG_DATASET_RUN_ID", source)
                self.assertIn("CRASHDIAG_ENV_FILE", source)
                if name in {"eval_sft.ipynb", "grpo.ipynb", "eval_grpo.ipynb"}:
                    self.assertIn("CRASHDIAG_CURRICULUM", source)
                self.assertIn("display(SVG", source)
                self.assertIn('"reports"', source)

    def test_eval_sft_and_eval_grpo_use_kaggle_secrets(self) -> None:
        for slug in PER_MODEL:
            for name in ("eval_sft.ipynb", "eval_grpo.ipynb"):
                source = _source(NOTEBOOK_ROOT / slug / name)
                self.assertIn("UserSecretsClient", source)
                self.assertIn("kaggle_secrets", source)
                self.assertIn("loaded Kaggle secret names", source)
            for name in ("sft.ipynb", "grpo.ipynb"):
                source = _source(NOTEBOOK_ROOT / slug / name)
                self.assertNotIn("UserSecretsClient", source)
                self.assertNotIn("kaggle_secrets", source)

    def test_eval_sft_requires_sft_run_id(self) -> None:
        for slug in PER_MODEL:
            source = _source(NOTEBOOK_ROOT / slug / "eval_sft.ipynb")
            self.assertIn('"SFT_RUN_ID": "CRASHDIAG_SFT_RUN_ID"', source)
            self.assertIn("kaggle_secret_errors", source)
            self.assertIn('LAUNCH_DIR / "env.txt"', source)

    def test_grpo_trains_and_evaluates_hard_v3_by_default(self) -> None:
        for slug in PER_MODEL:
            grpo = _source(NOTEBOOK_ROOT / slug / "grpo.ipynb")
            self.assertIn('"--train-file", str(DATASET_DIR / TRAIN_FILE)', grpo)
            self.assertIn('"--eval-file", str(DATASET_DIR / EVAL_FILE)', grpo)
            self.assertIn('"--max-steps", "24"', grpo)
            self.assertIn('"--num-generations", "2"', grpo)
            self.assertIn('"96"', grpo)
            self.assertIn("grpo-smoke", grpo)
            self.assertIn('TRAIN_FILE = "grpo_hard_train.jsonl"', grpo)

    def test_eval_all_baselines_covers_all_models(self) -> None:
        source = _source(NOTEBOOK_ROOT / "eval_all_baselines.ipynb")
        for slug, base in {
            "qwen2.5_14b": "Qwen/Qwen2.5-14B-Instruct",
            "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
            "qwen2.5_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
            "qwen2.5_0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        }.items():
            self.assertIn(f'"{slug}": "{base}"', source)
        self.assertIn('"--artifact-stage", "base-eval"', source)
        self.assertIn("ALL BASELINE RESULTS", source)
