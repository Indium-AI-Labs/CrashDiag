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
    "qwen2.5_3b",
}
NOTEBOOK_NAMES = {
    "grpo.ipynb",
    "eval_grpo.ipynb",
    "eval_base.ipynb",
    "run_all.ipynb",
}
TOP_LEVEL: set[str] = set()


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
                "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
            }[slug]
            for name in NOTEBOOK_NAMES:
                source = _source(model_dir / name)
                self.assertIn(f'BASE_MODEL = "{expected_base}"', source)
                self.assertIn(f'MODEL_SLUG = "{slug}"', source)
                self.assertIn('BUCKET_ID = "devaanshpa/CrashDiag"', source)
                self.assertIn("CRASHDIAG_DATASET_RUN_ID", source)
                self.assertIn("CRASHDIAG_ENV_FILE", source)
                if name in {"grpo.ipynb", "eval_grpo.ipynb", "eval_base.ipynb"}:
                    self.assertIn("CRASHDIAG_CURRICULUM", source)
                self.assertIn("display(SVG", source)
                self.assertIn("reports", source)

    def test_all_notebooks_use_kaggle_secrets(self) -> None:
        for slug in PER_MODEL:
            for name in NOTEBOOK_NAMES:
                source = _source(NOTEBOOK_ROOT / slug / name)
                self.assertIn("UserSecretsClient", source)
                self.assertIn("kaggle_secrets", source)
                self.assertIn("loaded Kaggle secret names", source)

    def test_grpo_does_not_require_sft_run_id(self) -> None:
        for slug in PER_MODEL:
            source = _source(NOTEBOOK_ROOT / slug / "run_all.ipynb")
            self.assertNotIn("SFT_RUN_ID", source)
            self.assertIn("CRASHDIAG_ENV_FILE", source)

    def test_grpo_trains_and_evaluates_v5_curriculum_by_default(self) -> None:
        for slug in PER_MODEL:
            grpo = _source(NOTEBOOK_ROOT / slug / "grpo.ipynb")
            self.assertIn("'--train-file', str(DATASET_DIR / TRAIN_FILE)", grpo)
            self.assertIn("'--eval-file', str(DATASET_DIR / EVAL_FILE)", grpo)
            self.assertIn("'--num-generations', '2'", grpo)
            self.assertIn("'--no-load-in-4bit'", grpo)
            self.assertIn("directly from the base model", grpo)
            self.assertIn('TRAIN_FILE = "grpo_train.jsonl"', grpo)
            self.assertIn('"--no-few-shot"', _source(NOTEBOOK_ROOT / slug / "eval_grpo.ipynb"))

    def test_evaluation_has_few_shot_prompting(self) -> None:
        from training.evaluate_jsonl import FEW_SHOT_MESSAGES, few_shot_prompt, SYSTEM_PROMPT

        self.assertIn(SYSTEM_PROMPT, [msg["content"] for msg in FEW_SHOT_MESSAGES])
        self.assertIn("wait_and_observe", json.dumps(FEW_SHOT_MESSAGES))
        wrapped = few_shot_prompt([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "real"}])
        self.assertEqual(wrapped[: len(FEW_SHOT_MESSAGES)], FEW_SHOT_MESSAGES)
        self.assertEqual(wrapped[-1]["content"], "real")
