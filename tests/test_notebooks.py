"""Static safety checks for the four supported model notebook folders."""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
MODELS = {
    "qwen2.5_3b_instruct": (
        "Qwen/Qwen2.5-3B-Instruct",
        {"eval_base.ipynb"},
    ),
    "gemma3_1b_it": ("google/gemma-3-1b-it", {"eval_base.ipynb"}),
    "deepseek_r1_distill_qwen_1.5b": (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        {"eval_base.ipynb"},
    ),
    "ministral3_3b_instruct_2512": (
        "mistralai/Ministral-3-3B-Instruct-2512",
        {"eval_base.ipynb"},
    ),
}


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


class ModelNotebookTests(unittest.TestCase):
    def test_notebooks_directory_contains_exactly_four_model_folders(self) -> None:
        folders = {path.name for path in NOTEBOOK_ROOT.iterdir() if path.is_dir()}
        self.assertEqual(folders, set(MODELS))
        root_files = {path.name for path in NOTEBOOK_ROOT.iterdir() if path.is_file()}
        self.assertEqual(root_files, {"sft.ipynb", "eval_sft.ipynb", "grpo.ipynb"})
        for slug, (_, expected_files) in MODELS.items():
            with self.subTest(slug=slug):
                actual_files = {path.name for path in (NOTEBOOK_ROOT / slug).iterdir() if path.is_file()}
                self.assertEqual(actual_files, expected_files)

    def test_all_notebooks_are_valid_python_and_do_not_embed_secrets(self) -> None:
        for slug, (_, names) in MODELS.items():
            for name in names:
                path = NOTEBOOK_ROOT / slug / name
                notebook = json.loads(path.read_text(encoding="utf-8"))
                for index, cell in enumerate(notebook["cells"]):
                    source = "".join(cell.get("source", []))
                    self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{12,}", source))
                    self.assertNotIn("--hf-token", source.lower())
                    self.assertNotIn("--sandbox-token", source.lower())
                    if cell.get("cell_type") == "code":
                        ast.parse(source, f"{path}:cell-{index}", "exec")

    def test_model_ids_and_scoped_run_ids_are_explicit(self) -> None:
        for slug, (model_id, names) in MODELS.items():
            for name in names:
                source = _notebook_source(NOTEBOOK_ROOT / slug / name)
                with self.subTest(slug=slug, notebook=name):
                    self.assertIn(f'BASE_MODEL = "{model_id}"', source)
                    self.assertIn(f'MODEL_SLUG = "{slug}"', source)
                    self.assertIn('BUCKET_ID = "devaanshpa/CrashDiag"', source)
        for name in ("sft.ipynb", "eval_sft.ipynb"):
            source = _notebook_source(NOTEBOOK_ROOT / name)
            self.assertIn('ZoneInfo("Asia/Kolkata")', source)

    def test_qwen_workflow_retains_signed_sft_and_hard_grpo_guards(self) -> None:
        sft = _notebook_source(NOTEBOOK_ROOT / "sft.ipynb")
        grpo = _notebook_source(NOTEBOOK_ROOT / "grpo.ipynb")
        sft_eval = _notebook_source(NOTEBOOK_ROOT / "eval_sft.ipynb")
        for marker in ('dataset_client.download_stage("datasets", DATASET_DIR)', "sft_main([", '"--model", BASE_MODEL'):
            self.assertIn(marker, sft)
        self.assertIn("TRAINER_COMMIT", sft)
        self.assertIn("artifact_commit != SOURCE_COMMIT", sft)
        for marker in ("calibrate_main", "SMOKE_GATE_VERIFIED=true", "promotion_gate(", "uploader.complete_run"):
            self.assertIn(marker, grpo)
        for marker in ("SFT_RUN_ID", "SFT adapter/base-model mismatch", '"--model", str(SFT_DIR)'):
            self.assertIn(marker, sft_eval)
        self.assertIn("EXPECTED_BASE_MODEL", grpo)
