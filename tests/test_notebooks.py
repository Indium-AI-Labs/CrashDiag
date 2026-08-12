"""Static safety checks for the one-model Qwen3-14B notebook workflow."""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
MODEL_DIR = NOTEBOOK_ROOT / "qwen3_14b"
NOTEBOOKS = {"eval_base.ipynb", "sft.ipynb", "eval_sft.ipynb", "grpo.ipynb"}


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


class ModelNotebookTests(unittest.TestCase):
    def test_only_the_qwen3_14b_workflow_is_present(self) -> None:
        self.assertEqual({path.name for path in NOTEBOOK_ROOT.iterdir()}, {"qwen3_14b"})
        self.assertEqual({path.name for path in MODEL_DIR.iterdir()}, NOTEBOOKS)

    def test_notebooks_are_safe_valid_python(self) -> None:
        for name in NOTEBOOKS:
            notebook = json.loads((MODEL_DIR / name).read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                source = "".join(cell.get("source", []))
                self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{12,}", source))
                if cell.get("cell_type") == "code":
                    ast.parse(source, f"{name}:cell-{index}", "exec")

    def test_notebooks_use_qwen3_14b_qlora_and_bucket_artifacts(self) -> None:
        for name in NOTEBOOKS:
            source = _source(MODEL_DIR / name)
            self.assertIn('BASE_MODEL = "Qwen/Qwen3-14B"', source)
            self.assertIn('MODEL_SLUG = "qwen3_14b"', source)
            self.assertIn('BUCKET_ID = "devaanshpa/CrashDiag"', source)
            self.assertIn("load_dotenv", source)
            self.assertIn("CRASHDIAG_ENV_FILE", source)
            self.assertNotIn("UserSecretsClient", source)
            self.assertNotIn("kaggle_secrets", source)
            self.assertIn("CRASHDIAG_DATASET_RUN_ID", source)
            self.assertIn('PYTORCH_ALLOC_CONF', source)
            self.assertIn('expandable_segments:True', source)
            self.assertNotIn("ArtifactConfig.from_env", source)
            self.assertNotIn('"--artifact-run-id"', source)
        self.assertIn('"--load-in-4bit"', _source(MODEL_DIR / "eval_base.ipynb"))
        self.assertIn('"--epochs", "2"', _source(MODEL_DIR / "sft.ipynb"))
        self.assertIn('"--max-length", "2048"', _source(MODEL_DIR / "sft.ipynb"))
        self.assertIn('"--lora-rank", "16"', _source(MODEL_DIR / "sft.ipynb"))
        self.assertIn('"--num_processes", "1"', _source(MODEL_DIR / "sft.ipynb"))
        self.assertIn('"--mixed_precision", "bf16"', _source(MODEL_DIR / "sft.ipynb"))
        grpo = _source(MODEL_DIR / "grpo.ipynb")
        self.assertIn('"--num_processes", "1"', grpo)
        self.assertIn('"--max-steps", "24"', grpo)
        self.assertIn('"--num-generations", "2"', grpo)
        self.assertIn('"--max-prompt-length", "1024"', grpo)
        self.assertIn('"--max-completion-length", "64"', grpo)
        self.assertIn('"96"', grpo)
        self.assertNotIn('"--artifact-stage"', _source(MODEL_DIR / "sft.ipynb"))
        for name in NOTEBOOKS:
            source = _source(MODEL_DIR / name)
            self.assertIn("display(SVG", source)
            self.assertIn('"reports"', source)
