"""Generate per-model Qwen2.5 notebook folders plus an all-baselines eval.

Each model gets a folder under notebooks/ with sft, eval_sft, grpo, and
eval_grpo notebooks (templated from the proven qwen3_14b workflow), and a
top-level notebooks/eval_all_baselines.ipynb evaluates every base model in
one run, uploading each report to the private HF bucket.

Regenerate after changing the templates with:
    python scripts/generate_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"

# model_slug -> huggingface id
MODELS: dict[str, str] = {
    "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
}

BUCKET_ID = "devaanshpa/CrashDiag"


def _nb(title: str, cells: list[str]) -> dict:
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n"]},
            *[
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None,
                    "source": [cell],
                }
                for cell in cells
            ],
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --- shared cell templates (parameterized by model) -------------------------

def _setup_cell(env_file: str, kaggle: bool, stage_aliases: str = "") -> str:
    kaggle_part = ""
    if kaggle:
        kaggle_part = f"""
try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None

KAGGLE_SECRET_ALIASES = {{
    "HF_TOKEN": "HF_TOKEN",
    "CRASHDIAG_DATASET_RUN_ID": "CRASHDIAG_DATASET_RUN_ID",
    "DATASET_RUN_ID": "CRASHDIAG_DATASET_RUN_ID",
    "CRASHDIAG_SANDBOX_URL": "CRASHDIAG_SANDBOX_URL",
    "CRASHDIAG_API_TOKEN": "CRASHDIAG_API_TOKEN",
    "CRASHDIAG_SANDBOX_TOKEN": "CRASHDIAG_API_TOKEN",
    "CRASHDIAG_SOURCE_COMMIT": "CRASHDIAG_SOURCE_COMMIT",
    "SOURCE_COMMIT": "CRASHDIAG_SOURCE_COMMIT",
{stage_aliases}
}}
loaded_kaggle_secrets = []
kaggle_secret_errors = {{}}
if UserSecretsClient is not None:
    secrets = UserSecretsClient()
    for secret_name, env_name in KAGGLE_SECRET_ALIASES.items():
        if os.environ.get(env_name):
            continue
        try:
            value = secrets.get_secret(secret_name)
        except Exception as exc:
            kaggle_secret_errors[secret_name] = f"{{type(exc).__name__}}: {{exc}}"
            continue
        if value:
            os.environ[env_name] = value
            loaded_kaggle_secrets.append(secret_name)
print("loaded Kaggle secret names:", loaded_kaggle_secrets or "none")
"""
    env_load = (
        f"if ENV_FILE.is_file():\n    load_dotenv(ENV_FILE, override=True)"
        if kaggle
        else (
            f"if not ENV_FILE.is_file():\n"
            f"    raise RuntimeError(f\"CrashDiag env file not found: {{ENV_FILE}}\")\n"
            f"load_dotenv(ENV_FILE, override=True)"
        )
    )
    return f'''import os, subprocess, sys
from pathlib import Path

LAUNCH_DIR = Path.cwd().resolve()
subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv>=1,<2"], check=True)
from dotenv import load_dotenv

ENV_FILE = Path(os.environ.get("CRASHDIAG_ENV_FILE", LAUNCH_DIR / "{env_file}")).expanduser()
if not ENV_FILE.is_absolute():
    ENV_FILE = (LAUNCH_DIR / ENV_FILE).resolve()
{env_load}
{kaggle_part}
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

REPO_URL = os.environ.get("CRASHDIAG_REPO_URL", "https://github.com/Indium-AI-Labs/CrashDiag.git")
SOURCE_COMMIT = os.environ.get("CRASHDIAG_SOURCE_COMMIT", "main")
WORKDIR = Path(os.environ.get("CRASHDIAG_WORKDIR", LAUNCH_DIR / "CrashDiag-runtime")).expanduser().resolve()
if (WORKDIR / ".git").is_dir():
    subprocess.run(["git", "-C", str(WORKDIR), "fetch", "origin", "main"], check=True)
elif WORKDIR.exists() and any(WORKDIR.iterdir()):
    raise RuntimeError(f"CRASHDIAG_WORKDIR exists and is not a Git checkout: {{WORKDIR}}")
else:
    subprocess.run(["git", "clone", REPO_URL, str(WORKDIR)], check=True)
subprocess.run(["git", "-C", str(WORKDIR), "checkout", SOURCE_COMMIT], check=True)
os.chdir(WORKDIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "bitsandbytes"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[train]"], check=True)
print(f"env_file={{ENV_FILE if ENV_FILE.is_file() else 'not present (using runtime/Kaggle secrets)'}}")
print("checked_out_source_commit=" + subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
'''


def _model_cell(model_slug: str, base_model: str, extra: str = "") -> str:
    return f'''from datetime import datetime
from zoneinfo import ZoneInfo
import os

BASE_MODEL = "{base_model}"
MODEL_SLUG = "{model_slug}"
BUCKET_ID = "{BUCKET_ID}"
DATASET_RUN_ID = os.environ.get("CRASHDIAG_DATASET_RUN_ID", "").strip()
def ist_run_id(stage):
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%dT%H%M%SIST") + f"-{{MODEL_SLUG}}-{{stage}}"
if not DATASET_RUN_ID:
    raise RuntimeError("Set CRASHDIAG_DATASET_RUN_ID to the fresh dataset-generation run ID.")
print(f"base_model={{BASE_MODEL}}")
print(f"dataset_run_id={{DATASET_RUN_ID}}")
{extra}'''


def _download_cell(model_slug: str, base_model: str, kind: str) -> str:
    return f'''from pathlib import Path
from training.artifacts import ArtifactConfig, ArtifactUploader

{_hard_curriculum_cell()}
DATASET_DIR = Path("artifacts/datasets")
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=DATASET_RUN_ID, token=os.environ["HF_TOKEN"])).download_stage("datasets", DATASET_DIR)
assert (DATASET_DIR / EVAL_FILE).is_file(), f"dataset stage missing {{EVAL_FILE}}; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
print(f"curriculum={{CURRICULUM}}")
print(f"eval_file={{EVAL_FILE}}")
'''


def _sft_eval_cell(model_slug: str, base_model: str) -> str:
    return f'''from training.evaluate_jsonl import main as evaluate_main

exit_code = evaluate_main([
    "--model", str(SFT_DIR), "--dataset", str(DATASET_DIR / EVAL_FILE),
    "--output-dir", "outputs/sft-eval", "--load-in-4bit", "--precision", "bf16",
    "--max-new-tokens", "64",
    "--sandbox-url", os.environ["CRASHDIAG_SANDBOX_URL"],
    "--artifact-bucket", BUCKET_ID, "--run-id", SFT_EVAL_RUN_ID, "--artifact-stage", "sft-eval",
    "--no-few-shot",
])
if exit_code: raise RuntimeError(f"SFT evaluation failed: {{exit_code}}")
'''


def _grpo_eval_cell(model_slug: str, base_model: str) -> str:
    return f'''from training.evaluate_jsonl import main as evaluate_main

exit_code = evaluate_main([
    "--model", str(GRPO_DIR), "--dataset", str(DATASET_DIR / EVAL_FILE),
    "--output-dir", "outputs/grpo-eval", "--load-in-4bit", "--precision", "bf16",
    "--max-new-tokens", "64",
    "--sandbox-url", os.environ["CRASHDIAG_SANDBOX_URL"],
    "--artifact-bucket", BUCKET_ID, "--run-id", GRPO_EVAL_RUN_ID, "--artifact-stage", "grpo-eval",
    "--no-few-shot",
])
if exit_code: raise RuntimeError(f"GRPO evaluation failed: {{exit_code}}")
'''


def _svg_cell(stage: str, run_var: str) -> str:
    return f'''from IPython.display import SVG, display

REPORTS_DIR = Path("outputs/{stage}") / "reports"
charts = sorted(REPORTS_DIR.glob("*.svg"))
if not charts:
    raise RuntimeError(f"No SVG charts were generated in {{REPORTS_DIR}}")
print(f"Uploaded reports: hf://buckets/{{BUCKET_ID}}/runs/{{{run_var}}}/{stage}/reports")
for chart in charts:
    display(SVG(filename=str(chart)))
'''


# --- per-model notebook builders --------------------------------------------

def build_sft(model_slug: str, base_model: str) -> dict:
    return _nb(
        f"{model_slug} QLoRA SFT",
        [
            _setup_cell("env.txt", kaggle=True),
            _model_cell(model_slug, base_model),
            f'''from pathlib import Path
from training.artifacts import ArtifactConfig, ArtifactUploader

SFT_RUN_ID = os.environ.get("CRASHDIAG_SFT_RUN_ID") or ist_run_id("sft")
DATASET_DIR = Path("artifacts/datasets")
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=DATASET_RUN_ID, token=os.environ["HF_TOKEN"])).download_stage("datasets", DATASET_DIR)
assert (DATASET_DIR / "sft_train.jsonl").is_file(), f"dataset stage missing sft_train.jsonl; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
print(f"SFT_RUN_ID={{SFT_RUN_ID}}")
''',
            f'''import subprocess, sys

command = [
    sys.executable, "-m", "accelerate.commands.launch",
    "--num_processes", "1", "--num_machines", "1",
    "--mixed_precision", "bf16", "--dynamo_backend", "no",
    "-m", "training.sft",
    "--model", BASE_MODEL,
    "--dataset", str(DATASET_DIR / "sft_train.jsonl"),
    "--eval-dataset", str(DATASET_DIR / "sft_eval.jsonl"),
    "--output-dir", "outputs/sft",
    "--epochs", "1",
    "--batch-size", "8",
    "--eval-batch-size", "8",
    "--gradient-accumulation-steps", "8",
    "--max-length", "512",
    "--learning-rate", "2e-4",
    "--lora-rank", "16", "--lora-alpha", "32",
    "--load-in-4bit",
    "--precision", "bf16",
    "--report-to", "none",
    "--artifact-bucket", BUCKET_ID,
    "--run-id", SFT_RUN_ID,
]
subprocess.run(command, check=True)
''',
            _svg_cell("sft", "SFT_RUN_ID"),
        ],
    )


def build_eval_sft(model_slug: str, base_model: str) -> dict:
    return _nb(
        f"{model_slug} SFT evaluation",
        [
            _setup_cell(
                "env.txt",
                kaggle=True,
                stage_aliases='''    "CRASHDIAG_SFT_RUN_ID": "CRASHDIAG_SFT_RUN_ID",
    "SFT_RUN_ID": "CRASHDIAG_SFT_RUN_ID",
    "CRASHDIAG_SFT_EVAL_RUN_ID": "CRASHDIAG_SFT_EVAL_RUN_ID",
''',
            ),
            _model_cell(
                model_slug,
                base_model,
                extra=f'''SFT_RUN_ID = os.environ.get("CRASHDIAG_SFT_RUN_ID", "").strip()
SFT_EVAL_RUN_ID = os.environ.get("CRASHDIAG_SFT_EVAL_RUN_ID") or ist_run_id("sft-eval")
if not SFT_RUN_ID:
    detail = kaggle_secret_errors.get("CRASHDIAG_SFT_RUN_ID") or kaggle_secret_errors.get("SFT_RUN_ID")
    raise RuntimeError("Missing CRASHDIAG_SFT_RUN_ID. Add and enable that Kaggle Secret "
                       "(or SFT_RUN_ID), then rerun from the first cell. "
                       f"Kaggle response: {{detail or 'secret was not returned'}}")
''',
            ),
            f'''from pathlib import Path
from training.artifacts import ArtifactConfig, ArtifactUploader

{_hard_curriculum_cell()}
DATASET_DIR, SFT_DIR = Path("artifacts/datasets"), Path("artifacts/sft")
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=DATASET_RUN_ID, token=os.environ["HF_TOKEN"])).download_stage("datasets", DATASET_DIR)
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=SFT_RUN_ID, token=os.environ["HF_TOKEN"])).download_stage("sft", SFT_DIR)
assert (DATASET_DIR / EVAL_FILE).is_file(), f"dataset stage missing {{EVAL_FILE}}; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
print(f"curriculum={{CURRICULUM}}")
print(f"eval_file={{EVAL_FILE}}")
''',
            _sft_eval_cell(model_slug, base_model),
            _svg_cell("sft-eval", "SFT_EVAL_RUN_ID"),
        ],
    )


def build_grpo(model_slug: str, base_model: str) -> dict:
    return _nb(
        f"{model_slug} GRPO",
        [
            _setup_cell(
                "env.txt",
                kaggle=True,
                stage_aliases='''    "CRASHDIAG_SFT_RUN_ID": "CRASHDIAG_SFT_RUN_ID",
    "SFT_RUN_ID": "CRASHDIAG_SFT_RUN_ID",
''',
            ),
            _model_cell(model_slug, base_model),
            f'''from pathlib import Path
from training.artifacts import ArtifactConfig, ArtifactUploader

SFT_RUN_ID = os.environ.get("CRASHDIAG_SFT_RUN_ID", "").strip()
if not SFT_RUN_ID: raise RuntimeError("Set CRASHDIAG_SFT_RUN_ID to the completed SFT run ID.")
GRPO_RUN_ID = os.environ.get("CRASHDIAG_GRPO_RUN_ID") or ist_run_id("grpo")
{_hard_curriculum_cell()}
DATASET_DIR, SFT_DIR = Path("artifacts/datasets"), Path("artifacts/sft")
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=DATASET_RUN_ID, token=os.environ["HF_TOKEN"])).download_stage("datasets", DATASET_DIR)
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=SFT_RUN_ID, token=os.environ["HF_TOKEN"])).download_stage("sft", SFT_DIR)
assert (DATASET_DIR / TRAIN_FILE).is_file(), f"dataset stage missing {{TRAIN_FILE}}; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
assert (DATASET_DIR / EVAL_FILE).is_file(), f"dataset stage missing {{EVAL_FILE}}; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
print(f"GRPO_RUN_ID={{GRPO_RUN_ID}}")
print(f"curriculum={{CURRICULUM}}")
print(f"train_file={{TRAIN_FILE}}")
print(f"eval_file={{EVAL_FILE}}")
''',
            f'''import subprocess, sys

common = [
    sys.executable, "-m", "accelerate.commands.launch",
    "--num_processes", "1", "--num_machines", "1",
    "--mixed_precision", "bf16", "--dynamo_backend", "no",
    "-m", "training.grpo", "--model", str(SFT_DIR),
    "--train-file", str(DATASET_DIR / TRAIN_FILE),
    "--eval-file", str(DATASET_DIR / EVAL_FILE),
    "--output-dir", "outputs/grpo", "--load-in-4bit", "--precision", "bf16",
    "--batch-size", "2", "--gradient-accumulation-steps", "4", "--num-generations", "2",
    "--max-prompt-length", "1024", "--max-completion-length", "64",
    "--max-steps", "24", "--artifact-bucket", BUCKET_ID, "--run-id", GRPO_RUN_ID,
    "--artifact-stage", "grpo-smoke", "--sandbox-url", os.environ["CRASHDIAG_SANDBOX_URL"],
]
subprocess.run(common, check=True)
full = common[:]
full[full.index("24")] = "96"
full[full.index("grpo-smoke")] = "grpo"
subprocess.run(full, check=True)
''',
            _grpo_eval_cell(model_slug, base_model),
            _svg_cell("grpo-eval", "GRPO_RUN_ID"),
        ],
    )


def build_eval_grpo(model_slug: str, base_model: str) -> dict:
    return _nb(
        f"{model_slug} GRPO evaluation",
        [
            _setup_cell(
                "env.txt",
                kaggle=True,
                stage_aliases='''    "CRASHDIAG_GRPO_RUN_ID": "CRASHDIAG_GRPO_RUN_ID",
    "GRPO_RUN_ID": "CRASHDIAG_GRPO_RUN_ID",
    "CRASHDIAG_GRPO_EVAL_RUN_ID": "CRASHDIAG_GRPO_EVAL_RUN_ID",
''',
            ),
            _model_cell(
                model_slug,
                base_model,
                extra=f'''GRPO_RUN_ID = os.environ.get("CRASHDIAG_GRPO_RUN_ID", "").strip()
if not GRPO_RUN_ID:
    detail = kaggle_secret_errors.get("CRASHDIAG_GRPO_RUN_ID") or kaggle_secret_errors.get("GRPO_RUN_ID")
    raise RuntimeError("Missing CRASHDIAG_GRPO_RUN_ID. Add and enable that Kaggle Secret "
                       "(or GRPO_RUN_ID), then rerun from the first cell. "
                       f"Kaggle response: {{detail or 'secret was not returned'}}")
GRPO_EVAL_RUN_ID = os.environ.get("CRASHDIAG_GRPO_EVAL_RUN_ID") or ist_run_id("grpo-eval")
print(f"grpo_run_id={{GRPO_RUN_ID}}")
print(f"grpo_eval_run_id={{GRPO_EVAL_RUN_ID}}")
''',
            ),
            f'''from pathlib import Path
from training.artifacts import ArtifactConfig, ArtifactUploader

{_hard_curriculum_cell()}
DATASET_DIR = Path("artifacts/datasets")
GRPO_DIR = Path("artifacts/grpo")
token = os.environ["HF_TOKEN"]
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=DATASET_RUN_ID, token=token)).download_stage("datasets", DATASET_DIR)
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=GRPO_RUN_ID, token=token)).download_stage("grpo", GRPO_DIR)
assert (DATASET_DIR / EVAL_FILE).is_file(), f"dataset stage missing {{EVAL_FILE}}; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
print(f"curriculum={{CURRICULUM}}")
print(f"eval_file={{EVAL_FILE}}")
''',
            _grpo_eval_cell(model_slug, base_model),
            _svg_cell("grpo-eval", "GRPO_EVAL_RUN_ID"),
        ],
    )


def build_eval_base(model_slug: str, base_model: str) -> dict:
    return _nb(
        f"{model_slug} base model evaluation",
        [
            _setup_cell(
                "env.txt",
                kaggle=True,
                stage_aliases='''    "CRASHDIAG_BASE_EVAL_RUN_ID": "CRASHDIAG_BASE_EVAL_RUN_ID",
    "BASE_EVAL_RUN_ID": "CRASHDIAG_BASE_EVAL_RUN_ID",
''',
            ),
            _model_cell(
                model_slug,
                base_model,
                extra='''BASE_EVAL_RUN_ID = os.environ.get("CRASHDIAG_BASE_EVAL_RUN_ID") or ist_run_id("base-eval")
print(f"base_eval_run_id={BASE_EVAL_RUN_ID}")
''',
            ),
            f'''from pathlib import Path
from training.artifacts import ArtifactConfig, ArtifactUploader

{_hard_curriculum_cell()}
DATASET_DIR = Path("artifacts/datasets")
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=DATASET_RUN_ID, token=os.environ["HF_TOKEN"])).download_stage("datasets", DATASET_DIR)
assert (DATASET_DIR / EVAL_FILE).is_file(), f"dataset stage missing {{EVAL_FILE}}; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
print(f"curriculum={{CURRICULUM}}")
print(f"eval_file={{EVAL_FILE}}")
''',
            f'''from training.evaluate_jsonl import main as evaluate_main

exit_code = evaluate_main([
    "--model", BASE_MODEL, "--dataset", str(DATASET_DIR / EVAL_FILE),
    "--output-dir", "outputs/base-eval", "--load-in-4bit", "--precision", "bf16",
    "--max-new-tokens", "64",
    "--sandbox-url", os.environ["CRASHDIAG_SANDBOX_URL"],
    "--artifact-bucket", BUCKET_ID, "--run-id", BASE_EVAL_RUN_ID, "--artifact-stage", "base-eval",
])
if exit_code: raise RuntimeError(f"Base model evaluation failed: {{exit_code}}")
''',
            _svg_cell("base-eval", "BASE_EVAL_RUN_ID"),
        ],
    )


# --- all-models training notebooks ------------------------------------------
#
# Both notebooks share ONE run ID for every model.  Each model's training
# uploads under its own stage folder inside that run, so a single run ID in
# the HF bucket contains one subfolder per model.

def _models_cell() -> str:
    entries = "\n".join(
        f'    "{slug}": "{base}",' for slug, base in MODELS.items()
    )
    return f'''BUCKET_ID = "{BUCKET_ID}"
MODELS = {{
{entries}
}}
DATASET_RUN_ID = os.environ.get("CRASHDIAG_DATASET_RUN_ID", "").strip()
ALL_RUN_ID = os.environ.get("CRASHDIAG_ALL_RUN_ID", "").strip() or ist_run_id("all")
if not DATASET_RUN_ID:
    raise RuntimeError("Set CRASHDIAG_DATASET_RUN_ID to the fresh dataset-generation run ID.")
print(f"models={{list(MODELS)}}")
print(f"dataset_run_id={{DATASET_RUN_ID}}")
print(f"ALL_RUN_ID={{ALL_RUN_ID}}")'''


def _models_setup_cell() -> str:
    return f'''from datetime import datetime
from zoneinfo import ZoneInfo
import os

def ist_run_id(stage):
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%dT%H%M%SIST") + f"-{{stage}}"
{_models_cell()}'''


def _datasets_cell() -> str:
    return f'''from pathlib import Path
from training.artifacts import ArtifactConfig, ArtifactUploader

DATASET_DIR = Path("artifacts/datasets")
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=DATASET_RUN_ID, token=os.environ["HF_TOKEN"])).download_stage("datasets", DATASET_DIR)
assert (DATASET_DIR / "sft_train.jsonl").is_file(), f"dataset stage missing sft_train.jsonl; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
print(f"sft_train={{DATASET_DIR / 'sft_train.jsonl'}}")'''


def _hard_curriculum_cell() -> str:
    return f'''CURRICULUM = os.environ.get("CRASHDIAG_CURRICULUM", "v5").strip().lower()
HARD = CURRICULUM in ("v5", "hard-v5", "hard-v4", "hard-v3")
TRAIN_FILE = "grpo_train.jsonl"
EVAL_FILE = "grpo_eval.jsonl"'''


def build_sft_all() -> dict:
    return _nb(
        "Qwen2.5 SFT all models (one run ID)",
        [
            _setup_cell("env.txt", kaggle=True),
            _models_setup_cell(),
            _datasets_cell(),
            f'''import subprocess, sys

results = {{}}
for slug, base_model in MODELS.items():
    output_dir = f"outputs/{{slug}}-sft"
    print(f"=== SFT {{base_model}} ({{slug}}) ===")
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1", "--num_machines", "1",
        "--mixed_precision", "bf16", "--dynamo_backend", "no",
        "-m", "training.sft",
        "--model", base_model,
        "--dataset", str(DATASET_DIR / "sft_train.jsonl"),
        "--eval-dataset", str(DATASET_DIR / "sft_eval.jsonl"),
        "--output-dir", output_dir,
        "--epochs", "1",
        "--batch-size", "8",
        "--eval-batch-size", "8",
        "--gradient-accumulation-steps", "8",
        "--max-length", "512",
        "--learning-rate", "2e-4",
        "--lora-rank", "16", "--lora-alpha", "32",
        "--load-in-4bit",
        "--precision", "bf16",
        "--report-to", "none",
        "--artifact-bucket", BUCKET_ID,
        "--run-id", ALL_RUN_ID,
        "--artifact-stage", slug,
    ]
    subprocess.run(command, check=True)
    import json
    report = json.loads((Path(output_dir) / "reports" / "metrics_summary.json").read_text(encoding="utf-8"))
    results[slug] = report

print("\\n=== ALL SFT RESULTS ===")
for slug, report in results.items():
    print(f"{{slug}}: {{report}}")''',
            f'''from IPython.display import SVG, display

for slug in MODELS:
    REPORTS_DIR = Path(f"outputs/{{slug}}-sft") / "reports"
    charts = sorted(REPORTS_DIR.glob("*.svg"))
    print(f"[{{slug}}] hf://buckets/{{BUCKET_ID}}/runs/{{ALL_RUN_ID}}/{{slug}}/reports")
    for chart in charts:
        display(SVG(filename=str(chart)))''',
        ],
    )


def build_grpo_all() -> dict:
    return _nb(
        "Qwen2.5 GRPO all models (one run ID)",
        [
            _setup_cell(
                "env.txt",
                kaggle=True,
                stage_aliases='''    "CRASHDIAG_SFT_RUN_ID": "CRASHDIAG_SFT_RUN_ID",
    "SFT_RUN_ID": "CRASHDIAG_SFT_RUN_ID",
''',
            ),
            _models_setup_cell(),
            f'''from pathlib import Path
from training.artifacts import ArtifactConfig, ArtifactUploader

SFT_RUN_ID = os.environ.get("CRASHDIAG_SFT_RUN_ID", "").strip()
if not SFT_RUN_ID: raise RuntimeError("Set CRASHDIAG_SFT_RUN_ID to the single SFT-all run ID.")
{_hard_curriculum_cell()}
DATASET_DIR, SFT_ROOT = Path("artifacts/datasets"), Path("artifacts/sft-all")
token = os.environ["HF_TOKEN"]
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=DATASET_RUN_ID, token=token)).download_stage("datasets", DATASET_DIR)
for slug in MODELS:
    stage_dir = SFT_ROOT / slug
    if stage_dir.exists():
        import shutil
        shutil.rmtree(stage_dir)
    ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=SFT_RUN_ID, token=token)).download_stage(slug, stage_dir)
assert (DATASET_DIR / TRAIN_FILE).is_file(), f"dataset stage missing {{TRAIN_FILE}}; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
assert (DATASET_DIR / EVAL_FILE).is_file(), f"dataset stage missing {{EVAL_FILE}}; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
print(f"SFT_RUN_ID={{SFT_RUN_ID}}")
print(f"ALL_RUN_ID={{ALL_RUN_ID}}")
print(f"curriculum={{CURRICULUM}}")
print(f"train_file={{TRAIN_FILE}}")
print(f"eval_file={{EVAL_FILE}}")''',
            f'''import subprocess, sys

for slug in MODELS:
    print(f"=== GRPO {{slug}} ===")
    common = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1", "--num_machines", "1",
        "--mixed_precision", "bf16", "--dynamo_backend", "no",
        "-m", "training.grpo", "--model", str(SFT_ROOT / slug),
        "--train-file", str(DATASET_DIR / TRAIN_FILE),
        "--eval-file", str(DATASET_DIR / EVAL_FILE),
        "--output-dir", f"outputs/{{slug}}-grpo", "--load-in-4bit", "--precision", "bf16",
        "--batch-size", "2", "--gradient-accumulation-steps", "4", "--num-generations", "2",
        "--max-prompt-length", "1024", "--max-completion-length", "64",
        "--max-steps", "24", "--artifact-bucket", BUCKET_ID, "--run-id", ALL_RUN_ID,
        "--artifact-stage", f"{{slug}}-grpo-smoke", "--sandbox-url", os.environ["CRASHDIAG_SANDBOX_URL"],
    ]
    subprocess.run(common, check=True)
    full = common[:]
    full[full.index("24")] = "96"
    full[full.index(f"{{slug}}-grpo-smoke")] = f"{{slug}}-grpo"
    subprocess.run(full, check=True)''',
            f'''from training.evaluate_jsonl import main as evaluate_main

for slug in MODELS:
    print(f"=== GRPO eval {{slug}} ===")
    exit_code = evaluate_main([
        "--model", f"outputs/{{slug}}-grpo", "--dataset", str(DATASET_DIR / EVAL_FILE),
        "--output-dir", f"outputs/{{slug}}-grpo-eval", "--load-in-4bit", "--precision", "bf16",
        "--max-new-tokens", "64",
        "--sandbox-url", os.environ["CRASHDIAG_SANDBOX_URL"],
        "--artifact-bucket", BUCKET_ID, "--run-id", ALL_RUN_ID, "--artifact-stage", f"{{slug}}-grpo-eval",
        "--no-few-shot",
    ])
    if exit_code: raise RuntimeError(f"GRPO evaluation failed for {{slug}}: {{exit_code}}")''',
            f'''from IPython.display import SVG, display

for slug in MODELS:
    REPORTS_DIR = Path(f"outputs/{{slug}}-grpo-eval") / "reports"
    charts = sorted(REPORTS_DIR.glob("*.svg"))
    print(f"[{{slug}}] hf://buckets/{{BUCKET_ID}}/runs/{{ALL_RUN_ID}}/{{slug}}-grpo-eval/reports")
    for chart in charts:
        display(SVG(filename=str(chart)))''',
        ],
    )


# --- all-baselines eval notebook --------------------------------------------

def build_eval_all_baselines() -> dict:
    model_entries = "\n".join(
        f'    "{slug}": "{base}",' for slug, base in MODELS.items()
    )
    cells = [
        _setup_cell("env.txt", kaggle=True),
        f'''from datetime import datetime
from zoneinfo import ZoneInfo
import os

BUCKET_ID = "{BUCKET_ID}"
MODELS = {{
{model_entries}
}}
DATASET_RUN_ID = os.environ.get("CRASHDIAG_DATASET_RUN_ID", "").strip()
def ist_run_id(stage, slug):
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%dT%H%M%SIST") + f"-{{slug}}-{{stage}}"
if not DATASET_RUN_ID:
    raise RuntimeError("Set CRASHDIAG_DATASET_RUN_ID to the fresh dataset-generation run ID.")
print(f"models={{list(MODELS)}}")
print(f"dataset_run_id={{DATASET_RUN_ID}}")
''',
        f'''from pathlib import Path
from training.artifacts import ArtifactConfig, ArtifactUploader

CURRICULUM = os.environ.get("CRASHDIAG_CURRICULUM", "hard-v4").strip().lower()
HARD = CURRICULUM in ("hard-v3", "hard-v4")
EVAL_FILE = "grpo_hard_eval.jsonl" if HARD else "grpo_eval.jsonl"
DATASET_DIR = Path("artifacts/datasets")
ArtifactUploader(ArtifactConfig(bucket_id=BUCKET_ID, run_id=DATASET_RUN_ID, token=os.environ["HF_TOKEN"])).download_stage("datasets", DATASET_DIR)
assert (DATASET_DIR / EVAL_FILE).is_file(), f"dataset stage missing {{EVAL_FILE}}; check CRASHDIAG_DATASET_RUN_ID={{DATASET_RUN_ID}}"
print(f"curriculum={{CURRICULUM}}")
print(f"eval_file={{EVAL_FILE}}")
''',
        f'''from training.evaluate_jsonl import main as evaluate_main

results = {{}}
for slug, base_model in MODELS.items():
    run_id = os.environ.get(f"CRASHDIAG_BASE_{{slug.upper().replace('.', '_')}}_RUN_ID") or ist_run_id("base-eval", slug)
    print(f"=== evaluating {{base_model}} ({{slug}}) -> {{run_id}} ===")
    exit_code = evaluate_main([
        "--model", base_model,
        "--dataset", str(DATASET_DIR / EVAL_FILE),
        "--output-dir", f"outputs/{{slug}}-base-eval",
        "--load-in-4bit",
        "--precision", "bf16",
        "--max-new-tokens", "64",
        "--sandbox-url", os.environ["CRASHDIAG_SANDBOX_URL"],
        "--artifact-bucket", BUCKET_ID,
        "--run-id", run_id,
        "--artifact-stage", "base-eval",
    ])
    if exit_code:
        raise RuntimeError(f"{{base_model}} baseline evaluation failed: {{exit_code}}")
    import json
    report = json.loads((Path(f"outputs/{{slug}}-base-eval") / "mechanical_evaluation.json").read_text(encoding="utf-8"))
    results[slug] = report["summary"]
    print(f"{{slug}}: {{report['summary']}}")

print("\\n=== ALL BASELINE RESULTS ===")
for slug, summary in results.items():
    print(f"{{slug}}: success={{summary['success_rate']:.1%}} strict_json={{summary['strict_json_rate']:.1%}} backend_error={{summary['backend_error_rate']:.1%}}")
''',
        f'''# Model-wise comparison across all baselines
from pathlib import Path as _P
import json as _j, html as _h

slugs = list(MODELS)
_chart_rows = []
for _slug in slugs:
    _p2 = _P(f"outputs/{{_slug}}-base-eval/mechanical_evaluation.json")
    if not _p2.exists():
        print(f"missing: {{_p2}}"); continue
    _data = _j.loads(_p2.read_text())
    _s = _data["summary"]
    _chart_rows.append((_slug, _s["success_rate"], _s["strict_json_rate"], _s["backend_error_rate"], dict(_data.get("per_fault", {{}}))))

for _slug, _, _, _, _per_fault in _chart_rows:
    print(f"\\n[{{_slug}}] per-fault:")
    for _fault in sorted(_per_fault):
        _v = _per_fault[_fault]
        print(f"  {{_fault}}: {{_v['resolved']}}/{{_v['episodes']}}  {{_v['success_rate']:.1%}}")

_width,_height,_left,_right,_top,_bottom = 960,480,78,24,70,120
_plot_w = _width - _left - _right
_plot_h = _height - _top - _bottom
_n = len(_chart_rows) or 1
_slot = _plot_w / _n
_bar = _slot * 0.34
_colors = {{"success": "#2563eb", "strict_json": "#16a34a"}}
_parts = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{{_width}}" height="{{_height}}" viewBox="0 0 {{_width}} {{_height}}" role="img">',
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    f'<text x="{{_width/2:.1f}}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600">Baseline comparison by model (hard-v4)</text>',
]
for _t in range(6):
    _r = _t/5; _y = _top + (1-_r)*_plot_h
    _parts += [f'<line x1="{{_left}}" y1="{{_y:.1f}}" x2="{{_width-_right}}" y2="{{_y:.1f}}" stroke="#e5e7eb" stroke-width="1"/>',
               f'<text x="{{_left-10}}" y="{{_y+4:.1f}}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#4b5563">{{_r:.0%}}</text>']
for _i,(_slug,_s,_strict,_err,_) in enumerate(_chart_rows):
    _x0 = _left + _i*_slot + (_slot - 2*_bar - 6)/2
    for _j,(_val,_c) in enumerate([(_s,_colors["success"]),(_strict,_colors["strict_json"])]):
        _bh = _val*_plot_h; _y = _top + _plot_h - _bh
        _parts += [f'<rect x="{{_x0+_j*(_bar+6):.1f}}" y="{{_y:.1f}}" width="{{_bar:.1f}}" height="{{_bh:.1f}}" fill="{{_c}}" rx="3"/>',
                   f'<text x="{{_x0+_j*(_bar+6)+_bar/2:.1f}}" y="{{max(_top+14,_y-8):.1f}}" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600">{{_val:.0%}}</text>']
    _cx = _left + _i*_slot + _slot/2
    _parts += [f'<text x="{{_cx:.1f}}" y="{{_height-_bottom+22}}" text-anchor="end" transform="rotate(-30 {{_cx:.1f}} {{_height-_bottom+22}})" font-family="sans-serif" font-size="12">{{_h.escape(_slug)}}</text>']
_parts += [f'<line x1="{{_left}}" y1="{{_top}}" x2="{{_left}}" y2="{{_height-_bottom}}" stroke="#111827" stroke-width="1.2"/>',
           f'<line x1="{{_left}}" y1="{{_height-_bottom}}" x2="{{_width-_right}}" y2="{{_height-_bottom}}" stroke="#111827" stroke-width="1.2"/>',
           f'<rect x="{{_left}}" y="{{55}}" width="18" height="9" fill="{{_colors["success"]}}"/><text x="{{_left+24}}" y="{{63}}" font-family="sans-serif" font-size="12">success_rate</text>',
           f'<rect x="{{_left+120}}" y="{{55}}" width="18" height="9" fill="{{_colors["strict_json"]}}"/><text x="{{_left+144}}" y="{{63}}" font-family="sans-serif" font-size="12">strict_json_rate</text>',
           "</svg>"]
_svg = _P("outputs/baselines_summary.svg")
_svg.parent.mkdir(parents=True, exist_ok=True)
_svg.write_text("\\n".join(_parts), encoding="utf-8")

from IPython.display import SVG, display
display(SVG(filename=str(_svg)))

for _slug, _s, _strict, _err, _ in _chart_rows:
    print(f"{{_slug:14s}}  success {{_s:.1%}}  strict_json {{_strict:.1%}}  backend_error {{_err:.1%}}")
''',
    ]
    return _nb("Qwen2.5 all-baselines evaluation", cells)


def main() -> int:
    import shutil

    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    # Remove obsolete multi-model and sweep notebooks from the previous workflow.
    for slug in ("qwen2.5_14b", "qwen2.5_7b", "qwen2.5_1.5b", "qwen2.5_0.5b", "qwen3_14b"):
        stale = NOTEBOOK_ROOT / slug
        if stale.is_dir():
            shutil.rmtree(stale)
            print(f"removed {stale}")
    for name in (
        "eval_all_baselines.ipynb",
        "eval_all_sft.ipynb",
        "eval_all_grpo.ipynb",
        "sft_all.ipynb",
        "grpo_all.ipynb",
    ):
        stale = NOTEBOOK_ROOT / name
        if stale.is_file():
            stale.unlink()
            print(f"removed {stale}")
    for slug, base in MODELS.items():
        folder = NOTEBOOK_ROOT / slug
        folder.mkdir(parents=True, exist_ok=True)
        specs = {
            "sft.ipynb": build_sft(slug, base),
            "eval_sft.ipynb": build_eval_sft(slug, base),
            "grpo.ipynb": build_grpo(slug, base),
            "eval_grpo.ipynb": build_eval_grpo(slug, base),
            "eval_base.ipynb": build_eval_base(slug, base),
        }
        for name, nb in specs.items():
            (folder / name).write_text(
                json.dumps(nb, indent=1) + "\n", encoding="utf-8"
            )
        print(f"wrote {folder}/ (sft, eval_sft, grpo, eval_grpo, eval_base)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
