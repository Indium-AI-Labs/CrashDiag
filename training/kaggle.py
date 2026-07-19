"""Launch CrashDiag training on Kaggle with secrets kept out of saved files.

Kaggle Secrets are read in this process and injected only into the child
environment.  The default child is the complete ``scripts/train.sh`` pipeline,
but any phase command can be supplied after ``--``.
"""

from __future__ import annotations

import argparse
import os
import secrets
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone

from .artifacts import PROJECT_ROOT


DEFAULT_BUCKET = "devaanshpa/CrashDiag"


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-kaggle-{secrets.token_hex(3)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sandbox-url",
        default=os.environ.get("CRASHDIAG_SANDBOX_URL"),
        help="public HTTPS URL of the Vultr sandbox",
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--run-id", default=os.environ.get("CRASHDIAG_RUN_ID"))
    parser.add_argument(
        "--hf-secret-name",
        default="HF_TOKEN",
        help="Kaggle Secret label containing the HF write token",
    )
    parser.add_argument(
        "--sandbox-secret-name",
        default="CRASHDIAG_SANDBOX_TOKEN",
        help="Kaggle Secret label containing the Vultr bearer token",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command after -- (default: bash scripts/train.sh)",
    )
    return parser


def launch(args: argparse.Namespace) -> int:
    if not args.sandbox_url or not str(args.sandbox_url).startswith("https://"):
        raise SystemExit("--sandbox-url must be the Vultr HTTPS endpoint")
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise SystemExit(
            "kaggle_secrets is available only inside a Kaggle notebook"
        ) from exc

    client = UserSecretsClient()
    try:
        hf_token = client.get_secret(args.hf_secret_name)
        sandbox_token = client.get_secret(args.sandbox_secret_name)
    except Exception as exc:
        raise SystemExit(
            "required Kaggle Secrets are unavailable; attach HF_TOKEN and "
            "CRASHDIAG_SANDBOX_TOKEN to the notebook"
        ) from exc
    if not hf_token or not sandbox_token:
        raise SystemExit("required Kaggle Secrets cannot be empty")

    run_id = args.run_id or _run_id()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        command = ["bash", "scripts/train.sh"]

    environment = os.environ.copy()
    environment.update(
        {
            "HF_TOKEN": hf_token,
            "CRASHDIAG_SANDBOX_TOKEN": sandbox_token,
            "CRASHDIAG_SANDBOX_URL": args.sandbox_url,
            "CRASHDIAG_HF_BUCKET_ID": args.bucket,
            "CRASHDIAG_ARTIFACT_PREFIX": "runs",
            "CRASHDIAG_ARTIFACT_LOCAL_ROOT": str(PROJECT_ROOT / "artifacts"),
            "CRASHDIAG_ARTIFACT_UPLOAD_POLICY": "required",
            "CRASHDIAG_RUN_ID": run_id,
        }
    )
    print(f"CrashDiag Kaggle run: {run_id}")
    print(f"private artifact bucket: {args.bucket}")
    print(f"sandbox: {args.sandbox_url}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    return launch(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_BUCKET", "build_parser", "launch", "main"]
