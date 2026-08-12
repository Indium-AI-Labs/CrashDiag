"""Reset local artifacts/data and the private HF storage bucket.

Use before starting a fresh Qwen3-14B run.  This is destructive:
- deletes local data/, outputs/, artifacts/, scratchpad contents
- deletes the HF storage bucket named by CRASHDIAG_HF_BUCKET_ID
  (defaults to devaanshpa/CrashDiag) and recreates it as private so
  subsequent uploads work.

The script reads HF_TOKEN from .env (or CRASHDIAG_ENV_FILE).  It refuses to
run without a confirmed token.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path | None = None) -> Path:
    env_path = path or Path(
        os.environ.get("CRASHDIAG_ENV_FILE", PROJECT_ROOT / ".env")
    )
    env_path = env_path.expanduser()
    if not env_path.is_absolute():
        env_path = (Path.cwd() / env_path).resolve()
    if not env_path.is_file():
        raise SystemExit(f".env not found: {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
    return env_path


def reset_local() -> None:
    for name in ("data", "outputs", "artifacts"):
        path = PROJECT_ROOT / name
        if path.is_dir():
            shutil.rmtree(path)
            print(f"removed {path}")
        else:
            print(f"skipped missing {path}")


def reset_bucket(bucket_id: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        info = api.bucket_info(bucket_id)
        exists = True
    except Exception as exc:  # noqa: BLE001 - 404 or transient failure
        if getattr(exc, "response", None) is not None and exc.response.status_code == 404:
            exists = False
        else:
            raise
    if exists:
        api.delete_bucket(bucket_id)
        print(f"deleted HF storage bucket {bucket_id!r}")
    else:
        print(f"HF storage bucket {bucket_id!r} did not exist")
    api.create_bucket(bucket_id, private=True, exist_ok=False)
    print(f"recreated private HF storage bucket {bucket_id!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket-id",
        default=os.environ.get("CRASHDIAG_HF_BUCKET_ID", "devaanshpa/CrashDiag"),
        help="HF storage bucket to clear (default: devaanshpa/CrashDiag)",
    )
    parser.add_argument(
        "--skip-bucket",
        action="store_true",
        help="only clear local data/outputs/artifacts, leave the HF bucket alone",
    )
    args = parser.parse_args()

    load_env_file()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN not set; add it to .env first")

    reset_local()
    if not args.skip_bucket:
        reset_bucket(args.bucket_id, token)
    else:
        print("skipped HF storage bucket (--skip-bucket)")
    print("reset complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
