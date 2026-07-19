"""Private Hugging Face Storage Bucket support for training artifacts.

The module stays dependency-light until an upload is requested.  In
particular, ``.env`` is loaded before ``huggingface_hub`` is imported and the
token is passed directly to ``HfApi``.  Tokens are never accepted as command
line arguments or written into manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_ARTIFACT_PREFIX = "runs"
UPLOAD_POLICIES = ("required", "best-effort", "disabled")
_SAFE_PATH_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_KEY = re.compile(
    r"token|secret|password|api[_-]?key|access[_-]?key|authorization|cookie|credential|private[_-]?key|bearer",
    re.IGNORECASE,
)


class ArtifactError(RuntimeError):
    """Raised when required private artifact persistence cannot be completed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _configured_env_file() -> Path:
    configured = os.environ.get("CRASHDIAG_ENV_FILE", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_ENV_FILE


def _parse_dotenv_fallback(path: Path) -> dict[str, str]:
    """Parse the small, conventional subset used by CrashDiag's ``.env``.

    ``python-dotenv`` is used when installed.  This fallback keeps local data
    generation dependency-free and deliberately never evaluates shell syntax.
    """

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ArtifactError(
                f"invalid .env assignment at {path}:{line_number}; expected NAME=value"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        if not _ENV_NAME.fullmatch(key):
            raise ArtifactError(f"invalid .env variable name at {path}:{line_number}")
        value = value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise ArtifactError(f"unterminated quoted value at {path}:{line_number}")
            value = value[1:-1]
            if quote == '"':
                value = (
                    value.replace(r"\n", "\n")
                    .replace(r"\r", "\r")
                    .replace(r"\t", "\t")
                    .replace(r'\"', '"')
                    .replace("\\\\", "\\")
                )
        else:
            # Whitespace followed by # starts a comment; a literal # inside a
            # token or URL is preserved.
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        values[key] = value
    return values


def load_env_file(path: str | os.PathLike[str] | None = None) -> Path:
    """Load one explicit env file without overriding injected environment.

    The default is the repository-root ``.env``.  Parent directories are not
    searched, which prevents an unrelated file from silently supplying a
    credential when commands are launched from elsewhere.
    """

    env_path = Path(path).expanduser() if path is not None else _configured_env_file()
    if not env_path.is_absolute():
        env_path = (Path.cwd() / env_path).resolve()
    if not env_path.is_file():
        return env_path

    try:
        from dotenv import dotenv_values
    except ImportError:
        values: Mapping[str, str | None] = _parse_dotenv_fallback(env_path)
    else:
        values = dotenv_values(env_path)

    for key, value in values.items():
        if value is not None and _ENV_NAME.fullmatch(key):
            os.environ.setdefault(key, value)
    return env_path


def preload_env(argv: Sequence[str] | None = None) -> Path:
    """Resolve ``--env-file`` early, then load it before Hub/ML imports."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-file", type=Path, default=_configured_env_file())
    args, _ = parser.parse_known_args(list(argv) if argv is not None else None)
    return load_env_file(args.env_file)


def add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common non-secret artifact options to a command parser."""

    group = parser.add_argument_group("private artifact storage")
    group.add_argument(
        "--env-file",
        type=Path,
        default=_configured_env_file(),
        help="dotenv file containing HF_TOKEN and bucket settings",
    )
    group.add_argument(
        "--artifact-bucket",
        default=None,
        metavar="NAMESPACE/BUCKET",
        help="private HF Storage Bucket (or CRASHDIAG_HF_BUCKET_ID)",
    )
    group.add_argument(
        "--artifact-prefix",
        default=None,
        help="remote prefix before the run ID (default: runs)",
    )
    group.add_argument(
        "--artifact-local-root",
        type=Path,
        default=None,
        help="local marker/log root (or CRASHDIAG_ARTIFACT_LOCAL_ROOT)",
    )
    group.add_argument(
        "--run-id",
        default=None,
        help="shared immutable run identifier (or CRASHDIAG_RUN_ID)",
    )
    group.add_argument(
        "--artifact-upload-policy",
        choices=UPLOAD_POLICIES,
        default=None,
        help="required, best-effort, or disabled; bucket configuration implies required",
    )
    group.add_argument(
        "--create-artifact-bucket",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="explicitly allow creation when the configured bucket does not exist",
    )


def _normalize_component(value: str, label: str) -> str:
    normalized = value.strip().strip("/")
    if not normalized or not _SAFE_PATH_PART.fullmatch(normalized):
        raise ArtifactError(
            f"{label} must contain only letters, digits, '.', '_' and '-'"
        )
    if normalized in {".", ".."}:
        raise ArtifactError(f"{label} cannot be {normalized!r}")
    return normalized


def _normalize_prefix(value: str) -> str:
    parts = value.strip().strip("/").split("/")
    if not parts or any(not part for part in parts):
        raise ArtifactError("artifact prefix cannot be empty")
    return "/".join(
        _normalize_component(part, "artifact prefix component") for part in parts
    )


def _normalize_bucket_id(value: str) -> str:
    parts = value.strip().strip("/").split("/")
    if len(parts) != 2:
        raise ArtifactError("artifact bucket must be NAMESPACE/BUCKET")
    return "/".join(
        _normalize_component(part, "artifact bucket component") for part in parts
    )


def _env_choice(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env_choice(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ArtifactError(f"{name} must be true or false")


@dataclass(frozen=True)
class ArtifactConfig:
    """Resolved private bucket settings; the credential is excluded from repr."""

    bucket_id: str
    run_id: str
    prefix: str = DEFAULT_ARTIFACT_PREFIX
    policy: str = "required"
    local_root: Path = PROJECT_ROOT / "artifacts"
    create_bucket: bool = False
    token: InitVar[str] = ""

    def __post_init__(self, token: str) -> None:
        object.__setattr__(self, "bucket_id", _normalize_bucket_id(self.bucket_id))
        object.__setattr__(self, "run_id", _normalize_component(self.run_id, "run ID"))
        object.__setattr__(self, "prefix", _normalize_prefix(self.prefix))
        if self.policy not in UPLOAD_POLICIES or self.policy == "disabled":
            raise ArtifactError("an enabled uploader requires required or best-effort policy")
        if not token.strip():
            raise ArtifactError("HF_TOKEN is required for private artifact uploads")
        object.__setattr__(self, "local_root", Path(self.local_root))
        object.__setattr__(self, "_token", token)

    def secret_token(self) -> str:
        """Return the in-memory credential without making it a dataclass field."""

        return str(getattr(self, "_token"))

    @property
    def remote_root(self) -> str:
        return f"{self.prefix}/{self.run_id}"

    @property
    def local_run_root(self) -> Path:
        return self.local_root / self.run_id


def artifact_config_from_args(args: argparse.Namespace) -> ArtifactConfig | None:
    """Resolve CLI and env settings after :func:`preload_env` has run."""

    bucket = args.artifact_bucket or _env_choice("CRASHDIAG_HF_BUCKET_ID")
    # Accept the shorter early prototype spelling as a compatibility alias.
    bucket = bucket or _env_choice("CRASHDIAG_HF_BUCKET")
    run_id = args.run_id or _env_choice("CRASHDIAG_RUN_ID")
    prefix = (
        args.artifact_prefix
        or _env_choice("CRASHDIAG_ARTIFACT_PREFIX")
        or DEFAULT_ARTIFACT_PREFIX
    )
    local_root_value = getattr(args, "artifact_local_root", None) or _env_choice(
        "CRASHDIAG_ARTIFACT_LOCAL_ROOT"
    )
    local_root = (
        Path(local_root_value).expanduser()
        if local_root_value is not None
        else PROJECT_ROOT / "artifacts"
    )
    if not local_root.is_absolute():
        local_root = (Path.cwd() / local_root).resolve()
    policy = (
        args.artifact_upload_policy
        or _env_choice("CRASHDIAG_ARTIFACT_UPLOAD_POLICY")
        or ("required" if bucket else "disabled")
    )
    if policy not in UPLOAD_POLICIES:
        raise ArtifactError(
            "CRASHDIAG_ARTIFACT_UPLOAD_POLICY must be required, best-effort, or disabled"
        )
    if policy == "disabled":
        return None

    missing: list[str] = []
    if not bucket:
        missing.append("CRASHDIAG_HF_BUCKET_ID")
    if not run_id:
        missing.append("CRASHDIAG_RUN_ID")
    token = _env_choice("HF_TOKEN")
    if not token:
        missing.append("HF_TOKEN")
    if missing:
        message = "artifact upload is enabled but these settings are missing: " + ", ".join(
            missing
        )
        if policy == "best-effort":
            warnings.warn(message + "; continuing with local artifacts only", stacklevel=2)
            return None
        raise ArtifactError(message)

    create_bucket = getattr(args, "create_artifact_bucket", None)
    if create_bucket is None:
        create_bucket = _env_bool("CRASHDIAG_CREATE_ARTIFACT_BUCKET", False)
    return ArtifactConfig(
        bucket_id=bucket,
        run_id=run_id,
        prefix=prefix,
        policy=policy,
        token=token,
        create_bucket=create_bucket,
        local_root=local_root,
    )


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
            if not _SECRET_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        lowered = value.lower()
        if (value.startswith("hf_") and len(value) > 12) or "bearer " in lowered:
            return "[REDACTED]"
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def runtime_metadata() -> dict[str, Any]:
    """Return useful, secret-free provenance for manifests and markers."""

    commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        candidate = result.stdout.strip()
        if candidate:
            commit = candidate
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                _sanitize(payload),
                temporary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            temporary.write("\n")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return path


def _status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _is_transient(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = _status_code(current)
        if status is not None:
            return status in {408, 409, 425, 429} or status >= 500
        if isinstance(current, (ConnectionError, TimeoutError, OSError)):
            return True
        if current.__class__.__module__.startswith("httpx") and (
            current.__class__.__name__.endswith("TransportError")
            or current.__class__.__name__.endswith("Timeout")
            or current.__class__.__name__.endswith("TimeoutException")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class ArtifactUploader:
    """Upload stage artifacts to one verified-private HF Storage Bucket."""

    def __init__(
        self,
        config: ArtifactConfig,
        *,
        api: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = 3,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be positive")
        self.config = config
        self._api_instance = api
        self._sleep = sleep
        self._attempts = attempts

    @property
    def api(self) -> Any:
        if self._api_instance is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise ArtifactError(
                    "huggingface_hub with Storage Bucket support is missing; "
                    "install `pip install -e .[train]`"
                ) from exc
            self._api_instance = HfApi(token=self.config.secret_token())
        return self._api_instance

    def _retry(self, operation: str, call: Callable[[], Any]) -> Any:
        for attempt in range(1, self._attempts + 1):
            try:
                return call()
            except Exception as exc:
                if attempt >= self._attempts or not _is_transient(exc):
                    raise ArtifactError(f"{operation} failed: {type(exc).__name__}") from exc
                self._sleep(float(2 ** (attempt - 1)))
        raise AssertionError("unreachable")

    def _run(self, operation: str, call: Callable[[], Any]) -> bool:
        try:
            call()
            return True
        except Exception as exc:
            if isinstance(exc, ArtifactError):
                if self.config.policy == "best-effort":
                    warnings.warn(str(exc), stacklevel=2)
                    return False
                raise
            error = ArtifactError(f"{operation} failed: {type(exc).__name__}")
            if self.config.policy == "best-effort":
                warnings.warn(str(error), stacklevel=2)
                return False
            raise error from exc

    def _ensure_private(self, *, allow_create: bool = True) -> None:
        try:
            info = self._retry(
                "inspect artifact bucket",
                lambda: self.api.bucket_info(self.config.bucket_id),
            )
        except ArtifactError as exc:
            cause = exc.__cause__
            if (
                not allow_create
                or not self.config.create_bucket
                or cause is None
                or _status_code(cause) != 404
            ):
                raise
            self._retry(
                "create private artifact bucket",
                lambda: self.api.create_bucket(
                    self.config.bucket_id,
                    private=True,
                    exist_ok=False,
                ),
            )
            info = self._retry(
                "inspect created artifact bucket",
                lambda: self.api.bucket_info(self.config.bucket_id),
            )
        if getattr(info, "private", None) is not True:
            raise ArtifactError(
                f"refusing to upload: HF bucket {self.config.bucket_id!r} is not private"
            )

    def _remote_paths(self, prefix: str) -> set[str]:
        items = self._retry(
            "list artifact bucket files",
            lambda: list(
                self.api.list_bucket_tree(
                    self.config.bucket_id,
                    prefix=prefix,
                    recursive=True,
                )
            ),
        )
        return {
            str(getattr(item, "path"))
            for item in items
            if getattr(item, "path", None) is not None
        }

    def _remote_exists(self, path: str) -> bool:
        return path in self._remote_paths(path)

    def _assert_run_open(self, stage: str | None = None) -> None:
        completed_run = f"{self.config.remote_root}/_SUCCESS.json"
        if self._remote_exists(completed_run):
            raise ArtifactError(
                f"run {self.config.run_id!r} is already complete; use a new run ID"
            )
        if stage is not None:
            completed_stage = (
                f"{self.config.remote_root}/{self._stage(stage)}/_SUCCESS.json"
            )
            if self._remote_exists(completed_stage):
                raise ArtifactError(
                    f"artifact stage {stage!r} is already complete; use a new run ID"
                )

    def _stage(self, stage: str) -> str:
        return _normalize_component(stage, "artifact stage")

    def remote_uri(self, stage: str | None = None) -> str:
        path = self.config.remote_root
        if stage is not None:
            path = f"{path}/{self._stage(stage)}"
        return f"hf://buckets/{self.config.bucket_id}/{path}"

    def _batch(self, additions: Sequence[tuple[Path, str]]) -> None:
        normalized = [(str(path), remote) for path, remote in additions]
        self._retry(
            "upload bucket files",
            lambda: self.api.batch_bucket_files(
                self.config.bucket_id,
                add=normalized,
            ),
        )

    def _marker_path(self, stage: str, name: str) -> Path:
        return self.config.local_run_root / self._stage(stage) / name

    def start_run(self, metadata: Mapping[str, Any] | None = None) -> bool:
        """Verify private write access and leave a run-level start marker."""

        def operation() -> None:
            self._ensure_private()
            self._assert_run_open()
            marker = _write_json(
                self.config.local_run_root / "_STARTED.json",
                {
                    "schema_version": 1,
                    "status": "started",
                    "run_id": self.config.run_id,
                    "created_at": _utc_now(),
                    "runtime": runtime_metadata(),
                    "metadata": metadata or {},
                },
            )
            self._batch([(marker, f"{self.config.remote_root}/_STARTED.json")])

        return self._run("artifact run preflight", operation)

    def start_stage(
        self, stage: str, metadata: Mapping[str, Any] | None = None
    ) -> bool:
        """Verify private write access before performing a stage's work."""

        stage_name = self._stage(stage)

        def operation() -> None:
            self._ensure_private()
            self._assert_run_open(stage_name)
            marker = _write_json(
                self._marker_path(stage_name, "_STARTED.json"),
                {
                    "schema_version": 1,
                    "status": "started",
                    "run_id": self.config.run_id,
                    "stage": stage_name,
                    "created_at": _utc_now(),
                    "runtime": runtime_metadata(),
                    "metadata": metadata or {},
                },
            )
            self._batch(
                [
                    (
                        marker,
                        f"{self.config.remote_root}/{stage_name}/_STARTED.json",
                    )
                ]
            )

        return self._run(f"artifact preflight for {stage_name}", operation)

    def _manifest(
        self,
        stage: str,
        files: Sequence[tuple[Path, str]],
        metadata: Mapping[str, Any] | None,
    ) -> Path:
        entries = [
            {
                "path": remote,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path, remote in sorted(files, key=lambda item: item[1])
        ]
        return _write_json(
            self._marker_path(stage, "manifest.json"),
            {
                "schema_version": 1,
                "run_id": self.config.run_id,
                "stage": stage,
                "created_at": _utc_now(),
                "files": entries,
                "metadata": metadata or {},
                "runtime": runtime_metadata(),
            },
        )

    def _success_marker(self, stage: str, manifest: Path) -> Path:
        return _write_json(
            self._marker_path(stage, "_SUCCESS.json"),
            {
                "schema_version": 1,
                "status": "complete",
                "run_id": self.config.run_id,
                "stage": stage,
                "completed_at": _utc_now(),
                "manifest_sha256": _sha256(manifest),
            },
        )

    def upload_files(
        self,
        files: Sequence[str | os.PathLike[str]],
        stage: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Upload exact files, a hash manifest, then ``_SUCCESS`` last."""

        stage_name = self._stage(stage)
        paths = [Path(path) for path in files]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ArtifactError("artifact files do not exist: " + ", ".join(missing))
        linked = [str(path) for path in paths if path.is_symlink()]
        if linked:
            raise ArtifactError("refusing symlink artifact files: " + ", ".join(linked))
        names = [path.name for path in paths]
        if len(set(names)) != len(names):
            raise ArtifactError("artifact files must have unique basenames")
        mapped = [(path, name) for path, name in zip(paths, names, strict=True)]

        def operation() -> None:
            self._ensure_private()
            manifest = self._manifest(stage_name, mapped, metadata)
            base = f"{self.config.remote_root}/{stage_name}"
            additions = [(path, f"{base}/{remote}") for path, remote in mapped]
            additions.append((manifest, f"{base}/manifest.json"))
            self._batch(additions)
            success = self._success_marker(stage_name, manifest)
            self._batch([(success, f"{base}/_SUCCESS.json")])

        return self._run(f"upload {stage_name} artifacts", operation)

    def upload_directory(
        self,
        directory: str | os.PathLike[str],
        stage: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        partial: bool = False,
    ) -> bool:
        """Incrementally sync a directory; final syncs get manifest and marker."""

        stage_name = self._stage(stage)
        root = Path(directory)
        if not root.is_dir():
            raise ArtifactError(f"artifact directory does not exist: {root}")
        linked = [str(path) for path in root.rglob("*") if path.is_symlink()]
        if linked:
            raise ArtifactError(
                "refusing artifact directory containing symlinks: " + ", ".join(linked)
            )

        def operation() -> None:
            self._ensure_private()
            self._retry(
                f"sync {stage_name} directory",
                lambda: self.api.sync_bucket(str(root), self.remote_uri(stage_name)),
            )
            if partial:
                return
            mapped = [
                (path, path.relative_to(root).as_posix())
                for path in root.rglob("*")
                if path.is_file() and not path.is_symlink()
            ]
            manifest = self._manifest(stage_name, mapped, metadata)
            base = f"{self.config.remote_root}/{stage_name}"
            self._batch([(manifest, f"{base}/manifest.json")])
            success = self._success_marker(stage_name, manifest)
            self._batch([(success, f"{base}/_SUCCESS.json")])

        return self._run(f"upload {stage_name} directory", operation)

    def complete_run(self, metadata: Mapping[str, Any] | None = None) -> bool:
        """Upload the run-level completion marker after every stage succeeds."""

        def operation() -> None:
            self._ensure_private()
            requested_stages = list((metadata or {}).get("stages", []))
            if not requested_stages:
                raise ArtifactError("at least one completed stage is required")
            missing_local = [
                stage
                for stage in requested_stages
                if not self._marker_path(str(stage), "_SUCCESS.json").is_file()
            ]
            missing_remote = [
                stage
                for stage in requested_stages
                if not self._remote_exists(
                    f"{self.config.remote_root}/{self._stage(str(stage))}/_SUCCESS.json"
                )
            ]
            if missing_local or missing_remote:
                raise ArtifactError(
                    "cannot complete run; missing stage success markers "
                    f"(local={missing_local}, remote={missing_remote})"
                )
            marker = _write_json(
                self.config.local_run_root / "_SUCCESS.json",
                {
                    "schema_version": 1,
                    "status": "complete",
                    "run_id": self.config.run_id,
                    "completed_at": _utc_now(),
                    "metadata": metadata or {},
                },
            )
            self._batch([(marker, f"{self.config.remote_root}/_SUCCESS.json")])

        return self._run("complete artifact run", operation)

    def download_run(
        self,
        destination: str | os.PathLike[str],
        *,
        allow_incomplete: bool = False,
    ) -> bool:
        """Download a run, requiring completion unless explicitly recovering."""

        target = Path(destination)

        def operation() -> None:
            self._ensure_private(allow_create=False)
            complete_path = f"{self.config.remote_root}/_SUCCESS.json"
            if not allow_incomplete and not self._remote_exists(complete_path):
                raise ArtifactError(
                    f"artifact run {self.config.run_id!r} has no run-level success marker"
                )
            if target.exists() and any(target.iterdir()):
                raise ArtifactError(f"download destination must be empty: {target}")
            target.mkdir(parents=True, exist_ok=True)
            self._retry(
                "download artifact run",
                lambda: self.api.sync_bucket(self.remote_uri(), str(target)),
            )
            self._verify_download(target, require_complete=not allow_incomplete)

        return self._run("download artifact run", operation)

    def download_stage(
        self,
        stage: str,
        destination: str | os.PathLike[str],
        *,
        include_paths: Sequence[str] | None = None,
    ) -> bool:
        """Download only signed files from one completed stage.

        ``include_paths`` permits a minimal cross-run handoff such as the final
        adapter and tokenizer without downloading retained optimizer
        checkpoints.  The stage manifest and success marker are always fetched
        first and verified; requested paths must be entries in that signed
        manifest.
        """

        stage_name = self._stage(stage)
        target = Path(destination)

        def operation() -> None:
            self._ensure_private(allow_create=False)
            if target.exists() and any(target.iterdir()):
                raise ArtifactError(f"download destination must be empty: {target}")
            target.mkdir(parents=True, exist_ok=True)
            remote_base = f"{self.config.remote_root}/{stage_name}"
            markers = [
                (f"{remote_base}/manifest.json", target / "manifest.json"),
                (f"{remote_base}/_SUCCESS.json", target / "_SUCCESS.json"),
            ]
            self._retry(
                f"download {stage_name} stage markers",
                lambda: self.api.download_bucket_files(
                    self.config.bucket_id,
                    markers,
                    raise_on_missing_files=True,
                ),
            )
            manifest = self._read_stage_manifest(target, stage_name)
            entries = manifest["files"]
            entry_map = {
                str(entry["path"]): entry
                for entry in entries
                if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
            }
            selected = (
                sorted(entry_map)
                if include_paths is None
                else [self._safe_manifest_relative(path) for path in include_paths]
            )
            if len(set(selected)) != len(selected):
                raise ArtifactError("stage include paths must be unique")
            missing = sorted(set(selected).difference(entry_map))
            if missing:
                raise ArtifactError(
                    "requested files are absent from the signed stage manifest: "
                    + ", ".join(missing)
                )
            downloads: list[tuple[str, Path]] = []
            for relative in selected:
                destination_path = target / Path(relative)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                downloads.append((f"{remote_base}/{relative}", destination_path))
            if downloads:
                self._retry(
                    f"download signed {stage_name} stage files",
                    lambda: self.api.download_bucket_files(
                        self.config.bucket_id,
                        downloads,
                        raise_on_missing_files=True,
                    ),
                )
            self._verify_stage_download(
                target,
                stage_name,
                include_paths=selected,
            )

        return self._run(f"download {stage_name} artifact stage", operation)

    def verify_local_stage(
        self,
        directory: str | os.PathLike[str],
        stage: str,
        *,
        include_paths: Sequence[str] | None = None,
    ) -> bool:
        """Verify a full or explicitly selected local stage handoff."""

        root = Path(directory)
        if not root.is_dir():
            raise ArtifactError(f"local artifact stage does not exist: {root}")
        selected = (
            None
            if include_paths is None
            else [self._safe_manifest_relative(path) for path in include_paths]
        )
        self._verify_stage_download(root, self._stage(stage), include_paths=selected)
        return True

    def verify_local_run(
        self,
        directory: str | os.PathLike[str],
        *,
        require_complete: bool = True,
    ) -> bool:
        """Verify manifests in an existing local run without contacting the Hub."""

        root = Path(directory)
        if not root.is_dir():
            raise ArtifactError(f"local artifact run does not exist: {root}")
        self._verify_download(root, require_complete=require_complete)
        return True

    @staticmethod
    def _safe_manifest_relative(value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ArtifactError("stage include paths must be non-empty strings")
        normalized = value.replace("\\", "/")
        if normalized.startswith("/"):
            raise ArtifactError(f"invalid stage include path: {value!r}")
        parts = normalized.split("/")
        if any(
            not part
            or part in {".", ".."}
            or not _SAFE_PATH_PART.fullmatch(part)
            for part in parts
        ):
            raise ArtifactError(f"invalid stage include path: {value!r}")
        return "/".join(parts)

    def _read_stage_manifest(self, root: Path, stage: str) -> dict[str, Any]:
        manifest_path = root / "manifest.json"
        success_path = root / "_SUCCESS.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            success = json.loads(success_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid completed stage markers: {stage}") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_id") != self.config.run_id
            or manifest.get("stage") != stage
            or not isinstance(manifest.get("files"), list)
        ):
            raise ArtifactError(f"artifact stage manifest identity mismatch: {stage}")
        if (
            not isinstance(success, Mapping)
            or success.get("status") != "complete"
            or success.get("run_id") != self.config.run_id
            or success.get("stage") != stage
            or success.get("manifest_sha256") != _sha256(manifest_path)
        ):
            raise ArtifactError(f"stage success marker does not match manifest: {stage}")
        return manifest

    def _verify_stage_download(
        self,
        root: Path,
        stage: str,
        *,
        include_paths: Sequence[str] | None,
    ) -> None:
        manifest = self._read_stage_manifest(root, stage)
        entry_map: dict[str, Mapping[str, Any]] = {}
        for entry in manifest["files"]:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
                raise ArtifactError(f"invalid artifact stage manifest: {stage}")
            relative = self._safe_manifest_relative(str(entry["path"]))
            if relative in entry_map:
                raise ArtifactError(f"duplicate path in artifact stage manifest: {relative}")
            entry_map[relative] = entry
        selected = sorted(entry_map) if include_paths is None else list(include_paths)
        missing_manifest = sorted(set(selected).difference(entry_map))
        if missing_manifest:
            raise ArtifactError(
                "requested files are absent from the signed stage manifest: "
                + ", ".join(missing_manifest)
            )
        for relative in selected:
            entry = entry_map[relative]
            candidate = root / Path(relative)
            if not candidate.is_file():
                raise ArtifactError(f"downloaded artifact is missing: {relative}")
            if candidate.stat().st_size != entry.get("bytes"):
                raise ArtifactError(f"downloaded artifact size mismatch: {relative}")
            if _sha256(candidate) != entry.get("sha256"):
                raise ArtifactError(f"downloaded artifact hash mismatch: {relative}")
        allowed = set(selected) | {"manifest.json", "_SUCCESS.json"}
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        extras = sorted(actual.difference(allowed))
        if extras:
            raise ArtifactError(
                "downloaded stage contains unsigned or unrequested files: "
                + ", ".join(extras)
            )

    def _verify_download(self, root: Path, *, require_complete: bool) -> None:
        run_success = root / "_SUCCESS.json"
        if require_complete and not run_success.is_file():
            raise ArtifactError("download is missing the run-level success marker")
        if run_success.is_file():
            try:
                run_marker = json.loads(run_success.read_text(encoding="utf-8"))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ArtifactError("invalid run-level success marker") from exc
            if (
                not isinstance(run_marker, Mapping)
                or run_marker.get("status") != "complete"
                or run_marker.get("run_id") != self.config.run_id
            ):
                raise ArtifactError("invalid run-level success marker")
        manifests = sorted(root.glob("*/manifest.json"))
        if not manifests and require_complete:
            raise ArtifactError("download contains no stage manifests")
        if not manifests and not any(path.is_file() for path in root.rglob("*")):
            raise ArtifactError("downloaded artifact prefix is empty")
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                entries = manifest["files"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"invalid artifact manifest: {manifest_path}") from exc
            if not isinstance(entries, list):
                raise ArtifactError(f"invalid artifact manifest: {manifest_path}")
            stage = manifest_path.parent.name
            if (
                manifest.get("run_id") != self.config.run_id
                or manifest.get("stage") != stage
            ):
                raise ArtifactError(
                    f"artifact manifest identity mismatch: {manifest_path}"
                )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ArtifactError(f"invalid artifact manifest: {manifest_path}")
                relative = Path(str(entry.get("path", "")))
                candidate = (manifest_path.parent / relative).resolve()
                try:
                    candidate.relative_to(manifest_path.parent.resolve())
                except ValueError as exc:
                    raise ArtifactError(
                        f"manifest path escapes its stage: {relative}"
                    ) from exc
                if not candidate.is_file():
                    raise ArtifactError(f"downloaded artifact is missing: {relative}")
                if candidate.stat().st_size != entry.get("bytes"):
                    raise ArtifactError(f"downloaded artifact size mismatch: {relative}")
                if _sha256(candidate) != entry.get("sha256"):
                    raise ArtifactError(f"downloaded artifact hash mismatch: {relative}")
            stage_success = manifest_path.parent / "_SUCCESS.json"
            if require_complete and not stage_success.is_file():
                raise ArtifactError(
                    f"completed run is missing a stage success marker: {stage}"
                )
            if stage_success.is_file():
                try:
                    stage_marker = json.loads(
                        stage_success.read_text(encoding="utf-8")
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ArtifactError(
                        f"invalid stage success marker: {stage_success}"
                    ) from exc
                if (
                    not isinstance(stage_marker, Mapping)
                    or stage_marker.get("status") != "complete"
                    or stage_marker.get("run_id") != self.config.run_id
                    or stage_marker.get("stage") != stage
                    or stage_marker.get("manifest_sha256") != _sha256(manifest_path)
                ):
                    raise ArtifactError(
                        f"stage success marker does not match its manifest: {stage}"
                    )
        for stage_success in root.glob("*/_SUCCESS.json"):
            if not (stage_success.parent / "manifest.json").is_file():
                raise ArtifactError(
                    f"stage success marker has no manifest: {stage_success.parent.name}"
                )


def make_checkpoint_upload_callback(
    uploader: ArtifactUploader | None,
    stage: str,
) -> Any | None:
    """Return a Transformers callback that syncs each rank-zero checkpoint."""

    if uploader is None:
        return None
    from transformers import TrainerCallback

    class CheckpointUploadCallback(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if getattr(state, "is_world_process_zero", False):
                uploader.upload_directory(args.output_dir, stage, partial=True)
            return control

    return CheckpointUploadCallback()


def process_is_world_zero() -> bool:
    """Best-effort rank check usable before Accelerate/torch imports."""

    value = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(value) == 0
    except ValueError:
        return value in {"", "0"}


def uploader_from_args(args: argparse.Namespace) -> ArtifactUploader | None:
    config = artifact_config_from_args(args)
    return ArtifactUploader(config) if config is not None else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    add_artifact_arguments(common)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "preflight",
        parents=[common],
        help="verify private bucket write access",
    )
    upload = subparsers.add_parser(
        "upload",
        parents=[common],
        help="sync one local directory",
    )
    upload.add_argument("--stage", required=True)
    upload.add_argument("--path", type=Path, required=True)
    upload.add_argument("--partial", action="store_true")
    complete = subparsers.add_parser(
        "complete",
        parents=[common],
        help="mark the whole run complete",
    )
    complete.add_argument("--stages", nargs="*", default=[])
    download = subparsers.add_parser(
        "download",
        parents=[common],
        help="download a run prefix",
    )
    download.add_argument("--destination", type=Path, required=True)
    download.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="recovery only: permit a partial run without a run-level success marker",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    preload_env(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        uploader = uploader_from_args(args)
        if uploader is None:
            raise ArtifactError("artifact upload is disabled")
        if args.command == "preflight":
            uploader.start_run()
        elif args.command == "upload":
            uploader.upload_directory(args.path, args.stage, partial=args.partial)
        elif args.command == "complete":
            uploader.complete_run({"stages": args.stages})
        elif args.command == "download":
            uploader.download_run(
                args.destination,
                allow_incomplete=args.allow_incomplete,
            )
        else:
            parser.error(f"unknown command: {args.command}")
    except ArtifactError as exc:
        parser.exit(2, f"artifact error: {exc}\n")
    print(f"artifact run: {uploader.remote_uri()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtifactConfig",
    "ArtifactError",
    "ArtifactUploader",
    "DEFAULT_ENV_FILE",
    "add_artifact_arguments",
    "artifact_config_from_args",
    "load_env_file",
    "make_checkpoint_upload_callback",
    "preload_env",
    "process_is_world_zero",
    "runtime_metadata",
    "uploader_from_args",
]
