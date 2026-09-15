"""Victim runtime whose health is derived from real processes and files.

This process runs inside a disposable Compose stack.  ``DockerSandbox`` talks to
it over the in-container HTTP control plane.  Tests may also construct it
directly; the mechanical contract matches :class:`MockSandbox` so existing
fault modules and v6 scenario preparation keep working.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from crashdiag.sandbox_apps.mock import MockSandbox


_APP_CHILD = (
    "import time, sys\n"
    "sys.stderr.write('crashdiag-app-child ready\\n')\n"
    "sys.stderr.flush()\n"
    "while True:\n"
    "    time.sleep(30)\n"
)
_WORKER_CHILD = (
    "import time, sys\n"
    "sys.stderr.write('crashdiag-worker-child ready\\n')\n"
    "sys.stderr.flush()\n"
    "while True:\n"
    "    time.sleep(30)\n"
)


class VictimRuntime(MockSandbox):
    """Mock contract backed by child processes, tmpfs fillers, and flag files."""

    DISK_QUOTA_BYTES = 8 * 1024 * 1024

    def __init__(self, root: str | Path | None = None) -> None:
        default_root = Path(os.environ.get("CRASHDIAG_VICTIM_ROOT", "/var/lib/crashdiag"))
        if root is not None:
            self._root = Path(root)
            self._root.mkdir(parents=True, exist_ok=True)
            self._owns_root = False
        elif default_root.is_dir() or os.environ.get("CRASHDIAG_VICTIM_ROOT"):
            self._root = default_root
            self._root.mkdir(parents=True, exist_ok=True)
            self._owns_root = False
        else:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="crashdiag-victim-")
            self._root = Path(self._tmpdir.name)
            self._owns_root = True
        self._disk_dir = Path(os.environ.get("CRASHDIAG_DISK_ROOT", str(self._root / "disk")))
        self._disk_dir.mkdir(parents=True, exist_ok=True)
        self._service_dir = self._root / "services"
        self._service_dir.mkdir(parents=True, exist_ok=True)
        self._perm_file = self._root / "app.key"
        self._cert_file = self._root / "tls.crt"
        self._image_file = self._root / "image.txt"
        self._env_file = self._root / "env.json"
        self._app_proc: subprocess.Popen[bytes] | None = None
        self._worker_proc: subprocess.Popen[bytes] | None = None
        super().__init__()
        self._write_perm_file(healthy=True)
        self._write_cert(healthy=True)
        self._image_file.write_text("crashdiag-victim:declared\n", encoding="utf-8")
        self._sync_service_files()
        self._sync_disk_file()
        self._start_app()
        self._start_worker()

    def close(self) -> None:
        """Stop child processes and release a privately owned work directory."""

        self._stop_proc("_app_proc")
        self._stop_proc("_worker_proc")
        tmpdir = getattr(self, "_tmpdir", None)
        if tmpdir is not None:
            tmpdir.cleanup()
            self._tmpdir = None

    def __enter__(self) -> "VictimRuntime":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _start_app(self) -> None:
        self._stop_proc("_app_proc")
        self._app_proc = subprocess.Popen(
            [sys.executable, "-c", _APP_CHILD],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.process_running = True
        self.last_exit_reason = None

    def _start_worker(self) -> None:
        self._stop_proc("_worker_proc")
        self._worker_proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER_CHILD],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.services["worker"] = True

    def _stop_proc(self, attr: str) -> None:
        proc: subprocess.Popen[bytes] | None = getattr(self, attr)
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        setattr(self, attr, None)

    def _app_alive(self) -> bool:
        return self._app_proc is not None and self._app_proc.poll() is None

    def _worker_alive(self) -> bool:
        return self._worker_proc is not None and self._worker_proc.poll() is None

    def _perm_ok(self) -> bool:
        if not self._perm_file.is_file():
            return False
        try:
            return self._perm_file.read_bytes() == b"service-account-key\n"
        except OSError:
            return False

    def _write_perm_file(self, *, healthy: bool) -> None:
        self._perm_file.write_bytes(
            b"service-account-key\n" if healthy else b"denied\n"
        )
        try:
            os.chmod(self._perm_file, 0o640 if healthy else 0o600)
        except OSError:
            pass

    def _write_cert(self, *, healthy: bool) -> None:
        if healthy:
            self._cert_file.write_text("-----BEGIN CERTIFICATE-----\nVALID\n", encoding="utf-8")
        elif self._cert_file.exists():
            self._cert_file.write_text("-----BEGIN CERTIFICATE-----\nEXPIRED\n", encoding="utf-8")

    def _sync_service_files(self) -> None:
        for name, healthy in self.services.items():
            path = self._service_dir / name
            path.write_text("1" if healthy else "0", encoding="utf-8")

    def _sync_disk_file(self) -> None:
        filler = self._disk_dir / "filler.bin"
        target = int(self.DISK_QUOTA_BYTES * max(0.0, min(self.disk_usage_percent, 100.0)) / 100.0)
        filler.write_bytes(b"\0" * target)

    def _measured_disk_percent(self) -> float:
        filler = self._disk_dir / "filler.bin"
        used = filler.stat().st_size if filler.is_file() else 0
        return min(100.0, 100.0 * used / float(self.DISK_QUOTA_BYTES))

    def observe(self) -> dict[str, Any]:
        self.process_running = self._app_alive()
        if not self.process_running and self.last_exit_reason is None:
            self.last_exit_reason = "exited"
        snapshot = super().observe()
        snapshot["runtime"] = {
            "backend": "docker-victim",
            "image": self._image_file.read_text(encoding="utf-8").strip(),
            "disk_bytes": int(self.DISK_QUOTA_BYTES * self.disk_usage_percent / 100.0),
        }
        return snapshot

    def health_check(self) -> dict[str, Any]:
        self.process_running = self._app_alive()
        return super().health_check()

    def trigger_oom_kill(self) -> None:
        self._stop_proc("_app_proc")
        super().trigger_oom_kill()

    def restart_app(self) -> dict[str, Any]:
        self._start_app()
        return super().restart_app()

    def restart_worker(self) -> dict[str, Any]:
        result = super().restart_worker()
        self._start_worker()
        self._sync_service_files()
        return result

    def redeploy_container(self) -> dict[str, Any]:
        result = super().redeploy_container()
        self._image_file.write_text("crashdiag-victim:declared\n", encoding="utf-8")
        self._start_app()
        self._sync_service_files()
        return result

    def set_disk_usage(self, percent: float) -> None:
        super().set_disk_usage(percent)
        self._sync_disk_file()

    def clear_disk(self, target_percent: float = 40.0) -> dict[str, Any]:
        result = super().clear_disk(target_percent)
        self._sync_disk_file()
        return result

    def set_service_state(self, name: str, healthy: bool) -> None:
        super().set_service_state(name, healthy)
        if name == "worker" and not healthy:
            self._stop_proc("_worker_proc")
        elif name == "worker" and healthy:
            self._start_worker()
        elif name == "permissions":
            self._write_perm_file(healthy=healthy)
        elif name == "tls":
            self._write_cert(healthy=healthy)
        elif name == "container" and not healthy:
            self._image_file.write_text("crashdiag-victim:drifted\n", encoding="utf-8")
        self._sync_service_files()

    def restore_file_permissions(self) -> dict[str, Any]:
        result = super().restore_file_permissions()
        self._write_perm_file(healthy=True)
        self._sync_service_files()
        return result

    def renew_tls_certificate(self) -> dict[str, Any]:
        result = super().renew_tls_certificate()
        self._write_cert(healthy=True)
        self._sync_service_files()
        return result

    def restore_tls_config(self) -> dict[str, Any]:
        result = super().restore_tls_config()
        self._write_cert(healthy=True)
        self._sync_service_files()
        return result

    def _restore_service(self, service: str, action: str) -> dict[str, Any]:
        result = super()._restore_service(service, action)
        if service == "worker":
            self._start_worker()
        elif service == "permissions":
            self._write_perm_file(healthy=True)
        elif service in {"tls", "tls_config"}:
            self._write_cert(healthy=True)
        elif service == "container":
            self._image_file.write_text("crashdiag-victim:declared\n", encoding="utf-8")
        elif service == "temp":
            for leftover in self._disk_dir.glob("tmp-*"):
                leftover.unlink(missing_ok=True)
        self._sync_service_files()
        return result


__all__ = ["VictimRuntime"]
