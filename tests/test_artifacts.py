"""Offline tests for private training-artifact persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from training.artifacts import (
    ArtifactConfig,
    ArtifactError,
    ArtifactUploader,
    artifact_config_from_args,
    build_parser,
    load_env_file,
    make_checkpoint_upload_callback,
)


class _BucketInfo:
    def __init__(self, private: bool) -> None:
        self.private = private


class _FakeApi:
    def __init__(self, *, private: bool = True) -> None:
        self.private = private
        self.operations: list[tuple[object, ...]] = []
        self.remote_paths: set[str] = set()
        self.bucket_files: dict[str, bytes] = {}
        self.download_writer = None

    def create_bucket(self, bucket_id: str, **kwargs: object) -> None:
        self.operations.append(("create", bucket_id, kwargs))

    def bucket_info(self, bucket_id: str) -> _BucketInfo:
        self.operations.append(("info", bucket_id))
        return _BucketInfo(self.private)

    def batch_bucket_files(
        self, bucket_id: str, *, add: list[tuple[str, str]]
    ) -> None:
        self.operations.append(("batch", bucket_id, add))
        self.remote_paths.update(remote for _, remote in add)
        for local, remote in add:
            self.bucket_files[remote] = Path(local).read_bytes()

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str, str | Path]],
        *,
        raise_on_missing_files: bool = False,
    ) -> None:
        self.operations.append(("download_files", bucket_id, files))
        for remote, local in files:
            if remote not in self.bucket_files:
                if raise_on_missing_files:
                    raise FileNotFoundError(remote)
                continue
            destination = Path(local)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.bucket_files[remote])

    def sync_bucket(self, source: str, destination: str) -> None:
        self.operations.append(("sync", source, destination))
        if source.startswith("hf://") and self.download_writer is not None:
            self.download_writer(Path(destination))

    def list_bucket_tree(
        self, bucket_id: str, *, prefix: str, recursive: bool
    ) -> list[object]:
        self.operations.append(("list", bucket_id, prefix, recursive))
        return [
            types.SimpleNamespace(path=path)
            for path in sorted(self.remote_paths)
            if path.startswith(prefix)
        ]


def _namespace(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "artifact_bucket": None,
        "artifact_prefix": None,
        "artifact_local_root": None,
        "run_id": None,
        "artifact_upload_policy": None,
        "create_artifact_bucket": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DotenvAndConfigTests(unittest.TestCase):
    def test_dotenv_does_not_override_injected_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "HF_TOKEN=from-file\nCRASHDIAG_HF_BUCKET_ID=owner/private\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HF_TOKEN": "from-runtime"}, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ["HF_TOKEN"], "from-runtime")
                self.assertEqual(
                    os.environ["CRASHDIAG_HF_BUCKET_ID"], "owner/private"
                )

    def test_bucket_configuration_implies_required_upload(self) -> None:
        environment = {
            "HF_TOKEN": "hf_private_value",
            "CRASHDIAG_HF_BUCKET_ID": "devaanshpa/CrashDiag",
            "CRASHDIAG_RUN_ID": "run-123",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = artifact_config_from_args(_namespace())
        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.policy, "required")
        self.assertEqual(config.remote_root, "runs/run-123")
        self.assertNotIn("hf_private_value", repr(config))
        self.assertNotIn("hf_private_value", repr(asdict(config)))

    def test_artifact_cli_accepts_settings_after_subcommand(self) -> None:
        args = build_parser().parse_args(
            [
                "preflight",
                "--artifact-bucket",
                "devaanshpa/CrashDiag",
                "--run-id",
                "run-123",
            ]
        )
        self.assertEqual(args.command, "preflight")
        self.assertEqual(args.artifact_bucket, "devaanshpa/CrashDiag")

    def test_missing_secret_fails_without_putting_secret_on_cli(self) -> None:
        environment = {
            "CRASHDIAG_HF_BUCKET_ID": "devaanshpa/CrashDiag",
            "CRASHDIAG_RUN_ID": "run-123",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ArtifactError, "HF_TOKEN"):
                artifact_config_from_args(_namespace())


class ArtifactUploaderTests(unittest.TestCase):
    def _config(self, root: Path, policy: str = "required") -> ArtifactConfig:
        return ArtifactConfig(
            bucket_id="devaanshpa/CrashDiag",
            run_id="run-123",
            prefix="runs",
            policy=policy,
            token="hf_do_not_serialize",
            local_root=root,
        )

    def test_public_bucket_is_rejected_before_any_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "result.json"
            artifact.write_text("{}\n", encoding="utf-8")
            api = _FakeApi(private=False)
            uploader = ArtifactUploader(self._config(root / "metadata"), api=api)

            with self.assertRaisesRegex(ArtifactError, "not private"):
                uploader.upload_files([artifact], "evaluation")

            self.assertFalse(any(item[0] in {"batch", "sync"} for item in api.operations))
            self.assertFalse(any(item[0] == "create" for item in api.operations))

    def test_stage_completion_can_be_queried_without_mutating_remote_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = _FakeApi()
            api.remote_paths.add("runs/run-123/calibration-wide-v1/_SUCCESS.json")
            uploader = ArtifactUploader(self._config(root / "metadata"), api=api)

            self.assertTrue(uploader.stage_is_complete("calibration-wide-v1"))
            self.assertFalse(uploader.stage_is_complete("grpo-hard"))
            self.assertFalse(any(item[0] in {"batch", "sync"} for item in api.operations))

    def test_files_manifest_then_success_marker_are_uploaded_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "sft_train.jsonl"
            second = root / "grpo_train.jsonl"
            first.write_text('{"sample":1}\n', encoding="utf-8")
            second.write_text('{"prompt":1}\n', encoding="utf-8")
            api = _FakeApi()
            metadata_root = root / "metadata"
            uploader = ArtifactUploader(self._config(metadata_root), api=api)

            self.assertTrue(
                uploader.upload_files(
                    [first, second],
                    "datasets",
                    metadata={"HF_TOKEN": "must-not-leak", "seed": 42},
                )
            )

            batches = [item for item in api.operations if item[0] == "batch"]
            self.assertEqual(len(batches), 2)
            first_additions = batches[0][2]
            final_additions = batches[1][2]
            self.assertEqual(
                {remote for _, remote in first_additions},
                {
                    "runs/run-123/datasets/sft_train.jsonl",
                    "runs/run-123/datasets/grpo_train.jsonl",
                    "runs/run-123/datasets/manifest.json",
                },
            )
            self.assertEqual(
                [remote for _, remote in final_additions],
                ["runs/run-123/datasets/_SUCCESS.json"],
            )

            manifest_path = metadata_root / "run-123" / "datasets" / "manifest.json"
            manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertNotIn("must-not-leak", manifest_text)
            self.assertNotIn("HF_TOKEN", manifest["metadata"])
            self.assertEqual(len(manifest["files"]), 2)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["files"]))

    def test_directory_sync_never_requests_remote_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            output.mkdir()
            (output / "adapter_config.json").write_text("{}", encoding="utf-8")
            api = _FakeApi()
            uploader = ArtifactUploader(self._config(root / "metadata"), api=api)

            uploader.upload_directory(output, "sft", partial=True)

            sync = next(item for item in api.operations if item[0] == "sync")
            self.assertEqual(sync[1], str(output))
            self.assertEqual(
                sync[2],
                "hf://buckets/devaanshpa/CrashDiag/runs/run-123/sft",
            )
            self.assertEqual(len(sync), 3)

    def test_selective_stage_download_verifies_signed_requested_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            checkpoint = output / "checkpoint-10"
            checkpoint.mkdir(parents=True)
            (output / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output / "adapter_model.safetensors").write_bytes(b"signed-adapter")
            (checkpoint / "optimizer.pt").write_bytes(b"large-checkpoint")
            api = _FakeApi()
            uploader = ArtifactUploader(self._config(root / "metadata"), api=api)
            uploader.upload_directory(output, "sft")

            # The fake sync records the directory but does not copy its payload,
            # so populate the same remote files a real bucket sync would hold.
            for path in output.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(output).as_posix()
                    remote = f"runs/run-123/sft/{relative}"
                    api.remote_paths.add(remote)
                    api.bucket_files[remote] = path.read_bytes()

            destination = root / "handoff"
            selected = ["adapter_config.json", "adapter_model.safetensors"]
            self.assertTrue(
                uploader.download_stage(
                    "sft",
                    destination,
                    include_paths=selected,
                )
            )
            self.assertTrue(
                uploader.verify_local_stage(
                    destination,
                    "sft",
                    include_paths=selected,
                )
            )
            self.assertTrue((destination / "adapter_model.safetensors").is_file())
            self.assertFalse((destination / "checkpoint-10").exists())
            downloaded = [
                operation
                for operation in api.operations
                if operation[0] == "download_files"
            ]
            self.assertEqual(len(downloaded), 2)

    def test_selective_stage_download_rejects_escaping_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uploader = ArtifactUploader(self._config(root / "metadata"), api=_FakeApi())
            for value in ("../secret", "/absolute", "C:/windows", "nested//file"):
                with self.subTest(value=value):
                    with self.assertRaises(ArtifactError):
                        uploader.verify_local_stage(
                            root,
                            "sft",
                            include_paths=[value],
                        )

    def test_final_directory_manifest_includes_nested_training_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            reports = output / "reports"
            reports.mkdir(parents=True)
            (output / "adapter_config.json").write_text("{}", encoding="utf-8")
            (reports / "loss.svg").write_text("<svg/>", encoding="utf-8")
            (reports / "metrics_summary.json").write_text(
                '{"loss":0.5}', encoding="utf-8"
            )
            api = _FakeApi()
            metadata_root = root / "metadata"
            uploader = ArtifactUploader(self._config(metadata_root), api=api)

            uploader.upload_directory(output, "sft")

            manifest_path = metadata_root / "run-123" / "sft" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {entry["path"] for entry in manifest["files"]},
                {
                    "adapter_config.json",
                    "reports/loss.svg",
                    "reports/metrics_summary.json",
                },
            )
            batches = [operation for operation in api.operations if operation[0] == "batch"]
            self.assertEqual(len(batches), 2)
            self.assertEqual(
                [remote for _, remote in batches[-1][2]],
                ["runs/run-123/sft/_SUCCESS.json"],
            )

    def test_checkpoint_callback_uploads_on_world_zero_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            output.mkdir()
            api = _FakeApi()
            uploader = ArtifactUploader(self._config(root / "metadata"), api=api)

            fake_transformers = types.ModuleType("transformers")
            fake_transformers.TrainerCallback = type("TrainerCallback", (), {})
            with patch.dict("sys.modules", {"transformers": fake_transformers}):
                callback = make_checkpoint_upload_callback(uploader, "grpo")

            args = types.SimpleNamespace(output_dir=str(output))
            control = object()
            callback.on_save(
                args,
                types.SimpleNamespace(is_world_process_zero=False),
                control,
            )
            self.assertFalse(any(item[0] == "sync" for item in api.operations))
            self.assertIs(
                callback.on_save(
                    args,
                    types.SimpleNamespace(is_world_process_zero=True),
                    control,
                ),
                control,
            )
            self.assertEqual(sum(item[0] == "sync" for item in api.operations), 1)

    def test_download_reverses_bucket_sync_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = _FakeApi()
            uploader = ArtifactUploader(self._config(root / "metadata"), api=api)
            destination = root / "download"
            api.remote_paths.add("runs/run-123/_SUCCESS.json")

            def write_download(target: Path) -> None:
                stage = target / "sft"
                stage.mkdir(parents=True)
                artifact = stage / "adapter_config.json"
                artifact.write_text("{}", encoding="utf-8")
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                manifest_path = stage / "manifest.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "run_id": "run-123",
                            "stage": "sft",
                            "files": [
                                {
                                    "path": artifact.name,
                                    "bytes": artifact.stat().st_size,
                                    "sha256": digest,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                manifest_digest = hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest()
                (stage / "_SUCCESS.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "run_id": "run-123",
                            "stage": "sft",
                            "manifest_sha256": manifest_digest,
                        }
                    ),
                    encoding="utf-8",
                )
                (target / "_SUCCESS.json").write_text(
                    json.dumps({"status": "complete", "run_id": "run-123"}),
                    encoding="utf-8",
                )

            api.download_writer = write_download

            uploader.download_run(destination)

            sync = next(item for item in api.operations if item[0] == "sync")
            self.assertEqual(
                sync,
                (
                    "sync",
                    "hf://buckets/devaanshpa/CrashDiag/runs/run-123",
                    str(destination),
                ),
            )
            operations_before_verify = len(api.operations)
            self.assertTrue(uploader.verify_local_run(destination))
            self.assertEqual(len(api.operations), operations_before_verify)

            (destination / "sft" / "manifest.json").write_text(
                '{"run_id":"run-123","stage":"sft","files":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ArtifactError, "does not match"):
                uploader.verify_local_run(destination)

    def test_incomplete_download_requires_explicit_recovery_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = _FakeApi()
            uploader = ArtifactUploader(self._config(root / "metadata"), api=api)

            with self.assertRaisesRegex(ArtifactError, "no run-level success"):
                uploader.download_run(root / "strict")

            def write_partial(target: Path) -> None:
                checkpoint = target / "sft" / "checkpoint-25"
                checkpoint.mkdir(parents=True)
                (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

            api.download_writer = write_partial
            self.assertTrue(
                uploader.download_run(root / "recovery", allow_incomplete=True)
            )

    def test_completed_stage_and_run_markers_cannot_be_blindly_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = _FakeApi()
            uploader = ArtifactUploader(self._config(root / "metadata"), api=api)
            api.remote_paths.add("runs/run-123/sft/_SUCCESS.json")

            with self.assertRaisesRegex(ArtifactError, "already complete"):
                uploader.start_stage("sft")

            api.remote_paths.remove("runs/run-123/sft/_SUCCESS.json")
            stages = ("datasets", "sft")
            for stage in stages:
                local = root / "metadata" / "run-123" / stage / "_SUCCESS.json"
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_text("{}", encoding="utf-8")
                api.remote_paths.add(f"runs/run-123/{stage}/_SUCCESS.json")

            self.assertTrue(uploader.complete_run({"stages": list(stages)}))
            self.assertIn("runs/run-123/_SUCCESS.json", api.remote_paths)

    def test_run_can_complete_from_verified_remote_stages_on_a_fresh_machine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = _FakeApi()
            stages = ("datasets", "calibration-wide-v1")
            api.remote_paths.update(
                f"runs/run-123/{stage}/_SUCCESS.json" for stage in stages
            )
            uploader = ArtifactUploader(self._config(root / "metadata"), api=api)

            self.assertTrue(uploader.complete_run({"stages": list(stages)}))
            self.assertIn("runs/run-123/_SUCCESS.json", api.remote_paths)


if __name__ == "__main__":
    unittest.main()
