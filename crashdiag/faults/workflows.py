"""Composite, multi-action CrashDiag workflows.

A :class:`WorkflowFault` injects several sub-faults and is resolved only when every
sub-fault returns to its healthy state *and* the sandbox reports healthy.  Each
workflow carries an ordered expert action list used only to prove mechanical
solvability during dataset generation; it is never serialized into GRPO rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import FaultModule
from .modules import (
    ApiKeyExpired,
    BadEnvVar,
    BrokenDBConnection,
    CacheConfigMisconfigured,
    CacheCorruption,
    CircuitBreakerOpen,
    ContainerImageDrift,
    CronSkipped,
    DatabaseConfigMisconfigured,
    DatabasePoolExhaustion,
    DNSResolutionFailure,
    DeadLetterBacklog,
    DependencyMismatch,
    DiskFull,
    EnvVarCaseMismatch,
    FeatureFlagMisconfiguration,
    FilePermissionFailure,
    LoadBalancerMisroute,
    LogVolumeExceeded,
    MessageQueueConnectionFailure,
    MissingSecret,
    NetworkConfigMisconfigured,
    ObjectStorageCredentialsFailure,
    OOMKill,
    PortProxyMisconfig,
    RateLimitMisconfiguration,
    RedisConnectionFailure,
    ReplicaLag,
    SchemaMigrationPending,
    SearchIndexStale,
    StaleSecretRotation,
    TLSCertificateFailure,
    TempFilePressure,
    TlsConfigMisconfigured,
    WorkerDown,
)


class WorkflowFault(FaultModule):
    """An ordered collection of sub-faults repaired by an ordered action list."""

    def __init__(
        self,
        name: str,
        difficulty: str,
        sub_faults: tuple[FaultModule, ...],
        actions: tuple[str, ...],
    ) -> None:
        if not name:
            raise ValueError("workflow name must be non-empty")
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be easy, medium, or hard")
        if not sub_faults:
            raise ValueError("workflow requires at least one sub-fault")
        if not actions:
            raise ValueError("workflow requires at least one expert action")
        if len(set(sub_faults)) != len(sub_faults):
            raise ValueError("workflow sub-faults must be distinct")
        self._name = name
        self._difficulty = difficulty
        self.sub_faults = sub_faults
        self.actions = actions

    @property
    def name(self) -> str:
        return self._name

    @property
    def difficulty(self) -> str:
        return self._difficulty

    @property
    def subfault_count(self) -> int:
        return len(self.sub_faults)

    def inject(self, instance: Any) -> None:
        for sub_fault in self.sub_faults:
            sub_fault.inject(instance)

    def local_resolved(self, instance: Any) -> bool:
        return all(sub_fault.local_resolved(instance) for sub_fault in self.sub_faults)

    def resolved_subfault_count(self, instance: Any) -> int:
        return sum(1 for sub_fault in self.sub_faults if sub_fault.local_resolved(instance))

    def is_resolved(self, instance: Any) -> bool:
        return self.local_resolved(instance) and self._mechanically_healthy(instance)

    @staticmethod
    def _mechanically_healthy(instance: Any) -> bool:
        health_check = getattr(instance, "health_check", None)
        if not callable(health_check):
            return False
        result = health_check()
        if isinstance(result, bool):
            return result
        if isinstance(result, Mapping):
            return result.get("healthy") is True
        return False


_W = WorkflowFault

WORKFLOWS: dict[str, WorkflowFault] = {
    workflow.name: workflow
    for workflow in (
        _W("oom_kill", "medium", (OOMKill(), CacheCorruption()), ("restart_app", "clear_cache")),
        _W("bad_env_var", "easy", (BadEnvVar(),), ("rollback_env_var",)),
        _W("broken_db_connection", "medium", (BrokenDBConnection(),), ("rollback_env_var",)),
        _W("dependency_mismatch", "hard", (DependencyMismatch(), OOMKill()), ("fix_dependency", "restart_app")),
        _W("disk_full", "medium", (DiskFull(),), ("clear_disk",)),
        _W("port_proxy_misconfig", "easy", (PortProxyMisconfig(),), ("fix_port_config",)),
        _W("missing_secret", "medium", (MissingSecret(),), ("rollback_env_var",)),
        _W("feature_flag_misconfiguration", "medium", (FeatureFlagMisconfiguration(), WorkerDown()), ("rollback_env_var", "restart_worker")),
        _W("redis_connection_failure", "medium", (RedisConnectionFailure(),), ("rollback_env_var",)),
        _W("message_queue_connection_failure", "medium", (MessageQueueConnectionFailure(),), ("rollback_env_var",)),
        _W("object_storage_credentials_failure", "medium", (ObjectStorageCredentialsFailure(),), ("rollback_env_var",)),
        _W("cache_corruption", "medium", (CacheCorruption(), WorkerDown()), ("clear_cache", "restart_worker")),
        _W("tls_certificate_failure", "hard", (TLSCertificateFailure(),), ("renew_tls_certificate",)),
        _W("file_permission_failure", "medium", (FilePermissionFailure(),), ("restore_file_permissions",)),
        _W("schema_migration_pending", "hard", (SchemaMigrationPending(), OOMKill()), ("apply_database_migration", "restart_app")),
        _W("database_pool_exhaustion", "hard", (DatabasePoolExhaustion(),), ("reset_database_pool",)),
        _W("dns_resolution_failure", "hard", (DNSResolutionFailure(),), ("restore_dns_configuration",)),
        _W("rate_limit_misconfiguration", "medium", (RateLimitMisconfiguration(),), ("restore_rate_limit_configuration",)),
        _W("zombie_process_accumulation", "hard", (WorkerDown(), OOMKill()), ("restart_worker", "restart_app")),
        _W("container_image_drift", "hard", (ContainerImageDrift(), OOMKill()), ("redeploy_container", "restart_app")),
        _W("memory_pressure_high", "hard", (OOMKill(), CacheCorruption()), ("restart_app", "clear_cache")),
        _W("cpu_throttling", "medium", (WorkerDown(), OOMKill()), ("restart_worker", "restart_app")),
        _W("file_descriptor_exhaustion", "hard", (OOMKill(), FilePermissionFailure()), ("restart_app", "restore_file_permissions")),
        _W("stale_secret_rotation", "hard", (StaleSecretRotation(),), ("rollback_env_var",)),
        _W("api_key_expired", "medium", (ApiKeyExpired(),), ("rollback_env_var",)),
        _W("env_var_case_mismatch", "easy", (EnvVarCaseMismatch(),), ("rollback_env_var",)),
        _W("transitive_dependency_conflict", "hard", (DependencyMismatch(), OOMKill()), ("fix_dependency", "restart_app")),
        _W("package_registry_unreachable", "medium", (DependencyMismatch(), DNSResolutionFailure()), ("fix_dependency", "restore_dns_configuration")),
        _W("inode_exhaustion", "hard", (TempFilePressure(), DiskFull()), ("clear_temp_files", "clear_disk")),
        _W("log_rotation_failure", "medium", (LogVolumeExceeded(), WorkerDown()), ("rotate_logs", "restart_worker")),
        _W("tmp_partition_full", "medium", (DiskFull(),), ("clear_disk",)),
        _W("load_balancer_misroute", "hard", (LoadBalancerMisroute(), OOMKill()), ("restore_load_balancer_config", "restart_app")),
        _W("firewall_port_block", "medium", (NetworkConfigMisconfigured(), PortProxyMisconfig()), ("restore_network_config", "fix_port_config")),
        _W("mtu_mismatch", "hard", (NetworkConfigMisconfigured(),), ("restore_network_config",)),
        _W("upstream_health_check_failure", "medium", (OOMKill(), LoadBalancerMisroute()), ("restart_app", "restore_load_balancer_config")),
        _W("replica_lag_exceeded", "hard", (ReplicaLag(), DatabasePoolExhaustion()), ("sync_replica", "reset_database_pool")),
        _W("read_write_split_misconfig", "hard", (DatabaseConfigMisconfigured(), OOMKill()), ("restore_database_config", "restart_app")),
        _W("connection_pool_size_too_small", "medium", (DatabasePoolExhaustion(),), ("reset_database_pool",)),
        _W("deadlock_transaction", "medium", (DatabasePoolExhaustion(), SchemaMigrationPending()), ("reset_database_pool", "apply_database_migration")),
        _W("queue_dead_letter_backlog", "hard", (DeadLetterBacklog(), WorkerDown()), ("flush_dead_letter_queue", "restart_worker")),
        _W("cache_eviction_policy_misconfig", "medium", (CacheConfigMisconfigured(), CacheCorruption()), ("restore_cache_config", "clear_cache")),
        _W("message_serialization_error", "medium", (DependencyMismatch(), WorkerDown()), ("fix_dependency", "restart_worker")),
        _W("rate_limiter_cascade_failure", "hard", (RateLimitMisconfiguration(), CircuitBreakerOpen()), ("restore_rate_limit_configuration", "reset_circuit_breaker")),
        _W("circuit_breaker_open", "hard", (CircuitBreakerOpen(),), ("reset_circuit_breaker",)),
        _W("feature_flag_dependency_chain", "hard", (FeatureFlagMisconfiguration(), OOMKill()), ("rollback_env_var", "restart_app")),
        _W("cron_job_skipped", "medium", (CronSkipped(), WorkerDown()), ("restore_cron_schedule", "restart_worker")),
        _W("search_index_stale", "medium", (SearchIndexStale(), WorkerDown()), ("rebuild_index", "restart_worker")),
        _W("tls_chain_incomplete", "hard", (TlsConfigMisconfigured(), TLSCertificateFailure()), ("restore_tls_config", "renew_tls_certificate")),
        _W("certificate_authority_missing", "hard", (TLSCertificateFailure(), FilePermissionFailure()), ("renew_tls_certificate", "restore_file_permissions")),
        _W("hsts_misconfig", "medium", (TlsConfigMisconfigured(),), ("restore_tls_config",)),
        _W("monitoring_agent_offline", "easy", (WorkerDown(),), ("restart_worker",)),
        _W("configuration_drift_detected", "hard", (BadEnvVar(), DependencyMismatch(), PortProxyMisconfig()), ("rollback_env_var", "fix_dependency", "fix_port_config")),
    )
}

WORKFLOW_NAMES = tuple(WORKFLOWS)


def workflow_for_name(name: str) -> WorkflowFault:
    """Return the workflow definition for a stable task name."""

    if not isinstance(name, str):
        raise TypeError("workflow name must be a string")
    try:
        return WORKFLOWS[name]
    except KeyError as exc:
        raise ValueError(f"unknown workflow {name!r}") from exc


__all__ = ["WORKFLOW_NAMES", "WORKFLOWS", "WorkflowFault", "workflow_for_name"]
