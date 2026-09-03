# CrashDiag v6 Task Catalog

The v6 curriculum has **52 tasks**. Each task injects one or more *sub-faults* and is
resolved by an **ordered multi-action workflow**. Resolution is mechanical: every
sub-fault must return to its healthy state and the sandbox must report `healthy: true`.

Difficulty tiers are `easy`, `medium`, `hard`.

| # | task name | category | difficulty | sub-faults | expert workflow |
|---|---|---|---|---|---|
| 1 | oom_kill | process | medium | 2 | restart_app, clear_cache |
| 2 | bad_env_var | environment | easy | 1 | rollback_env_var |
| 3 | broken_db_connection | database | medium | 1 | rollback_env_var |
| 4 | dependency_mismatch | dependency | hard | 2 | fix_dependency, restart_app |
| 5 | disk_full | disk | medium | 1 | clear_disk |
| 6 | port_proxy_misconfig | network | easy | 1 | fix_port_config |
| 7 | missing_secret | secret | medium | 1 | rollback_env_var |
| 8 | feature_flag_misconfiguration | environment | medium | 2 | rollback_env_var, restart_worker |
| 9 | redis_connection_failure | cache | medium | 1 | rollback_env_var |
| 10 | message_queue_connection_failure | queue | medium | 1 | rollback_env_var |
| 11 | object_storage_credentials_failure | secret | medium | 1 | rollback_env_var |
| 12 | cache_corruption | cache | medium | 2 | clear_cache, restart_worker |
| 13 | tls_certificate_failure | tls | hard | 1 | renew_tls_certificate |
| 14 | file_permission_failure | security | medium | 1 | restore_file_permissions |
| 15 | schema_migration_pending | database | hard | 2 | apply_database_migration, restart_app |
| 16 | database_pool_exhaustion | database | hard | 1 | reset_database_pool |
| 17 | dns_resolution_failure | network | hard | 1 | restore_dns_configuration |
| 18 | rate_limit_misconfiguration | network | medium | 1 | restore_rate_limit_configuration |
| 19 | zombie_process_accumulation | process | hard | 2 | restart_worker, restart_app |
| 20 | container_image_drift | deployment | hard | 2 | redeploy_container, restart_app |
| 21 | memory_pressure_high | resource | hard | 2 | restart_app, clear_cache |
| 22 | cpu_throttling | resource | medium | 2 | restart_worker, restart_app |
| 23 | file_descriptor_exhaustion | resource | hard | 2 | restart_app, restore_file_permissions |
| 24 | stale_secret_rotation | secret | hard | 1 | rollback_env_var |
| 25 | api_key_expired | secret | medium | 1 | rollback_env_var |
| 26 | env_var_case_mismatch | environment | easy | 1 | rollback_env_var |
| 27 | transitive_dependency_conflict | dependency | hard | 2 | fix_dependency, restart_app |
| 28 | package_registry_unreachable | dependency | medium | 2 | fix_dependency, restore_dns_configuration |
| 29 | inode_exhaustion | disk | hard | 2 | clear_temp_files, clear_disk |
| 30 | log_rotation_failure | disk | medium | 2 | rotate_logs, restart_worker |
| 31 | tmp_partition_full | disk | medium | 1 | clear_disk |
| 32 | load_balancer_misroute | network | hard | 2 | restore_load_balancer_config, restart_app |
| 33 | firewall_port_block | network | medium | 2 | restore_network_config, fix_port_config |
| 34 | mtu_mismatch | network | hard | 1 | restore_network_config |
| 35 | upstream_health_check_failure | network | medium | 2 | restart_app, restore_load_balancer_config |
| 36 | replica_lag_exceeded | database | hard | 2 | sync_replica, reset_database_pool |
| 37 | read_write_split_misconfig | database | hard | 2 | restore_database_config, restart_app |
| 38 | connection_pool_size_too_small | database | medium | 1 | reset_database_pool |
| 39 | deadlock_transaction | database | medium | 2 | reset_database_pool, apply_database_migration |
| 40 | queue_dead_letter_backlog | queue | hard | 2 | flush_dead_letter_queue, restart_worker |
| 41 | cache_eviction_policy_misconfig | cache | medium | 2 | restore_cache_config, clear_cache |
| 42 | message_serialization_error | queue | medium | 2 | fix_dependency, restart_worker |
| 43 | rate_limiter_cascade_failure | network | hard | 2 | restore_rate_limit_configuration, reset_circuit_breaker |
| 44 | circuit_breaker_open | network | hard | 1 | reset_circuit_breaker |
| 45 | feature_flag_dependency_chain | environment | hard | 2 | rollback_env_var, restart_app |
| 46 | cron_job_skipped | deployment | medium | 2 | restore_cron_schedule, restart_worker |
| 47 | search_index_stale | search | medium | 2 | rebuild_index, restart_worker |
| 48 | tls_chain_incomplete | tls | hard | 2 | restore_tls_config, renew_tls_certificate |
| 49 | certificate_authority_missing | tls | hard | 2 | renew_tls_certificate, restore_file_permissions |
| 50 | hsts_misconfig | tls | medium | 1 | restore_tls_config |
| 51 | monitoring_agent_offline | observability | easy | 1 | restart_worker |
| 52 | configuration_drift_detected | deployment | hard | 3 | rollback_env_var, fix_dependency, fix_port_config |

## Workflow rules

- The policy returns one JSON object with an ordered `actions` array. Actions are
  executed in the order they appear.
- The expert workflow above is the *reference* solution used only for mechanical
  validation during dataset generation. It is never serialized into GRPO rows.
- Every element of the array uses the same bounded schema as the single-action v4
  contract: `{"action": "<name>", "parameters": {}}`.
- `parameters` is always `{}` for every repair action. The sandbox restores declared
  or historical values; the policy must never guess names, versions, ports, or
  thresholds.
