# CrashDiag v5 Action Space

The v5 sandbox exposes **27 repair actions** plus the safe no-op fallback
`wait_and_observe`. Every repair action takes `parameters: {}` only. Actions restore
declared or historical sandbox state; a model-supplied parameter is never trusted.

## Existing actions (carried over from v4)

| action | restores | mechanical verification |
|---|---|---|
| `restart_app` | application process to running | `process.running == true` |
| `rollback_env_var` | env var to its known-good value | `environment.variables == environment.expected` |
| `fix_dependency` | dependency to its required version | `dependencies.installed == dependencies.required` |
| `clear_disk` | disk usage below threshold | `disk.used_percent < disk.healthy_below_percent` |
| `fix_port_config` | proxy target to app port | `network.proxy_target_port == network.app_port` |
| `clear_cache` | cache service healthy | `services.cache == true` |
| `renew_tls_certificate` | TLS certificate valid | `services.tls == true` |
| `restore_file_permissions` | file permissions correct | `services.permissions == true` |
| `apply_database_migration` | schema migration applied | `services.migration == true` |
| `reset_database_pool` | DB pool reset | `services.db_pool == true` |
| `restore_dns_configuration` | DNS resolution working | `services.dns == true` |
| `restore_rate_limit_configuration` | rate-limit config restored | `services.rate_limit == true` |
| `wait_and_observe` | nothing (no-op fallback) | no state change |

## New actions (added in v5)

| action | restores | mechanical verification |
|---|---|---|
| `restart_worker` | background worker / cron / queue-consumer / monitor processes | `services.worker == true` |
| `redeploy_container` | container image to declared version | `services.container == true` |
| `clear_temp_files` | inode / temp-file pressure | `services.temp == true` |
| `rotate_logs` | log volume below health threshold | `services.logs == true` |
| `restore_load_balancer_config` | LB upstream route + health-check config | `services.lb_config == true` |
| `restore_network_config` | MTU / firewall port / network config | `services.network_config == true` |
| `sync_replica` | DB replica lag below threshold | `services.replica == true` |
| `restore_database_config` | read/write split + replica topology + pool sizing | `services.db_config == true` |
| `flush_dead_letter_queue` | dead-letter backlog to zero | `services.dead_letter == true` |
| `restore_cache_config` | cache eviction policy | `services.cache_config == true` |
| `reset_circuit_breaker` | open circuit breaker to closed | `services.circuit_breaker == true` |
| `restore_cron_schedule` | cron jobs enabled | `services.cron == true` |
| `rebuild_index` | search index fresh | `services.search_index == true` |
| `restore_tls_config` | TLS chain completeness / CA / HSTS | `services.tls_config == true` |

## Shared rules

- `parameters` must be `{}`. `fix_dependency` parameters are additionally stripped at
  parse time so a model can never override the declared dependency lock.
- The full allowlist is defined in `crashdiag/agents.py` (`ACTION_SPACE`) and mirrored
  in `crashdiag/sandbox_apps/mock.py` (`SandboxBackend.ACTIONS`).
- Actions mutate real `MockSandbox` state and are recomputed into `health_check()` on
  every observation. There is no hidden "resolved" flag.
- The remote sandbox validates every action against the same allowlist and rejects
  anything outside it with `404 unknown_action`.
