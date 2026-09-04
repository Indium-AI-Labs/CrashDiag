#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${repo_root}/.env"
hud_bin="${HUD_BIN:-${HOME}/.local/bin/hud}"

if [[ ! -x "${hud_bin}" ]]; then
  echo "HUD CLI not found at ${hud_bin}. Install it with: uv tool install hud --python 3.12" >&2
  exit 1
fi
if [[ ! -f "${env_file}" ]]; then
  echo "Missing ${env_file}" >&2
  exit 1
fi

# Read only HUD_API_KEY. Never source the repository .env: it also contains
# credentials that must not be forwarded to a third-party build service.
hud_key_line="$(grep -m1 '^HUD_API_KEY=' "${env_file}" || true)"
if [[ -z "${hud_key_line}" ]]; then
  echo "HUD_API_KEY is missing from ${env_file}" >&2
  exit 1
fi
export HUD_API_KEY="${hud_key_line#HUD_API_KEY=}"
HUD_API_KEY="${HUD_API_KEY%$'\r'}"
HUD_API_KEY="${HUD_API_KEY%\"}"
HUD_API_KEY="${HUD_API_KEY#\"}"
HUD_API_KEY="${HUD_API_KEY%\'}"
HUD_API_KEY="${HUD_API_KEY#\'}"
export HUD_API_KEY

cd "${repo_root}"

if [[ "${HUD_SKIP_DEPLOY:-0}" != "1" ]]; then
  # HUD prefers compose.yaml when both Compose and Dockerfile recipes exist.
  # Stage only the HUD runtime so the repository's standalone service Compose
  # file cannot be selected accidentally.
  stage_dir="$(mktemp -d -t crashdiag-hud.XXXXXXXX)"
  cleanup() {
    case "${stage_dir}" in
      /tmp/crashdiag-hud.*) rm -rf -- "${stage_dir}" ;;
    esac
  }
  trap cleanup EXIT
  cp Dockerfile.hud "${stage_dir}/Dockerfile.hud"
  cp LICENSE env.py "${stage_dir}/"
  cp -R crashdiag training "${stage_dir}/"

  "${hud_bin}" deploy "${stage_dir}" --no-env
fi
for batch_index in 0 1 2 3; do
  CRASHDIAG_HUD_TRAIN_BATCH="${batch_index}" \
    "${hud_bin}" sync tasks crashdiag-v6-train tasks_train_batch.py --yes
done
"${hud_bin}" sync tasks crashdiag-v6-eval tasks.py --yes
