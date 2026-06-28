#!/usr/bin/env bash
set -euo pipefail

export ODS_REPO_URL="${ODS_REPO_URL:-https://github.com/Light-Heart-Labs/DreamServer.git}"
export ODS_REF="${ODS_REF:-codex/pr1626-merge-test-20260628}"
export ODS_BOOTSTRAP_REF="${ODS_BOOTSTRAP_REF:-$ODS_REF}"

echo "[pr1626] ODS_REPO_URL=$ODS_REPO_URL" >&2
echo "[pr1626] ODS_REF=$ODS_REF" >&2

curl -fsSL "https://raw.githubusercontent.com/Light-Heart-Labs/DreamServer/codex/pr1626-merge-test-20260628/ods/get-ods.sh" \
  | bash -s -- "$@"
