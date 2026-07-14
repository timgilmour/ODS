#!/usr/bin/env bash
# Keep public install commands and provenance guidance aligned.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

CANONICAL_ENDPOINT="https://install.osmantic.com/ods.sh"
LEGACY_RAW_ENDPOINT="https://raw.githubusercontent.com/Light-Heart-Labs/ODS/main/ods/get-ods.sh"
LEGACY_CLONE_URL="https://github.com/Light-Heart-Labs/ODS.git"

fail() {
    echo "[FAIL] $*"
    exit 1
}

pass() {
    echo "[PASS] $*"
}

require_literal() {
    local file="$1"
    local literal="$2"
    local description="$3"

    grep -qF -- "$literal" "$file" \
        || fail "$description missing from ${file#"$REPO_ROOT"/}"
}

reject_literal() {
    local file="$1"
    local literal="$2"
    local description="$3"

    if grep -qF -- "$literal" "$file"; then
        fail "$description remains in ${file#"$REPO_ROOT"/}"
    fi
}

install_docs=(
    "$REPO_ROOT/README.md"
    "$ROOT_DIR/README.md"
    "$ROOT_DIR/QUICKSTART.md"
    "$ROOT_DIR/docs/FAQ.md"
    "$ROOT_DIR/docs/INSTALLER_TRUST.md"
    "$ROOT_DIR/get-ods.sh"
)

clone_docs=(
    "$REPO_ROOT/README.md"
    "$ROOT_DIR/README.md"
    "$ROOT_DIR/QUICKSTART.md"
    "$ROOT_DIR/docs/INSTALLER_TRUST.md"
)

for file in "${install_docs[@]}"; do
    [[ -f "$file" ]] || fail "Expected install document missing: $file"
    require_literal "$file" "$CANONICAL_ENDPOINT" "Canonical install endpoint"
    reject_literal "$file" "$LEGACY_RAW_ENDPOINT" "Legacy raw bootstrap endpoint"
done

for file in "${clone_docs[@]}"; do
    reject_literal "$file" "$LEGACY_CLONE_URL" "Legacy clone URL"
done

trust_doc="$ROOT_DIR/docs/INSTALLER_TRUST.md"
require_literal "$trust_doc" 'currently `main`' "Default branch guidance"
require_literal "$trust_doc" 'ODS_REF=' "Release-tag pinning guidance"
require_literal "$trust_doc" 'git checkout AUDITED_COMMIT_SHA' "Exact-commit guidance"
require_literal "$trust_doc" 'not a separate stable release channel' "Hosted-versus-raw channel guidance"

pass "Install commands and provenance guidance are consistent"
