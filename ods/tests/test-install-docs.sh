#!/usr/bin/env bash
# Keep public install commands and provenance guidance aligned.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

CANONICAL_ENDPOINT="https://install.osmantic.com/ods.sh"
CANONICAL_REPO_URL="https://github.com/Osmantic/ODS.git"

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
done

for file in "${clone_docs[@]}"; do
    require_literal "$file" "$CANONICAL_REPO_URL" "Canonical clone URL"
done

legacy_brand_matches="$(
    git -C "$REPO_ROOT" grep -n -I -i -E \
        'light[-_ ]?heart[-_ ]?labs' \
        -- . ':!ods/tests/test-install-docs.sh' || true
)"
if [[ -n "$legacy_brand_matches" ]]; then
    echo "[FAIL] Legacy organization references remain:"
    echo "$legacy_brand_matches"
    exit 1
fi

trust_doc="$ROOT_DIR/docs/INSTALLER_TRUST.md"
require_literal "$trust_doc" 'currently `main`' "Default branch guidance"
require_literal "$trust_doc" 'ODS_REF=' "Release-tag pinning guidance"
require_literal "$trust_doc" 'git checkout AUDITED_COMMIT_SHA' "Exact-commit guidance"
require_literal "$trust_doc" 'not a separate stable release channel' "Hosted-versus-raw channel guidance"

pass "Install commands and provenance guidance are consistent"
