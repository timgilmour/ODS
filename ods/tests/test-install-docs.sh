#!/usr/bin/env bash
# Keep public install commands and provenance guidance aligned.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

CANONICAL_ENDPOINT="https://install.osmantic.com/ods.sh"
CANONICAL_REPO_URL="https://github.com/Osmantic/ODS.git"
STABLE_VERSION="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["release"]["version"])' \
        "$ROOT_DIR/manifest.json"
)"
STABLE_TAG="v$STABLE_VERSION"

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

compatible_ref_docs=(
    "$REPO_ROOT/README.md"
    "$ROOT_DIR/README.md"
    "$ROOT_DIR/QUICKSTART.md"
    "$ROOT_DIR/docs/FAQ.md"
)

for file in "${compatible_ref_docs[@]}"; do
    require_literal "$file" 'compatible ref with `ODS_REF`' "Compatible bootstrap ref guidance"
done

legacy_brand_matches="$(
    git -C "$REPO_ROOT" grep -n -I -i -E \
        'light[-_ ]?heart[-_ ]?labs' \
        -- . ':!ods/tests/test-install-docs.sh' || true
)"
unexpected_legacy_brand_matches=""
while IFS= read -r match; do
    [[ -n "$match" ]] || continue
    case "$match" in
        SECURITY_AUDIT.md:*'`Light-Heart-Labs/ODS` public repository as named at audit time'*) ;;
        installer/src-tauri/src/installer.rs:*'const TRANSFERRED_REPO_URL: &str = "https://github.com/Light-Heart-Labs/ODS.git";'*) ;;
        *) unexpected_legacy_brand_matches+="${match}"$'\n' ;;
    esac
done <<<"$legacy_brand_matches"
if [[ -n "$unexpected_legacy_brand_matches" ]]; then
    echo "[FAIL] Unexpected legacy organization references remain:"
    printf '%s' "$unexpected_legacy_brand_matches"
    exit 1
fi
require_literal "$REPO_ROOT/SECURITY_AUDIT.md" \
    '`Light-Heart-Labs/ODS` public repository as named at audit time' \
    "Historical audit scope"
require_literal "$REPO_ROOT/installer/src-tauri/src/installer.rs" \
    'const TRANSFERRED_REPO_URL: &str = "https://github.com/Light-Heart-Labs/ODS.git";' \
    "Transferred checkout compatibility"

trust_doc="$ROOT_DIR/docs/INSTALLER_TRUST.md"
release_doc="$ROOT_DIR/docs/RELEASE_CHANNELS.md"
require_literal "$trust_doc" 'currently `main`' "Default branch guidance"
require_literal "$trust_doc" 'ODS_REF=' "Release-tag pinning guidance"
require_literal "$trust_doc" 'git checkout AUDITED_COMMIT_SHA' "Exact-commit guidance"
require_literal "$trust_doc" 'not a separate stable release channel' "Hosted-versus-raw channel guidance"
require_literal "$REPO_ROOT/README.md" "\`$STABLE_TAG\` is the current stable release" "README stable release"
require_literal "$release_doc" "current stable release is \`$STABLE_TAG\`" "Release channel stable release"
require_literal "$trust_doc" "--branch $STABLE_TAG $CANONICAL_REPO_URL" "Manual stable clone"
require_literal "$trust_doc" 'predates that repository layout' "Stable layout guidance"

if grep -qF "ODS_REF=$STABLE_TAG" "$REPO_ROOT/README.md" "$trust_doc"; then
    fail "$STABLE_TAG must not be documented through the incompatible sparse-checkout bootstrap"
fi

legacy_product_doc_matches="$(
    git -C "$REPO_ROOT" grep -n -I -i -E \
        'dream[ _-]?server|dreamserver|dream[ _-]?fleet' \
        -- README.md ods/README.md ods/QUICKSTART.md ods/docs || true
)"
if [[ -n "$legacy_product_doc_matches" ]]; then
    echo "[FAIL] Legacy product branding remains in public documentation:"
    echo "$legacy_product_doc_matches"
    exit 1
fi

legacy_product_files="$(
    git -C "$REPO_ROOT" grep -l -I -i -E \
        'dream[ _-]?server|dreamserver|dream[ _-]?fleet' \
        -- . ':!ods/tests/test-install-docs.sh' | sort || true
)"
expected_legacy_product_files="$(
    printf '%s\n' \
        .gitleaksignore \
        ods/get-ods.sh \
        ods/installers/phases/01-preflight.sh \
        ods/installers/windows/install-windows.ps1 \
        ods/tests/contracts/test-amd-lemonade-contracts.sh \
        ods/tests/contracts/test-installer-contracts.sh \
        ods/tests/fleet-incus-vm.sh \
        ods/tests/fleet-multi-distro.sh \
        ods/tests/test-fleet-host-lock.sh |
        sort
)"
if [[ "$legacy_product_files" != "$expected_legacy_product_files" ]]; then
    echo "[FAIL] Legacy product identifiers must remain confined to compatibility fingerprints"
    printf '%s\n' "$legacy_product_files"
    exit 1
fi

for fleet_script in \
    "$ROOT_DIR/tests/fleet-incus-vm.sh" \
    "$ROOT_DIR/tests/fleet-multi-distro.sh"; do
    if bash "$fleet_script" --help | grep -qiE 'dream[ _-]?server|dreamserver|dream[ _-]?fleet'; then
        fail "Legacy product branding exposed by ${fleet_script#"$REPO_ROOT"/} --help"
    fi
done

pass "Install commands and provenance guidance are consistent"
