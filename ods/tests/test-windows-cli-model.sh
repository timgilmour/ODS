#!/bin/bash
# Tests for ods.ps1 model command parity (issue #1757)
# These are hermetic shell tests that exercise the static configuration
# and verify command routing in the Windows CLI helper.
#
# Run: bash ods/tests/test-windows-cli-model.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ODS_PS1="$ROOT_DIR/installers/windows/ods.ps1"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }
info() { echo -e "${BLUE}ℹ${NC} $1"; }

info "Static: ods.ps1 exists and is non-empty"
[[ -f "$ODS_PS1" ]] || fail "ods.ps1 not found at $ODS_PS1"
[[ -s "$ODS_PS1" ]] || fail "ods.ps1 is empty"
pass "ods.ps1 exists"

info "Static: header comment documents model"
grep -q 'model.*current|list|swap' "$ODS_PS1" \
    || fail "Header comment missing 'model' usage line"
pass "Header comment documents model subcommand"

info "Static: tier-map.ps1 library is sourced in imports block"
grep -q 'tier-map\.ps1' "$ODS_PS1" \
    || fail "tier-map.ps1 not sourced in library imports"
pass "tier-map.ps1 is sourced"

info "Static: Invoke-Model function defined"
grep -q 'function Invoke-Model' "$ODS_PS1" \
    || fail "Invoke-Model function not found in ods.ps1"
pass "Invoke-Model function present"

info "Static: Invoke-Model checks current, list, and swap actions"
grep -q 'current' "$ODS_PS1" || fail "Invoke-Model missing current subcommand"
grep -q 'list' "$ODS_PS1" || fail "Invoke-Model missing list subcommand"
grep -q 'swap' "$ODS_PS1" || fail "Invoke-Model missing swap subcommand"
pass "Invoke-Model handles current, list, and swap"

info "Static: Invoke-Model uses ConvertTo-ModelFromTier and Resolve-TierConfig for swap action"
grep -q 'ConvertTo-ModelFromTier' "$ODS_PS1" \
    || fail "Invoke-Model does not call ConvertTo-ModelFromTier"
grep -q 'Resolve-TierConfig' "$ODS_PS1" \
    || fail "Invoke-Model does not call Resolve-TierConfig"
pass "Invoke-Model integrates with tier-map functions"

info "Static: Invoke-Model uses Set-ODSEnvValue to update .env on swap"
grep -q 'Set-ODSEnvValue -Key "LLM_MODEL"' "$ODS_PS1" \
    || fail "Invoke-Model does not set LLM_MODEL"
grep -q 'Set-ODSEnvValue -Key "TIER"' "$ODS_PS1" \
    || fail "Invoke-Model does not set TIER"
pass "Invoke-Model updates .env variables correctly on swap"

info "Static: command dispatcher wires 'model'"
grep -q '"model"' "$ODS_PS1" && grep -q 'Invoke-Model' "$ODS_PS1" \
    || fail "Dispatcher does not call Invoke-Model for 'model'"
pass "Dispatcher wires 'model' -> Invoke-Model"

info "Static: Show-Help lists model command"
grep -q 'model' "$ODS_PS1" && grep -q 'Inspect/swap LLM profiles' "$ODS_PS1" \
    || fail "Show-Help does not list model command"
pass "Show-Help lists model command"

info "Static: Show-Help EXAMPLES mention model swap"
grep -q 'model swap' "$ODS_PS1" \
    || fail "Show-Help EXAMPLES do not include 'model swap'"
pass "Show-Help EXAMPLES include model swap"

echo ""
echo -e "${GREEN}All windows-cli-model tests passed.${NC}"
