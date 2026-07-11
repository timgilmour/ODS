#!/bin/bash
# ODS Integration Test Suite
# Validates all services are working end-to-end
#
# Usage: ./tests/test-integration.sh [--verbose] [--quick]

# Note: Intentionally NOT using set -e here — test functions return 1 on failure
# and we want to continue running all tests, tracking results via PASSED/FAILED counters
set -uo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check for required dependencies
command -v jq >/dev/null 2>&1 || { echo -e "${RED}✗${NC} jq is required but not installed. Install with: apt-get install jq (Debian/Ubuntu) or brew install jq (macOS)"; exit 1; }

# Config
VERBOSE=${VERBOSE:-false}
QUICK=${QUICK:-false}
TIMEOUT=10
PASSED=0
FAILED=0
SKIPPED=0

# Parse args
for arg in "$@"; do
    case $arg in
        --verbose|-v) VERBOSE=true ;;
        --quick|-q) QUICK=true ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--quick]"
            echo "  --verbose  Show detailed output"
            echo "  --quick    Skip slow tests"
            exit 0
            ;;
    esac
done

# Load environment and resolve ports
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

load_env() {
    local env_file="$ROOT_DIR/.env"
    if [[ -f "$env_file" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            line="${line%[$'\r']}"
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"
            [[ "$line" =~ ^# ]] && continue
            [[ -z "$line" ]] && continue
            if [[ "$line" == *"="* ]]; then
                local key="${line%%=*}"
                local val="${line#*=}"
                key="${key#"${key%%[![:space:]]*}"}"
                key="${key%"${key##*[![:space:]]}"}"
                val="${val#\"}"
                val="${val%\"}"
                val="${val#\'}"
                val="${val%\'}"
                if [[ -n "$key" && -z "${!key:-}" ]]; then
                    export "$key"="$val"
                fi
            fi
        done < "$env_file"
    fi
}
load_env

# Retrieve DASHBOARD_API_KEY from text file if not set in .env
if [[ -z "${DASHBOARD_API_KEY:-}" ]]; then
    key_file="$ROOT_DIR/data/dashboard-api-key.txt"
    if [[ -f "$key_file" ]]; then
        DASHBOARD_API_KEY=$(cat "$key_file" | tr -d '\r\n ' || true)
        export DASHBOARD_API_KEY
    fi
fi

# Logging
log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_pass() { echo -e "${GREEN}✓${NC} $1"; ((PASSED++)); }
log_fail() { echo -e "${RED}✗${NC} $1"; ((FAILED++)); }
log_skip() { echo -e "${YELLOW}○${NC} $1 (skipped)"; ((SKIPPED++)); }
log_verbose() { $VERBOSE && echo -e "  ${NC}$1" || true; }

# Test helpers
test_http() {
    local name="$1"
    local url="$2"
    local expected="${3:-200}"
    local method="${4:-GET}"
    local data="${5:-}"
    
    local args=(-s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT")
    [[ -n "$data" ]] && args+=(-X "$method" -H "Content-Type: application/json" -d "$data")
    if [[ -n "${DASHBOARD_API_KEY:-}" && "$url" == *":${DASHBOARD_API_PORT:-3002}"* ]]; then
        args+=(-H "Authorization: Bearer $DASHBOARD_API_KEY")
    fi
    
    local code
    code=$(curl "${args[@]}" "$url" 2>/dev/null) || code="000"
    
    if [[ "$code" == "$expected" ]]; then
        log_pass "$name"
        return 0
    else
        log_fail "$name (expected $expected, got $code)"
        return 1
    fi
}

test_json() {
    local name="$1"
    local url="$2"
    local jq_filter="$3"
    
    local args=(-s --max-time "$TIMEOUT")
    if [[ -n "${DASHBOARD_API_KEY:-}" && "$url" == *":${DASHBOARD_API_PORT:-3002}"* ]]; then
        args+=(-H "Authorization: Bearer $DASHBOARD_API_KEY")
    fi
    
    local response
    response=$(curl "${args[@]}" "$url" 2>/dev/null) || response=""
    
    if echo "$response" | jq -e "$jq_filter" >/dev/null 2>&1; then
        log_pass "$name"
        local summary
        summary=$(echo "$response" | jq -c '.' 2>/dev/null || echo "$response")
        log_verbose "Response: ${summary:0:100}"
        return 0
    else
        log_fail "$name (jq filter failed: $jq_filter)"
        log_verbose "Response: ${response:0:100}"
        return 1
    fi
}

test_llm() {
    local name="$1"
    local url="$2"
    local prompt="$3"
    
    local data
    data=$(jq -n --arg prompt "$prompt" '{
        model: "qwen2.5-32b-instruct",
        messages: [{role: "user", content: $prompt}],
        max_tokens: 50,
        stream: false
    }')
    
    local response
    response=$(curl -s --max-time 30 -X POST \
        -H "Content-Type: application/json" \
        -d "$data" \
        "$url/v1/chat/completions" 2>/dev/null) || response=""
    
    if echo "$response" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
        local content
        content=$(echo "$response" | jq -r '.choices[0].message.content' | head -c 100)
        log_pass "$name"
        log_verbose "Response: $content"
        return 0
    else
        log_fail "$name (no valid response)"
        log_verbose "Response: $response"
        return 1
    fi
}

# Header
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ODS Integration Tests                              ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ========================================
# Dashboard API Tests
# ========================================
echo -e "${BLUE}▸ Dashboard API${NC}"

test_http "API health check" "http://localhost:${DASHBOARD_API_PORT:-3002}/health"
test_json "API status endpoint" "http://localhost:${DASHBOARD_API_PORT:-3002}/api/status" '.gpu or .services'
test_json "GPU metrics" "http://localhost:${DASHBOARD_API_PORT:-3002}/gpu" '.name and .memory_used_mb'
test_json "Service list" "http://localhost:${DASHBOARD_API_PORT:-3002}/services" '. | length > 0'

# ========================================
# Model API Tests
# ========================================
echo ""
echo -e "${BLUE}▸ Model Manager API${NC}"

test_json "Model catalog" "http://localhost:${DASHBOARD_API_PORT:-3002}/api/models" '.models | length > 0'
test_json "VRAM info in catalog" "http://localhost:${DASHBOARD_API_PORT:-3002}/api/models" '.gpu.vramTotal > 0'

# ========================================
# Workflow API Tests
# ========================================
echo ""
echo -e "${BLUE}▸ Workflow API${NC}"

test_json "Workflow catalog" "http://localhost:${DASHBOARD_API_PORT:-3002}/api/workflows" '.workflows | length > 0'
test_json "Workflow categories" "http://localhost:${DASHBOARD_API_PORT:-3002}/api/workflows" '.categories | keys | length > 0'

# ========================================
# Voice API Tests
# ========================================
echo ""
echo -e "${BLUE}▸ Voice API${NC}"

test_json "Voice status" "http://localhost:${DASHBOARD_API_PORT:-3002}/api/voice/status" '.services'

# ========================================
# Core Service Tests
# ========================================
echo ""
echo -e "${BLUE}▸ Core Services${NC}"

# llama-server
if ! $QUICK; then
    test_http "llama-server health" "http://localhost:${OLLAMA_PORT:-8080}/health"
    test_llm "llama-server inference" "http://localhost:${OLLAMA_PORT:-8080}" "Say hello in exactly 3 words."
else
    log_skip "llama-server inference test"
fi

# n8n
test_http "n8n health" "http://localhost:${N8N_PORT:-5678}/healthz" || log_skip "n8n not running"

# Qdrant
test_http "Qdrant health" "http://localhost:${QDRANT_PORT:-6333}/" || log_skip "Qdrant not running"

# ========================================
# Voice Services Tests
# ========================================
echo ""
echo -e "${BLUE}▸ Voice Services${NC}"

test_http "Whisper STT" "http://localhost:${WHISPER_PORT:-9000}/health" || log_skip "Whisper not running"
test_http "Kokoro TTS" "http://localhost:${TTS_PORT:-8880}/health" || log_skip "Kokoro not running"
test_http "LiveKit" "http://localhost:${LIVEKIT_PORT:-7880}/" || log_skip "LiveKit not running"

# ========================================
# Dashboard UI Tests
# ========================================
echo ""
echo -e "${BLUE}▸ Dashboard UI${NC}"

test_http "Dashboard serves" "http://localhost:${DASHBOARD_PORT:-3001}/"

# ========================================
# Summary
# ========================================
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
TOTAL=$((PASSED + FAILED + SKIPPED))
echo -e "Results: ${GREEN}$PASSED passed${NC} / ${RED}$FAILED failed${NC} / ${YELLOW}$SKIPPED skipped${NC} ($TOTAL total)"

if [[ $FAILED -gt 0 ]]; then
    echo -e "${RED}Some tests failed. Check the output above.${NC}"
    exit 1
else
    echo -e "${GREEN}All active tests passed!${NC}"
    exit 0
fi
