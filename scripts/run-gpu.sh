#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd -- "$REPOSITORY_ROOT"

readonly -a SERVICES=(image-api upscale-worker background-worker generation-worker)
readonly -a GPU_WORKERS=(upscale-worker background-worker generation-worker)
readonly STARTUP_TIMEOUT_SECONDS="${IMAGE_API_STARTUP_TIMEOUT_SECONDS-300}"
readonly HEALTH_LOG_TAIL_LINES=80
health_file=""

require_command() {
    local command_name=$1
    local install_hint=$2
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Error: %s is required. %s\n' "$command_name" "$install_hint" >&2
        exit 127
    fi
}

require_command docker 'Install Docker Engine and ensure docker is on PATH.'
require_command curl 'Install curl and ensure curl is on PATH.'
require_command python3 'Install Python 3 and ensure python3 is on PATH.'

readonly -a COMPOSE=(docker compose -f compose.yml)
if ! "${COMPOSE[@]}" version >/dev/null 2>&1; then
    printf 'Error: docker compose is required. Install the Docker Compose plugin and verify `docker compose version`.\n' >&2
    exit 127
fi
if [[ ! "$STARTUP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Error: IMAGE_API_STARTUP_TIMEOUT_SECONDS must be a strict positive integer; got %q.\n' "$STARTUP_TIMEOUT_SECONDS" >&2
    exit 2
fi
if ! "${COMPOSE[@]}" config --quiet; then
    printf 'Error: production Compose configuration is invalid; fix compose.yml or its required environment before startup.\n' >&2
    exit 1
fi

diagnostics() {
    printf '\nStartup verification failed. Current Compose state:\n' >&2
    "${COMPOSE[@]}" ps >&2 || true
    printf '\nRecent service logs (last %s lines per service):\n' "$HEALTH_LOG_TAIL_LINES" >&2
    "${COMPOSE[@]}" logs --tail "$HEALTH_LOG_TAIL_LINES" --no-color "${SERVICES[@]}" >&2 || true
}

cleanup() {
    if [[ -n "$health_file" ]]; then
        rm -f -- "$health_file"
    fi
}

trap cleanup EXIT

fail() {
    printf 'Error: %s\n' "$1" >&2
    return 1
}

wait_for_services() {
    local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
    local service cid state container_status health_status
    local all_healthy

    while true; do
        all_healthy=1
        for service in "${SERVICES[@]}"; do
            if ! cid="$("${COMPOSE[@]}" ps --all -q "$service")"; then
                fail "could not resolve the $service container"
                return 1
            fi
            if [[ -z "$cid" ]]; then
                fail "missing container for required service $service"
                return 1
            fi
            if ! state="$(docker inspect --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid")"; then
                fail "could not inspect container for required service $service"
                return 1
            fi
            read -r container_status health_status <<<"$state"
            printf '%s: container=%s health=%s\n' "$service" "$container_status" "$health_status"

            case "$container_status" in
                exited | dead)
                    fail "$service container is $container_status"
                    return 1
                    ;;
            esac
            if [[ "$health_status" == unhealthy ]]; then
                fail "$service container is unhealthy"
                return 1
            fi
            if [[ "$container_status" != running || "$health_status" != healthy ]]; then
                all_healthy=0
            fi
        done

        if ((all_healthy)); then
            printf 'All required Compose services are healthy.\n'
            return 0
        fi
        if ((SECONDS >= deadline)); then
            fail "timed out after ${STARTUP_TIMEOUT_SECONDS}s waiting for all required services to become healthy"
            return 1
        fi
        sleep 1
    done
}

verify_cuda() {
    local service=$1
    local device_names
    if ! device_names="$("${COMPOSE[@]}" exec -T "$service" python -c 'import torch; assert torch.cuda.is_available(), "torch.cuda.is_available() returned false"; names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]; assert names, "CUDA is available but no CUDA devices were reported"; print(" | ".join(names))')"; then
        fail "CUDA verification failed for $service"
        return 1
    fi
    if [[ -z "$device_names" ]]; then
        fail "CUDA verification for $service returned no device names"
        return 1
    fi
    printf 'GPU worker %s CUDA devices: %s\n' "$service" "$device_names"
}

validate_gateway_health() {
    local published_address port health_url
    if ! published_address="$("${COMPOSE[@]}" port image-api 8000)"; then
        fail 'could not resolve the published image-api port'
        return 1
    fi
    published_address="${published_address%%$'\n'*}"
    port="${published_address##*:}"
    if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
        fail "docker compose returned an invalid image-api port mapping: $published_address"
        return 1
    fi
    health_url="http://127.0.0.1:${port}/health"
    printf 'Querying gateway health at %s (published mapping: %s)\n' "$health_url" "$published_address"
    if ! health_file="$(mktemp)"; then
        fail 'could not create a temporary file for the gateway health response'
        return 1
    fi
    if ! curl --fail --silent --show-error --max-time 15 --output "$health_file" "$health_url"; then
        fail 'gateway health request failed'
        return 1
    fi
    if ! python3 - "$health_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"gateway health validation failed: invalid JSON: {exc}") from exc

if not isinstance(payload, dict):
    raise SystemExit("gateway health validation failed: response must be a JSON object")
if payload.get("service") != "image-api":
    raise SystemExit("gateway health validation failed: service must equal image-api")
if payload.get("status") != "ok":
    raise SystemExit("gateway health validation failed: status must equal ok")
capabilities = payload.get("capabilities")
if not isinstance(capabilities, dict):
    raise SystemExit("gateway health validation failed: capabilities must be an object")
for capability in ("upscale", "background-removal", "generation"):
    details = capabilities.get(capability)
    if not isinstance(details, dict):
        raise SystemExit(f"gateway health validation failed: missing {capability} capability")
    if details.get("ready") is not True:
        raise SystemExit(f"gateway health validation failed: {capability} is not ready")
    if details.get("device") != "cuda":
        raise SystemExit(f"gateway health validation failed: {capability} device must equal cuda")
generation = capabilities["generation"]
if generation.get("weightsAvailable") is not True:
    raise SystemExit("gateway health validation failed: generation weights are unavailable")
if generation.get("workerAvailable") is not True:
    raise SystemExit("gateway health validation failed: generation worker is unavailable")
gpu_lane = payload.get("gpuLane")
if not isinstance(gpu_lane, dict) or gpu_lane.get("active") is not False:
    raise SystemExit("gateway health validation failed: GPU lane must be inactive")
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
PY
    then
        fail 'gateway health validation failed'
        return 1
    fi
    printf 'Gateway health contract verified.\n'
}

printf 'Starting the production Compose stack without rebuilding images...\n'
if ! "${COMPOSE[@]}" up -d; then
    diagnostics
    exit 1
fi
if ! wait_for_services; then
    diagnostics
    exit 1
fi
for worker in "${GPU_WORKERS[@]}"; do
    if ! verify_cuda "$worker"; then
        diagnostics
        exit 1
    fi
done
if ! validate_gateway_health; then
    diagnostics
    exit 1
fi
printf 'GPU Compose startup verification completed successfully.\n'
