#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd -- "$REPOSITORY_ROOT"

if ! command -v docker >/dev/null 2>&1; then
    printf 'Error: docker is required. Install Docker Engine and ensure docker is on PATH.\n' >&2
    exit 127
fi

readonly -a COMPOSE=(docker compose -f compose.yml)
if ! "${COMPOSE[@]}" version >/dev/null 2>&1; then
    printf 'Error: docker compose is required. Install the Docker Compose plugin and verify `docker compose version`.\n' >&2
    exit 127
fi

if ! "${COMPOSE[@]}" config --quiet; then
    printf 'Error: production Compose configuration is invalid; fix compose.yml or its required environment before rebuilding.\n' >&2
    exit 1
fi

printf 'Rebuilding production images with updated base images...\n'
"${COMPOSE[@]}" build --pull "$@"
printf 'Production image rebuild completed. No containers were started, stopped, or recreated.\n'
