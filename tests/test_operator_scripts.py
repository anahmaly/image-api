from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (ROOT / "scripts" / "rebuild-images.sh", ROOT / "scripts" / "run-gpu.sh")
SERVICES = ("image-api", "upscale-worker", "background-worker", "generation-worker")
GPU_WORKERS = ("upscale-worker", "background-worker", "generation-worker")


def _healthy_payload() -> dict[str, object]:
    return {
        "service": "image-api",
        "status": "ok",
        "capabilities": {
            "upscale": {"ready": True, "device": "cuda"},
            "background-removal": {"ready": True, "device": "cuda"},
            "generation": {
                "ready": True,
                "device": "cuda",
                "weightsAvailable": True,
                "workerAvailable": True,
            },
        },
        "gpuLane": {"active": False, "activeCapability": None},
    }


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fake_environment(tmp_path: Path, **overrides: str) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "invocations.log"
    _write_executable(
        bin_dir / "docker",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s|docker %s\\n' "$PWD" "$*" >> "$FAKE_LOG"
if [[ "$1" == "compose" ]]; then
  shift
  [[ "$1" == "-f" && "$2" == "compose.yml" ]] || exit 91
  shift 2
  case "$1" in
    version) exit 0 ;;
    config) [[ "${FAKE_CONFIG_FAIL:-0}" == 1 ]] && exit 41; exit 0 ;;
    build) exit 0 ;;
    up) [[ "$2" == "-d" && $# == 2 ]] || exit 92; exit 0 ;;
    ps)
      if [[ "${2:-}" == "--all" && "${3:-}" == "-q" ]]; then
        service="$4"
        [[ "${FAKE_MISSING_SERVICE:-}" == "$service" ]] || printf 'cid-%s\\n' "$service"
      else
        printf 'NAME STATUS\\nimage-api running\\n'
      fi
      exit 0
      ;;
    exec)
      [[ "$2" == "-T" ]] || exit 93
      service="$3"
      if [[ "${FAKE_CUDA_FALSE_SERVICE:-}" == "$service" ]]; then
        printf 'CUDA unavailable for %s\\n' "$service" >&2
        exit 42
      fi
      printf 'Fake GPU %s\\n' "$service"
      exit 0
      ;;
    port) [[ "$2" == "image-api" && "$3" == "8000" ]] || exit 94; printf '0.0.0.0:19000\\n'; exit 0 ;;
    logs) exit 0 ;;
  esac
elif [[ "$1" == "inspect" ]]; then
  service="${*: -1}"
  service="${service#cid-}"
  if [[ "${FAKE_UNHEALTHY_SERVICE:-}" == "$service" ]]; then
    printf 'running unhealthy\\n'
  elif [[ "${FAKE_STARTING_SERVICE:-}" == "$service" ]]; then
    printf 'running starting\\n'
  elif [[ "${FAKE_EXITED_SERVICE:-}" == "$service" ]]; then
    printf 'exited none\\n'
  else
    printf 'running healthy\\n'
  fi
  exit 0
fi
exit 95
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s|curl %s\\n' "$PWD" "$*" >> "$FAKE_LOG"
out=''
while (($#)); do
  if [[ "$1" == "--output" ]]; then out="$2"; shift 2; else shift; fi
done
[[ -n "$out" ]] || exit 96
printf '%s' "$FAKE_HEALTH_JSON" > "$out"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_LOG": str(log),
            "FAKE_HEALTH_JSON": json.dumps(_healthy_payload(), separators=(",", ":")),
        }
    )
    env.update(overrides)
    return env, log


def _run(
    script: Path, env: dict[str, str], cwd: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_scripts_are_executable_and_parse_as_bash() -> None:
    for script in SCRIPTS:
        assert os.access(script, os.X_OK)
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_rebuild_runs_from_root_validates_and_forwards_arguments(tmp_path: Path) -> None:
    env, log = _fake_environment(tmp_path)
    result = _run(SCRIPTS[0], env, tmp_path, "--no-cache", "background-worker", "image-api")

    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert calls == [
        f"{ROOT}|docker compose -f compose.yml version",
        f"{ROOT}|docker compose -f compose.yml config --quiet",
        f"{ROOT}|docker compose -f compose.yml build --pull --no-cache background-worker image-api",
    ]
    assert not any(" up" in call or " down" in call for call in calls)


def test_rebuild_stops_when_production_config_is_invalid(tmp_path: Path) -> None:
    env, log = _fake_environment(tmp_path, FAKE_CONFIG_FAIL="1")
    result = _run(SCRIPTS[0], env, tmp_path)

    assert result.returncode != 0
    assert "production Compose configuration" in result.stderr
    calls = log.read_text()
    assert " build " not in calls
    assert " up" not in calls and " down" not in calls


def test_run_starts_only_production_stack_and_verifies_every_contract(tmp_path: Path) -> None:
    env, log = _fake_environment(tmp_path)
    result = _run(SCRIPTS[1], env, tmp_path)

    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert all(call.startswith(f"{ROOT}|") for call in calls)
    assert not any("compose.test.yml" in call or "--build" in call for call in calls)
    assert f"{ROOT}|docker compose -f compose.yml up -d" in calls
    for service in SERVICES:
        assert f"{ROOT}|docker compose -f compose.yml ps --all -q {service}" in calls
    for service in GPU_WORKERS:
        assert any(
            f"docker compose -f compose.yml exec -T {service} python -c" in call for call in calls
        )
    assert f"{ROOT}|docker compose -f compose.yml port image-api 8000" in calls
    assert any("curl --fail" in call and "http://127.0.0.1:19000/health" in call for call in calls)
    assert "GPU worker generation-worker CUDA devices: Fake GPU generation-worker" in result.stdout
    assert "Gateway health contract verified" in result.stdout
    assert not any(" down" in call for call in calls)


@pytest.mark.parametrize(
    ("override", "value", "expected"),
    [
        ("FAKE_MISSING_SERVICE", "generation-worker", "missing container"),
        ("FAKE_UNHEALTHY_SERVICE", "background-worker", "unhealthy"),
        ("FAKE_EXITED_SERVICE", "upscale-worker", "exited"),
    ],
)
def test_run_fails_terminal_or_missing_service_with_diagnostics(
    tmp_path: Path, override: str, value: str, expected: str
) -> None:
    env, log = _fake_environment(tmp_path, **{override: value})
    result = _run(SCRIPTS[1], env, tmp_path)

    assert result.returncode != 0
    assert expected in result.stderr
    calls = log.read_text()
    assert "docker compose -f compose.yml ps" in calls
    assert "docker compose -f compose.yml logs --tail" in calls
    assert " down" not in calls


def test_run_times_out_waiting_for_health_and_prints_diagnostics(tmp_path: Path) -> None:
    env, log = _fake_environment(
        tmp_path,
        FAKE_STARTING_SERVICE="generation-worker",
        IMAGE_API_STARTUP_TIMEOUT_SECONDS="1",
    )
    result = _run(SCRIPTS[1], env, tmp_path)

    assert result.returncode != 0
    assert "timed out" in result.stderr
    assert "generation-worker: container=running health=starting" in result.stdout
    assert "docker compose -f compose.yml ps" in log.read_text()
    assert " down" not in log.read_text()


@pytest.mark.parametrize("timeout", ["0", "-1", "1.5", "nope", ""])
def test_run_rejects_invalid_timeout_before_start(tmp_path: Path, timeout: str) -> None:
    env, log = _fake_environment(tmp_path, IMAGE_API_STARTUP_TIMEOUT_SECONDS=timeout)
    result = _run(SCRIPTS[1], env, tmp_path)

    assert result.returncode != 0
    assert "positive integer" in result.stderr
    assert " up " not in log.read_text()


def test_run_fails_false_cuda_and_does_not_tear_down(tmp_path: Path) -> None:
    env, log = _fake_environment(tmp_path, FAKE_CUDA_FALSE_SERVICE="background-worker")
    result = _run(SCRIPTS[1], env, tmp_path)

    assert result.returncode != 0
    assert "CUDA verification failed for background-worker" in result.stderr
    assert "docker compose -f compose.yml ps" in log.read_text()
    assert " down" not in log.read_text()


@pytest.mark.parametrize(
    "mutation",
    [
        ("service", "other"),
        ("status", "degraded"),
        ("capabilities.upscale.ready", False),
        ("capabilities.background-removal.device", "cpu"),
        ("capabilities.generation.weightsAvailable", False),
        ("capabilities.generation.workerAvailable", False),
        ("gpuLane.active", True),
    ],
)
def test_run_rejects_invalid_gateway_health_contract(
    tmp_path: Path, mutation: tuple[str, object]
) -> None:
    payload = _healthy_payload()
    path, value = mutation
    target: dict[str, object] = payload
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[keys[-1]] = value
    env, log = _fake_environment(
        tmp_path,
        FAKE_HEALTH_JSON=json.dumps(payload, separators=(",", ":")),
    )
    result = _run(SCRIPTS[1], env, tmp_path)

    assert result.returncode != 0
    assert "gateway health validation failed" in result.stderr
    assert "docker compose -f compose.yml ps" in log.read_text()
    assert " down" not in log.read_text()
