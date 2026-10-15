"""Launch and inspect deployed Modal SFT v3 jobs without a long-lived client."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable
import uuid

import modal
from modal.exception import TimeoutError as ModalTimeoutError


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "nexus-sft-candidate-v3"
REGISTRY_NAME = "nexus-sft-candidate-v3-launches"
CANARY_FUNCTION = "run_orchestration_canary"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
LOCAL_RESULTS = ROOT / "evals/results/modal-sft-v3-launches"


class ModalControlError(RuntimeError):
    """The deployed-job control contract was violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or not RUN_ID_PATTERN.fullmatch(run_id)
        or run_id in {".", ".."}
    ):
        raise ModalControlError("run_id must contain 1-128 safe filename characters")
    return run_id


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def local_record_path(run_id: str) -> Path:
    return LOCAL_RESULTS / safe_run_id(run_id) / "launch.json"


def reserve_and_spawn_canary(
    run_id: str,
    duration_seconds: int,
    *,
    registry: Any,
    function: Any,
    now: Callable[[], str] = utc_now,
    commit: str | None = None,
    record_path: Path | None = None,
) -> dict[str, Any]:
    run_id = safe_run_id(run_id)
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, int)
        or not 30 <= duration_seconds <= 180
    ):
        raise ModalControlError("duration_seconds must be an integer from 30 to 180")
    record_path = record_path or local_record_path(run_id)
    reservation = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "orchestration_canary",
        "state": "reserved",
        "reservation_id": str(uuid.uuid4()),
        "reserved_at": now(),
        "deployment": {
            "app_name": APP_NAME,
            "function_name": CANARY_FUNCTION,
        },
        "arguments": {"duration_seconds": duration_seconds},
        "source_commit": commit or _git_commit(),
        "gpu_authorized": False,
        "final_holdout_inference_authorized": False,
    }
    if not registry.put(run_id, reservation, skip_if_exists=True):
        existing = registry.get(run_id)
        raise ModalControlError(
            f"run_id already reserved; refusing duplicate spawn: {existing}"
        )
    _write_json(record_path, reservation)
    try:
        call = function.spawn(run_id, duration_seconds)
    except BaseException as error:
        failed = {
            **reservation,
            "state": "spawn_failed",
            "failed_at": now(),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        registry.put(run_id, failed)
        _write_json(record_path, failed)
        raise
    launched = {
        **reservation,
        "state": "spawned",
        "spawned_at": now(),
        "function_call_id": call.object_id,
    }
    registry.put(run_id, launched)
    _write_json(record_path, launched)
    return launched


def poll_launch(
    run_id: str,
    *,
    registry: Any,
    call_from_id: Callable[[str], Any],
    now: Callable[[], str] = utc_now,
    record_path: Path | None = None,
) -> dict[str, Any]:
    run_id = safe_run_id(run_id)
    record_path = record_path or local_record_path(run_id)
    record = registry.get(run_id)
    if not isinstance(record, dict):
        raise ModalControlError(f"No launch reservation exists for {run_id}")
    call_id = record.get("function_call_id")
    if not isinstance(call_id, str) or not call_id:
        _write_json(record_path, record)
        return record
    call = call_from_id(call_id)
    try:
        result = call.get(timeout=0)
    except ModalTimeoutError:
        running = {**record, "state": "running", "last_polled_at": now()}
        registry.put(run_id, running)
        _write_json(record_path, running)
        return running
    except BaseException as error:
        failed = {
            **record,
            "state": "remote_failed",
            "failed_at": now(),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        registry.put(run_id, failed)
        _write_json(record_path, failed)
        raise
    completed = {
        **record,
        "state": "complete",
        "completed_at": now(),
        "result": result,
    }
    registry.put(run_id, completed)
    _write_json(record_path, completed)
    return completed


def _registry():
    return modal.Dict.from_name(REGISTRY_NAME, create_if_missing=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch = subparsers.add_parser("launch-canary")
    launch.add_argument("run_id")
    launch.add_argument("--duration-seconds", type=int, default=90)
    status = subparsers.add_parser("status")
    status.add_argument("run_id")
    args = parser.parse_args()
    try:
        if args.command == "launch-canary":
            record = reserve_and_spawn_canary(
                args.run_id,
                args.duration_seconds,
                registry=_registry(),
                function=modal.Function.from_name(APP_NAME, CANARY_FUNCTION),
            )
        else:
            record = poll_launch(
                args.run_id,
                registry=_registry(),
                call_from_id=modal.FunctionCall.from_id,
            )
    except (OSError, ValueError, ModalControlError) as error:
        parser.error(str(error))
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
