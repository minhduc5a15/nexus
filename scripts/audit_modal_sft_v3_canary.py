"""Audit the completed Modal deployed-spawn canary and its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.eval_proposals import strict_json


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "evals/sft_v3_orchestration_canary_v1.json"


class CanaryAuditError(ValueError):
    """The canary decision or raw evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryAuditError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def validate(path: Path = DECISION) -> dict[str, Any]:
    decision = load(path)
    _require(decision.get("schema_version") == 1, "Unsupported canary schema")
    _require(decision.get("status") == "passed", "Canary did not pass")
    _require(
        decision.get("run_id") == "sft-v3-orchestration-canary-20261016a",
        "Canary run ID changed",
    )
    result = decision.get("result", {})
    _require(result.get("heartbeats") == 3, "Canary heartbeat count changed")
    _require(result.get("gpu_used") is False, "Canary used a GPU")
    _require(result.get("model_loaded") is False, "Canary loaded a model")
    _require(result.get("training_authorized") is False, "Canary authorized training")
    _require(
        result.get("final_holdout_inference") is False,
        "Canary opened final holdout inference",
    )
    gates = decision.get("gates", {})
    _require(gates and all(value is True for value in gates.values()), "A canary gate failed")
    launch = decision.get("launch", {})
    _require(launch.get("launcher_returned_without_waiting") is True, "Launcher waited for remote result")
    _require(launch.get("duplicate_launch_exit_code") == 2, "Duplicate launch was not rejected")
    _require(launch.get("automatic_retry_performed") is False, "Canary retried automatically")
    billing = decision.get("billing", {})
    breakdown = billing.get("breakdown_usd", {})
    _require(breakdown.get("gpu") == 0.0, "Billing contains GPU cost")
    _require(
        math.isclose(
            sum(breakdown.values()),
            billing.get("deploy_and_canary_metered_cost_usd", -1),
            abs_tol=1e-12,
        ),
        "Canary billing arithmetic changed",
    )
    artifacts = decision.get("artifacts", {})
    for path_key, hash_key in (
        ("raw_report_path", "raw_report_sha256"),
        ("launch_record_path", "launch_record_sha256"),
    ):
        raw_path = ROOT / artifacts.get(path_key, "")
        if raw_path.is_file():
            _require(sha256(raw_path) == artifacts.get(hash_key), f"{path_key} hash mismatch")
    _require(
        decision.get("next_gate")
        == "design_and_review_explicit_cross_run_resume_from_checkpoint_25_before_requesting_new_gpu_budget",
        "Unexpected next gate",
    )
    return {
        "protocol_id": decision["protocol_id"],
        "status": decision["status"],
        "run_id": decision["run_id"],
        "function_call_id": launch["function_call_id"],
        "heartbeats": result["heartbeats"],
        "elapsed_seconds": result["elapsed_seconds"],
        "metered_cost_usd": billing["deploy_and_canary_metered_cost_usd"],
        "gpu_cost_usd": breakdown["gpu"],
        "training_authorized": result["training_authorized"],
        "final_holdout_inference": result["final_holdout_inference"],
        "next_gate": decision["next_gate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, default=DECISION)
    args = parser.parse_args()
    try:
        summary = validate(args.decision.resolve())
    except (OSError, ValueError, KeyError, TypeError, CanaryAuditError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Audit only: no Modal call, GPU, model, training or holdout inference was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
