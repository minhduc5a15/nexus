"""Audit the blocked cross-run resume proposal before budget approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.eval_proposals import strict_json


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "evals/sft_candidate_training_resume_proposal_v3.json"
ACTIVE_PLAN = ROOT / "evals/sft_candidate_training_plan_v3.json"


class ResumeProposalError(ValueError):
    """The cross-run resume proposal is unsafe or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResumeProposalError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def validate(path: Path = PROPOSAL) -> dict[str, Any]:
    proposal = load(path)
    active = load(ACTIVE_PLAN)
    _require(active.get("training_authorized") is False, "Active training must remain blocked")
    _require(proposal.get("status") == "ready_for_budget_review", "Proposal status changed")
    _require(proposal.get("training_authorized") is False, "Proposal unexpectedly authorizes training")
    run_ids = proposal.get("target_run_ids", {})
    _require(
        run_ids
        == {
            "train": "sft-candidate-v3-train-20261016d",
            "development": "sft-candidate-v3-development-20261016d",
        },
        "Target run IDs changed",
    )
    source = proposal.get("resume_source", {})
    _require(source.get("source_run_id") == "sft-candidate-v3-train-20261014c", "Source run changed")
    _require(source.get("checkpoint_step") == 25, "Source checkpoint step changed")
    _require(source.get("planned_final_step") == 300, "Final step changed")
    _require(source.get("remaining_optimizer_steps") == 275, "Remaining-step arithmetic changed")
    _require(
        source.get("directory_sha256")
        == "e37986c9877f4bba225cec1b0e252dae5be5c243d0ad8a1e3923cd10f337d797",
        "Source checkpoint hash changed",
    )
    inspection = ROOT / source.get("inspection_path", "")
    if inspection.is_file():
        _require(sha256(inspection) == source.get("inspection_sha256"), "Inspection artifact hash mismatch")
        inspected = load(inspection)
        _require(inspected.get("trainer_global_step") == 25, "Inspected trainer step changed")
        _require(inspected.get("directory_sha256") == source["directory_sha256"], "Inspected directory hash mismatch")
        _require(inspected.get("gpu_used") is False and inspected.get("model_loaded") is False, "Inspector used model compute")
    execution = proposal.get("execution", {})
    _require(execution.get("transport") == "deployed_function_spawn", "Transport changed")
    _require(execution.get("atomic_duplicate_guard") is True, "Duplicate guard changed")
    _require(execution.get("automatic_retry") is False, "Automatic retry changed")
    seconds = execution.get("maximum_planned_gpu_seconds")
    _require(seconds == 10200, "Maximum runtime changed")
    budget = proposal.get("budget", {})
    _require(budget.get("pricing_checked_on") == "2026-10-16", "Pricing date changed")
    maximum = (
        seconds * budget["t4_usd_per_hour"] / 3600
        + seconds * 2 * budget["cpu_core_usd_per_hour"] / 3600
        + seconds * 16 * budget["memory_gib_usd_per_hour"] / 3600
    )
    _require(math.isclose(maximum, budget.get("maximum_compute_usd", -1), abs_tol=1e-12), "Budget arithmetic changed")
    _require(maximum < budget.get("requested_additional_cost_cap_usd", -1) == 2.45, "Requested cap changed")
    scope = proposal.get("scope", {})
    _require(scope.get("train") is True and scope.get("development") is True, "Train/development scope changed")
    _require(not any(value for key, value in scope.items() if key not in {"train", "development"}), "Proposal opens later stages")
    gates = proposal.get("gates_before_authorization", {})
    _require(gates.get("deployed_spawn_canary_passed") is True, "Spawn canary gate failed")
    _require(gates.get("checkpoint_inspected_in_place") is True, "Checkpoint inspection gate failed")
    _require(gates.get("cross_run_resume_unit_tests_passed") is True, "Resume unit-test gate failed")
    _require(gates.get("full_test_suite_passed") is True, "Full test-suite gate failed")
    _require(gates.get("active_manifest_updated_and_hash_locked") is False, "Active manifest changed before approval")
    _require(gates.get("project_owner_approved_additional_cost") is False, "Proposal claims unreceived approval")
    _require(proposal.get("next_gate") == "project_owner_budget_approval", "Unexpected next gate")
    return {
        "proposal_id": proposal["proposal_id"],
        "status": proposal["status"],
        "target_run_ids": run_ids,
        "source_checkpoint_step": source["checkpoint_step"],
        "remaining_optimizer_steps": source["remaining_optimizer_steps"],
        "source_checkpoint_sha256": source["directory_sha256"],
        "maximum_compute_usd": maximum,
        "requested_additional_cost_cap_usd": budget["requested_additional_cost_cap_usd"],
        "training_authorized": False,
        "final_holdout_v2_inference": scope["final_holdout_v2_inference"],
        "next_gate": proposal["next_gate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=PROPOSAL)
    args = parser.parse_args()
    try:
        summary = validate(args.proposal.resolve())
    except (OSError, ValueError, KeyError, TypeError, ResumeProposalError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Review artifact only: no Modal call, GPU, training, development or holdout inference was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
