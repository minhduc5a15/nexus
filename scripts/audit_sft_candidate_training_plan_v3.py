"""Audit the offline QLoRA v3 training proposal without starting Modal."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from scripts.audit_sft_candidate_v3 import audit_dataset, verify_manifest, verify_review
from scripts.eval_proposals import ACTION_TO_TOOL, load_cases, strict_json


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "evals/sft_candidate_training_plan_v3.json"
ATTEMPT = ROOT / "evals/sft_candidate_training_attempt_v3.json"
REPLACEMENT_ATTEMPT = ROOT / "evals/sft_candidate_training_attempt_v3_replacement.json"
REPLACEMENT_AUTHORIZATION = ROOT / "evals/sft_candidate_training_replacement_v3.json"
ACTIVE_AUTHORIZATION = ROOT / "evals/sft_candidate_training_resume_authorization_v3.json"
REENTRANT_ATTEMPT = ROOT / "evals/sft_candidate_training_attempt_v3_reentrant.json"


class TrainingPlanV3Error(ValueError):
    """The proposed v3 protocol no longer matches its reviewable contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingPlanV3Error(message)


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _probe_balance(plan: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    ids = plan["training"].get("teacher_forced_probe_ids")
    _require(isinstance(ids, list) and len(ids) == len(set(ids)) == 40, "Probe must contain 40 unique IDs")
    by_id = {case["id"]: case for case in dataset["cases"]}
    _require(all(case_id in by_id for case_id in ids), "Probe references an unknown training case")
    selected = [by_id[case_id] for case_id in ids]
    actions = Counter(case["action"] for case in selected if case["behavior"] == "call")
    behaviors = Counter(case["behavior"] for case in selected)
    _require(actions == Counter({action: 4 for action in ACTION_TO_TOOL}), "Probe must contain four calls per action")
    _require(behaviors == Counter({"call": 28, "clarify": 7, "no_action": 5}), "Unexpected probe behavior balance")
    return {"behaviors": dict(behaviors), "calls_per_action": dict(actions)}


def validate_plan(path: Path = PLAN) -> dict[str, Any]:
    plan = load_json(path)
    _require(plan.get("schema_version") == 1, "Unsupported plan schema")
    _require(plan.get("protocol_id") == "nexus-sft-qlora-candidate-v3", "Unexpected protocol ID")

    dataset_spec = plan.get("dataset", {})
    dataset_path = ROOT / dataset_spec.get("path", "")
    review_path = ROOT / dataset_spec.get("review_path", "")
    data_manifest_path = ROOT / dataset_spec.get("data_manifest_path", "")
    _require(dataset_path == ROOT / "evals/sft_candidate_v3.json", "Unexpected training dataset")
    _require(dataset_spec.get("sha256") == sha256(dataset_path), "Training dataset hash mismatch")
    _require(dataset_spec.get("review_sha256") == sha256(review_path), "Training review hash mismatch")
    _require(dataset_spec.get("data_manifest_sha256") == sha256(data_manifest_path), "Data manifest hash mismatch")
    dataset = load_json(dataset_path)
    _require(audit_dataset(dataset_path)["cases"] == dataset_spec.get("count") == 600, "Training dataset is incomplete")
    _require(len(verify_review(review_path)["reviewed_case_ids"]) == 600, "Training review is incomplete")
    verify_manifest(data_manifest_path)

    training = plan.get("training", {})
    _require(training.get("per_device_batch_size") == 1, "Physical batch size changed")
    _require(training.get("gradient_accumulation_steps") == 4, "Gradient accumulation changed")
    _require(training.get("effective_batch_size") == 4, "Effective batch size changed")
    _require(training.get("dataset_passes") == 2, "Dataset pass count changed")
    expected_steps = training["dataset_passes"] * dataset_spec["count"] // training["effective_batch_size"]
    _require(expected_steps == training.get("maximum_steps") == 300, "Optimizer-step arithmetic changed")
    _require(training.get("optimizer_steps_per_epoch") == 150, "Epoch checkpoint changed")
    _require(training.get("save_steps") == [150, 300], "Checkpoint schedule changed")
    _require(training.get("recovery_checkpoint_interval_steps") == 25, "Recovery checkpoint interval changed")
    _require(training.get("recovery_checkpoint_retention") == 1, "Recovery checkpoint retention changed")
    _require(
        training.get("resume_scope")
        == "same_modal_input_or_hash_locked_cross_run_checkpoint",
        "Recovery scope changed",
    )
    resume_source = {
        "source_run_id": "sft-candidate-v3-train-20261014c",
        "source_modal_input_id": "in-01M4WE97X2S868S67G25JGMZV4:1791955869603-0",
        "checkpoint_step": 25,
        "directory_sha256": "e37986c9877f4bba225cec1b0e252dae5be5c243d0ad8a1e3923cd10f337d797",
    }
    _require(training.get("cross_run_resume") == resume_source, "Cross-run resume source changed")
    _require(training.get("warmup_steps") == 15, "Warmup ratio changed")
    _require(training.get("automatic_retry") is False, "Automatic retry must remain disabled")
    probe = _probe_balance(plan, dataset)

    development = plan.get("development", {})
    development_path = ROOT / development.get("path", "")
    _require(development.get("sha256") == sha256(development_path), "Development hash mismatch")
    _, development_cases = load_cases(development_path)
    _require(len(development_cases) == development.get("count") == 80, "Development set is incomplete")
    _require(development.get("candidates") == ["step-150", "step-300"], "Candidate checkpoints changed")

    diagnostic = plan.get("diagnostic_after_selection", {})
    diagnostic_path = ROOT / diagnostic.get("path", "")
    _require(diagnostic.get("sha256") == sha256(diagnostic_path), "Exposed diagnostic hash mismatch")
    _require(diagnostic.get("included_in_this_budget") is False, "Diagnostic must remain outside this run")

    holdout = plan.get("final_holdout", {})
    holdout_path = ROOT / holdout.get("path", "")
    review_path = ROOT / holdout.get("human_review_path", "")
    _require(holdout.get("sha256") == sha256(holdout_path), "Final holdout hash mismatch")
    _require(holdout.get("human_review_sha256") == sha256(review_path), "Final holdout review hash mismatch")
    _require(holdout.get("model_inference_authorized") is False, "Final holdout must remain closed")

    execution = plan.get("execution", {})
    _require(execution.get("planned_stages") == ["preflight", "train", "development"], "Unexpected stage order")
    gpu_seconds = execution.get("train_timeout_seconds") + execution.get("development_timeout_seconds")
    _require(gpu_seconds == execution.get("maximum_planned_gpu_seconds") == 10200, "GPU timeout arithmetic changed")
    _require(execution.get("automatic_retry") is False, "Execution retry changed")
    _require(execution.get("runner_status") == "authorized_resume_pending_launch", "Runner status changed")
    _require(execution.get("runner_path") == "modal_sft_candidate_v3.py", "Runner path changed")
    authorized_runs = {
        "train": "sft-candidate-v3-train-20261016d",
        "development": "sft-candidate-v3-development-20261016d",
    }
    _require(execution.get("authorized_run_ids") == authorized_runs, "Authorized run IDs changed")
    _require(
        execution.get("active_authorization_path")
        == "evals/sft_candidate_training_resume_authorization_v3.json",
        "Active authorization path changed",
    )
    _require(
        execution.get("active_authorization_sha256")
        == sha256(ACTIVE_AUTHORIZATION),
        "Active authorization hash changed",
    )

    runtime = plan["runtime"]
    budget = plan["budget"]
    gpu = gpu_seconds * budget["t4_usd_per_hour"] / 3600
    cpu = gpu_seconds * runtime["cpu_cores"] * budget["cpu_core_usd_per_hour"] / 3600
    memory = gpu_seconds * runtime["training_memory_gib"] * budget["memory_gib_usd_per_hour"] / 3600
    _require(math.isclose(gpu, budget["maximum_gpu_usd"], abs_tol=1e-12), "GPU budget arithmetic changed")
    _require(math.isclose(cpu, budget["maximum_cpu_usd"], abs_tol=1e-12), "CPU budget arithmetic changed")
    _require(math.isclose(memory, budget["maximum_memory_usd"], abs_tol=1e-12), "Memory budget arithmetic changed")
    _require(math.isclose(gpu + cpu + memory, budget["maximum_compute_usd"], abs_tol=1e-12), "Total budget arithmetic changed")
    _require(budget["maximum_compute_usd"] < budget["requested_additional_cost_cap_usd"] == 2.45, "Requested cap changed")

    _require(plan.get("training_authorized") is True, "Checkpoint-resumed training is not authorized")
    _require(
        plan.get("authorization_basis")
        == "Project owner explicitly approved the locked cross-run resume and additional 2.45 USD Modal cap on 2026-10-19.",
        "Training authorization basis changed",
    )
    _require(plan.get("next_gate") == "deploy_and_spawn_authorized_training_run", "Unexpected next gate")
    scope = plan.get("scope", {})
    _require(scope.get("train") is True and scope.get("development") is True, "Train/development scope changed")
    _require(not any(value for key, value in scope.items() if key not in {"train", "development"}), "Plan opens an unauthorized later stage")
    active = load_json(ACTIVE_AUTHORIZATION)
    _require(active.get("status") == "active", "Active authorization status changed")
    _require(active.get("authorized_run_ids") == authorized_runs, "Active authorization run IDs changed")
    _require(active.get("resume_source") == resume_source, "Active resume source changed")
    _require(active.get("additional_cost_cap_usd") == 2.45, "Active authorization cap changed")
    _require(active.get("execution", {}).get("transport") == "deployed_function_spawn", "Active transport changed")
    _require(active.get("execution", {}).get("atomic_duplicate_guard") is True, "Active duplicate guard changed")
    _require(active.get("execution", {}).get("project_automatic_retry") is False, "Active project retry rule changed")
    _require(active.get("execution", {}).get("provider_same_input_resume") is True, "Active provider resume rule changed")
    _require(active.get("execution", {}).get("recovery_checkpoint_interval_steps") == 25, "Active recovery interval changed")
    _require(active.get("scope", {}).get("final_holdout_v2_inference") is False, "Active authorization includes final holdout")
    _require(active.get("consumed") is False, "Active authorization was already consumed")
    reentrant_attempt = load_json(REENTRANT_ATTEMPT)
    _require(reentrant_attempt.get("status") == "interrupted", "Reentrant attempt status changed")
    _require(
        reentrant_attempt.get("run_id") == "sft-candidate-v3-train-20261014c",
        "Reentrant attempt run ID changed",
    )
    _require(reentrant_attempt.get("authorized_commit") == "a71471c", "Reentrant attempt commit changed")
    _require(
        reentrant_attempt.get("authorized_manifest_sha256")
        == "2174284d2e95bad9625dc26d23f5598991f96b2f127f4c991574392101608fe5",
        "Reentrant attempt manifest provenance changed",
    )
    reentrant_progress = reentrant_attempt.get("observed_progress", {})
    _require(reentrant_progress.get("last_logged_optimizer_step") == 31, "Reentrant logged progress changed")
    _require(reentrant_progress.get("last_durable_optimizer_step") == 25, "Reentrant durable progress changed")
    _require(reentrant_progress.get("operator_cancellation_observed") is True, "Operator cancellation record changed")
    _require(reentrant_progress.get("retained_candidate_adapter_paths") == [], "Unexpected retained candidate adapter")
    reentrant_cost = reentrant_attempt.get("cost", {})
    _require(
        math.isclose(reentrant_cost.get("modal_app_metered_cost_usd", -1), 0.21459993, abs_tol=1e-12),
        "Reentrant metered cost changed",
    )
    _require(
        math.isclose(
            sum(reentrant_cost.get("modal_app_metered_breakdown_usd", {}).values()),
            reentrant_cost.get("modal_app_metered_cost_usd", -1),
            abs_tol=1e-12,
        ),
        "Reentrant cost breakdown is inconsistent",
    )
    _require(reentrant_attempt.get("gates", {}).get("new_run_authorized") is False, "A new paid run must remain blocked")
    reentrant_raw_state = ROOT / "evals/results/sft-candidate-v3-train-20261014c/sft-candidate-v3/train-state.json"
    if reentrant_raw_state.is_file():
        _require(
            reentrant_attempt["state"]["raw_state_sha256"] == sha256(reentrant_raw_state),
            "Reentrant raw state hash mismatch",
        )
    replacement_runs = {
        "train": "sft-candidate-v3-train-20261014b",
        "development": "sft-candidate-v3-development-20261014b",
    }
    replacement = load_json(REPLACEMENT_AUTHORIZATION)
    _require(replacement.get("status") == "consumed", "Replacement authorization status changed")
    _require(replacement.get("authorized_run_ids") == replacement_runs, "Replacement authorization run IDs changed")
    _require(replacement.get("additional_cost_cap_usd") == 2.6, "Replacement authorization cap changed")
    _require(replacement.get("execution", {}).get("transport") == "modal_run_detach", "Replacement transport changed")
    _require(replacement.get("execution", {}).get("automatic_retry") is False, "Replacement retry rule changed")
    _require(replacement.get("scope", {}).get("final_holdout_v2_inference") is False, "Replacement must not authorize final holdout")
    _require(replacement.get("consumed") is True, "Replacement authorization was not consumed")
    replacement_attempt = load_json(REPLACEMENT_ATTEMPT)
    _require(replacement_attempt.get("status") == "interrupted", "Replacement attempt status changed")
    _require(replacement_attempt.get("run_id") == replacement_runs["train"], "Replacement attempt run ID changed")
    _require(
        replacement_attempt.get("authorized_manifest_sha256")
        == "940ad04e5b39a4dc863ce3010ae8ca1e4259c3e3ce560c2c2f4c9765258e8669",
        "Replacement attempt manifest provenance changed",
    )
    replacement_progress = replacement_attempt.get("observed_progress", {})
    _require(replacement_progress.get("last_logged_optimizer_step") == 68, "Replacement progress changed")
    _require(replacement_progress.get("provider_preemption_observed") is True, "Provider preemption record changed")
    _require(replacement_progress.get("restart_rejected_by_existing_state_guard") is True, "Restart failure record changed")
    _require(replacement_progress.get("retained_adapter_paths") == [], "Unexpected retained replacement adapter")
    replacement_cost = replacement_attempt.get("cost", {})
    _require(
        math.isclose(
            replacement_cost.get("modal_app_metered_cost_usd", -1),
            0.45206372,
            abs_tol=1e-12,
        ),
        "Replacement metered cost changed",
    )
    _require(
        math.isclose(
            sum(replacement_cost.get("modal_app_metered_breakdown_usd", {}).values()),
            replacement_cost.get("modal_app_metered_cost_usd", -1),
            abs_tol=1e-12,
        ),
        "Replacement cost breakdown is inconsistent",
    )
    replacement_gates = replacement_attempt.get("gates", {})
    _require(replacement_gates.get("replacement_authorization_consumed") is True, "Replacement authorization consumption changed")
    _require(replacement_gates.get("third_run_authorized") is False, "Third run must remain blocked")
    _require(replacement_gates.get("final_holdout_v2_inference_authorized") is False, "Final holdout must remain closed")
    replacement_raw_state = ROOT / "evals/results/sft-candidate-v3-train-20261014b/sft-candidate-v3/train-state.json"
    if replacement_raw_state.is_file():
        _require(
            replacement_attempt["state"]["raw_state_sha256"]
            == sha256(replacement_raw_state),
            "Replacement raw state hash mismatch",
        )
    attempt = load_json(ATTEMPT)
    _require(attempt.get("status") == "interrupted", "Training attempt status changed")
    _require(attempt.get("run_id") == "sft-candidate-v3-train-20261014a", "Training attempt run ID changed")
    _require(attempt.get("authorized_manifest_sha256") == "d48709b19d6cae099b3fd11ece64af2e8f1bb03bec6bc6045dc5869120f40860", "Authorized manifest provenance changed")
    progress = attempt.get("observed_progress", {})
    _require(progress.get("last_logged_optimizer_step") == 184, "Interrupted progress changed")
    _require(progress.get("retained_adapter_paths") == [] and progress.get("train_report_written") is False, "Interrupted artifact state changed")
    cost = attempt.get("cost", {})
    _require(
        math.isclose(cost.get("modal_app_metered_cost_usd", -1), 1.0208283, abs_tol=1e-12),
        "Interrupted attempt metered cost changed",
    )
    _require(
        math.isclose(
            sum(cost.get("modal_app_metered_breakdown_usd", {}).values()),
            cost.get("modal_app_metered_cost_usd", -1),
            abs_tol=1e-12,
        ),
        "Interrupted attempt cost breakdown is inconsistent",
    )
    gates = attempt.get("gates", {})
    _require(gates.get("original_authorization_consumed") is True, "Original authorization was not consumed")
    _require(gates.get("replacement_run_authorized") is False, "Replacement run must remain blocked")
    _require(gates.get("final_holdout_v2_inference_authorized") is False, "Final holdout must remain closed")
    raw_state = ROOT / "evals/results/sft-candidate-v3-train-20261014a/sft-candidate-v3/train-state.json"
    if raw_state.is_file():
        _require(attempt["state"]["raw_state_sha256"] == sha256(raw_state), "Downloaded raw state hash mismatch")
    return {
        "protocol_id": plan["protocol_id"],
        "dataset_cases": dataset_spec["count"],
        "optimizer_steps": training["maximum_steps"],
        "microbatches": training["maximum_steps"] * training["gradient_accumulation_steps"],
        "checkpoints": development["candidates"],
        "probe": probe,
        "maximum_planned_gpu_seconds": gpu_seconds,
        "maximum_compute_usd": budget["maximum_compute_usd"],
        "requested_additional_cost_cap_usd": budget["requested_additional_cost_cap_usd"],
        "training_authorized": True,
        "authorized_run_ids": authorized_runs,
        "final_holdout_v2_inference_authorized": False,
        "runner_authored": True,
        "attempt": {
            "run_id": attempt["run_id"],
            "status": attempt["status"],
            "last_logged_optimizer_step": progress["last_logged_optimizer_step"],
            "retained_adapters": len(progress["retained_adapter_paths"]),
            "metered_cost_usd": cost["modal_app_metered_cost_usd"],
            "replacement_run_authorized_at_interruption": gates["replacement_run_authorized"],
        },
        "replacement": {
            "authorization_id": replacement["authorization_id"],
            "additional_cost_cap_usd": replacement["additional_cost_cap_usd"],
            "consumed": replacement["consumed"],
            "attempt": {
                "status": replacement_attempt["status"],
                "last_logged_optimizer_step": replacement_progress[
                    "last_logged_optimizer_step"
                ],
                "metered_cost_usd": replacement_cost[
                    "modal_app_metered_cost_usd"
                ],
                "retained_adapters": len(
                    replacement_progress["retained_adapter_paths"]
                ),
                "third_run_authorized": replacement_gates[
                    "third_run_authorized"
                ],
            },
        },
        "active_authorization": {
            "authorization_id": active["authorization_id"],
            "additional_cost_cap_usd": active["additional_cost_cap_usd"],
            "consumed": active["consumed"],
            "status": active["status"],
        },
        "reentrant_attempt": {
            "status": reentrant_attempt["status"],
            "last_logged_optimizer_step": reentrant_progress["last_logged_optimizer_step"],
            "last_durable_optimizer_step": reentrant_progress["last_durable_optimizer_step"],
            "metered_cost_usd": reentrant_cost["modal_app_metered_cost_usd"],
            "candidate_adapters": len(reentrant_progress["retained_candidate_adapter_paths"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    args = parser.parse_args()
    try:
        summary = validate_plan(args.plan.resolve())
    except (OSError, ValueError, KeyError, TypeError, TrainingPlanV3Error) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline proposal only: no Modal function, model, GPU, final holdout, tool or database was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
