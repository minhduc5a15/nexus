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
    _require(execution.get("runner_status") == "authorized_for_one_train_and_development_run", "Runner status changed")
    _require(execution.get("runner_path") == "modal_sft_candidate_v3.py", "Runner path changed")

    runtime = plan["runtime"]
    budget = plan["budget"]
    prices = (
        budget["t4_usd_per_second"],
        budget["cpu_core_usd_per_second"],
        budget["memory_gib_usd_per_second"],
    )
    gpu = gpu_seconds * prices[0]
    cpu = (gpu_seconds + execution["model_download_timeout_seconds"]) * runtime["cpu_cores"] * prices[1]
    memory = (
        gpu_seconds * runtime["training_memory_gib"]
        + execution["model_download_timeout_seconds"] * runtime["model_download_memory_gib"]
    ) * prices[2]
    _require(math.isclose(gpu, budget["maximum_gpu_usd"], abs_tol=1e-12), "GPU budget arithmetic changed")
    _require(math.isclose(cpu, budget["maximum_cpu_usd"], abs_tol=1e-12), "CPU budget arithmetic changed")
    _require(math.isclose(memory, budget["maximum_memory_usd"], abs_tol=1e-12), "Memory budget arithmetic changed")
    _require(math.isclose(gpu + cpu + memory, budget["maximum_compute_usd"], abs_tol=1e-12), "Total budget arithmetic changed")
    _require(budget["maximum_compute_usd"] < budget["requested_total_cost_cap_usd"] == 2.6, "Requested cap changed")

    _require(plan.get("training_authorized") is True, "Approved protocol must authorize training")
    _require(
        plan.get("authorization_basis")
        == "Project owner explicitly approved the 2.60 USD Modal cap on 2026-10-14.",
        "Training authorization basis changed",
    )
    _require(plan.get("next_gate") == "run_train_then_development_once_without_automatic_retry", "Unexpected next gate")
    _require(not any(plan.get("scope", {}).values()), "Draft scope must exclude external effects and rollout")
    return {
        "protocol_id": plan["protocol_id"],
        "dataset_cases": dataset_spec["count"],
        "optimizer_steps": training["maximum_steps"],
        "microbatches": training["maximum_steps"] * training["gradient_accumulation_steps"],
        "checkpoints": development["candidates"],
        "probe": probe,
        "maximum_planned_gpu_seconds": gpu_seconds,
        "maximum_compute_usd": budget["maximum_compute_usd"],
        "requested_total_cost_cap_usd": budget["requested_total_cost_cap_usd"],
        "training_authorized": True,
        "final_holdout_v2_inference_authorized": False,
        "runner_authored": True,
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
