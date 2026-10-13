"""Validate the frozen final-holdout protocol, reports and acceptance gates."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.eval_proposals import ACTION_TO_TOOL, load_cases, strict_json
from scripts.review_holdout_labels import load_review, validate_review
from scripts.score_sft_generation import score_proposal_text, summarize_proposal_texts


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals/sft_final_holdout_manifest_v1.json"
RUNNER = ROOT / "modal_sft_final_holdout.py"
EXPECTED_MANIFEST_SHA256 = "05a36512e6cffc0f33556969fdecb6057f02ab08aa35ec2a34a4c4c6d9d0d8cc"


class FinalHoldoutProtocolError(ValueError):
    """Final holdout evidence violates the frozen protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalHoldoutProtocolError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _assignment(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise FinalHoldoutProtocolError(f"Runner assignment missing: {name}")


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    try:
        return ast.literal_eval(_assignment(tree, name))
    except (ValueError, TypeError) as error:
        raise FinalHoldoutProtocolError(f"Runner assignment is not literal: {name}") from error


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise FinalHoldoutProtocolError(f"Runner function missing: {name}")


def _modal_keywords(function: ast.FunctionDef) -> dict[str, Any]:
    for decorator in function.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "function"
        ):
            return {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in decorator.keywords
                if keyword.arg in {"gpu", "cpu", "memory", "timeout"}
            }
    raise FinalHoldoutProtocolError(f"Modal decorator missing: {function.name}")


def _image_local_paths(tree: ast.Module, assignment_name: str) -> list[str]:
    paths = []
    for call in ast.walk(_assignment(tree, assignment_name)):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in {"add_local_file", "add_local_dir"}
            and call.args
        ):
            paths.append(ast.literal_eval(call.args[0]))
    return paths


def _id_mismatch(row: dict[str, Any]) -> bool:
    calls = row.get("parsed", {}).get("calls", [])
    accepted = row.get("accepted_calls", [])
    if len(calls) != 1:
        return False
    arguments = calls[0].get("arguments")
    if not isinstance(arguments, dict) or "id" not in arguments:
        return False
    expected_ids = {
        call["arguments"]["id"]
        for call in accepted
        if "id" in call.get("arguments", {})
    }
    return bool(expected_ids) and arguments["id"] not in expected_ids


def _safety_errors(row: dict[str, Any]) -> set[str]:
    tags = set(row.get("error_tags", []))
    result = tags & {"unexpected_call", "multiple_calls", "wrong_tool"}
    if _id_mismatch(row):
        result.add("id_mismatch")
    return result


def validate_train_evidence(
    report: dict[str, Any],
    candidate: dict[str, Any],
    *,
    report_sha256: str,
    adapter_directory_sha256: str,
) -> dict[str, str]:
    """Validate the selected adapter without consulting development or holdout output."""

    _require(report_sha256 == candidate["train_report_sha256"], "Train report hash mismatch")
    _require(adapter_directory_sha256 == candidate["adapter_directory_sha256"], "Adapter directory hash mismatch")
    adapter = report.get("adapters", {}).get(candidate["label"], {})
    _require(report.get("run_id") == candidate["train_run_id"], "Train run ID mismatch")
    _require(report.get("state", {}).get("status") == "complete", "Train report is incomplete")
    _require(adapter.get("directory_sha256") == candidate["adapter_directory_sha256"], "Train report adapter directory mismatch")
    _require(adapter.get("state_sha256_reloaded") == candidate["adapter_state_sha256"], "Train report adapter state mismatch")
    return {
        "run_id": candidate["train_run_id"],
        "label": candidate["label"],
        "directory_sha256": candidate["adapter_directory_sha256"],
        "state_sha256": candidate["adapter_state_sha256"],
    }


def evaluate_automatic_acceptance(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    _require(len(baseline_rows) == len(candidate_rows) == 120, "Acceptance needs 120 paired cases")
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    candidate_by_id = {row["id"]: row for row in candidate_rows}
    _require(list(baseline_by_id) == list(candidate_by_id), "Base/candidate case order changed")
    _require(len(baseline_by_id) == 120, "Duplicate case IDs in acceptance input")
    base_summary = summarize_proposal_texts(baseline_rows)
    candidate_summary = summarize_proposal_texts(candidate_rows)
    action_rows = [row for row in candidate_rows if row["behavior"] == "call"]
    action_exact = sum(row["automatic_status"] == "pass" for row in action_rows)
    _require(len(action_rows) == acceptance["candidate_action_total"] == 84, "Action case total changed")

    baseline_failures = base_summary["fail"]
    candidate_failures = candidate_summary["fail"]
    required_reduction = (
        0
        if baseline_failures == 0
        else math.ceil(acceptance["minimum_error_reduction_fraction"] * baseline_failures)
    )
    error_reduction_pass = (
        candidate_failures == 0
        if baseline_failures == 0
        else baseline_failures - candidate_failures >= required_reduction
    )
    action_regressions = [
        action
        for action in ACTION_TO_TOOL
        if candidate_summary["by_action"][action]["pass"]
        < base_summary["by_action"][action]["pass"]
    ]
    control_regressions = [
        behavior
        for behavior in ("clarify", "no_action")
        if candidate_summary["by_behavior"][behavior]["pass"]
        < base_summary["by_behavior"][behavior]["pass"]
    ]
    safety_regressions = []
    for case_id in baseline_by_id:
        before = _safety_errors(baseline_by_id[case_id])
        after = _safety_errors(candidate_by_id[case_id])
        new_errors = sorted(after - before)
        if new_errors:
            safety_regressions.append({"id": case_id, "new_errors": new_errors})
    generation_error_ids = [
        row["id"]
        for row in baseline_rows + candidate_rows
        if "generation_error" in row.get("error_tags", [])
    ]
    gates = {
        "complete_without_generation_error": not generation_error_ids,
        "candidate_action_exact_minimum": action_exact >= acceptance["candidate_action_exact_minimum"],
        "minimum_error_reduction": error_reduction_pass,
        "no_action_regression": not action_regressions,
        "controls_not_below_baseline": not control_regressions,
        "no_candidate_only_safety_regression": not safety_regressions,
    }
    return {
        "baseline_failures": baseline_failures,
        "candidate_failures": candidate_failures,
        "required_error_reduction": required_reduction,
        "observed_error_reduction": baseline_failures - candidate_failures,
        "candidate_action_exact": action_exact,
        "candidate_action_total": len(action_rows),
        "action_regressions": action_regressions,
        "control_regressions": control_regressions,
        "candidate_only_safety_regressions": safety_regressions,
        "generation_error_ids": generation_error_ids,
        "gates": gates,
        "automatic_eligible": all(gates.values()),
        "manual_noncall_review": "pending",
        "gguf_rollout_authorized": False,
    }


def validate_protocol(
    root: Path = ROOT,
    manifest_path: Path = MANIFEST,
    runner_path: Path = RUNNER,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    runner_path = runner_path.resolve()
    _require(sha256(manifest_path) == EXPECTED_MANIFEST_SHA256, "Final manifest hash changed")
    manifest = load_json(manifest_path)
    _require(manifest.get("schema_version") == 1, "Unsupported final protocol schema")
    _require(manifest.get("protocol_id") == "nexus-sft-final-holdout-v1", "Unexpected protocol ID")
    _require(manifest.get("holdout_authorized") is True, "Final holdout is not authorized")

    holdout = manifest["holdout"]
    dataset_path = root / holdout["path"]
    review_path = root / holdout["human_review_path"]
    _require(sha256(dataset_path) == holdout["sha256"], "Holdout dataset hash changed")
    _require(sha256(review_path) == holdout["human_review_sha256"], "Human review hash changed")
    _, cases = load_cases(dataset_path)
    _require(len(cases) == holdout["count"] == 120, "Holdout case count changed")
    behavior = Counter(case["expected"]["behavior"] for case in cases)
    actions = Counter(
        case["expected"]["action"]
        for case in cases
        if case["expected"]["behavior"] == "call"
    )
    _require(behavior == Counter({"call": 84, "no_action": 24, "clarify": 12}), "Holdout behavior balance changed")
    _require(actions == Counter({action: 12 for action in ACTION_TO_TOOL}), "Holdout action balance changed")
    review_summary = validate_review(load_review(review_path), dataset_path, require_complete=True)
    _require(review_summary["counts"]["approved"] == holdout["human_review_approved"] == 120, "Human review is incomplete")

    candidate = manifest["candidate"]
    selection_path = root / candidate["selection_review_path"]
    _require(sha256(selection_path) == candidate["selection_review_sha256"], "Selection review hash changed")
    selection = load_json(selection_path)
    _require(selection["decision"]["selected_checkpoint"] == candidate["label"] == "step-200", "Candidate selection changed")
    _require(candidate["adapter_state_sha256"] == "d5d597b6d1e9990e6031af86b129bf0df486cbd3795b000f09f83610458a6473", "Adapter state changed")

    contract = manifest["contract"]
    _require(contract["prompt_version"] == "v13" and contract["tool_routing"] == "all", "Prompt or routing changed")
    _require(contract["do_sample"] is False and contract["temperature"] == 0.0, "Generation is not greedy")
    _require(contract["maximum_new_tokens"] == 128 and contract["batch_size"] == 1, "Generation limits changed")
    runtime = manifest["runtime"]
    _require(runtime["gpu"] == "T4" and runtime["cpu_cores"] == 2 and runtime["memory_gib"] == 32, "Modal resources changed")
    _require(runtime["timeout_seconds"] == 2400, "Final timeout changed")
    budget = manifest["budget"]
    seconds = runtime["timeout_seconds"]
    gpu = seconds * budget["t4_usd_per_second"]
    cpu = seconds * runtime["cpu_cores"] * budget["cpu_core_usd_per_second"]
    memory = seconds * runtime["memory_gib"] * budget["memory_gib_usd_per_second"]
    _require(math.isclose(gpu, budget["maximum_gpu_usd"], abs_tol=1e-12), "GPU budget arithmetic changed")
    _require(math.isclose(cpu, budget["maximum_cpu_usd"], abs_tol=1e-12), "CPU budget arithmetic changed")
    _require(math.isclose(memory, budget["maximum_memory_usd"], abs_tol=1e-12), "Memory budget arithmetic changed")
    _require(math.isclose(gpu + cpu + memory, budget["maximum_compute_usd"], abs_tol=1e-12), "Compute budget arithmetic changed")
    _require(budget["maximum_compute_usd"] < budget["authorized_total_usd"] == 0.7, "Authorized budget changed")
    _require(budget["automatic_retry"] is False, "Automatic retry must remain disabled")

    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    constants = {
        name: _literal_assignment(tree, name)
        for name in (
            "PROTOCOL_ID",
            "MANIFEST_SHA256",
            "MODEL_ID",
            "MODEL_REVISION",
            "HOLDOUT_SHA256",
            "HUMAN_REVIEW_SHA256",
            "MAX_NEW_TOKENS",
        )
    }
    _require(constants["PROTOCOL_ID"] == manifest["protocol_id"], "Runner protocol changed")
    _require(constants["MANIFEST_SHA256"] == EXPECTED_MANIFEST_SHA256, "Runner manifest lock changed")
    _require(constants["MODEL_ID"] == manifest["model"]["id"] and constants["MODEL_REVISION"] == manifest["model"]["revision"], "Runner model lock changed")
    _require(constants["HOLDOUT_SHA256"] == holdout["sha256"], "Runner holdout lock changed")
    _require(constants["HUMAN_REVIEW_SHA256"] == holdout["human_review_sha256"], "Runner human-review lock changed")
    _require(constants["MAX_NEW_TOKENS"] == contract["maximum_new_tokens"], "Runner token limit changed")
    _require(
        _modal_keywords(_function(tree, "run_final_holdout"))
        == {"gpu": "T4", "cpu": 2, "memory": 32768, "timeout": 2400},
        "Final Modal resources changed",
    )
    paths = _image_local_paths(tree, "holdout_image")
    _require(holdout["path"] in paths and holdout["human_review_path"] in paths, "Holdout image is missing locked input")
    _require(not any("development_v1.json" in path or "sft_candidate_v2.json" in path for path in paths), "Holdout image includes development or training data")
    runner_text = runner_path.read_text(encoding="utf-8")
    _require("if value.get(\"holdout_authorized\") is not True" in runner_text, "Remote authorization guard missing")
    _require("if not protocol[\"holdout_authorized\"]" in runner_text, "Local authorization guard missing")
    _require("result_volume.commit()" in runner_text and "persist_every_case" in json.dumps(manifest), "Per-case persistence contract missing")
    for forbidden in ("sqlite3", "run_turn(", "create_task(", "delete_task(", "update_task("):
        _require(forbidden not in runner_text, f"Runner contains forbidden side-effect path: {forbidden}")
    defaults = [ast.literal_eval(value) for value in _function(tree, "main").args.defaults]
    _require(defaults and defaults[0] == "preflight", "Runner default stage must be preflight")

    return {
        "protocol_id": manifest["protocol_id"],
        "manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256(manifest_path)},
        "runner": {"path": str(runner_path.relative_to(root)), "sha256": sha256(runner_path), "default_stage": "preflight"},
        "holdout": {"path": holdout["path"], "sha256": holdout["sha256"], "cases": 120},
        "human_review": review_summary,
        "candidate": {"label": candidate["label"], "adapter_state_sha256": candidate["adapter_state_sha256"]},
        "resources": {"gpu": "T4", "cpu": 2, "memory_gib": 32, "timeout_seconds": 2400},
        "maximum_compute_usd": budget["maximum_compute_usd"],
        "authorized_total_usd": budget["authorized_total_usd"],
        "holdout_authorized": True,
        "tools_or_database_executed": False,
        "gguf_rollout_authorized": False,
    }


def validate_report(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_run_id: str,
) -> dict[str, Any]:
    _require(report.get("schema_version") == 1, "Unsupported final report schema")
    _require(report.get("protocol_id") == manifest["protocol_id"], "Report protocol mismatch")
    _require(report.get("run_id") == expected_run_id and report.get("stage") == "final_holdout", "Report identity mismatch")
    _require(report.get("dataset", {}).get("sha256") == manifest["holdout"]["sha256"], "Report holdout mismatch")
    _, cases = load_cases(ROOT / manifest["holdout"]["path"])
    results = report.get("results", {})
    _require(set(results) == {"baseline", "step-200"}, "Report configuration set changed")
    for label, value in results.items():
        rows = value.get("cases", [])
        _require(len(rows) == 120, f"Incomplete {label} report")
        _require([row.get("id") for row in rows] == [case["id"] for case in cases], f"{label} case order changed")
        for case, row in zip(cases, rows):
            if "generation_error" in row.get("error_tags", []):
                _require(row.get("generated_text") is None and row.get("generation_error"), f"Invalid generation error row: {case['id']}")
                continue
            rescored = score_proposal_text(case, row.get("generated_text"))
            for field in (
                "id",
                "family",
                "behavior",
                "action",
                "prompt",
                "generated_text",
                "parsed",
                "accepted_calls",
                "automatic_status",
                "proposal_correct",
                "error_tags",
                "reply_review",
            ):
                _require(row.get(field) == rescored[field], f"{label} score mismatch: {case['id']} {field}")
        _require(value.get("summary") == summarize_proposal_texts(rows), f"{label} summary mismatch")
    expected_decision = evaluate_automatic_acceptance(
        results["baseline"]["cases"],
        results["step-200"]["cases"],
        manifest["acceptance"],
    )
    _require(report.get("automatic_decision") == expected_decision, "Automatic decision mismatch")
    provenance = report.get("provenance", {})
    _require(provenance.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256, "Report manifest provenance mismatch")
    _require(provenance.get("holdout_sha256") == manifest["holdout"]["sha256"], "Report holdout provenance mismatch")
    _require(provenance.get("tools_or_database_executed") is False and provenance.get("response_repair") is False, "Report used forbidden integration or repair")
    _require(report.get("manual_noncall_review") == "pending", "Raw report must not self-authorize manual review")
    return {
        "run_id": expected_run_id,
        "summaries": {label: value["summary"] for label, value in results.items()},
        "automatic_decision": expected_decision,
        "manual_noncall_review": "pending",
        "gguf_rollout_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        summary = validate_protocol(manifest_path=args.manifest.resolve())
        if args.report:
            _require(bool(args.run_id), "--report requires --run-id")
            summary["report"] = validate_report(
                load_json(args.report.resolve()),
                load_json(args.manifest.resolve()),
                expected_run_id=args.run_id,
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline audit only: no Modal function, model, GPU, tool, database or holdout inference was called.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
