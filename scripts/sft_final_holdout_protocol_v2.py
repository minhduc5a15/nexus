"""Validate the locked final-holdout v2 protocol and its raw report."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from scripts.eval_proposals import ACTION_TO_TOOL, load_cases, strict_json
from scripts.review_holdout_labels import load_review, validate_review
from scripts.score_sft_generation import score_proposal_text, summarize_proposal_texts


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals/sft_final_holdout_manifest_v2.json"
RUNNER = ROOT / "modal_sft_final_holdout_v2.py"
EXPECTED_MANIFEST_SHA256 = "ea268886950af27c89437513f39dcd7b3d540fe3d9406f9594299043a8531c65"


class FinalHoldoutV2Error(ValueError):
    """Final holdout v2 evidence violates its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalHoldoutV2Error(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _literal(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError) as error:
                raise FinalHoldoutV2Error(f"Runner assignment is not literal: {name}") from error
    raise FinalHoldoutV2Error(f"Runner assignment missing: {name}")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise FinalHoldoutV2Error(f"Runner function missing: {name}")


def _modal_resources(function: ast.FunctionDef) -> dict[str, Any]:
    for decorator in function.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "function"
        ):
            return {
                item.arg: ast.literal_eval(item.value)
                for item in decorator.keywords
                if item.arg in {"gpu", "cpu", "memory", "timeout", "retries"}
            }
    raise FinalHoldoutV2Error("Modal function decorator is missing")


def _image_paths(tree: ast.Module) -> list[str]:
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "holdout_image" for target in node.targets)
    )
    paths = []
    for call in ast.walk(assignment.value):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in {"add_local_file", "add_local_dir"}
            and call.args
        ):
            paths.append(ast.literal_eval(call.args[0]))
    return paths


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
    _require(manifest.get("protocol_id") == "nexus-sft-final-holdout-v2", "Protocol identity changed")
    _require(manifest.get("holdout_authorized") is True, "Final holdout is not authorized")

    holdout = manifest["holdout"]
    dataset_path = root / holdout["path"]
    review_path = root / holdout["human_review_path"]
    _require(sha256(dataset_path) == holdout["sha256"], "Holdout dataset hash changed")
    _require(sha256(review_path) == holdout["human_review_sha256"], "Human review hash changed")
    _, cases = load_cases(dataset_path)
    _require(len(cases) == holdout["count"] == 120, "Holdout case count changed")
    behaviors = Counter(case["expected"]["behavior"] for case in cases)
    actions = Counter(
        case["expected"]["action"]
        for case in cases
        if case["expected"]["behavior"] == "call"
    )
    _require(behaviors == Counter({"call": 84, "clarify": 12, "no_action": 24}), "Behavior balance changed")
    _require(actions == Counter({action: 12 for action in ACTION_TO_TOOL}), "Action balance changed")
    review = validate_review(load_review(review_path), dataset_path, require_complete=True)
    _require(review["counts"]["approved"] == 120, "Human review is incomplete")

    selection = manifest["candidate"]["selection_basis"]
    _require(sha256(root / selection["development_path"]) == selection["development_sha256"], "Development evidence changed")
    _require(sha256(root / selection["semantic_review_path"]) == selection["semantic_review_sha256"], "Semantic review changed")
    _require(manifest["candidate"]["label"] == "v4-step-320", "Selected candidate changed")
    _require(manifest["incumbent"]["label"] == "v3-step-300", "Incumbent changed")

    contract = manifest["contract"]
    _require(contract["prompt_version"] == "v13" and contract["tool_routing"] == "all", "Request contract changed")
    _require(contract["do_sample"] is False and contract["temperature"] == 0.0, "Generation is not greedy")
    _require(contract["maximum_new_tokens"] == 128 and contract["batch_size"] == 1, "Generation bounds changed")
    runtime = manifest["runtime"]
    budget = manifest["budget"]
    seconds = runtime["timeout_seconds"]
    expected = (
        seconds * budget["t4_usd_per_second"]
        + seconds * runtime["cpu_cores"] * budget["cpu_core_usd_per_second"]
        + seconds * runtime["memory_gib"] * budget["memory_gib_usd_per_second"]
    )
    _require(math.isclose(expected, budget["maximum_compute_usd"], abs_tol=1e-12), "Budget arithmetic changed")
    _require(expected < budget["authorized_total_usd"] == 0.7, "Budget authorization changed")
    _require(budget["automatic_retry"] is False, "Automatic retry must remain disabled")

    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    locks = {
        name: _literal(tree, name)
        for name in (
            "PROTOCOL_ID", "MANIFEST_SHA256", "MODEL_ID", "MODEL_REVISION",
            "HOLDOUT_SHA256", "HUMAN_REVIEW_SHA256", "RUN_ID", "MAX_NEW_TOKENS",
        )
    }
    _require(locks["PROTOCOL_ID"] == manifest["protocol_id"], "Runner protocol lock changed")
    _require(locks["MANIFEST_SHA256"] == EXPECTED_MANIFEST_SHA256, "Runner manifest lock changed")
    _require(locks["MODEL_ID"] == manifest["model"]["id"] and locks["MODEL_REVISION"] == manifest["model"]["revision"], "Runner model lock changed")
    _require(locks["HOLDOUT_SHA256"] == holdout["sha256"], "Runner dataset lock changed")
    _require(locks["HUMAN_REVIEW_SHA256"] == holdout["human_review_sha256"], "Runner review lock changed")
    _require(locks["RUN_ID"] == manifest["execution"]["run_id"], "Runner run ID changed")
    _require(locks["MAX_NEW_TOKENS"] == contract["maximum_new_tokens"], "Runner token limit changed")
    _require(
        _modal_resources(_function(tree, "run_final_holdout"))
        == {"gpu": "T4", "cpu": 2, "memory": 32768, "timeout": 2400, "retries": 0},
        "Modal resources changed",
    )
    paths = _image_paths(tree)
    _require(holdout["path"] in paths and holdout["human_review_path"] in paths, "Holdout image lacks frozen inputs")
    forbidden_paths = ("development", "sft_candidate_v3.json", "sft_candidate_v4_draft.json", "holdout_v1")
    _require(not any(any(marker in path for marker in forbidden_paths) for path in paths), "Holdout image contains diagnostic or training data")
    runner_text = runner_path.read_text(encoding="utf-8")
    for forbidden in ("sqlite3", "run_turn(", "create_task(", "delete_task(", "update_task("):
        _require(forbidden not in runner_text, f"Runner contains side-effect path: {forbidden}")
    defaults = [ast.literal_eval(value) for value in _function(tree, "main").args.defaults]
    _require(defaults == ["preflight"], "Runner default stage must be preflight")
    _require("run_final_holdout.spawn(RUN_ID)" in runner_text, "Runner must submit asynchronously")

    return {
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha256(runner_path),
        "holdout_sha256": holdout["sha256"],
        "approved_labels": review["counts"]["approved"],
        "configurations": manifest["execution"]["configurations"],
        "run_id": manifest["execution"]["run_id"],
        "maximum_compute_usd": expected,
        "holdout_authorized": True,
        "tools_or_database_executed": False,
    }


def _safety_errors(row: dict[str, Any]) -> set[str]:
    result = set(row.get("error_tags", [])) & {"unexpected_call", "multiple_calls", "wrong_tool"}
    calls = row.get("parsed", {}).get("calls", [])
    accepted = row.get("accepted_calls", [])
    if len(calls) == 1 and isinstance(calls[0].get("arguments"), dict) and "id" in calls[0]["arguments"]:
        expected_ids = {call["arguments"]["id"] for call in accepted if "id" in call.get("arguments", {})}
        if expected_ids and calls[0]["arguments"]["id"] not in expected_ids:
            result.add("id_mismatch")
    return result


def evaluate_acceptance(models: dict[str, dict[str, Any]], acceptance: dict[str, Any]) -> dict[str, Any]:
    labels = ("base", "v3-step-300", "v4-step-320")
    rows = {label: models[label]["cases"] for label in labels}
    _require(all(len(value) == 120 for value in rows.values()), "Acceptance needs 120 rows per model")
    ids = {label: [row["id"] for row in value] for label, value in rows.items()}
    _require(ids["base"] == ids["v3-step-300"] == ids["v4-step-320"], "Case order changed")
    summaries = {label: summarize_proposal_texts(value) for label, value in rows.items()}
    candidate_rows = rows["v4-step-320"]
    action_exact = sum(row["automatic_status"] == "pass" for row in candidate_rows if row["behavior"] == "call")
    base_failures = summaries["base"]["fail"]
    candidate_failures = summaries["v4-step-320"]["fail"]
    required_reduction = 0 if base_failures == 0 else math.ceil(acceptance["minimum_error_reduction_from_base_fraction"] * base_failures)
    action_regressions = {}
    for reference in ("base", "v3-step-300"):
        action_regressions[reference] = [
            action for action in ACTION_TO_TOOL
            if summaries["v4-step-320"]["by_action"][action]["pass"]
            < summaries[reference]["by_action"][action]["pass"]
        ]
    control_regressions = {}
    for reference in ("base", "v3-step-300"):
        control_regressions[reference] = [
            behavior for behavior in ("clarify", "no_action")
            if summaries["v4-step-320"]["by_behavior"][behavior]["pass"]
            < summaries[reference]["by_behavior"][behavior]["pass"]
        ]
    safety_regressions = []
    for index, candidate in enumerate(candidate_rows):
        before = _safety_errors(rows["base"][index]) | _safety_errors(rows["v3-step-300"][index])
        new = sorted(_safety_errors(candidate) - before)
        if new:
            safety_regressions.append({"id": candidate["id"], "new_errors": new})
    generation_errors = {
        label: [row["id"] for row in value if "generation_error" in row.get("error_tags", [])]
        for label, value in rows.items()
    }
    gates = {
        "complete_without_generation_error": not any(generation_errors.values()),
        "candidate_action_exact_minimum": action_exact >= acceptance["candidate_action_exact_minimum"],
        "minimum_error_reduction_from_base": base_failures - candidate_failures >= required_reduction,
        "candidate_total_not_below_incumbent": summaries["v4-step-320"]["pass"] >= summaries["v3-step-300"]["pass"],
        "no_action_regression_vs_base_or_incumbent": not any(action_regressions.values()),
        "controls_not_below_base_or_incumbent": not any(control_regressions.values()),
        "no_candidate_only_safety_regression": not safety_regressions,
    }
    return {
        "summaries": summaries,
        "candidate_action_exact": action_exact,
        "candidate_action_total": acceptance["candidate_action_total"],
        "base_failures": base_failures,
        "candidate_failures": candidate_failures,
        "required_error_reduction": required_reduction,
        "observed_error_reduction": base_failures - candidate_failures,
        "action_regressions": action_regressions,
        "control_regressions": control_regressions,
        "candidate_only_safety_regressions": safety_regressions,
        "generation_error_ids": generation_errors,
        "gates": gates,
        "automatic_eligible": all(gates.values()),
        "manual_noncall_review": "pending",
        "gguf_rollout_authorized": False,
    }


def validate_report(report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("schema_version") == 1, "Report schema changed")
    _require(report.get("protocol_id") == manifest["protocol_id"], "Report protocol changed")
    _require(report.get("run_id") == manifest["execution"]["run_id"], "Report run ID changed")
    _require(report.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256, "Report manifest changed")
    _require(report.get("dataset", {}).get("sha256") == manifest["holdout"]["sha256"], "Report dataset changed")
    _require(report.get("contract") == manifest["contract"], "Report generation contract changed")
    _require(report.get("state", {}).get("status") == "complete", "Report is incomplete")
    provenance = report.get("provenance", {})
    _require(provenance.get("tools_or_database_executed") is False, "Report executed application side effects")
    _require(provenance.get("response_repair") is False and provenance.get("generation_retry") is False, "Report repaired or retried responses")
    _, cases = load_cases(ROOT / manifest["holdout"]["path"])
    for label in manifest["execution"]["configurations"]:
        rows = report.get("models", {}).get(label, {}).get("cases", [])
        _require(len(rows) == 120, f"Incomplete rows for {label}")
        for case, row in zip(cases, rows):
            _require(row.get("id") == case["id"], f"Case order changed for {label}")
            if row.get("generation_error") is None:
                rescored = score_proposal_text(case, row.get("generated_text"))
                for key in ("automatic_status", "proposal_correct", "error_tags", "parsed"):
                    _require(row.get(key) == rescored.get(key), f"Raw score mismatch for {label}/{case['id']}/{key}")
        _require(report["models"][label]["summary"] == summarize_proposal_texts(rows), f"Summary mismatch for {label}")
    decision = evaluate_acceptance(report["models"], manifest["acceptance"])
    return {
        "run_id": report["run_id"],
        "report_sha256": None,
        "automatic_decision": decision,
        "manual_noncall_review_required": 36 * 3,
        "gguf_rollout_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = validate_protocol()
    if args.report:
        report = load_json(args.report)
        summary["report"] = validate_report(report, load_json(MANIFEST))
        summary["report"]["report_sha256"] = sha256(args.report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
