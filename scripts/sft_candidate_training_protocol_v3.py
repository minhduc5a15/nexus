"""Validate the blocked, reentrant SFT candidate v3 protocol and artifacts."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.audit_sft_candidate_v3 import audit_dataset, verify_manifest, verify_review
from scripts.eval_proposals import ACTION_TO_TOOL, load_cases, strict_json
from scripts.score_sft_generation import summarize_proposal_texts


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals/sft_candidate_training_plan_v3.json"
RUNNER = ROOT / "modal_sft_candidate_v3.py"
EXPECTED_MANIFEST_SHA256 = "2d8c1ea2451b7b68dc67c684cbf376d5f5e2363e278c66b58b7c8d61ddbfad3a"
EXPECTED_MODEL = {
    "id": "Qwen/Qwen3-4B-Instruct-2507",
    "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
}


class CandidateProtocolError(ValueError):
    """The candidate protocol or one of its artifacts violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateProtocolError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    _require(bool(files), f"Adapter directory is empty: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(candidate)))
    return digest.hexdigest()


def _assignment(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise CandidateProtocolError(f"Runner assignment missing: {name}")


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    try:
        return ast.literal_eval(_assignment(tree, name))
    except (ValueError, TypeError) as error:
        raise CandidateProtocolError(f"Runner assignment is not literal: {name}") from error


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise CandidateProtocolError(f"Runner function missing: {name}")


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
                if keyword.arg in {"gpu", "timeout", "cpu", "memory"}
            }
    raise CandidateProtocolError(f"Modal decorator missing: {function.name}")


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


def _validate_probe(manifest: dict[str, Any], dataset: dict[str, Any]) -> dict[str, int]:
    ids = manifest["training"].get("teacher_forced_probe_ids")
    _require(isinstance(ids, list) and len(ids) == len(set(ids)) == 40, "Teacher-forced probe must contain 40 unique cases")
    by_id = {case["id"]: case for case in dataset["cases"]}
    _require(all(case_id in by_id for case_id in ids), "Teacher-forced probe references an unknown case")
    selected = [by_id[case_id] for case_id in ids]
    call_actions = Counter(
        case["action"] for case in selected if case["behavior"] == "call"
    )
    clarify_actions = Counter(
        case["action"] for case in selected if case["behavior"] == "clarify"
    )
    behaviors = Counter(case["behavior"] for case in selected)
    _require(call_actions == Counter({action: 4 for action in ACTION_TO_TOOL}), "Probe must contain four calls per action")
    _require(clarify_actions == Counter({action: 1 for action in ACTION_TO_TOOL}), "Probe must contain one clarification per action")
    _require(behaviors == Counter({"call": 28, "clarify": 7, "no_action": 5}), "Unexpected probe behavior balance")
    return dict(behaviors)


def validate_protocol(
    root: Path = ROOT,
    manifest_path: Path = MANIFEST,
    runner_path: Path = RUNNER,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    runner_path = runner_path.resolve()
    _require(sha256(manifest_path) == EXPECTED_MANIFEST_SHA256, "Candidate training manifest hash changed")
    manifest = load_json(manifest_path)
    _require(manifest.get("schema_version") == 1, "Unsupported candidate protocol schema")
    _require(manifest.get("protocol_id") == "nexus-sft-qlora-candidate-v3", "Unexpected candidate protocol ID")
    _require(manifest.get("model") == EXPECTED_MODEL, "Candidate base model lock changed")

    dataset_spec = manifest.get("dataset", {})
    dataset_path = root / dataset_spec.get("path", "")
    review_path = root / dataset_spec.get("review_path", "")
    data_manifest_path = root / dataset_spec.get("data_manifest_path", "")
    _require(dataset_path == root / "evals/sft_candidate_v3.json", "Unexpected candidate dataset path")
    _require(dataset_spec.get("sha256") == sha256(dataset_path), "Candidate dataset hash changed")
    _require(dataset_spec.get("count") == 600, "Candidate dataset count changed")
    _require(dataset_spec.get("review_sha256") == sha256(review_path), "Candidate review hash changed")
    _require(dataset_spec.get("data_manifest_sha256") == sha256(data_manifest_path), "Candidate data manifest hash changed")
    data_audit = audit_dataset(dataset_path)
    data_review = verify_review(review_path)
    data_manifest = verify_manifest(data_manifest_path)
    _require(data_audit["cases"] == 600 and len(data_review["reviewed_case_ids"]) == 600, "Candidate review is incomplete")
    _require(data_manifest["training_authorized"] is False, "Data checkpoint unexpectedly authorizes training")

    development = manifest.get("development", {})
    development_path = root / development.get("path", "")
    _require(development_path == root / "evals/model_proposal_development_v1.json", "Unexpected development path")
    _require(development.get("sha256") == sha256(development_path), "Development dataset hash changed")
    _, development_cases = load_cases(development_path)
    _require(len(development_cases) == development.get("count") == 80, "Development count changed")
    _require(development.get("candidates") == ["step-150", "step-300"], "Candidate checkpoint set changed")

    contract = manifest.get("contract", {})
    _require(
        contract
        == {
            "prompt_version": "v13",
            "tool_routing": "all",
            "maximum_tool_calls": 1,
            "assistant_only_loss": True,
            "maximum_sequence_tokens": 2048,
        },
        "Candidate training contract changed",
    )
    training = manifest.get("training", {})
    _require(training.get("seed") == 42 and training.get("automatic_retry") is False, "Seed or retry rule changed")
    _require(training.get("per_device_batch_size") == 1, "Physical batch size changed")
    _require(training.get("gradient_accumulation_steps") == 4, "Gradient accumulation changed")
    _require(training.get("effective_batch_size") == 4, "Effective batch size changed")
    _require(training.get("maximum_steps") == 300, "Optimizer step count changed")
    _require(training.get("save_steps") == [150, 300], "Checkpoint schedule changed")
    _require(training.get("recovery_checkpoint_interval_steps") == 25, "Recovery checkpoint interval changed")
    _require(training.get("recovery_checkpoint_retention") == 1, "Recovery checkpoint retention changed")
    _require(training.get("resume_scope") == "same_modal_input_id_only", "Recovery resume scope changed")
    expected_steps = (
        training["dataset_passes"] * dataset_spec["count"]
        // training["effective_batch_size"]
    )
    _require(expected_steps == training["maximum_steps"], "Epoch/step arithmetic changed")
    probe_balance = _validate_probe(manifest, load_json(dataset_path))

    execution = manifest.get("execution", {})
    _require(execution.get("planned_stages") == ["preflight", "train", "development"], "Unexpected stage order")
    train_timeout = execution.get("train_timeout_seconds")
    development_timeout = execution.get("development_timeout_seconds")
    maximum_seconds = execution.get("maximum_planned_gpu_seconds")
    _require(train_timeout + development_timeout == maximum_seconds == 10200, "GPU timeout budget changed")
    authorized_runs = execution.get("authorized_run_ids", {})
    _require(
        authorized_runs
        == {
            "train": "sft-candidate-v3-train-20261014b",
            "development": "sft-candidate-v3-development-20261014b",
        },
        "Authorized replacement run IDs changed",
    )
    authorization_path = root / execution.get("replacement_authorization_path", "")
    _require(
        execution.get("replacement_authorization_sha256") == sha256(authorization_path),
        "Replacement authorization hash changed",
    )
    authorization = load_json(authorization_path)
    _require(authorization.get("status") == "consumed", "Replacement authorization status changed")
    _require(authorization.get("authorized_run_ids") == authorized_runs, "Replacement authorization run IDs changed")
    _require(authorization.get("additional_cost_cap_usd") == 2.6, "Replacement cost cap changed")
    _require(authorization.get("execution", {}).get("transport") == "modal_run_detach", "Replacement transport changed")
    _require(authorization.get("execution", {}).get("automatic_retry") is False, "Replacement retry rule changed")
    _require(authorization.get("scope", {}).get("final_holdout_v2_inference") is False, "Replacement authorizes final holdout")
    _require(authorization.get("consumed") is True, "Replacement authorization was not consumed")
    budget = manifest.get("budget", {})
    _require(math.isclose(maximum_seconds * budget.get("t4_usd_per_second", -1), budget.get("maximum_gpu_usd", -1), abs_tol=1e-9), "GPU cost arithmetic changed")
    _require(budget.get("requested_total_cost_cap_usd") == 2.6, "Requested total cost cap changed")
    _require(budget["requested_total_cost_cap_usd"] > budget["maximum_compute_usd"], "Total cost cap must exceed the compute ceiling")
    _require(budget.get("pricing_checked_on") == "2026-10-14", "Pricing verification date changed")

    _require(manifest.get("training_authorized") is False, "Consumed replacement must block training")
    _require(
        manifest.get("authorization_basis")
        == "The authorized detached replacement was consumed by sft-candidate-v3-train-20261014b and interrupted by provider preemption before a recovery checkpoint.",
        "Candidate training authorization basis changed",
    )
    scope = manifest.get("scope", {})
    _require(scope and not any(scope.values()), "Candidate protocol scope must exclude side effects, holdout and rollout")

    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    constants = {
        name: _literal_assignment(tree, name)
        for name in (
            "PROTOCOL_ID",
            "MANIFEST_SHA256",
            "MODEL_ID",
            "MODEL_REVISION",
            "CANDIDATE_SHA256",
            "DEVELOPMENT_SHA256",
            "MAX_SEQUENCE_TOKENS",
            "MAX_STEPS",
            "MAX_NEW_TOKENS",
        )
    }
    _require(constants["PROTOCOL_ID"] == manifest["protocol_id"], "Runner protocol ID changed")
    _require(constants["MANIFEST_SHA256"] == EXPECTED_MANIFEST_SHA256, "Runner manifest lock changed")
    _require(constants["MODEL_ID"] == manifest["model"]["id"], "Runner model ID changed")
    _require(constants["MODEL_REVISION"] == manifest["model"]["revision"], "Runner model revision changed")
    _require(constants["CANDIDATE_SHA256"] == dataset_spec["sha256"], "Runner dataset lock changed")
    _require(constants["DEVELOPMENT_SHA256"] == development["sha256"], "Runner development lock changed")
    _require(constants["MAX_SEQUENCE_TOKENS"] == contract["maximum_sequence_tokens"], "Runner sequence limit changed")
    _require(constants["MAX_STEPS"] == training["maximum_steps"], "Runner step count changed")
    _require(constants["MAX_NEW_TOKENS"] == development["generation"]["maximum_new_tokens"], "Runner generation limit changed")
    expected_resources = {
        "gpu": "T4",
        "cpu": manifest["runtime"]["cpu_cores"],
        "memory": manifest["runtime"]["training_memory_gib"] * 1024,
    }
    _require(
        _modal_keywords(_function(tree, "run_train"))
        == {**expected_resources, "timeout": train_timeout},
        "Train Modal resources changed",
    )
    _require(
        _modal_keywords(_function(tree, "run_development"))
        == {**expected_resources, "timeout": development_timeout},
        "Development Modal resources changed",
    )
    training_paths = _image_local_paths(tree, "training_image")
    development_paths = _image_local_paths(tree, "development_image")
    _require(not any("development" in path.casefold() for path in training_paths), "Training image contains development data")
    _require(not any("holdout" in path.casefold() for path in training_paths + development_paths), "Runner image contains final holdout data")
    _require("evals/model_proposal_development_v1.json" in development_paths, "Development image is missing its locked dataset")
    runner_text = runner_path.read_text(encoding="utf-8")
    _require(
        runner_text.count('if manifest.get("training_authorized") is not True') == 2,
        "Remote training authorization guard missing",
    )
    _require("if not protocol[\"training_authorized\"]" in runner_text, "Local training authorization guard missing")
    _require(
        'trainer_output = output / "trainer-checkpoints"' in runner_text
        and "CommitCheckpointCallback(" in runner_text
        and "modal.current_input_id()" in runner_text
        and "resume_from_checkpoint=" in runner_text
        and "save_only_model=False" in runner_text
        and 'checkpoint / ".nexus-complete"' in runner_text
        and 'output_dir=f"/tmp/nexus-sft-candidate-v3-' not in runner_text,
        "Reentrant checkpoint contract missing",
    )
    defaults = [ast.literal_eval(value) for value in _function(tree, "main").args.defaults]
    _require(defaults and defaults[0] == "preflight", "Runner default stage must be preflight")

    return {
        "protocol_id": manifest["protocol_id"],
        "manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256(manifest_path)},
        "runner": {"path": str(runner_path.relative_to(root)), "sha256": sha256(runner_path), "default_stage": "preflight"},
        "dataset": {"path": dataset_spec["path"], "sha256": dataset_spec["sha256"], "count": 600},
        "development": {"path": development["path"], "sha256": development["sha256"], "count": 80},
        "probe_balance": probe_balance,
        "optimizer_steps": training["maximum_steps"],
        "microbatches": training["maximum_steps"] * training["gradient_accumulation_steps"],
        "checkpoints": development["candidates"],
        "gpu": manifest["runtime"]["gpu"],
        "maximum_planned_gpu_seconds": maximum_seconds,
        "maximum_planned_gpu_cost_usd": budget["maximum_gpu_usd"],
        "requested_total_cost_cap_usd": budget["requested_total_cost_cap_usd"],
        "training_authorized": False,
        "authorized_run_ids": authorized_runs,
        "final_holdout_inference": False,
    }


def validate_train_report(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_run_id: str,
    adapter_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    _require(report.get("protocol_id") == manifest["protocol_id"], "Train report protocol mismatch")
    _require(report.get("stage") == "train" and report.get("run_id") == expected_run_id, "Unexpected train report identity")
    _require(report.get("dataset", {}).get("sha256") == manifest["dataset"]["sha256"], "Train report dataset mismatch")
    _require(report.get("state", {}).get("status") == "complete", "Train report is incomplete")
    gates = report.get("gates", {})
    _require(gates and all(value is True for value in gates.values()), "Train gates did not all pass")
    _require(report.get("training", {}).get("completed_steps") == 300, "Train step count mismatch")
    adapters = report.get("adapters", {})
    _require(set(adapters) == set(manifest["development"]["candidates"]), "Train checkpoints changed")
    final = adapters["step-300"]
    _require(final.get("state_sha256_after") == final.get("state_sha256_reloaded"), "Final adapter reload mismatch")
    if adapter_paths is not None:
        _require(set(adapter_paths) == set(adapters), "Adapter path set changed")
        for label, path in adapter_paths.items():
            _require(path.is_dir(), f"Adapter directory is missing: {label}")
            _require(directory_sha256(path) == adapters[label].get("directory_sha256"), f"Adapter directory hash mismatch: {label}")
    provenance = report.get("provenance", {})
    _require(provenance.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256, "Train provenance manifest mismatch")
    _require(provenance.get("final_holdout_inference") is False, "Train report used final holdout")
    return {"run_id": expected_run_id, "completed_steps": 300, "adapters": adapters}


def validate_development_report(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_run_id: str,
    expected_train_run_id: str,
) -> dict[str, Any]:
    _require(report.get("protocol_id") == manifest["protocol_id"], "Development protocol mismatch")
    _require(report.get("stage") == "development" and report.get("run_id") == expected_run_id, "Unexpected development identity")
    _require(report.get("train_run_id") == expected_train_run_id, "Development train gate mismatch")
    _require(report.get("state", {}).get("status") == "complete", "Development report is incomplete")
    gates = report.get("gates", {})
    _require(gates and all(value is True for value in gates.values()), "Development gates did not all pass")
    baseline = report.get("baseline", {})
    _require(len(baseline.get("cases", [])) == 80, "Baseline development cases incomplete")
    _require(baseline.get("summary") == summarize_proposal_texts(baseline["cases"]), "Baseline summary mismatch")
    candidates = report.get("candidates", {})
    _require(set(candidates) == set(manifest["development"]["candidates"]), "Development candidate set changed")
    for label, value in candidates.items():
        _require(len(value.get("cases", [])) == 80, f"{label} development cases incomplete")
        _require(value.get("summary") == summarize_proposal_texts(value["cases"]), f"{label} summary mismatch")
    expected_comparisons = {
        label: {
            "pass_delta": value["summary"]["pass"] - baseline["summary"]["pass"],
            "unexpected_call_delta": value["summary"]["unexpected_calls"] - baseline["summary"]["unexpected_calls"],
        }
        for label, value in candidates.items()
    }
    _require(report.get("comparisons") == expected_comparisons, "Development comparisons mismatch")
    _require(report.get("provenance", {}).get("final_holdout_inference") is False, "Development used final holdout")
    return {"run_id": expected_run_id, "baseline": baseline["summary"], "candidates": {label: value["summary"] for label, value in candidates.items()}, "comparisons": expected_comparisons}


def validate_development_review(
    review: dict[str, Any],
    development_report: dict[str, Any],
    train_report: dict[str, Any],
    *,
    development_report_sha256: str,
    train_report_sha256: str,
) -> dict[str, Any]:
    """Bind the manual non-call decisions and checkpoint selection to raw output."""

    _require(review.get("schema_version") == 1, "Unsupported candidate development review schema")
    _require(review.get("review_id") == "nexus-sft-candidate-development-review-v3", "Unexpected candidate development review ID")
    source = review.get("source", {})
    _require(source.get("train_run_id") == train_report.get("run_id"), "Review train run mismatch")
    _require(source.get("development_run_id") == development_report.get("run_id"), "Review development run mismatch")
    _require(source.get("train_report_sha256") == train_report_sha256, "Review train report hash mismatch")
    _require(source.get("development_report_sha256") == development_report_sha256, "Review development report hash mismatch")
    _require(source.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256, "Review manifest hash mismatch")
    expected_adapter_hashes = {
        label: value["state_sha256_reloaded"]
        for label, value in train_report.get("adapters", {}).items()
    }
    _require(source.get("adapter_state_sha256") == expected_adapter_hashes, "Review adapter hashes mismatch")
    expected_automatic = {
        "baseline": development_report["baseline"]["summary"],
        "candidates": {
            label: value["summary"]
            for label, value in development_report["candidates"].items()
        },
        "comparisons": development_report["comparisons"],
    }
    _require(review.get("automatic_summary") == expected_automatic, "Review automatic summary mismatch")

    sets = {
        "baseline": development_report["baseline"],
        **development_report["candidates"],
    }
    expected_ids = [
        case["id"]
        for case in sets["baseline"]["cases"]
        if case["behavior"] != "call"
    ]
    _require(len(expected_ids) == 24, "Development review must cover 24 non-call cases")
    semantic = {}
    for label in ("baseline", "step-150", "step-300"):
        raw_by_id = {case["id"]: case for case in sets[label]["cases"]}
        rows = review.get("reviews", {}).get(label)
        _require(isinstance(rows, list), f"Missing {label} manual reviews")
        _require([row.get("id") for row in rows] == expected_ids, f"{label} review coverage changed")
        for row in rows:
            _require(row.get("decision") in {"pass", "fail"}, f"Invalid {label} review decision")
            _require(isinstance(row.get("reason"), str) and row["reason"].strip(), f"Missing {label} review reason")
            raw = raw_by_id[row["id"]]
            _require(row.get("generated_text") == raw["generated_text"], f"{label} review text mismatch: {row['id']}")
            _require(row.get("automatic_status") == raw["automatic_status"], f"{label} automatic status mismatch: {row['id']}")
        semantic[label] = {
            "pass": sum(row["decision"] == "pass" for row in rows),
            "fail": sum(row["decision"] == "fail" for row in rows),
            "total": len(rows),
        }
    _require(review.get("semantic_noncall_summary") == semantic, "Semantic non-call summary mismatch")

    baseline_by_id = {case["id"]: case for case in sets["baseline"]["cases"]}
    final_by_id = {case["id"]: case for case in sets["step-300"]["cases"]}
    expected_changes = {
        "fixed_automatic_case_ids": [
            case_id
            for case_id in baseline_by_id
            if baseline_by_id[case_id]["automatic_status"] == "fail"
            and final_by_id[case_id]["automatic_status"] == "pass"
        ],
        "regressed_automatic_case_ids": [
            case_id
            for case_id in baseline_by_id
            if baseline_by_id[case_id]["automatic_status"] == "pass"
            and final_by_id[case_id]["automatic_status"] == "fail"
        ],
    }
    _require(review.get("step_300_status_changes") == expected_changes, "Step-300 status changes mismatch")
    decision = review.get("decision", {})
    selected = decision.get("selected_checkpoint")
    _require(selected in {"step-150", "step-300"}, "Unexpected checkpoint selection")
    _require(decision.get("rollout_eligible") is False, "Development review cannot authorize rollout")
    _require(decision.get("open_final_holdout") is False, "Review must keep final holdout closed")
    eligible = [
        label
        for label in ("step-150", "step-300")
        if sets[label]["summary"]["unexpected_calls"]
        <= sets["baseline"]["summary"]["unexpected_calls"]
        and semantic[label]["pass"] >= semantic["baseline"]["pass"]
    ]
    _require(bool(eligible), "No checkpoint passes the safety and semantic gates")
    expected_selected = max(
        eligible,
        key=lambda label: (
            sets[label]["summary"]["pass"],
            label == "step-150",
        ),
    )
    _require(selected == expected_selected, "Checkpoint selection rule mismatch")
    return {
        "review_id": review["review_id"],
        "semantic_noncall_summary": semantic,
        "selected_checkpoint": selected,
        "step_300_status_changes": expected_changes,
        "rollout_eligible": False,
        "final_holdout_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--train-report", type=Path)
    parser.add_argument("--train-run-id")
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--development-run-id")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    try:
        summary = validate_protocol(manifest_path=args.manifest.resolve())
        manifest = load_json(args.manifest.resolve())
        if args.train_report:
            _require(bool(args.train_run_id), "--train-report requires --train-run-id")
            summary["train"] = validate_train_report(load_json(args.train_report.resolve()), manifest, expected_run_id=args.train_run_id)
        if args.development_report:
            _require(bool(args.development_run_id), "--development-report requires --development-run-id")
            _require(bool(args.train_run_id), "--development-report requires --train-run-id")
            summary["development_run"] = validate_development_report(load_json(args.development_report.resolve()), manifest, expected_run_id=args.development_run_id, expected_train_run_id=args.train_run_id)
        if args.review:
            _require(args.train_report is not None, "--review requires --train-report")
            _require(args.development_report is not None, "--review requires --development-report")
            train_path = args.train_report.resolve()
            development_path = args.development_report.resolve()
            summary["manual_review"] = validate_development_review(
                load_json(args.review.resolve()),
                load_json(development_path),
                load_json(train_path),
                development_report_sha256=sha256(development_path),
                train_report_sha256=sha256(train_path),
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline preflight only: no Modal function, GPU, model, tool, database or final holdout was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
