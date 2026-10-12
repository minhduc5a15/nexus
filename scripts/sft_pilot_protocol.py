"""Validate the locked SFT pilot protocol and its Modal artifacts offline."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.audit_sft_pilot import audit_dataset, verify_review
from scripts.eval_proposals import load_cases, strict_json
from scripts.score_sft_generation import summarize_proposal_texts


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals/sft_pilot_training_manifest_v1.json"
RUNNER = ROOT / "modal_sft_pilot.py"
DEVELOPMENT_REVIEW = ROOT / "evals/sft_pilot_development_review_v1.json"
EXPECTED_MANIFEST_SHA256 = "332090d1b02ea8021a1033fc8f83ece9f573c5a9ad5b6ed3432c0758bda14629"
EXPECTED_MODEL = {
    "id": "Qwen/Qwen3-4B-Instruct-2507",
    "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
}


class PilotProtocolError(ValueError):
    """The pilot protocol or one of its artifacts violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotProtocolError(message)


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
    raise PilotProtocolError(f"Runner assignment missing: {name}")


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    try:
        return ast.literal_eval(_assignment(tree, name))
    except (ValueError, TypeError) as error:
        raise PilotProtocolError(f"Runner assignment is not literal: {name}") from error


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise PilotProtocolError(f"Runner function missing: {name}")


def _modal_keywords(function: ast.FunctionDef) -> dict[str, Any]:
    for decorator in function.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "function"
        ):
            result = {}
            for keyword in decorator.keywords:
                if keyword.arg in {"gpu", "timeout"}:
                    result[keyword.arg] = ast.literal_eval(keyword.value)
            return result
    raise PilotProtocolError(f"Modal decorator missing: {function.name}")


def _image_local_paths(tree: ast.Module, assignment_name: str) -> list[str]:
    node = _assignment(tree, assignment_name)
    paths = []
    for call in ast.walk(node):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in {"add_local_file", "add_local_dir"}
            and call.args
        ):
            try:
                paths.append(ast.literal_eval(call.args[0]))
            except (ValueError, TypeError) as error:
                raise PilotProtocolError(
                    f"Non-literal local path in {assignment_name}"
                ) from error
    return paths


def validate_protocol(
    root: Path = ROOT,
    manifest_path: Path = MANIFEST,
    runner_path: Path = RUNNER,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    runner_path = runner_path.resolve()
    _require(sha256(manifest_path) == EXPECTED_MANIFEST_SHA256, "Pilot manifest hash changed")
    manifest = load_json(manifest_path)
    _require(manifest.get("schema_version") == 1, "Unsupported pilot protocol schema")
    _require(manifest.get("protocol_id") == "nexus-sft-qlora-pilot-v1", "Unexpected pilot protocol ID")
    _require(manifest.get("model") == EXPECTED_MODEL, "Pilot base model lock changed")
    _require(manifest.get("training_authorized") is True, "Pilot training is not authorized")

    dataset = manifest.get("dataset", {})
    dataset_path = root / dataset.get("path", "")
    review_path = root / dataset.get("review_path", "")
    _require(dataset_path == root / "evals/sft_pilot_v1.json", "Unexpected pilot dataset path")
    _require(dataset.get("sha256") == sha256(dataset_path), "Pilot dataset hash changed")
    _require(dataset.get("count") == 100, "Pilot dataset count changed")
    _require(review_path == root / "evals/sft_pilot_review_v1.json", "Unexpected pilot review path")
    _require(dataset.get("review_sha256") == sha256(review_path), "Pilot review hash changed")
    audit = audit_dataset(dataset_path)
    review = verify_review(review_path)
    _require(audit["cases"] == 100 and len(review["reviewed_case_ids"]) == 100, "Pilot review is incomplete")

    development = manifest.get("development", {})
    development_path = root / development.get("path", "")
    _require(development_path == root / "evals/model_proposal_development_v1.json", "Unexpected development path")
    _require(development.get("sha256") == sha256(development_path), "Development dataset hash changed")
    _, development_cases = load_cases(development_path)
    _require(len(development_cases) == development.get("count") == 80, "Development count changed")

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
        "Pilot training contract changed",
    )
    training = manifest.get("training", {})
    _require(
        training.get("maximum_steps") == 200
        and training.get("seed") == 42
        and training.get("learning_rate") == 0.0001
        and training.get("automatic_retry") is False,
        "Pilot training settings changed",
    )
    execution = manifest.get("execution", {})
    _require(execution.get("stages") == ["preflight", "train", "development"], "Unexpected stage order")
    train_timeout = execution.get("train_timeout_seconds")
    development_timeout = execution.get("development_timeout_seconds")
    maximum_seconds = execution.get("maximum_planned_gpu_seconds")
    _require(
        all(type(value) is int and value > 0 for value in (train_timeout, development_timeout, maximum_seconds)),
        "Invalid GPU timeout budget",
    )
    _require(train_timeout + development_timeout <= maximum_seconds, "Function timeouts exceed GPU budget")
    expected_cost = maximum_seconds * execution.get("t4_price_usd_per_second", -1)
    _require(
        math.isclose(expected_cost, execution.get("maximum_planned_gpu_cost_usd", -1), abs_tol=1e-9),
        "Maximum GPU cost arithmetic changed",
    )
    scope = manifest.get("scope", {})
    _require(
        scope
        == {
            "tools_or_database_executed": False,
            "policy_used_for_gold": False,
            "final_holdout_packaged": False,
            "final_holdout_inference": False,
            "gguf_export": False,
            "deployment_rollout": False,
        },
        "Pilot scope changed",
    )

    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    constants = {
        name: _literal_assignment(tree, name)
        for name in (
            "PROTOCOL_ID",
            "MANIFEST_SHA256",
            "MODEL_ID",
            "MODEL_REVISION",
            "PILOT_SHA256",
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
    _require(constants["PILOT_SHA256"] == dataset["sha256"], "Runner pilot lock changed")
    _require(constants["DEVELOPMENT_SHA256"] == development["sha256"], "Runner development lock changed")
    _require(constants["MAX_SEQUENCE_TOKENS"] == contract["maximum_sequence_tokens"], "Runner sequence limit changed")
    _require(constants["MAX_STEPS"] == training["maximum_steps"], "Runner step count changed")
    _require(constants["MAX_NEW_TOKENS"] == development["generation"]["maximum_new_tokens"], "Runner generation limit changed")
    _require(
        _modal_keywords(_function(tree, "run_train"))
        == {"gpu": "T4", "timeout": train_timeout},
        "Train Modal resources changed",
    )
    _require(
        _modal_keywords(_function(tree, "run_development"))
        == {"gpu": "T4", "timeout": development_timeout},
        "Development Modal resources changed",
    )
    training_paths = _image_local_paths(tree, "training_image")
    development_paths = _image_local_paths(tree, "development_image")
    _require(not any("development" in path.casefold() for path in training_paths), "Training image contains development data")
    _require(not any("holdout" in path.casefold() for path in training_paths + development_paths), "Runner image contains final holdout data")
    _require(
        "evals/model_proposal_development_v1.json" in development_paths,
        "Development image is missing its locked dataset",
    )
    main = _function(tree, "main")
    defaults = [ast.literal_eval(value) for value in main.args.defaults]
    _require(defaults and defaults[0] == "preflight", "Runner default stage must be preflight")
    return {
        "protocol_id": manifest["protocol_id"],
        "manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256(manifest_path)},
        "runner": {"path": str(runner_path.relative_to(root)), "sha256": sha256(runner_path), "default_stage": "preflight"},
        "dataset": {"path": dataset["path"], "sha256": dataset["sha256"], "count": dataset["count"]},
        "development": {"path": development["path"], "sha256": development["sha256"], "count": development["count"]},
        "model": manifest["model"],
        "training_steps": training["maximum_steps"],
        "gpu": manifest["runtime"]["gpu"],
        "maximum_planned_gpu_seconds": maximum_seconds,
        "maximum_planned_gpu_cost_usd": execution["maximum_planned_gpu_cost_usd"],
        "final_holdout_inference": False,
    }


def validate_train_report(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_run_id: str,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    _require(report.get("protocol_id") == manifest["protocol_id"], "Train report protocol mismatch")
    _require(report.get("stage") == "train" and report.get("run_id") == expected_run_id, "Unexpected train report identity")
    _require(report.get("model") == manifest["model"], "Train report model mismatch")
    _require(report.get("dataset", {}).get("sha256") == manifest["dataset"]["sha256"], "Train report dataset mismatch")
    _require(report.get("state", {}).get("status") == "complete", "Train report is incomplete")
    gates = report.get("gates", {})
    _require(gates and all(value is True for value in gates.values()), "Train gates did not all pass")
    training = report.get("training", {})
    _require(training.get("completed_steps") == manifest["training"]["maximum_steps"], "Train step count mismatch")
    _require(isinstance(training.get("training_loss"), int | float) and math.isfinite(training["training_loss"]), "Train loss is not finite")
    adapter = report.get("adapter", {})
    _require(adapter.get("state_sha256_after") == adapter.get("state_sha256_reloaded"), "Adapter reload state mismatch")
    if adapter_path is not None:
        _require(adapter_path.is_dir(), "Adapter directory is missing")
        _require(directory_sha256(adapter_path) == adapter.get("directory_sha256"), "Adapter directory hash mismatch")
    provenance = report.get("provenance", {})
    _require(provenance.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256, "Train provenance manifest mismatch")
    _require(provenance.get("final_holdout_inference") is False, "Train report used final holdout")
    return {
        "run_id": expected_run_id,
        "report_sha256": None,
        "adapter_directory_sha256": adapter.get("directory_sha256"),
        "adapter_state_sha256": adapter.get("state_sha256_after"),
        "completed_steps": training["completed_steps"],
    }


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
    _require(report.get("dataset", {}).get("sha256") == manifest["development"]["sha256"], "Development dataset mismatch")
    _require(report.get("state", {}).get("status") == "complete", "Development report is incomplete")
    gates = report.get("gates", {})
    _require(gates and all(value is True for value in gates.values()), "Development gates did not all pass")
    for label in ("baseline", "candidate"):
        cases = report.get(label, {}).get("cases", [])
        _require(len(cases) == manifest["development"]["count"], f"{label} development cases incomplete")
        _require(report[label].get("summary") == summarize_proposal_texts(cases), f"{label} summary mismatch")
    expected_comparison = {
        "pass_delta": report["candidate"]["summary"]["pass"] - report["baseline"]["summary"]["pass"],
        "unexpected_call_delta": report["candidate"]["summary"]["unexpected_calls"] - report["baseline"]["summary"]["unexpected_calls"],
    }
    _require(report.get("comparison") == expected_comparison, "Development comparison mismatch")
    _require(report.get("provenance", {}).get("final_holdout_inference") is False, "Development used final holdout")
    return {
        "run_id": expected_run_id,
        "train_run_id": expected_train_run_id,
        "baseline": report["baseline"]["summary"],
        "candidate": report["candidate"]["summary"],
        "comparison": expected_comparison,
        "reply_review_required": report["candidate"]["summary"]["reply_review_pending"],
    }


def validate_development_review(
    review: dict[str, Any],
    report: dict[str, Any],
    *,
    report_sha256: str,
    train_report_sha256: str,
    failed_state_sha256: str,
) -> dict[str, Any]:
    """Bind every manual non-call decision to immutable raw output."""

    _require(review.get("schema_version") == 1, "Unsupported development review schema")
    _require(review.get("review_id") == "nexus-sft-pilot-development-review-v1", "Unexpected development review ID")
    source = review.get("source", {})
    _require(source.get("development_run_id") == report.get("run_id"), "Review run ID mismatch")
    _require(source.get("development_report_sha256") == report_sha256, "Review report hash mismatch")
    _require(source.get("train_report_sha256") == train_report_sha256, "Review train hash mismatch")
    _require(review.get("failed_run", {}).get("state_sha256") == failed_state_sha256, "Failed-run evidence hash mismatch")
    _require(review.get("automatic_summary") == {
        "baseline": report["baseline"]["summary"],
        "candidate": report["candidate"]["summary"],
        "comparison": report["comparison"],
    }, "Review automatic summary mismatch")
    expected_ids = [
        case["id"]
        for case in report["candidate"]["cases"]
        if case["behavior"] != "call"
    ]
    semantic = {}
    for label in ("baseline", "candidate"):
        raw_by_id = {case["id"]: case for case in report[label]["cases"]}
        rows = review.get(f"{label}_reviews")
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
    _require(review.get("semantic_noncall_summary") == semantic, "Semantic review summary mismatch")
    decision = review.get("decision", {})
    _require(decision.get("rollout_eligible") is False, "Unsafe pilot rollout decision")
    _require(decision.get("open_final_holdout") is False, "Review must keep final holdout closed")
    return {
        "review_id": review["review_id"],
        "source_report_sha256": report_sha256,
        "semantic_noncall_summary": semantic,
        "rollout_eligible": False,
        "final_holdout_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--train-report", type=Path)
    parser.add_argument("--train-run-id")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--development-run-id")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--failed-development-state", type=Path)
    args = parser.parse_args()
    try:
        summary = validate_protocol(manifest_path=args.manifest.resolve())
        manifest = load_json(args.manifest.resolve())
        if args.train_report:
            _require(bool(args.train_run_id), "--train-report requires --train-run-id")
            train_path = args.train_report.resolve()
            train = load_json(train_path)
            summary["train"] = validate_train_report(
                train,
                manifest,
                expected_run_id=args.train_run_id,
                adapter_path=args.adapter_dir.resolve() if args.adapter_dir else None,
            )
            summary["train"]["report_sha256"] = sha256(train_path)
        if args.development_report:
            _require(bool(args.development_run_id), "--development-report requires --development-run-id")
            _require(bool(args.train_run_id), "--development-report requires --train-run-id")
            development_path = args.development_report.resolve()
            development = load_json(development_path)
            summary["development_run"] = validate_development_report(
                development,
                manifest,
                expected_run_id=args.development_run_id,
                expected_train_run_id=args.train_run_id,
            )
            summary["development_run"]["report_sha256"] = sha256(development_path)
        if args.review:
            _require(args.train_report is not None, "--review requires --train-report")
            _require(args.development_report is not None, "--review requires --development-report")
            _require(
                args.failed_development_state is not None,
                "--review requires --failed-development-state",
            )
            failed_path = args.failed_development_state.resolve()
            summary["manual_review"] = validate_development_review(
                load_json(args.review.resolve()),
                load_json(args.development_report.resolve()),
                report_sha256=sha256(args.development_report.resolve()),
                train_report_sha256=sha256(args.train_report.resolve()),
                failed_state_sha256=sha256(failed_path),
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline preflight only: no Modal function, model, GPU, tool, database or final holdout was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
