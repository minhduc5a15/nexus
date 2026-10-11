"""Frozen protocol and report checks for SFT free-generation sanity."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.score_sft_generation import summarize
from scripts.sft_sanity_protocol import file_sha256, load_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evals/sft_generation_manifest_v1.json"
DEFAULT_REVIEW = ROOT / "evals/sft_generation_review_v1.json"
RUNNER = ROOT / "modal_sft_sanity.py"


class GenerationProtocolError(ValueError):
    """The generation protocol or report is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationProtocolError(message)


def _repo_file(root: Path, value: object, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} path is required")
    relative = Path(value)
    _require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe {label} path")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()) and path.is_file(), f"Missing {label}")
    return path


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise GenerationProtocolError(f"Runner function not found: {name}")


def _modal_literals(node: ast.FunctionDef) -> dict[str, object]:
    for decorator in node.decorator_list:
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
    raise GenerationProtocolError(f"Modal decorator not found: {node.name}")


def validate_protocol(
    *, root: Path = ROOT, manifest_path: Path | None = None
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = (manifest_path or root / "evals/sft_generation_manifest_v1.json").resolve()
    manifest = load_json(manifest_path)
    _require(manifest.get("schema_version") == 1, "Unsupported generation protocol schema")
    _require(manifest.get("protocol_id") == "nexus-sft-generation-sanity-v1", "Unexpected generation protocol ID")
    dataset = manifest.get("dataset", {})
    dataset_path = _repo_file(root, dataset.get("path"), "dataset")
    _require(file_sha256(dataset_path) == dataset.get("sha256"), "Generation dataset hash mismatch")
    _require(len(load_json(dataset_path).get("cases", [])) == dataset.get("count") == 16, "Generation dataset count mismatch")
    training = manifest.get("training_manifest", {})
    training_path = _repo_file(root, training.get("path"), "training manifest")
    _require(file_sha256(training_path) == training.get("sha256"), "Training manifest hash mismatch")
    source = manifest.get("source_overfit", {})
    _require(source.get("run_id") == "sft-sanity-overfit-20261006c", "Unexpected source overfit run")
    _require(source.get("probe_run_id") == "sft-sanity-probe-20261006d", "Unexpected source probe run")
    for field in ("report_sha256", "adapter_directory_sha256", "adapter_state_sha256"):
        value = source.get(field)
        _require(isinstance(value, str) and len(value) == 64, f"Invalid {field}")
    for field in ("report_path", "adapter_path"):
        value = source.get(field)
        _require(isinstance(value, str) and value and not Path(value).is_absolute() and ".." not in Path(value).parts, f"Unsafe {field}")
    generation = manifest.get("generation")
    _require(generation == {
        "gpu": "T4",
        "seed": 42,
        "do_sample": False,
        "maximum_new_tokens": 128,
        "batch_size": 1,
        "automatic_retry": False,
        "timeout_seconds": 1200,
    }, "Generation settings changed")
    scope = manifest.get("scope", {})
    _require(scope.get("training_records_only") is True, "Generation must stay on training records")
    _require(all(value is False for key, value in scope.items() if key != "training_records_only"), "Generation scope includes evaluation, tools, DB or rollout")

    runner = root / "modal_sft_sanity.py"
    _require(runner.is_file(), "Missing Modal SFT runner")
    source_text = runner.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(runner))
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    _require(assignments.get("GENERATION_PROTOCOL_ID") == manifest["protocol_id"], "Runner generation protocol ID differs")
    _require(assignments.get("GENERATION_MANIFEST_SHA256") == file_sha256(manifest_path), "Runner generation manifest hash differs")
    _require(assignments.get("GENERATION_MAX_NEW_TOKENS") == generation["maximum_new_tokens"], "Runner generation token limit differs")
    _require(
        _modal_literals(_function(tree, "run_generation"))
        == {"gpu": generation["gpu"], "timeout": generation["timeout_seconds"]},
        "Generation decorator differs from manifest",
    )
    function_source = ast.get_source_segment(source_text, _function(tree, "run_generation")) or ""
    for required in ("validate_overfit_report", "adapter_directory_sha256", "adapter_state_sha256"):
        _require(required in function_source, f"Generation runner is missing {required}")
    _require("do_sample=False" in source_text, "Generation runner must disable sampling")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"add_local_file", "add_local_dir"} or not node.args:
            continue
        try:
            local_path = str(ast.literal_eval(node.args[0])).casefold()
        except (TypeError, ValueError):
            continue
        _require("holdout" not in local_path and "development" not in local_path, "Runner image contains evaluation data")
    return {
        "protocol_id": manifest["protocol_id"],
        "manifest": {"path": str(manifest_path.relative_to(root)), "sha256": file_sha256(manifest_path)},
        "runner": {"path": str(runner.relative_to(root)), "sha256": file_sha256(runner)},
        "dataset": {"sha256": dataset["sha256"], "count": dataset["count"]},
        "source_overfit": source,
        "generation": generation,
        "final_holdout_inference": False,
    }


def validate_report(
    report: dict[str, Any], manifest: dict[str, Any], *, expected_run_id: str
) -> dict[str, Any]:
    _require(report.get("protocol_id") == manifest["protocol_id"], "Generation report protocol mismatch")
    _require(report.get("run_id") == expected_run_id, "Generation run ID mismatch")
    _require(report.get("stage") == "generation", "Expected generation report")
    _require(report.get("dataset_sha256") == manifest["dataset"]["sha256"], "Generation report dataset mismatch")
    _require(report.get("generation_manifest_sha256") == file_sha256(DEFAULT_MANIFEST), "Generation report manifest mismatch")
    _require(report.get("generation") == manifest["generation"], "Generation report settings mismatch")
    _require(report.get("state", {}).get("status") == "complete", "Generation report is incomplete")
    _require(report.get("provenance", {}).get("final_holdout_inference") is False, "Generation used final holdout")
    adapter = report.get("adapter", {})
    locked = manifest["source_overfit"]
    _require(adapter.get("directory_sha256") == locked["adapter_directory_sha256"], "Generation adapter directory mismatch")
    _require(adapter.get("state_sha256") == locked["adapter_state_sha256"], "Generation adapter state mismatch")
    gates = report.get("gates", {})
    required_gates = {
        "source_overfit_valid",
        "adapter_directory_hash_matches",
        "adapter_state_hash_matches",
        "baseline_completed",
        "candidate_completed",
    }
    _require(set(gates) == required_gates and all(gates.values()), "Generation infrastructure gate failed")
    dataset = load_json(ROOT / manifest["dataset"]["path"])
    expected_ids = [case["id"] for case in dataset["cases"]]
    summaries = {}
    for name in ("baseline", "candidate"):
        block = report.get(name, {})
        cases = block.get("cases", [])
        _require([case.get("id") for case in cases] == expected_ids, f"{name} case order mismatch")
        recomputed = summarize(cases)
        _require(block.get("summary") == recomputed, f"{name} summary mismatch")
        summaries[name] = recomputed
    comparison = report.get("comparison", {})
    _require(
        comparison == {
            "proposal_correct_delta": summaries["candidate"]["proposal_correct"] - summaries["baseline"]["proposal_correct"],
            "exact_token_sequence_delta": summaries["candidate"]["exact_token_sequences"] - summaries["baseline"]["exact_token_sequences"],
        },
        "Generation comparison mismatch",
    )
    return {
        "run_id": expected_run_id,
        "baseline": summaries["baseline"],
        "candidate": summaries["candidate"],
        "comparison": comparison,
    }


def validate_manual_review(path: Path = DEFAULT_REVIEW) -> dict[str, Any]:
    review = load_json(path)
    _require(review.get("schema_version") == 1, "Unsupported manual review schema")
    _require(review.get("review_id") == "nexus-sft-generation-manual-review-v1", "Unexpected manual review ID")
    source = review.get("source", {})
    _require(source.get("run_id") == "sft-sanity-generation-20261008b", "Unexpected reviewed generation run")
    report_hash = source.get("report_sha256")
    _require(isinstance(report_hash, str) and len(report_hash) == 64, "Invalid reviewed report hash")
    dataset = load_json(ROOT / "evals/sft_sanity_v1.json")
    expected_ids = [case["id"] for case in dataset["cases"] if case["behavior"] != "call"]
    cases = review.get("cases", [])
    _require([case.get("id") for case in cases] == expected_ids, "Manual review case order mismatch")
    recomputed = {}
    for block in ("baseline", "candidate"):
        results = [case.get(block, {}).get("result") for case in cases]
        _require(all(result in {"pass", "fail"} for result in results), f"Invalid {block} manual result")
        recomputed[block] = {
            "pass": results.count("pass"),
            "fail": results.count("fail"),
            "total": len(results),
        }
    _require(review.get("summary") == recomputed, "Manual review summary mismatch")
    return {"review_id": review["review_id"], "source": source, "summary": recomputed}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--skip-review", action="store_true")
    args = parser.parse_args()
    try:
        protocol = validate_protocol()
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        if not args.skip_review:
            print(json.dumps(
                {"manual_review": validate_manual_review()},
                ensure_ascii=False,
                indent=2,
            ))
        if args.report:
            if not args.run_id:
                parser.error("--report requires --run-id")
            report = load_json(args.report.resolve())
            manifest = load_json(DEFAULT_MANIFEST)
            print(json.dumps(
                validate_report(report, manifest, expected_run_id=args.run_id),
                ensure_ascii=False,
                indent=2,
            ))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print("Offline validation only: no model, GPU, tool or database was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
