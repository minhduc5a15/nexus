"""Validate downloaded v4 train/development artifacts without model access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.audit_sft_candidate_v4_execution import PLAN, validate_protocol
from scripts.eval_proposals import load_cases, strict_json
from scripts.score_sft_generation import score_proposal_text, summarize_proposal_texts


ROOT = Path(__file__).resolve().parents[1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(path: Path) -> str:
    _require(path.is_dir(), f"Missing adapter directory: {path}")
    digest = hashlib.sha256()
    for item in sorted(file for file in path.rglob("*") if file.is_file()):
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def validate_train_report(report_path: Path, *, check_adapter_files: bool = True) -> dict:
    plan = strict_json(PLAN.read_bytes())
    report = strict_json(report_path.read_bytes())
    _require(report.get("protocol_id") == "nexus-sft-candidate-v4" and report.get("stage") == "train", "Wrong train report")
    _require(report.get("run_id") == plan["run_ids"]["train"], "Train run ID changed")
    _require(report.get("plan_sha256") == sha256(PLAN), "Train plan hash changed")
    _require(report.get("dataset_sha256") == plan["dataset"]["sha256"], "Train dataset hash changed")
    _require(report.get("model") == {"id": plan["model"]["id"], "revision": plan["model"]["revision"]}, "Train base model changed")
    _require(report.get("state", {}).get("status") == "complete" and report.get("state", {}).get("stage") == "train", "Train is incomplete")
    training = report.get("training", {})
    _require(training.get("completed_steps") == 320, "Train optimizer step count changed")
    _require(type(training.get("loss")) in (float, int) and math.isfinite(training["loss"]), "Invalid train loss")
    history = training.get("history")
    _require(isinstance(history, list) and history and history[-1].get("step") == 320, "Training history is incomplete")
    adapter = report.get("adapter", {})
    before, after = adapter.get("state_sha256_before"), adapter.get("state_sha256_after")
    _require(all(isinstance(item, str) and len(item) == 64 for item in (before, after)), "Adapter state hash missing")
    _require(before != after and adapter.get("state_sha256_reloaded") == after, "Adapter update/reload gate failed")
    resume = plan["resume"]
    _require(report.get("resume") == {
        "source_run_id": resume["source_run_id"],
        "checkpoint_step": resume["checkpoint_step"],
        "trainer_state_sha256": resume["trainer_state_sha256"],
    }, "Train resume provenance changed")
    _require(report.get("final_holdout_inference") is False, "Train report claims holdout inference")
    if check_adapter_files:
        actual_adapter = report_path.parent / "adapter-step-320"
        _require(directory_sha256(actual_adapter) == adapter.get("directory_sha256"), "Adapter directory hash changed")
    return {
        "run_id": report["run_id"], "report_sha256": sha256(report_path),
        "completed_steps": 320, "training_loss": training["loss"],
        "adapter_state_sha256": after,
    }


def validate_development_report(report_path: Path, train_report_path: Path) -> dict:
    plan = strict_json(PLAN.read_bytes())
    train = validate_train_report(train_report_path)
    report = strict_json(report_path.read_bytes())
    _require(report.get("protocol_id") == "nexus-sft-candidate-v4" and report.get("stage") == "development", "Wrong development report")
    _require(report.get("run_id") == plan["run_ids"]["development"], "Development run ID changed")
    _require(report.get("plan_sha256") == sha256(PLAN), "Development plan hash changed")
    _require(report.get("dataset_sha256") == plan["development"]["sha256"], "Development dataset hash changed")
    _require(report.get("candidate_train_report_sha256") == train["report_sha256"], "Candidate train report changed")
    _require(report.get("source_v3_train_report_sha256") == plan["source_v3"]["train_report_sha256"], "V3 source report changed")
    _require(report.get("generation") == plan["generation"], "Generation settings changed")
    _require(report.get("state", {}).get("status") == "complete" and report.get("state", {}).get("stage") == "development", "Development is incomplete")
    _require(report.get("no_tool_execution") is True and report.get("final_holdout_inference") is False, "Development scope changed")
    _, cases = load_cases(ROOT / plan["development"]["path"])
    expected_ids = [case["id"] for case in cases]
    models = report.get("models", {})
    _require(set(models) == {"base", "v3-step-300", "v4-step-320"}, "Development model set changed")
    summaries = {}
    for name, value in models.items():
        rows = value.get("cases")
        _require(isinstance(rows, list) and [row.get("id") for row in rows] == expected_ids, f"{name} case order changed")
        for case, row in zip(cases, rows):
            expected = score_proposal_text(case, row.get("generated_text"))
            for field in ("id", "prompt", "parsed", "accepted_calls", "automatic_status", "error_tags", "reply_review"):
                _require(row.get(field) == expected[field], f"{name} scoring changed at {case['id']}: {field}")
            _require(type(row.get("prompt_tokens")) is int and row["prompt_tokens"] > 0, f"{name} prompt token count invalid")
            _require(type(row.get("generated_tokens")) is int and row["generated_tokens"] >= 0, f"{name} generation token count invalid")
            _require(type(row.get("generation_seconds")) in (float, int) and row["generation_seconds"] >= 0, f"{name} latency invalid")
        summary = summarize_proposal_texts(rows)
        _require(value.get("summary") == summary, f"{name} summary changed")
        summaries[name] = summary
    return {
        "run_id": report["run_id"], "report_sha256": sha256(report_path),
        "scores": {name: value["pass"] for name, value in summaries.items()},
        "reply_review_pending": {name: value["reply_review_pending"] for name, value in summaries.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "development"))
    parser.add_argument("report", type=Path)
    parser.add_argument("--train-report", type=Path)
    args = parser.parse_args()
    validate_protocol()
    if args.stage == "development":
        if args.train_report is None:
            parser.error("--train-report is required for development")
        result = validate_development_report(args.report, args.train_report)
    else:
        result = validate_train_report(args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
