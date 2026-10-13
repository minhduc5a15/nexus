"""Audit the failure taxonomy and frozen quotas for the next SFT dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / "evals/sft_dataset_v3_design.json"


class DatasetV3DesignError(ValueError):
    """The v3 design no longer matches its evidence or declared quotas."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetV3DesignError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(spec: dict[str, Any]) -> Path:
    path = ROOT / spec.get("path", "")
    _require(path.resolve().is_relative_to(ROOT.resolve()), "Source path escapes repository")
    return path


def _case_map(path: Path) -> dict[str, dict[str, Any]]:
    cases = _load(path).get("cases")
    _require(isinstance(cases, list), f"Missing cases: {path}")
    result = {case.get("id"): case for case in cases}
    _require(None not in result and len(result) == len(cases), f"Duplicate or missing case ID: {path}")
    return result


def _review_failures(path: Path, label: str) -> set[str]:
    rows = _load(path).get("reviews", {}).get(label)
    _require(isinstance(rows, list), f"Missing {label} review: {path}")
    return {row["id"] for row in rows if row.get("decision") == "fail"}


def _raw_failures(path: Path, split: str) -> set[str]:
    report = _load(path)
    if split == "development":
        rows = report.get("candidates", {}).get("step-200", {}).get("cases")
    else:
        rows = report.get("results", {}).get("step-200", {}).get("cases")
    _require(isinstance(rows, list), f"Missing step-200 cases: {path}")
    return {row["id"] for row in rows if row.get("automatic_status") == "fail"}


def audit_design(path: Path = DEFAULT_DESIGN) -> dict[str, Any]:
    design = _load(path)
    _require(design.get("schema_version") == 1, "Unsupported design schema")
    _require(design.get("design_id") == "nexus-sft-dataset-v3-design-v1", "Unexpected design ID")

    sources = design.get("sources")
    _require(isinstance(sources, dict), "Missing evidence sources")
    required_sources = {
        "candidate_v2",
        "development",
        "development_review",
        "development_raw_report",
        "final_holdout_v1",
        "final_holdout_v1_review",
        "final_holdout_v1_raw_report",
    }
    _require(set(sources) == required_sources, "Evidence source set changed")
    for name, spec in sources.items():
        source_path = _source_path(spec)
        if spec.get("tracked", True) or source_path.exists():
            _require(source_path.exists(), f"Missing source: {name}")
            _require(spec.get("sha256") == _sha256(source_path), f"Source hash mismatch: {name}")

    cases_by_split = {
        "development": _case_map(_source_path(sources["development"])),
        "final_holdout_v1": _case_map(_source_path(sources["final_holdout_v1"])),
    }
    semantic_failures = {
        "development": _review_failures(_source_path(sources["development_review"]), "step-200"),
        "final_holdout_v1": _review_failures(_source_path(sources["final_holdout_v1_review"]), "step-200"),
    }
    automatic_failures: dict[str, set[str]] = {}
    for split, source_name in (
        ("development", "development_raw_report"),
        ("final_holdout_v1", "final_holdout_v1_raw_report"),
    ):
        raw_path = _source_path(sources[source_name])
        if raw_path.exists():
            automatic_failures[split] = _raw_failures(raw_path, split)

    taxonomy = design.get("taxonomy")
    _require(isinstance(taxonomy, list) and taxonomy, "Missing taxonomy")
    category_ids = [row.get("id") for row in taxonomy]
    _require(None not in category_ids and len(category_ids) == len(set(category_ids)), "Duplicate taxonomy category")
    category_set = set(category_ids)

    evidence = design.get("evidence")
    _require(isinstance(evidence, list) and evidence, "Missing failure evidence")
    evidence_keys: set[tuple[str, str]] = set()
    observed_categories: Counter[str] = Counter()
    automatic_count = 0
    semantic_count = 0
    for row in evidence:
        split = row.get("split")
        case_id = row.get("case_id")
        key = (split, case_id)
        _require(split in cases_by_split and case_id in cases_by_split[split], f"Unknown evidence case: {key}")
        _require(key not in evidence_keys, f"Duplicate evidence case: {key}")
        evidence_keys.add(key)
        categories = row.get("categories")
        _require(isinstance(categories, list) and categories, f"Missing categories: {key}")
        _require(set(categories) <= category_set, f"Unknown category: {key}")
        observed_categories.update(categories)
        signals = row.get("signals")
        _require(isinstance(signals, list) and signals, f"Missing signals: {key}")
        if any(signal.startswith("automatic_") for signal in signals):
            automatic_count += 1
            if split in automatic_failures:
                _require(case_id in automatic_failures[split], f"Automatic evidence is not a failure: {key}")
        if "semantic_reply_fail" in signals:
            semantic_count += 1
            _require(case_id in semantic_failures[split], f"Semantic evidence is not a reviewed failure: {key}")

    for row in taxonomy:
        _require(row.get("evidence_cases") == observed_categories[row["id"]], f"Evidence count mismatch: {row['id']}")
        _require(isinstance(row.get("v3_new_examples"), int) and row["v3_new_examples"] > 0, f"Invalid v3 allocation: {row['id']}")
    _require(sum(row["v3_new_examples"] for row in taxonomy) == 200, "Taxonomy allocation must total 200")

    dataset = design.get("dataset_v3", {})
    _require(dataset.get("preserved_candidate_v2_cases") == 400, "v2 prefix count changed")
    _require(dataset.get("new_cases") == 200 and dataset.get("total_cases") == 600, "v3 total changed")
    behaviors = dataset.get("behavior_balance", {})
    actions = dataset.get("action_balance", {})
    _require(sum(behaviors.values()) == 600, "v3 behavior balance does not total 600")
    _require(behaviors.get("call") == sum(actions.values()) == 420, "v3 call/action balance mismatch")
    _require(set(actions) == {"create", "list", "complete", "edit", "delete", "deadline", "due_query"}, "v3 action set changed")
    _require(dataset.get("training_authorized") is False, "Design must not authorize training")

    holdout = design.get("final_holdout_v2", {})
    _require(holdout.get("required") is True, "A new final holdout is required")
    _require(holdout.get("total_cases") == 120, "Final holdout v2 size changed")
    _require(holdout.get("action_cases") == sum(holdout.get("action_balance", {}).values()) == 84, "Final holdout action balance mismatch")
    _require(holdout.get("control_cases") == 36, "Final holdout control count changed")
    _require(sum(holdout.get("behavior_balance", {}).values()) == 120, "Final holdout behavior balance mismatch")
    _require(holdout.get("model_inference_authorized") is False, "Final holdout inference must remain closed")

    return {
        "design_id": design["design_id"],
        "design_sha256": _sha256(path),
        "evidence_cases": len(evidence),
        "automatic_failure_cases": automatic_count,
        "semantic_failure_cases": semantic_count,
        "taxonomy_categories": len(taxonomy),
        "v3_cases": dataset["total_cases"],
        "v3_new_cases": dataset["new_cases"],
        "final_holdout_v2_cases": holdout["total_cases"],
        "training_authorized": False,
        "final_holdout_v2_inference_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    args = parser.parse_args()
    try:
        summary = audit_design(args.design.resolve())
    except (OSError, ValueError, KeyError, TypeError, DatasetV3DesignError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline design audit only: no model, GPU, tool, database or holdout inference was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
