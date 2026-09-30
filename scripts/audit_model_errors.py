"""Validate the manual attribution of frozen model-proposal development failures."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.eval_proposals import strict_json


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "evals/model_proposal_error_audit_v1.json"
CONFIGURATIONS = ("q4_all", "q4_classified", "q8_all", "q8_classified")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _case_projection(case: dict[str, Any]) -> dict[str, Any]:
    routing = case.get("routing")
    if not isinstance(routing, dict):
        raise ValueError(f"Case {case.get('id')} has no routing trace")
    return {
        "automatic_status": case.get("automatic_status"),
        "proposed_calls": case.get("proposed_calls"),
        "model_reply": case.get("model_reply"),
        "error_tags": case.get("error_tags"),
        "routing": {
            "mode": routing.get("mode"),
            "request_kind": routing.get("request_kind"),
            "tools": routing.get("tools"),
            "fallback": routing.get("fallback"),
        },
    }


def summarize(cases: list[dict[str, Any]], configurations=CONFIGURATIONS) -> dict[str, Any]:
    owners: Counter[str] = Counter()
    causes: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    for case in cases:
        dispositions[case["training_disposition"]] += 1
        labels[case["label_review"]["status"]] += 1
        for attribution in case["failure_attribution"].values():
            owners[attribution["owner"]] += 1
            causes[attribution["primary_cause"]] += 1
    return {
        "unique_failed_cases": len(cases),
        "failure_observations": sum(len(case["failed_configurations"]) for case in cases),
        "configuration_failures": {
            name: sum(case["observations"][name]["automatic_status"] == "fail" for case in cases)
            for name in configurations
        },
        "owner_counts": dict(sorted(owners.items())),
        "primary_cause_counts": dict(sorted(causes.items())),
        "training_disposition_counts": dict(sorted(dispositions.items())),
        "label_review_counts": dict(sorted(labels.items())),
    }


def validate_audit(
    audit_path: Path = AUDIT_PATH,
    reports_root: Path | None = None,
) -> dict[str, Any]:
    audit = _object(audit_path)
    if audit.get("schema_version") != 1:
        raise ValueError("Unsupported error-audit schema version")
    source = audit.get("source_run")
    taxonomy = audit.get("taxonomy")
    cases = audit.get("cases")
    if not isinstance(source, dict) or not isinstance(taxonomy, dict) or not isinstance(cases, list):
        raise ValueError("Audit requires source_run, taxonomy, and cases")
    if source.get("holdout_inference") is not False:
        raise ValueError("Audit must state that holdout inference did not occur")
    if source.get("dataset") != "evals/model_proposal_development_v1.json":
        raise ValueError("Error audit may only use the frozen development dataset")
    dataset_path = ROOT / source["dataset"]
    if sha256(dataset_path) != source.get("dataset_sha256"):
        raise ValueError("Development dataset hash mismatch")

    enum_names = ("owners", "primary_causes", "training_dispositions", "label_statuses")
    enums: dict[str, set[str]] = {}
    for name in enum_names:
        values = taxonomy.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"Taxonomy {name} must contain unique non-empty strings")
        enums[name] = set(values)

    report_specs = source.get("reports")
    if not isinstance(report_specs, dict) or tuple(report_specs) != CONFIGURATIONS:
        raise ValueError("Source reports must use the four frozen configurations in order")
    resolved_root = reports_root or ROOT / source.get("reports_root", "")
    reports: dict[str, dict[str, Any]] = {}
    report_cases: dict[str, dict[str, dict[str, Any]]] = {}
    ordered_ids: list[str] | None = None
    for name in CONFIGURATIONS:
        spec = report_specs[name]
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid source report specification: {name}")
        relative = spec.get("relative_path")
        if not isinstance(relative, str) or "holdout" in relative.casefold():
            raise ValueError("Source report path must be a development report")
        path = resolved_root / relative
        if sha256(path) != spec.get("sha256"):
            raise ValueError(f"Source report hash mismatch: {name}")
        report = _object(path)
        if report.get("prompt_version") != spec.get("prompt_version"):
            raise ValueError(f"Prompt version mismatch: {name}")
        if report.get("tool_routing") != spec.get("routing"):
            raise ValueError(f"Routing mode mismatch: {name}")
        if report.get("model_sha256") != spec.get("model_sha256"):
            raise ValueError(f"Model hash mismatch: {name}")
        if report.get("dataset", {}).get("sha256") != source.get("dataset_sha256"):
            raise ValueError(f"Report used a different dataset: {name}")
        if report.get("summary", {}).get("automatic") != spec.get("automatic"):
            raise ValueError(f"Recorded automatic summary mismatch: {name}")
        raw_cases = report.get("cases")
        if not isinstance(raw_cases, list) or len(raw_cases) != 80:
            raise ValueError(f"Expected 80 development cases: {name}")
        ids = [case.get("id") for case in raw_cases]
        if any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Report case IDs must be unique strings: {name}")
        if ordered_ids is None:
            ordered_ids = ids
        elif ids != ordered_ids:
            raise ValueError("Source reports do not contain the same ordered cases")
        reports[name] = report
        report_cases[name] = {case["id"]: case for case in raw_cases}

    assert ordered_ids is not None
    failure_ids = {
        case_id
        for name in CONFIGURATIONS
        for case_id, case in report_cases[name].items()
        if case.get("automatic_status") == "fail"
    }
    audit_ids = [case.get("id") for case in cases]
    expected_order = [case_id for case_id in ordered_ids if case_id in failure_ids]
    if audit_ids != expected_order or len(audit_ids) != len(set(audit_ids)):
        raise ValueError("Audit cases must exactly cover failed cases in development order")

    for case in cases:
        case_id = case["id"]
        baseline = report_cases[CONFIGURATIONS[0]][case_id]
        for field in ("family", "prompt", "expected"):
            if case.get(field) != baseline.get(field):
                raise ValueError(f"Immutable case field mismatch for {case_id}: {field}")
        observations = case.get("observations")
        if not isinstance(observations, dict) or tuple(observations) != CONFIGURATIONS:
            raise ValueError(f"Every configuration needs one observation: {case_id}")
        for name in CONFIGURATIONS:
            if observations[name] != _case_projection(report_cases[name][case_id]):
                raise ValueError(f"Observation differs from source report: {case_id}/{name}")

        failed = [
            name for name in CONFIGURATIONS
            if observations[name]["automatic_status"] == "fail"
        ]
        if case.get("failed_configurations") != failed:
            raise ValueError(f"Failed-configuration list mismatch: {case_id}")
        attribution = case.get("failure_attribution")
        if not isinstance(attribution, dict) or set(attribution) != set(failed):
            raise ValueError(f"Each failure needs exactly one attribution: {case_id}")
        for name, item in attribution.items():
            if not isinstance(item, dict) or set(item) != {"owner", "primary_cause", "secondary_causes"}:
                raise ValueError(f"Invalid attribution shape: {case_id}/{name}")
            owner = item["owner"]
            cause = item["primary_cause"]
            secondary = item["secondary_causes"]
            if owner not in enums["owners"] or cause not in enums["primary_causes"]:
                raise ValueError(f"Unknown attribution taxonomy: {case_id}/{name}")
            if cause.startswith("routing_") != (owner == "routing"):
                raise ValueError(f"Routing cause/owner disagreement: {case_id}/{name}")
            if (
                not isinstance(secondary, list)
                or any(not isinstance(value, str) or not value for value in secondary)
                or len(secondary) != len(set(secondary))
            ):
                raise ValueError(f"Secondary causes must be unique strings: {case_id}/{name}")

        review = case.get("label_review")
        if (
            not isinstance(review, dict)
            or review.get("status") not in enums["label_statuses"]
            or not isinstance(review.get("evidence"), str)
            or not review["evidence"].strip()
        ):
            raise ValueError(f"Case requires a label decision with evidence: {case_id}")
        disposition = case.get("training_disposition")
        if disposition not in enums["training_dispositions"]:
            raise ValueError(f"Unknown training disposition: {case_id}")
        if not isinstance(case.get("training_need"), str) or not case["training_need"].strip():
            raise ValueError(f"Case requires a concrete training or architecture need: {case_id}")
        owners = {item["owner"] for item in attribution.values()}
        causes = {item["primary_cause"] for item in attribution.values()}
        if causes == {"routing_wrong_exposure"} and disposition != "architecture_only":
            raise ValueError(f"Wrong-exposure case cannot drive SFT: {case_id}")
        if disposition == "architecture_only" and causes != {"routing_wrong_exposure"}:
            raise ValueError(f"Architecture-only case is not purely wrong exposure: {case_id}")
        if disposition == "sft_target" and "model" not in owners:
            raise ValueError(f"SFT target needs observed model evidence: {case_id}")
        if review["status"] != "accepted" and disposition == "sft_target":
            raise ValueError(f"Disputed/excluded labels cannot drive SFT: {case_id}")

    computed = summarize(cases)
    if audit.get("summary") != computed:
        raise ValueError("Stored audit summary does not match case attribution")
    if computed["configuration_failures"] != {
        name: report_specs[name]["automatic"]["fail"] for name in CONFIGURATIONS
    }:
        raise ValueError("Audit failure counts do not match source reports")
    return computed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--reports-root", type=Path)
    args = parser.parse_args()
    try:
        summary = validate_audit(args.audit, args.reports_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
