"""Audit final proposal holdout v2 without exposing it to a model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

from scripts.build_model_proposal_holdout_v2 import render_document
from scripts.eval_proposals import ACTION_TO_TOOL, load_cases, strict_json
from scripts.review_holdout_labels import load_review, validate_review


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLDOUT = ROOT / "evals/model_proposal_holdout_v2.json"
DEFAULT_MANIFEST = ROOT / "evals/model_proposal_holdout_manifest_v2.json"
DEFAULT_REVIEW = ROOT / "evals/model_proposal_holdout_human_review_v2.json"
TRAINING = ROOT / "evals/sft_candidate_v3.json"
DEVELOPMENT = ROOT / "evals/model_proposal_development_v1.json"
EXPOSED_HOLDOUT = ROOT / "evals/model_proposal_holdout_v1.json"
THRESHOLD = 0.92


class HoldoutV2Error(ValueError):
    """Final holdout v2 violates its frozen separation contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldoutV2Error(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _mask(prompt: str, arguments: dict[str, Any]) -> str:
    value = _normalize(prompt)
    for field in ("content", "when"):
        raw = arguments.get(field)
        if isinstance(raw, str) and raw:
            value = value.replace(_normalize(raw), f"<{field}>")
    value = re.sub(r"(?<!\w)\d+(?!\w)", "<id>", value)
    return value


def _evaluation_records(path: Path) -> list[dict[str, Any]]:
    _, cases = load_cases(path)
    rows = []
    for case in cases:
        calls = case["expected"]["accepted_calls"]
        arguments = calls[0]["arguments"] if calls else {}
        rows.append({
            "source": str(path.relative_to(ROOT)),
            "id": case["id"],
            "family": case["family"],
            "prompt": case["prompt"],
            "normalized": _normalize(case["prompt"]),
            "masked": _mask(case["prompt"], arguments),
        })
    return rows


def _training_records(path: Path) -> list[dict[str, Any]]:
    document = strict_json(path.read_bytes())
    rows = []
    for case in document["cases"]:
        prompt = case["messages"][0]["content"]
        calls = case["messages"][1].get("tool_calls", [])
        arguments = calls[0]["function"]["arguments"] if calls else {}
        rows.append({
            "source": str(path.relative_to(ROOT)),
            "id": case["id"],
            "family": case["family"],
            "prompt": prompt,
            "normalized": _normalize(prompt),
            "masked": _mask(prompt, arguments),
        })
    return rows


def audit_holdout(path: Path = DEFAULT_HOLDOUT) -> dict[str, Any]:
    source, cases = load_cases(path)
    document = strict_json(source)
    _require(document.get("split") == "final_holdout" and document.get("version") == "v2", "Unexpected holdout identity")
    _require(len(cases) == 120, "Final holdout v2 must contain 120 cases")
    if path.resolve() == DEFAULT_HOLDOUT.resolve():
        _require(path.read_text(encoding="utf-8") == render_document(), "Tracked holdout v2 is not reproducible")

    behaviors = Counter(case["expected"]["behavior"] for case in cases)
    actions = Counter(case["expected"]["action"] for case in cases if case["expected"]["behavior"] == "call")
    _require(behaviors == Counter({"call": 84, "clarify": 12, "no_action": 24}), f"Unexpected behavior balance: {dict(behaviors)}")
    _require(actions == Counter({action: 12 for action in ACTION_TO_TOOL}), f"Unexpected action balance: {dict(actions)}")
    prompts = [case["prompt"] for case in cases]
    _require(len(prompts) == len(set(prompts)), "Duplicate holdout v2 prompt")
    forbidden = ("data(", "time(", "§", "payload", "bằng chuỗi sau", "tôi chính thức yêu cầu")
    synthetic = [case["id"] for case in cases if any(marker in case["prompt"].casefold() for marker in forbidden)]
    _require(not synthetic, f"Synthetic prompt markers remain: {synthetic}")
    for case in cases:
        prompt = case["prompt"]
        for call in case["expected"]["accepted_calls"]:
            arguments = call["arguments"]
            for field in ("content", "when"):
                if field in arguments:
                    _require(arguments[field] in prompt, f"Gold {field} is not grounded: {case['id']}")
            if "id" in arguments:
                ids = {int(value) for value in re.findall(r"(?<!\d)\d+(?!\d)", prompt)}
                _require(arguments["id"] in ids, f"Gold ID is not grounded: {case['id']}")

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        families[case["family"]].append(case)
    _require(max(map(len, families.values())) <= 4, "A holdout family contains more than four cases")
    for family, members in families.items():
        labels = {(case["expected"]["behavior"], case["expected"]["action"]) for case in members}
        _require(len(labels) == 1, f"Family crosses labels: {family}")

    references = _training_records(TRAINING) + _evaluation_records(DEVELOPMENT) + _evaluation_records(EXPOSED_HOLDOUT)
    reference_families = {row["family"] for row in references}
    _require(not (set(families) & reference_families), "A final v2 family name already exists")
    exact = []
    masked = []
    near = []
    maximum = 0.0
    for case in cases:
        calls = case["expected"]["accepted_calls"]
        arguments = calls[0]["arguments"] if calls else {}
        normalized = _normalize(case["prompt"])
        shaped = _mask(case["prompt"], arguments)
        if any(normalized == row["normalized"] for row in references):
            exact.append(case["id"])
        if any(shaped == row["masked"] for row in references):
            masked.append(case["id"])
        score, nearest = max(
            (
                (SequenceMatcher(None, shaped, row["masked"], autojunk=False).ratio(), row)
                for row in references
            ),
            key=lambda item: item[0],
        )
        maximum = max(maximum, score)
        if score >= THRESHOLD:
            near.append({"id": case["id"], "score": round(score, 6), "nearest_source": nearest["source"], "nearest_id": nearest["id"]})
    _require(not exact, f"Exact prompt leakage: {exact}")
    _require(not masked, f"Masked prompt leakage: {masked}")
    _require(not near, f"Near prompt leakage: {near}")

    return {
        "dataset_id": "nexus-model-proposal-final-holdout-v2",
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": _sha256(path),
        "cases": len(cases),
        "behavior_balance": dict(behaviors),
        "action_balance": dict(actions),
        "families": len(families),
        "largest_family": max(map(len, families.values())),
        "separation_references": {
            "training_v3": {"path": str(TRAINING.relative_to(ROOT)), "sha256": _sha256(TRAINING), "cases": 600},
            "development_v1": {"path": str(DEVELOPMENT.relative_to(ROOT)), "sha256": _sha256(DEVELOPMENT), "cases": 80},
            "exposed_holdout_v1": {"path": str(EXPOSED_HOLDOUT.relative_to(ROOT)), "sha256": _sha256(EXPOSED_HOLDOUT), "cases": 120},
        },
        "leakage": {"method": "NFKC/casefold; mask gold content/when and standalone numbers; SequenceMatcher", "threshold": THRESHOLD, "exact": exact, "masked": masked, "near": near, "maximum_similarity": round(maximum, 6)},
        "model_inference_authorized": False,
    }


def verify_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = strict_json(path.read_bytes())
    _require(isinstance(manifest, dict), "Holdout manifest must be an object")
    _require(manifest.get("schema_version") == 1, "Unsupported holdout manifest schema")
    _require(manifest.get("manifest_id") == "nexus-model-proposal-final-holdout-v2", "Unexpected holdout manifest ID")
    holdout = manifest.get("holdout", {})
    _require(ROOT / holdout.get("path", "") == DEFAULT_HOLDOUT, "Manifest holdout path changed")
    _require(holdout.get("sha256") == _sha256(DEFAULT_HOLDOUT) and holdout.get("cases") == 120, "Manifest holdout lock changed")
    for field, expected in (("builder", "scripts/build_model_proposal_holdout_v2.py"), ("auditor", "scripts/audit_model_proposal_holdout_v2.py")):
        spec = manifest.get(field, {})
        _require(spec.get("path") == expected, f"Unexpected {field} path")
        _require(spec.get("sha256") == _sha256(ROOT / expected), f"{field} hash changed")
    review_spec = manifest.get("human_review", {})
    _require(ROOT / review_spec.get("path", "") == DEFAULT_REVIEW, "Manifest review path changed")
    _require(review_spec.get("sha256") == _sha256(DEFAULT_REVIEW), "Manifest review hash changed")
    reviewed = validate_review(load_review(DEFAULT_REVIEW), DEFAULT_HOLDOUT, require_complete=True)
    _require(reviewed["counts"] == {"approved": 120, "needs_correction": 0, "pending": 0}, "Human review is not fully approved")
    gates = manifest.get("gates", {})
    _require(gates.get("human_label_review_complete") is True and gates.get("dataset_locked") is True, "Dataset/review gate is open")
    _require(gates.get("training_authorized") is False, "Holdout manifest must not authorize training")
    _require(gates.get("final_holdout_inference_authorized_by_protocol") is False, "Holdout inference must remain protocol-closed")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        summary = audit_holdout(args.holdout.resolve())
        if args.holdout.resolve() == DEFAULT_HOLDOUT.resolve():
            verify_manifest(args.manifest.resolve())
            summary["manifest_sha256"] = _sha256(args.manifest.resolve())
            summary["human_review"] = {
                "sha256": _sha256(DEFAULT_REVIEW),
                "approved": 120,
                "complete": True,
            }
    except (OSError, ValueError, KeyError, TypeError, HoldoutV2Error) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline audit only: no model, GPU, policy, tool, database or holdout inference was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
