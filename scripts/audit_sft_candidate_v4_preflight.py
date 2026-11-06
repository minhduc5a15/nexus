"""Verify the offline v4 checkpoint without running inference or training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.audit_model_proposal_development_v3 import audit_dataset as audit_development
from scripts.audit_sft_candidate_v4_draft import audit_dataset as audit_training
from scripts.audit_sft_candidate_v4_owner_review import audit_review
from scripts.audit_sft_data import canonical_sha256
from scripts.eval_proposals import strict_json
from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.agent.tools import TOOL_DEFINITIONS


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals/sft_candidate_v4_offline_preflight.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locked_file(root: Path, section: dict, path_key: str, hash_key: str) -> Path:
    relative = section.get(path_key)
    _require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), f"Invalid {path_key}")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), f"File escapes repository: {relative}")
    _require(section.get(hash_key) == _sha256(path), f"Hash changed: {relative}")
    return path


def validate_manifest(manifest: Any, root: Path = ROOT) -> dict:
    _require(isinstance(manifest, dict), "Manifest must be an object")
    _require(manifest.get("schema_version") == 1 and manifest.get("protocol_id") == "nexus-sft-candidate-v4-offline-preflight", "Unexpected manifest identity")
    dataset, development = manifest.get("dataset"), manifest.get("development")
    comparison, training = manifest.get("comparison"), manifest.get("training_design")
    tokenizer, execution = manifest.get("tokenizer_validation"), manifest.get("execution_design")
    gates = manifest.get("gates")
    _require(all(isinstance(item, dict) for item in (dataset, development, comparison, training, tokenizer, execution, gates)), "Missing protocol section")
    _require(dataset.get("path") == "evals/sft_candidate_v4_draft.json", "Unexpected training data path")
    _require(dataset.get("builder_path") == "scripts/build_sft_candidate_v4_draft.py", "Unexpected training builder")
    _require(dataset.get("owner_review_path") == "evals/results/sft_candidate_v4_owner_review.json", "Unexpected owner review path")
    _require(development.get("path") == "evals/model_proposal_development_v3.json", "Unexpected development path")
    _require(development.get("builder_path") == "scripts/build_model_proposal_development_v3.py", "Unexpected development builder")
    data_path = _locked_file(root, dataset, "path", "sha256")
    _locked_file(root, dataset, "builder_path", "builder_sha256")
    review_path = _locked_file(root, dataset, "owner_review_path", "owner_review_sha256")
    dev_path = _locked_file(root, development, "path", "sha256")
    _locked_file(root, development, "builder_path", "builder_sha256")
    _require((dataset.get("cases"), dataset.get("source_v3_cases"), dataset.get("new_cases"), dataset.get("owner_approved_new_cases")) == (640, 600, 40, 40), "Unexpected training counts")
    _require(development.get("cases") == 40, "Unexpected development count")
    owner = audit_review(review_path, data_path, require_complete=True)
    _require(owner["approved"] == 40, "Owner review incomplete")
    data = audit_training(data_path)
    dev = audit_development(dev_path)
    _require(data["cases"] == 640 and dev["cases"] == 40, "Data audit count changed")
    _require(comparison.get("base_model_id") == "Qwen/Qwen3-4B-Instruct-2507", "Base model changed")
    _require(comparison.get("base_model_revision") == "cdbee75f17c01a7cc42f958dc650907174af0554", "Base revision changed")
    _require(comparison.get("baseline_adapter_checkpoint") == "v3-step-300", "Baseline checkpoint changed")
    _require(comparison.get("baseline_adapter_state_sha256") == "62b5c4f7073fdca99963708bc9c7f428e486046b539371aa1d5349f0131db96e", "Baseline adapter changed")
    _require(comparison.get("candidate_origin") == "base model plus reviewed v4 dataset; no v3 adapter continuation", "Candidate origin changed")
    _require(comparison.get("candidate_target_steps") == 320, "Candidate target changed")
    _require({key: comparison.get(key) for key in ("prompt_version", "tool_routing", "seed", "do_sample", "maximum_new_tokens", "proposal_only")} == {
        "prompt_version": "v13", "tool_routing": "all", "seed": 42,
        "do_sample": False, "maximum_new_tokens": 128, "proposal_only": True,
    }, "Evaluation contract changed")
    _require(comparison.get("prompt_sha256") == hashlib.sha256(SYSTEM_PROMPTS["v13"].encode()).hexdigest(), "Prompt v13 changed")
    _require(comparison.get("tools_sha256") == canonical_sha256(TOOL_DEFINITIONS), "Tool schema changed")
    renderer_files = manifest.get("renderer_files_sha256")
    _require(isinstance(renderer_files, dict) and set(renderer_files) == {
        "scripts/audit_sft_data.py", "scripts/inspect_sft_tokens.py", "scripts/score_sft_generation.py"
    }, "Renderer file set changed")
    for relative, expected_hash in renderer_files.items():
        _require(_sha256(root / relative) == expected_hash, f"Renderer changed: {relative}")
    _require(training.get("inherits_other_hyperparameters_from") == "evals/sft_candidate_training_plan_v3.json", "Training reference changed")
    _locked_file(root, training, "inherits_other_hyperparameters_from", "training_reference_sha256")
    _require({key: training.get(key) for key in ("dataset_passes", "per_device_batch_size", "gradient_accumulation_steps", "effective_batch_size", "optimizer_steps", "adapter_rank", "adapter_alpha", "learning_rate", "training_runner_locked")} == {
        "dataset_passes": 2, "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4, "effective_batch_size": 4,
        "optimizer_steps": 320, "adapter_rank": 16, "adapter_alpha": 32,
        "learning_rate": 0.0001, "training_runner_locked": False,
    }, "Training design changed")
    _require(training["dataset_passes"] * dataset["cases"] // training["effective_batch_size"] == training["optimizer_steps"], "Optimizer step arithmetic changed")
    _require(tokenizer == {
        "local_result_path": "evals/results/sft_candidate_v4_token_audit.json",
        "local_result_sha256": "55cfe0b8da131af0f997c190f60705065267613a1a5db2784476b3110ee9c6e6",
        "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "transformers_version": "4.57.1",
        "local_cache_only": True,
        "cases_checked": 640,
        "maximum_sequence_tokens": 2048,
        "maximum_observed_tokens": 1368,
        "median_observed_tokens": 1318.0,
        "supervised_tokens_total": 16050,
        "assistant_only_mask": "verified_by_exact_token_prefix",
    }, "Tokenizer validation changed")
    token_result_path = _locked_file(root, tokenizer, "local_result_path", "local_result_sha256")
    token_result = strict_json(token_result_path.read_bytes())
    _require(token_result.get("count") == tokenizer["cases_checked"]
             and token_result.get("maximum_total_tokens") == tokenizer["maximum_observed_tokens"]
             and token_result.get("median_total_tokens") == tokenizer["median_observed_tokens"]
             and token_result.get("supervised_tokens", {}).get("sum") == tokenizer["supervised_tokens_total"]
             and token_result.get("assistant_only_mask") == tokenizer["assistant_only_mask"], "Tokenizer result contents changed")
    _require(tokenizer["maximum_observed_tokens"] <= tokenizer["maximum_sequence_tokens"], "Sequence limit exceeded")
    _require(execution.get("planned_stages") == ["train", "development"], "Execution stages changed")
    _require(execution.get("gpu") == "T4" and execution.get("cpu_cores") == 2 and execution.get("memory_gib") == 16, "Execution resources changed")
    _require(execution.get("train_timeout_seconds") == 8400 and execution.get("development_timeout_seconds") == 1800, "Execution timeout changed")
    _require(execution.get("automatic_retry") is False, "Automatic retry changed")
    _require(execution.get("pricing_checked_on") == "2026-11-03" and execution.get("pricing_source") == "https://modal.com/pricing", "Pricing provenance changed")
    _require({key: execution.get(key) for key in ("t4_usd_per_second", "cpu_core_usd_per_second", "memory_gib_usd_per_second")} == {
        "t4_usd_per_second": 0.000164,
        "cpu_core_usd_per_second": 0.0000131,
        "memory_gib_usd_per_second": 0.00000222,
    }, "Pricing rate changed")
    seconds = execution["train_timeout_seconds"] + execution["development_timeout_seconds"]
    compute_ceiling = seconds * (
        execution["t4_usd_per_second"]
        + execution["cpu_cores"] * execution["cpu_core_usd_per_second"]
        + execution["memory_gib"] * execution["memory_gib_usd_per_second"]
    )
    _require(math.isclose(compute_ceiling, execution.get("maximum_planned_compute_usd", -1), abs_tol=1e-9), "Compute estimate changed")
    _require(execution.get("proposed_total_budget_usd") == 2.6 and execution["proposed_total_budget_usd"] > compute_ceiling, "Proposed budget changed")
    _require(execution.get("budget_authorized") is False, "Budget must remain unapproved")
    _require(gates == {
        "owner_label_review_complete": True,
        "development_assistant_review_complete": True,
        "training_authorized": False,
        "development_inference_authorized": False,
        "final_holdout_inference_authorized": False,
    }, "Preflight gate changed")
    return {
        "dataset_sha256": dataset["sha256"],
        "owner_review_sha256": dataset["owner_review_sha256"],
        "owner_approved": owner["approved"],
        "development_sha256": development["sha256"],
        "development_cases": dev["cases"],
        "target_optimizer_steps": training["optimizer_steps"],
        "maximum_observed_tokens": tokenizer["maximum_observed_tokens"],
        "maximum_planned_compute_usd": execution["maximum_planned_compute_usd"],
        "proposed_total_budget_usd": execution["proposed_total_budget_usd"],
        "training_authorized": False,
        "inference_authorized": False,
    }


def audit_preflight(manifest_path: Path = MANIFEST, root: Path = ROOT) -> dict:
    result = validate_manifest(strict_json(manifest_path.read_bytes()), root)
    return {"manifest_sha256": _sha256(manifest_path), **result}


def main() -> int:
    print(json.dumps(audit_preflight(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
