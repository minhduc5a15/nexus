"""Audit the v4 Modal runner and locked plan without contacting Modal."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.agent.tools import TOOL_DEFINITIONS
from scripts.audit_sft_candidate_v4_preflight import audit_preflight
from scripts.audit_sft_data import canonical_sha256
from scripts.eval_proposals import load_cases, strict_json


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "evals/sft_candidate_v4_execution_plan.json"
RUNNER = ROOT / "modal_sft_candidate_v4.py"
SUPPORT = (
    "scripts/__init__.py", "scripts/audit_proposal_data.py", "scripts/audit_sft_data.py",
    "scripts/eval_proposals.py", "scripts/eval_qwen.py", "scripts/inspect_sft_tokens.py",
    "scripts/score_sft_generation.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise ValueError(f"Runner constant missing: {name}")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"Runner function missing: {name}")


def _resources(function: ast.FunctionDef) -> dict:
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "function":
            return {item.arg: ast.literal_eval(item.value) for item in decorator.keywords
                    if item.arg in {"gpu", "timeout", "cpu", "memory", "retries"}}
    raise ValueError(f"Modal decorator missing: {function.name}")


def _packaged_paths(tree: ast.Module) -> tuple[list[str], list[str], int]:
    files, directories, variable_files = [], [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or not node.args:
            continue
        if node.func.attr == "add_local_file":
            if isinstance(node.args[0], ast.Name) and node.args[0].id == "relative":
                variable_files += 1
            else:
                files.append(ast.literal_eval(node.args[0]))
        elif node.func.attr == "add_local_dir":
            directories.append(ast.literal_eval(node.args[0]))
    return files, directories, variable_files


def validate_protocol(plan_path: Path = PLAN, runner_path: Path = RUNNER) -> dict:
    preflight = audit_preflight()
    plan = strict_json(plan_path.read_bytes())
    _require(plan.get("schema_version") == 1 and plan.get("protocol_id") == "nexus-sft-candidate-v4", "Plan identity changed")
    _require(plan.get("preflight") == {"path": "evals/sft_candidate_v4_offline_preflight.json", "sha256": preflight["manifest_sha256"]}, "Preflight lock changed")
    _require(plan.get("dataset") == {"path": "evals/sft_candidate_v4_draft.json", "sha256": preflight["dataset_sha256"], "count": 640}, "Training dataset changed")
    _require(plan.get("development") == {"path": "evals/model_proposal_development_v3.json", "sha256": preflight["development_sha256"], "count": 40}, "Development dataset changed")
    _require(plan.get("model") == {"id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554", "start_from": "base"}, "Base model changed")
    _require(plan.get("source_v3") == {"run_id": "sft-candidate-v3-train-20261016d", "train_report_sha256": "4f7f82c75a1c394a788a0ed3c4645d17fa1f0e161f33a9391ae7b8f0e05577dd", "step_300_adapter_state_sha256": "62b5c4f7073fdca99963708bc9c7f428e486046b539371aa1d5349f0131db96e"}, "V3 baseline changed")
    generation = plan.get("generation")
    _require(generation == {
        "prompt_version": "v13", "tool_routing": "all",
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPTS["v13"].encode()).hexdigest(),
        "tools_sha256": canonical_sha256(TOOL_DEFINITIONS),
        "seed": 42, "do_sample": False, "maximum_new_tokens": 128, "proposal_only": True,
    }, "Generation contract changed")
    _require(plan.get("training") == {
        "seed": 42, "dataset_passes": 2, "maximum_steps": 320,
        "per_device_batch_size": 1, "gradient_accumulation_steps": 4,
        "effective_batch_size": 4, "maximum_sequence_tokens": 2048,
        "learning_rate": 0.0001, "optimizer": "paged_adamw_8bit",
        "scheduler": "constant_with_warmup", "warmup_steps": 15,
        "weight_decay": 0.01, "adapter_rank": 16, "adapter_alpha": 32,
        "recovery_checkpoint_interval_steps": 25, "automatic_retry": False,
    }, "Training configuration changed")
    _require(plan["training"]["dataset_passes"] * 640 // plan["training"]["effective_batch_size"] == 320, "Optimizer arithmetic changed")
    _require(plan.get("resources") == {"gpu": "T4", "cpu_cores": 2, "memory_mib": 16384, "train_timeout_seconds": 1800, "development_timeout_seconds": 800}, "Resource design changed")
    _require(plan.get("resume") == {
        "source_run_id": "sft-candidate-v4-train-20261104c",
        "source_plan_sha256": "b10c8ec43b0a04e0a776d3989c606a56967cd72f10136950e9afe2bcd0a4c690",
        "checkpoint_step": 275,
        "checkpoint_path": "sft-candidate-v4-train-20261104c/nexus-sft-candidate-v4/trainer-checkpoints/checkpoint-275",
        "trainer_state_sha256": "002ae377f6721dd2e04156d73724ec91525e8d7970fad5ddbe5323a8df8b337b",
    }, "Resume contract changed")
    _require(plan.get("run_ids") == {"train": "sft-candidate-v4-train-20261106d", "development": "sft-candidate-v4-development-20261106d"}, "Run IDs changed")
    _require(plan.get("authorization") == {
        "proposed_total_budget_usd": 2.6, "approved_total_budget_usd": 2.6,
        "approved_by": "project_owner", "approved_on": "2026-11-03",
        "training_authorized": True, "development_inference_authorized": True,
        "final_holdout_inference_authorized": False,
    }, "Authorization gate changed")
    expected_files = list(SUPPORT) + ["src", "modal_sft_candidate_v4.py", "evals/sft_candidate_v4_execution_plan.json"]
    _require(plan.get("image_files") == {"shared": expected_files, "train_only": ["evals/sft_candidate_v4_draft.json"], "development_only": ["evals/model_proposal_development_v3.json"]}, "Image allowlist changed")
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    _require(_literal(tree, "APP_NAME") == plan["protocol_id"], "Runner app changed")
    _require(_literal(tree, "PLAN_SHA256") == _sha(plan_path), "Runner plan hash changed")
    for constant, expected in (
        ("MODEL_ID", plan["model"]["id"]), ("MODEL_REVISION", plan["model"]["revision"]),
        ("TRAIN_DATA_SHA256", plan["dataset"]["sha256"]),
        ("DEVELOPMENT_SHA256", plan["development"]["sha256"]),
        ("MAX_SEQUENCE_TOKENS", 2048), ("MAX_STEPS", 320), ("MAX_NEW_TOKENS", 128),
        ("TRAIN_RUN_ID", plan["run_ids"]["train"]),
        ("DEVELOPMENT_RUN_ID", plan["run_ids"]["development"]),
        ("SOURCE_V3_RUN_ID", plan["source_v3"]["run_id"]),
        ("SOURCE_V3_REPORT_SHA256", plan["source_v3"]["train_report_sha256"]),
        ("SOURCE_V3_ADAPTER_SHA256", plan["source_v3"]["step_300_adapter_state_sha256"]),
        ("RESUME_RUN_ID", plan["resume"]["source_run_id"]),
        ("RESUME_STEP", plan["resume"]["checkpoint_step"]),
        ("RESUME_TRAINER_STATE_SHA256", plan["resume"]["trainer_state_sha256"]),
        ("RESUME_CHECKPOINT_PATH", plan["resume"]["checkpoint_path"]),
        ("SUPPORT_FILES", tuple(SUPPORT)),
    ):
        _require(_literal(tree, constant) == expected, f"Runner {constant} changed")
    _require(_resources(_function(tree, "run_train")) == {"gpu": "T4", "timeout": 1800, "cpu": 2, "memory": 16384, "retries": 0}, "Train function resources changed")
    _require(_resources(_function(tree, "run_development")) == {"gpu": "T4", "timeout": 800, "cpu": 2, "memory": 16384, "retries": 0}, "Development function resources changed")
    files, directories, variable_files = _packaged_paths(tree)
    _require(sorted(files) == sorted(["modal_sft_candidate_v4.py", "evals/sft_candidate_v4_execution_plan.json", "evals/sft_candidate_v4_draft.json", "evals/model_proposal_development_v3.json"]) and directories == ["src"] and variable_files == 1, "Runner image packages unexpected files")
    source = runner_path.read_text(encoding="utf-8")
    _require("for relative in SUPPORT_FILES:" in source and 'base_image.add_local_file(relative, "/repo/" + relative)' in source, "Support file packaging changed")
    _require('plan["authorization"].get("training_authorized") is not True' in source, "Remote authorization guard missing")
    _require('_authorize(stage, chosen)' in source, "Local authorization guard missing")
    _require("run_train.spawn(chosen)" in source and "run_development.spawn(chosen)" in source, "Durable asynchronous submission missing")
    _require("run_train.remote(chosen)" not in source and "run_development.remote(chosen)" not in source, "Synchronous submission must remain disabled")
    _require("trainer.train(resume_from_checkpoint=str(resume_checkpoint))" in source, "Checkpoint resume path missing")
    _require("score_proposal_text(case, raw_text)" in source and "build_model_request(" in source, "Proposal generation path changed")
    _require("result_volume.commit()" in source and "_claim_fresh_output(" in source, "Durable artifact guard missing")
    _require("final_holdout" not in " ".join(files + directories + list(SUPPORT)), "Final holdout is packaged")
    _, dev_cases = load_cases(ROOT / plan["development"]["path"])
    _require(len(dev_cases) == 40, "Development cases changed")
    return {
        "protocol_id": plan["protocol_id"], "plan_sha256": _sha(plan_path),
        "runner_sha256": _sha(runner_path), "preflight_sha256": preflight["manifest_sha256"],
        "training_cases": 640, "development_cases": 40,
        "packaged_paths": plan["image_files"],
        "training_authorized": True, "development_inference_authorized": True,
        "final_holdout_inference_authorized": False,
    }


def main() -> int:
    print(json.dumps(validate_protocol(), ensure_ascii=False, indent=2))
    print("Offline audit only: no Modal function, GPU or model was called.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
