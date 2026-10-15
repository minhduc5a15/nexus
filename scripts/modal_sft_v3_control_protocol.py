"""Audit the deployed-spawn canary without contacting Modal."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from scripts.eval_proposals import strict_json


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_sft_candidate_v3.py"
CONTROL = ROOT / "scripts/modal_sft_v3_control.py"
PLAN = ROOT / "evals/sft_candidate_training_plan_v3.json"
DEMO = ROOT / "scripts/demo/38_modal_spawn_canary.sh"


class ModalControlProtocolError(ValueError):
    """The orchestration canary contract changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModalControlProtocolError(message)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ModalControlProtocolError(f"Function missing: {name}")


def _modal_resources(function: ast.FunctionDef) -> dict[str, Any]:
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
    raise ModalControlProtocolError("Canary Modal decorator missing")


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ModalControlProtocolError(f"Assignment missing: {name}")


def validate_protocol() -> dict[str, Any]:
    plan = strict_json(PLAN.read_bytes())
    _require(plan.get("training_authorized") is False, "Training must remain blocked")
    _require(
        plan.get("final_holdout", {}).get("model_inference_authorized") is False,
        "Final holdout must remain closed",
    )
    runner_tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    resources = _modal_resources(
        _function(runner_tree, "run_orchestration_canary")
    )
    _require(
        resources == {"timeout": 300, "cpu": 0.25, "memory": 256},
        "Canary must remain CPU-only and bounded",
    )
    inspector_resources = _modal_resources(
        _function(runner_tree, "inspect_recovery_checkpoint")
    )
    _require(
        inspector_resources == {"timeout": 300, "cpu": 0.25, "memory": 512},
        "Checkpoint inspector must remain CPU-only and bounded",
    )
    control_tree = ast.parse(CONTROL.read_text(encoding="utf-8"))
    _require(
        _literal_assignment(control_tree, "APP_NAME") == "nexus-sft-candidate-v3",
        "Deployed app name changed",
    )
    _require(
        _literal_assignment(control_tree, "REGISTRY_NAME")
        == "nexus-sft-candidate-v3-launches",
        "Launch registry changed",
    )
    _require(
        _literal_assignment(control_tree, "CANARY_FUNCTION")
        == "run_orchestration_canary",
        "Canary function name changed",
    )
    attributes = [
        node.attr for node in ast.walk(control_tree) if isinstance(node, ast.Attribute)
    ]
    _require("spawn" in attributes, "Launcher must use Function.spawn")
    _require("from_id" in attributes, "Status must restore FunctionCall by ID")
    _require("cancel" not in attributes, "Control script must not cancel remote calls")
    control_text = CONTROL.read_text(encoding="utf-8")
    _require(
        "skip_if_exists=True" in control_text,
        "Launch reservation must reject duplicate run IDs",
    )
    demo_text = DEMO.read_text(encoding="utf-8")
    _require("modal\" deploy" in demo_text, "Demo must deploy before spawning")
    _require("launch-canary" in demo_text and " status " in demo_text, "Demo stages changed")
    return {
        "app_name": "nexus-sft-candidate-v3",
        "registry_name": "nexus-sft-candidate-v3-launches",
        "function": "run_orchestration_canary",
        "resources": resources,
        "checkpoint_inspector_resources": inspector_resources,
        "atomic_duplicate_guard": True,
        "launcher_waits_for_remote_result": False,
        "status_can_cancel": False,
        "training_authorized": False,
        "final_holdout_inference_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        summary = validate_protocol()
    except (OSError, ValueError, SyntaxError, ModalControlProtocolError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Offline audit only: no Modal API, function, model, GPU or holdout was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
