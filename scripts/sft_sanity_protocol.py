"""Frozen checks and artifact gates for the NEXUS QLoRA sanity run."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from nexus.agent.prompts import SYSTEM_PROMPTS
from nexus.agent.tools import TOOL_DEFINITIONS
from scripts.audit_sft_data import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evals/sft_sanity_training_manifest_v1.json"
RUNNER = "modal_sft_sanity.py"


class SftProtocolError(ValueError):
    """The tracked training protocol or a gate artifact is inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash artifact names and bytes so save/load evidence is portable."""

    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise SftProtocolError(f"Artifact directory is empty: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(candidate)))
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SftProtocolError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SftProtocolError(f"Expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SftProtocolError(message)


def _repo_file(root: Path, value: object, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} path must be a non-empty string")
    relative = Path(value)
    _require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe {label} path")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), f"{label} escapes repository")
    _require(path.is_file(), f"Missing {label}: {path}")
    return path


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SftProtocolError(f"Runner function not found: {name}")


def _modal_keywords(node: ast.FunctionDef) -> dict[str, object]:
    for decorator in node.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "function"
        ):
            values: dict[str, object] = {}
            for keyword in decorator.keywords:
                if keyword.arg in {"gpu", "timeout"}:
                    values[keyword.arg] = ast.literal_eval(keyword.value)
            return values
    raise SftProtocolError(f"Modal function decorator missing: {node.name}")


def _named_call_literals(node: ast.FunctionDef, name: str) -> dict[str, object]:
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        function_name = candidate.func.id if isinstance(candidate.func, ast.Name) else None
        if function_name != name:
            continue
        result: dict[str, object] = {}
        for keyword in candidate.keywords:
            if keyword.arg is None:
                continue
            try:
                result[keyword.arg] = ast.literal_eval(keyword.value)
            except (TypeError, ValueError):
                pass
        return result
    raise SftProtocolError(f"Runner call not found: {name}")


def _runner_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = root / RUNNER
    _require(path.is_file(), f"Missing {RUNNER}")
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        raise SftProtocolError(f"Cannot parse runner: {error}") from error

    assignments: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    expected = {
        "PROTOCOL_ID": manifest["protocol_id"],
        "MODEL_ID": manifest["model"]["id"],
        "MODEL_REVISION": manifest["model"]["revision"],
        "DATASET_SHA256": manifest["dataset"]["sha256"],
        "MAX_SEQUENCE_TOKENS": manifest["contract"]["maximum_sequence_tokens"],
        "MAX_STEPS": manifest["training"]["maximum_steps"],
    }
    for name, value in expected.items():
        _require(assignments.get(name) == value, f"Runner constant differs from manifest: {name}")

    execution = manifest["execution"]
    ensure = _modal_keywords(_function(tree, "ensure_model"))
    _require("gpu" not in ensure, "Model download must not allocate a GPU")
    _require(
        _modal_keywords(_function(tree, "run_probe"))
        == {"gpu": manifest["runtime"]["gpu"], "timeout": execution["probe_timeout_seconds"]},
        "Probe decorator differs from manifest",
    )
    _require(
        _modal_keywords(_function(tree, "run_overfit"))
        == {"gpu": manifest["runtime"]["gpu"], "timeout": execution["overfit_timeout_seconds"]},
        "Overfit decorator differs from manifest",
    )
    main = _function(tree, "main")
    positional = [*main.args.posonlyargs, *main.args.args]
    defaults = {
        argument.arg: ast.literal_eval(default)
        for argument, default in zip(positional[-len(main.args.defaults) :], main.args.defaults)
    }
    _require(defaults.get("stage") == "preflight", "Runner must default to preflight")
    overfit_source = ast.get_source_segment(source, _function(tree, "run_overfit")) or ""
    _require("validate_probe_gate" in overfit_source, "Overfit does not validate its probe gate")
    sft = _named_call_literals(_function(tree, "run_overfit"), "SFTConfig")
    _require(
        {key: sft.get(key) for key in ("fp16", "bf16", "optim", "gradient_checkpointing")}
        == {
            "fp16": False,
            "bf16": False,
            "optim": manifest["training"]["optimizer"],
            "gradient_checkpointing": manifest["training"]["gradient_checkpointing"],
        },
        "SFTTrainer precision or optimizer differs from manifest",
    )
    load_source = ast.get_source_segment(source, _function(tree, "_load_qlora_model")) or ""
    _require(
        "parameter.data.to(dtype=torch.float32)" in load_source,
        "Runner does not enforce float32 adapter parameters",
    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"add_local_file", "add_local_dir"} or not node.args:
            continue
        try:
            local_path = ast.literal_eval(node.args[0])
        except (TypeError, ValueError):
            continue
        _require("holdout" not in str(local_path).casefold(), "Runner image must not include holdout data")
        _require("development" not in str(local_path).casefold(), "Runner image must not include development data")
    return {"path": RUNNER, "sha256": file_sha256(path), "default_stage": "preflight"}


def validate_protocol(
    *, root: Path = ROOT, manifest_path: Path | None = None
) -> dict[str, Any]:
    """Validate all immutable inputs without importing Modal or allocating a GPU."""

    root = root.resolve()
    manifest_path = (manifest_path or root / "evals/sft_sanity_training_manifest_v1.json").resolve()
    manifest = load_json(manifest_path)
    _require(manifest.get("schema_version") == 1, "Unsupported SFT protocol schema")
    _require(manifest.get("protocol_id") == "nexus-sft-qlora-sanity-v1", "Unexpected protocol ID")

    dataset_spec = manifest.get("dataset")
    _require(isinstance(dataset_spec, dict), "dataset must be an object")
    dataset_path = _repo_file(root, dataset_spec.get("path"), "dataset")
    _require(file_sha256(dataset_path) == dataset_spec.get("sha256"), "SFT dataset hash mismatch")
    dataset = load_json(dataset_path)
    _require(len(dataset.get("cases", [])) == dataset_spec.get("count") == 16, "SFT case count mismatch")

    data_manifest_spec = manifest.get("data_manifest")
    _require(isinstance(data_manifest_spec, dict), "data_manifest must be an object")
    data_manifest_path = _repo_file(root, data_manifest_spec.get("path"), "data manifest")
    _require(file_sha256(data_manifest_path) == data_manifest_spec.get("sha256"), "SFT data manifest hash mismatch")
    _require(load_json(data_manifest_path)["dataset"]["sha256"] == dataset_spec["sha256"], "Data manifest points to another dataset")

    model = manifest.get("model")
    _require(model == dataset.get("base_model"), "Training model differs from the dataset lock")
    contract = manifest.get("contract")
    _require(isinstance(contract, dict), "contract must be an object")
    _require(contract.get("prompt_version") == "v13" and contract.get("tool_routing") == "all", "Unexpected SFT contract")
    prompt_hash = hashlib.sha256(SYSTEM_PROMPTS["v13"].encode("utf-8")).hexdigest()
    _require(prompt_hash == contract.get("prompt_sha256"), "Prompt hash mismatch")
    _require(canonical_sha256(TOOL_DEFINITIONS) == contract.get("tools_sha256"), "Tool schema hash mismatch")
    _require(contract.get("assistant_only_labels") is True, "Assistant-only labels must remain enabled")
    _require(contract.get("maximum_sequence_tokens") == 2048, "Unexpected sequence limit")

    runtime = manifest.get("runtime")
    _require(isinstance(runtime, dict) and runtime.get("gpu") == "T4", "Sanity GPU must be T4")
    _require(runtime.get("compute_dtype") == "float16", "T4 compute dtype must be float16")
    expected_packages = {
        "torch": "2.8.0",
        "transformers": "4.57.1",
        "trl": "0.27.0",
        "peft": "0.18.0",
        "bitsandbytes": "0.48.2",
        "accelerate": "1.12.0",
        "datasets": "4.4.1",
    }
    _require(runtime.get("packages") == expected_packages, "Training package lock changed")
    _require(manifest.get("quantization") == {
        "load_in_4bit": True,
        "type": "nf4",
        "double_quantization": True,
    }, "Quantization config changed")
    _require(manifest.get("adapter") == {
        "type": "lora",
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "parameter_dtype": "float32",
        "bias": "none",
        "target_modules": "all-linear",
    }, "Adapter config changed")
    _require(manifest.get("training") == {
        "seed": 42,
        "optimizer": "paged_adamw_8bit",
        "learning_rate": 0.0002,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "maximum_steps": 64,
        "gradient_checkpointing": True,
        "trainer_amp": False,
        "automatic_retry": False,
    }, "Training config changed")

    execution = manifest.get("execution")
    _require(isinstance(execution, dict), "execution must be an object")
    _require(execution.get("stages") == ["preflight", "probe", "overfit"], "Unexpected stage order")
    _require(execution.get("default_stage") == "preflight", "Preflight must be default")
    _require(execution.get("overfit_requires_completed_probe") is True, "Probe gate is required")
    _require(execution.get("maximum_planned_gpu_seconds") == 3600, "GPU time budget changed")
    _require(execution.get("probe_timeout_seconds") + execution.get("overfit_timeout_seconds") == 3600, "Stage timeouts differ from budget")
    pricing = execution.get("pricing_snapshot", {})
    expected_cost = execution["maximum_planned_gpu_seconds"] * pricing.get("gpu_usd_per_second", -1)
    _require(math.isclose(expected_cost, pricing.get("maximum_planned_gpu_cost_usd", -1), rel_tol=0, abs_tol=1e-9), "Pricing arithmetic mismatch")
    scope = manifest.get("scope", {})
    _require(all(value is False for value in scope.values()), "Sanity run cannot include evaluation, tools, DB or rollout")

    runner = _runner_summary(root, manifest)
    return {
        "protocol_id": manifest["protocol_id"],
        "manifest": {"path": str(manifest_path.relative_to(root)), "sha256": file_sha256(manifest_path)},
        "runner": runner,
        "dataset": {"path": str(dataset_path.relative_to(root)), "sha256": file_sha256(dataset_path), "count": 16},
        "model": model,
        "gpu": runtime["gpu"],
        "maximum_planned_gpu_seconds": execution["maximum_planned_gpu_seconds"],
        "maximum_planned_gpu_cost_usd": pricing["maximum_planned_gpu_cost_usd"],
        "final_holdout_inference": False,
    }


def validate_probe_gate(
    report: dict[str, Any], manifest: dict[str, Any], *, expected_run_id: str
) -> dict[str, Any]:
    """Require concrete gradient and adapter evidence before the overfit stage."""

    _require(report.get("protocol_id") == manifest["protocol_id"], "Probe protocol mismatch")
    _require(report.get("run_id") == expected_run_id, "Probe run ID mismatch")
    _require(report.get("stage") == "probe", "Expected a probe report")
    _require(report.get("dataset_sha256") == manifest["dataset"]["sha256"], "Probe dataset mismatch")
    _require(report.get("model") == manifest["model"], "Probe model mismatch")
    _require(
        report.get("provenance", {}).get("training_manifest_sha256")
        == file_sha256(DEFAULT_MANIFEST),
        "Probe training manifest mismatch",
    )
    expected_configuration = {
        key: manifest[key] for key in ("runtime", "quantization", "adapter", "training")
    }
    _require(report.get("configuration") == expected_configuration, "Probe configuration mismatch")
    _require(report.get("state", {}).get("status") == "complete", "Probe is not complete")
    gates = report.get("gates", {})
    required = manifest["gates"]["probe"]
    _require(set(gates) == set(required), "Probe gate set mismatch")
    failed = [name for name in required if gates.get(name) is not True]
    _require(not failed, f"Probe gates failed: {failed}")
    return {
        "run_id": expected_run_id,
        "adapter_sha256": report["adapter"]["directory_sha256"],
        "trainable_parameters": report["parameters"]["trainable"],
        "loss": report["gradient_probe"]["loss"],
    }


def validate_overfit_report(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_run_id: str,
    expected_probe_run_id: str,
) -> dict[str, Any]:
    """Re-check a completed overfit report without loading model weights."""

    _require(report.get("protocol_id") == manifest["protocol_id"], "Overfit protocol mismatch")
    _require(report.get("run_id") == expected_run_id, "Overfit run ID mismatch")
    _require(report.get("stage") == "overfit", "Expected an overfit report")
    _require(report.get("dataset_sha256") == manifest["dataset"]["sha256"], "Overfit dataset mismatch")
    _require(report.get("model") == manifest["model"], "Overfit model mismatch")
    provenance = report.get("provenance", {})
    _require(provenance.get("training_manifest_sha256") == file_sha256(DEFAULT_MANIFEST), "Overfit training manifest mismatch")
    _require(provenance.get("final_holdout_inference") is False, "Overfit report used final holdout")
    _require(report.get("state", {}).get("status") == "complete", "Overfit run is not complete")
    approved = report.get("approved_probe", {})
    _require(approved.get("run_id") == expected_probe_run_id, "Overfit probe gate mismatch")

    gates = report.get("gates", {})
    required = manifest["gates"]["overfit"]
    _require(set(gates) == set(required), "Overfit gate set mismatch")
    failed = [name for name in required if gates.get(name) is not True]
    _require(not failed, f"Overfit gates failed: {failed}")
    training = report.get("training", {})
    _require(training.get("maximum_steps") == manifest["training"]["maximum_steps"], "Overfit maximum step mismatch")
    _require(training.get("completed_steps") == manifest["training"]["maximum_steps"], "Overfit did not complete every step")
    _require(math.isfinite(training.get("training_loss", math.nan)), "Overfit training loss is not finite")
    _require(training.get("runtime_seconds", 0) > 0, "Overfit runtime must be positive")
    before = report.get("teacher_forced_before", {})
    after = report.get("teacher_forced_after", {})
    _require(before.get("cases") == after.get("cases") == manifest["dataset"]["count"], "Overfit metric case count mismatch")
    _require(before.get("supervised_tokens") == after.get("supervised_tokens", -1) > 0, "Overfit supervised token count mismatch")
    _require(after.get("mean_loss", math.inf) < before.get("mean_loss", -math.inf), "Teacher-forced loss did not decrease")
    _require(0 <= after.get("token_accuracy", -1) <= 1, "Invalid post-training token accuracy")
    adapter = report.get("adapter", {})
    _require(adapter.get("state_sha256_before") != adapter.get("state_sha256_after"), "Adapter state did not change")
    _require(adapter.get("state_sha256_after") == adapter.get("state_sha256_reloaded"), "Reloaded adapter state differs")
    parameters = report.get("parameters", {})
    _require(parameters.get("only_lora_trainable") is True, "Overfit trained a non-LoRA parameter")
    _require(parameters.get("trainable_dtypes") == ["torch.float32"], "Overfit adapter dtype mismatch")
    return {
        "run_id": expected_run_id,
        "probe_run_id": expected_probe_run_id,
        "steps": training["completed_steps"],
        "mean_loss_before": before["mean_loss"],
        "mean_loss_after": after["mean_loss"],
        "token_accuracy_before": before["token_accuracy"],
        "token_accuracy_after": after["token_accuracy"],
        "exact_sequences_before": before["exact_sequences"],
        "exact_sequences_after": after["exact_sequences"],
        "adapter_state_sha256": adapter["state_sha256_after"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--probe-run-id")
    parser.add_argument("--overfit-report", type=Path)
    parser.add_argument("--overfit-run-id")
    parser.add_argument("--approved-probe-run-id")
    args = parser.parse_args()
    try:
        summary = validate_protocol(manifest_path=args.manifest.resolve())
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    manifest = load_json(args.manifest.resolve())
    if args.probe_report:
        if not args.probe_run_id:
            parser.error("--probe-report requires --probe-run-id")
        probe = load_json(args.probe_report.resolve())
        print(json.dumps(
            {"probe": validate_probe_gate(probe, manifest, expected_run_id=args.probe_run_id)},
            ensure_ascii=False,
            indent=2,
        ))
    if args.overfit_report:
        if not args.overfit_run_id or not args.approved_probe_run_id:
            parser.error("--overfit-report requires --overfit-run-id and --approved-probe-run-id")
        overfit = load_json(args.overfit_report.resolve())
        print(json.dumps(
            {"overfit": validate_overfit_report(
                overfit,
                manifest,
                expected_run_id=args.overfit_run_id,
                expected_probe_run_id=args.approved_probe_run_id,
            )},
            ensure_ascii=False,
            indent=2,
        ))
    print("Offline preflight only: no Modal function, model, GPU, tool or database was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
