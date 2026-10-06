"""Frozen protocol checks for the stronger-model control experiment."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evals/model_control_manifest_v1.json"


class ProtocolError(ValueError):
    """The tracked control protocol or a gate artifact is inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected a JSON object in {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _repo_path(root: Path, relative: object) -> Path:
    _require(isinstance(relative, str) and bool(relative), "Dataset path must be a non-empty string")
    candidate = Path(relative)
    _require(not candidate.is_absolute() and ".." not in candidate.parts, f"Unsafe dataset path: {relative}")
    resolved = (root / candidate).resolve()
    _require(resolved.is_relative_to(root.resolve()), f"Dataset escapes repository: {relative}")
    return resolved


def _validate_dataset(root: Path, spec: object, name: str) -> tuple[Path, dict[str, Any]]:
    _require(isinstance(spec, dict), f"datasets.{name} must be an object")
    path = _repo_path(root, spec.get("path"))
    _require(path.exists(), f"Missing {name} dataset: {path}")
    expected_hash = spec.get("sha256")
    _require(isinstance(expected_hash, str) and len(expected_hash) == 64, f"Invalid {name} SHA-256")
    _require(file_sha256(path) == expected_hash, f"{name} dataset hash mismatch")
    document = load_json(path)
    cases = document.get("cases")
    _require(isinstance(cases, list), f"{name} dataset cases must be a list")
    _require(len(cases) == spec.get("count"), f"{name} dataset count mismatch")
    ids = [case.get("id") for case in cases if isinstance(case, dict)]
    _require(len(ids) == len(cases) and all(isinstance(case_id, str) for case_id in ids), f"Invalid {name} case ID")
    _require(len(ids) == len(set(ids)), f"Duplicate {name} case ID")
    return path, document


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ProtocolError(f"Runner function not found: {name}")


def _modal_function_keywords(node: ast.FunctionDef) -> dict[str, object]:
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
    raise ProtocolError(f"Modal function decorator not found: {node.name}")


def _validate_runner(root: Path, manifest: dict[str, Any]) -> tuple[Path, str]:
    runner = root / "modal_control_benchmark.py"
    _require(runner.exists(), "Missing modal_control_benchmark.py")
    try:
        tree = ast.parse(runner.read_text(encoding="utf-8"), filename=str(runner))
    except (OSError, SyntaxError) as error:
        raise ProtocolError(f"Cannot parse control runner: {error}") from error

    assignments: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    model = manifest["model"]
    expected_assignments = {
        "PROTOCOL_ID": manifest["protocol_id"],
        "MODEL_KEY": model["id"],
        "MODEL_FILENAME": model["filename"],
        "MODEL_URL": model["url"],
        "MODEL_SIZE": model["size_bytes"],
        "MODEL_SHA256": model["sha256"],
        "LLAMA_COMMIT": manifest["runtime"]["llama_commit"],
        "CANARY_SHA256": manifest["datasets"]["canary"]["sha256"],
        "DEVELOPMENT_SHA256": manifest["datasets"]["development"]["sha256"],
    }
    for name, expected in expected_assignments.items():
        _require(assignments.get(name) == expected, f"Runner constant differs from manifest: {name}")

    execution = manifest["execution"]
    ensure_keywords = _modal_function_keywords(_function_node(tree, "ensure_model"))
    _require("gpu" not in ensure_keywords, "Model download must not allocate a GPU")
    canary_keywords = _modal_function_keywords(_function_node(tree, "run_canary"))
    development_keywords = _modal_function_keywords(_function_node(tree, "run_development"))
    _require(
        canary_keywords == {"gpu": manifest["runtime"]["gpu"], "timeout": execution["canary_timeout_seconds"]},
        "Canary decorator differs from manifest",
    )
    _require(
        development_keywords
        == {"gpu": manifest["runtime"]["gpu"], "timeout": execution["development_timeout_seconds"]},
        "Development decorator differs from manifest",
    )

    main = _function_node(tree, "main")
    positional = [*main.args.posonlyargs, *main.args.args]
    default_by_name = {
        argument.arg: ast.literal_eval(default)
        for argument, default in zip(positional[-len(main.args.defaults) :], main.args.defaults)
    }
    _require(default_by_name.get("stage") == "preflight", "Runner default stage must be preflight")
    development_source = ast.get_source_segment(runner.read_text(encoding="utf-8"), _function_node(tree, "run_development")) or ""
    _require("validate_canary_gate" in development_source, "Development runner does not validate the canary gate")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_local_file" or not node.args:
            continue
        try:
            local_path = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        _require("holdout" not in str(local_path).casefold(), "Runner image must not include a holdout file")
    return runner, file_sha256(runner)


def validate_control_protocol(
    *, root: Path = ROOT, manifest_path: Path | None = None
) -> dict[str, Any]:
    """Validate the immutable inputs without importing Modal or calling a model."""

    root = root.resolve()
    manifest_path = (manifest_path or root / "evals/model_control_manifest_v1.json").resolve()
    manifest = load_json(manifest_path)

    _require(manifest.get("schema_version") == 1, "Unsupported control protocol schema")
    _require(manifest.get("protocol_id") == "qwen3-30b-a3b-control-v1", "Unexpected protocol ID")
    _require(manifest.get("holdout_inference") is False, "Control protocol must not use final holdout")

    model = manifest.get("model")
    _require(isinstance(model, dict), "model must be an object")
    _require(model.get("base_model") == "Qwen/Qwen3-30B-A3B-Instruct-2507", "Unexpected base model")
    _require(model.get("artifact_repository") == "ggml-org/Qwen3-30B-A3B-Instruct-2507-Q8_0-GGUF", "Unexpected artifact repository")
    revision = model.get("artifact_revision")
    _require(isinstance(revision, str) and len(revision) == 40, "Artifact revision must be a full commit")
    _require(revision in str(model.get("url", "")), "Model URL must contain the locked artifact revision")
    _require(model.get("filename") == "qwen3-30b-a3b-instruct-2507-q8_0.gguf", "Unexpected filename")
    _require(model.get("size_bytes") == 32_483_930_432, "Unexpected model size")
    _require(model.get("sha256") == "66cbfc7624628ca4725a6a0581b95f2155aa6aa073a6ed4ff8d158794ee8fee3", "Unexpected model hash")
    _require(model.get("quantization") == "Q8_0", "Unexpected quantization")

    runtime = manifest.get("runtime")
    _require(isinstance(runtime, dict), "runtime must be an object")
    _require(runtime.get("llama_commit") == "5266f24da75dc449bd56cbed7addb9c8e4a6a73e", "llama.cpp commit changed")
    _require(runtime.get("gpu") == "A100-80GB", "Control GPU must be A100-80GB")
    _require(runtime.get("cuda_architecture") == "80", "Unexpected CUDA architecture")
    _require(runtime.get("context_size") == 4096 and runtime.get("parallel") == 1, "Runtime shape changed")
    _require(runtime.get("prompt_version") == "v13", "Control prompt must be v13")
    _require(runtime.get("tool_routing") == "all", "Control routing must expose all tools")
    expected_settings = {
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": 256,
        "seed": 42,
        "enable_thinking": False,
    }
    _require(runtime.get("settings") == expected_settings, "Sampling settings changed")

    datasets = manifest.get("datasets")
    _require(isinstance(datasets, dict) and set(datasets) == {"canary", "development"}, "Only canary and development datasets are allowed")
    canary_path, canary = _validate_dataset(root, datasets["canary"], "canary")
    development_path, development = _validate_dataset(root, datasets["development"], "development")
    _require("holdout" not in str(canary_path).casefold(), "Canary path must not reference holdout")
    _require("holdout" not in str(development_path).casefold(), "Development path must not reference holdout")

    development_cases = {case["id"]: case for case in development["cases"]}
    canary_ids = datasets["canary"].get("case_ids")
    _require(isinstance(canary_ids, list) and len(canary_ids) == 10, "Canary must lock ten case IDs")
    _require([case["id"] for case in canary["cases"]] == canary_ids, "Canary order differs from manifest")
    derived = canary.get("derived_from")
    _require(
        derived == {"path": datasets["development"]["path"], "sha256": datasets["development"]["sha256"]},
        "Canary source metadata differs from development lock",
    )
    for case in canary["cases"]:
        _require(case["id"] in development_cases, f"Canary case not found in development: {case['id']}")
        _require(case == development_cases[case["id"]], f"Canary case changed from development: {case['id']}")

    execution = manifest.get("execution")
    _require(isinstance(execution, dict), "execution must be an object")
    _require(execution.get("stages") == ["preflight", "canary", "development"], "Unexpected stage order")
    _require(execution.get("default_stage") == "preflight", "Preflight must be the default stage")
    _require(execution.get("development_requires_explicit_canary_run_id") is True, "Development must require a canary run ID")
    _require(execution.get("development_requires_error_free_canary") is True, "Development must require an error-free canary")
    _require(execution.get("automatic_retry") is False, "Automatic retry must stay disabled")
    planned_seconds = execution.get("maximum_planned_gpu_seconds")
    canary_seconds = execution.get("canary_timeout_seconds")
    development_seconds = execution.get("development_timeout_seconds")
    _require(
        all(type(value) is int and value > 0 for value in (planned_seconds, canary_seconds, development_seconds)),
        "GPU timeout values must be positive integers",
    )
    _require(canary_seconds + development_seconds <= planned_seconds, "Function timeouts exceed planned GPU seconds")
    pricing = execution.get("pricing_snapshot")
    _require(isinstance(pricing, dict), "Missing pricing snapshot")
    expected_cost = planned_seconds * pricing.get("gpu_usd_per_second", -1)
    _require(
        math.isclose(expected_cost, pricing.get("maximum_planned_gpu_cost_usd", -1), rel_tol=0, abs_tol=1e-9),
        "Pricing snapshot arithmetic mismatch",
    )
    runner_path, runner_hash = _validate_runner(root, manifest)

    return {
        "protocol_id": manifest["protocol_id"],
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": file_sha256(manifest_path),
        "runner": {
            "path": str(runner_path.relative_to(root)),
            "sha256": runner_hash,
            "default_stage": "preflight",
        },
        "model": {
            "id": model["id"],
            "artifact_revision": model["artifact_revision"],
            "size_bytes": model["size_bytes"],
            "sha256": model["sha256"],
        },
        "runtime": {
            "gpu": runtime["gpu"],
            "prompt_version": runtime["prompt_version"],
            "tool_routing": runtime["tool_routing"],
            "llama_commit": runtime["llama_commit"],
        },
        "datasets": {
            "canary": {"count": len(canary["cases"]), "sha256": datasets["canary"]["sha256"]},
            "development": {"count": len(development["cases"]), "sha256": datasets["development"]["sha256"]},
        },
        "holdout_inference": False,
        "maximum_planned_gpu_seconds": planned_seconds,
        "maximum_planned_gpu_cost_usd": pricing["maximum_planned_gpu_cost_usd"],
    }


def validate_canary_gate(
    report: dict[str, Any], manifest: dict[str, Any], *, expected_run_id: str
) -> dict[str, Any]:
    """Require an error-free, protocol-matching canary before development inference."""

    _require(report.get("protocol_id") == manifest.get("protocol_id"), "Canary protocol ID mismatch")
    _require(report.get("run_id") == expected_run_id, "Canary run ID mismatch")
    _require(report.get("block") == "control-canary", "Not a control canary report")
    runtime = manifest["runtime"]
    model = manifest["model"]
    dataset = manifest["datasets"]["canary"]
    _require(report.get("prompt_version") == runtime["prompt_version"], "Canary prompt mismatch")
    _require(report.get("tool_routing") == runtime["tool_routing"], "Canary routing mismatch")
    _require(report.get("settings") == runtime["settings"], "Canary settings mismatch")
    _require(report.get("model_sha256") == model["sha256"], "Canary model hash mismatch")
    _require(report.get("dataset") == {"path": dataset["path"], "sha256": dataset["sha256"]}, "Canary dataset mismatch")
    cases = report.get("cases")
    _require(isinstance(cases, list), "Canary cases must be a list")
    _require([case.get("id") for case in cases if isinstance(case, dict)] == dataset["case_ids"], "Canary cases or order changed")
    _require(all(case.get("automatic_status") != "error" for case in cases), "Canary contains a generation error")
    _require(report.get("state", {}).get("status") == "complete", "Canary did not complete")
    passed = sum(case.get("automatic_status") == "pass" for case in cases)
    failed = sum(case.get("automatic_status") == "fail" for case in cases)
    _require(passed + failed == len(cases), "Canary has an unknown automatic status")
    return {"run_id": expected_run_id, "cases": len(cases), "passed": passed, "failed": failed}


def validate_development_report(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_run_id: str,
    expected_canary_gate: dict[str, Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate a completed development artifact against the frozen protocol."""

    _require(report.get("protocol_id") == manifest.get("protocol_id"), "Development protocol ID mismatch")
    _require(report.get("run_id") == expected_run_id, "Development run ID mismatch")
    _require(report.get("block") == "control-development", "Not a control development report")
    runtime = manifest["runtime"]
    model = manifest["model"]
    dataset = manifest["datasets"]["development"]
    _require(report.get("prompt_version") == runtime["prompt_version"], "Development prompt mismatch")
    _require(report.get("tool_routing") == runtime["tool_routing"], "Development routing mismatch")
    _require(report.get("settings") == runtime["settings"], "Development settings mismatch")
    _require(report.get("model_sha256") == model["sha256"], "Development model hash mismatch")
    _require(
        report.get("dataset") == {"path": dataset["path"], "sha256": dataset["sha256"]},
        "Development dataset mismatch",
    )
    _require(report.get("approved_canary") == expected_canary_gate, "Development canary approval mismatch")

    development = load_json(_repo_path(root.resolve(), dataset["path"]))
    expected_ids = [case["id"] for case in development["cases"]]
    cases = report.get("cases")
    _require(isinstance(cases, list), "Development cases must be a list")
    _require(
        [case.get("id") for case in cases if isinstance(case, dict)] == expected_ids,
        "Development cases or order changed",
    )
    _require(
        all(case.get("automatic_status") != "error" for case in cases),
        "Development contains a generation error",
    )
    state = report.get("state", {})
    _require(state.get("status") == "complete", "Development did not complete")
    _require(state.get("scored_completed") == len(cases), "Development completion count mismatch")
    passed = sum(case.get("automatic_status") == "pass" for case in cases)
    failed = sum(case.get("automatic_status") == "fail" for case in cases)
    _require(passed + failed == len(cases), "Development has an unknown automatic status")
    automatic = report.get("summary", {}).get("automatic")
    _require(
        automatic == {"pass": passed, "fail": failed, "error": 0},
        "Development summary does not match case results",
    )
    return {
        "run_id": expected_run_id,
        "cases": len(cases),
        "passed": passed,
        "failed": failed,
        "canary_run_id": expected_canary_gate["run_id"],
    }
