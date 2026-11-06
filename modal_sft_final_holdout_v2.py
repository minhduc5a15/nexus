"""Run the frozen base, v3 incumbent and v4 candidate on final holdout v2 once."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import modal


APP_NAME = "nexus-sft-final-holdout-v2"
PROTOCOL_ID = "nexus-sft-final-holdout-v2"
MANIFEST_SHA256 = "ea268886950af27c89437513f39dcd7b3d540fe3d9406f9594299043a8531c65"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
HOLDOUT_SHA256 = "a1b9aead11ca80eeab34a827c2f1047b34a7c1c1f491de4123b42930ec2100ed"
HUMAN_REVIEW_SHA256 = "72db4c29ff3e0f3e4bccfa254b53ef2fdf85df9c78ef8867a9d574ad5898c0ec"
RUN_ID = "sft-final-holdout-v2-20261107a"
MAX_NEW_TOKENS = 128


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("nexus-hf-cache", create_if_missing=True)
result_volume = modal.Volume.from_name("nexus-eval-results", create_if_missing=True)

packages = (
    "torch==2.8.0",
    "transformers==4.57.1",
    "peft==0.18.0",
    "bitsandbytes==0.48.2",
    "accelerate==1.12.0",
    "jinja2==3.1.6",
)

holdout_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*packages)
    .env(
        {
            "HF_HOME": "/hf-cache",
            "HF_HUB_CACHE": "/hf-cache/hub",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir("src", "/repo/src", ignore=["**/__pycache__/**"])
    .add_local_dir("scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
    .add_local_file(
        "evals/model_proposal_holdout_v2.json",
        "/repo/evals/model_proposal_holdout_v2.json",
    )
    .add_local_file(
        "evals/model_proposal_holdout_human_review_v2.json",
        "/repo/evals/model_proposal_holdout_human_review_v2.json",
    )
    .add_local_file(
        "evals/sft_final_holdout_manifest_v2.json",
        "/repo/evals/sft_final_holdout_manifest_v2.json",
    )
    .add_local_file(
        "modal_sft_final_holdout_v2.py",
        "/repo/modal_sft_final_holdout_v2.py",
    )
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise RuntimeError(f"Adapter directory is empty: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(candidate)))
    return digest.hexdigest()


def _adapter_state_sha256(model: Any) -> str:
    import torch
    from peft import get_peft_model_state_dict

    digest = hashlib.sha256()
    state = get_peft_model_state_dict(model)
    if not state:
        raise RuntimeError("PEFT returned an empty adapter state")
    for name in sorted(state):
        tensor = state[name].detach().contiguous().to(dtype=torch.float32, device="cpu")
        digest.update(f"{name}\0{tuple(tensor.shape)}\0{tensor.dtype}\0".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _manifest() -> dict[str, Any]:
    path = Path("/repo/evals/sft_final_holdout_manifest_v2.json")
    if _sha256(path) != MANIFEST_SHA256:
        raise RuntimeError("Frozen final-holdout manifest hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Frozen final-holdout protocol ID mismatch")
    if value.get("holdout_authorized") is not True:
        raise PermissionError("Final holdout has not been authorized")
    if value.get("execution", {}).get("run_id") != RUN_ID:
        raise RuntimeError("Frozen run ID mismatch")
    return value


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _load_base():
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir="/hf-cache/hub",
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.config.use_cache = True
    return model


def _load_inputs(manifest: dict[str, Any]):
    sys.path[:0] = ["/repo", "/repo/src"]
    from scripts.eval_proposals import load_cases
    from scripts.review_holdout_labels import load_review, validate_review

    dataset = Path("/repo/evals/model_proposal_holdout_v2.json")
    review = Path("/repo/evals/model_proposal_holdout_human_review_v2.json")
    if _sha256(dataset) != HOLDOUT_SHA256:
        raise RuntimeError("Frozen holdout dataset hash mismatch")
    if _sha256(review) != HUMAN_REVIEW_SHA256:
        raise RuntimeError("Frozen human-review hash mismatch")
    _, cases = load_cases(dataset)
    review_summary = validate_review(load_review(review), dataset, require_complete=True)
    if len(cases) != manifest["holdout"]["count"]:
        raise RuntimeError("Frozen holdout case count mismatch")
    if review_summary["counts"]["approved"] != 120:
        raise RuntimeError("Human label review is incomplete")
    return cases, review_summary


def _validate_adapter_evidence(spec: dict[str, Any]) -> Path:
    report_path = Path(spec["train_report_path"])
    adapter_path = Path(spec["remote_adapter_path"])
    if not report_path.is_file() or not adapter_path.is_dir():
        raise FileNotFoundError(f"Frozen evidence is missing for {spec['label']}")
    if _sha256(report_path) != spec["train_report_sha256"]:
        raise RuntimeError(f"Train report hash mismatch for {spec['label']}")
    if _directory_sha256(adapter_path) != spec["adapter_directory_sha256"]:
        raise RuntimeError(f"Adapter directory hash mismatch for {spec['label']}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("run_id") != spec["train_run_id"]:
        raise RuntimeError(f"Train run ID mismatch for {spec['label']}")
    if report.get("state", {}).get("status") != "complete":
        raise RuntimeError(f"Train report is incomplete for {spec['label']}")
    return adapter_path


def _output() -> Path:
    return Path("/results") / RUN_ID / APP_NAME


def _progress(output: Path, label: str, expected_ids: list[str]) -> dict[str, Any]:
    path = output / f"progress-{label}.json"
    if not path.exists():
        return {
            "protocol_id": PROTOCOL_ID,
            "manifest_sha256": MANIFEST_SHA256,
            "holdout_sha256": HOLDOUT_SHA256,
            "run_id": RUN_ID,
            "configuration": label,
            "cases": [],
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = {
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": MANIFEST_SHA256,
        "holdout_sha256": HOLDOUT_SHA256,
        "run_id": RUN_ID,
        "configuration": label,
    }
    if any(value.get(key) != expected for key, expected in identity.items()):
        raise RuntimeError(f"Progress identity mismatch for {label}")
    rows = value.get("cases")
    if not isinstance(rows, list):
        raise RuntimeError(f"Progress rows are invalid for {label}")
    ids = [row.get("id") for row in rows]
    if ids != expected_ids[: len(ids)] or len(ids) > len(expected_ids):
        raise RuntimeError(f"Progress sequence mismatch for {label}")
    return value


def _generation_error(case: dict[str, Any], error: Exception, elapsed: float) -> dict[str, Any]:
    expected = case["expected"]
    return {
        "id": case["id"],
        "family": case["family"],
        "behavior": expected["behavior"],
        "action": expected["action"],
        "prompt": case["prompt"],
        "generated_text": None,
        "parsed": {"calls": [], "reply": None, "errors": ["generation_error"]},
        "accepted_calls": expected["accepted_calls"],
        "automatic_status": "fail",
        "proposal_correct": False,
        "error_tags": ["generation_error"],
        "reply_review": {"required": expected["behavior"] != "call", "status": "not_evaluated"},
        "prompt_tokens": None,
        "generated_tokens": None,
        "generated_token_ids": None,
        "generation_seconds": elapsed,
        "routing": None,
        "generation_error": {"type": type(error).__name__, "message": str(error)},
    }


def _generate_missing(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    progress: dict[str, Any],
    path: Path,
) -> None:
    import torch
    from nexus.agent.client import build_model_request, default_model_settings
    from nexus.agent.routing import ToolRoutingMode
    from scripts.score_sft_generation import score_proposal_text

    model.eval()
    completed = len(progress["cases"])
    with torch.inference_mode():
        for index, case in enumerate(cases[completed:], start=completed):
            started = time.perf_counter()
            try:
                payload, routing = build_model_request(
                    case["prompt"],
                    prompt_version="v13",
                    settings=default_model_settings(
                        model_id="qwen3-4b-instruct-2507", temperature=0.0
                    ),
                    tool_routing=ToolRoutingMode.ALL,
                )
                prompt_ids = tokenizer.apply_chat_template(
                    payload["messages"],
                    tools=payload["tools"],
                    add_generation_prompt=True,
                    tokenize=True,
                )
                inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
                output = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                elapsed = time.perf_counter() - started
                generated_ids = output[0, inputs.shape[1] :].detach().cpu().tolist()
                raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                scored = score_proposal_text(case, raw_text)
                scored.update(
                    {
                        "prompt_tokens": len(prompt_ids),
                        "generated_tokens": len(generated_ids),
                        "generated_token_ids": generated_ids,
                        "generation_seconds": elapsed,
                        "routing": routing,
                        "generation_error": None,
                    }
                )
                del inputs, output
            except Exception as error:
                scored = _generation_error(case, error, time.perf_counter() - started)
            progress["cases"].append(scored)
            _write_json(path, progress)
            result_volume.commit()
            print(
                f"[{index + 1:03d}/{len(cases)}] {progress['configuration']} "
                f"{case['id']}: {scored['automatic_status']} {scored['error_tags']}",
                flush=True,
            )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = [row["generation_seconds"] for row in rows if row["generation_error"] is None]
    prompt = [row["prompt_tokens"] for row in rows if row["prompt_tokens"] is not None]
    generated = [row["generated_tokens"] for row in rows if row["generated_tokens"] is not None]
    return {
        "generation_errors": sum(row["generation_error"] is not None for row in rows),
        "latency_seconds": {
            "median": statistics.median(times) if times else None,
            "p95": _percentile(times, 0.95),
            "total": sum(times),
        },
        "tokens": {
            "prompt_total": sum(prompt),
            "generated_total": sum(generated),
            "generated_per_second": sum(generated) / sum(times) if times and sum(times) else None,
        },
    }


@app.function(
    image=holdout_image,
    gpu="T4",
    cpu=2,
    memory=32768,
    timeout=2400,
    retries=0,
    volumes={"/hf-cache": cache_volume, "/results": result_volume},
)
def run_final_holdout(run_id: str) -> dict[str, Any]:
    """Evaluate missing rows only and persist every raw response atomically."""

    if run_id != RUN_ID:
        raise ValueError("Run ID is not authorized by the frozen manifest")
    sys.path[:0] = ["/repo", "/repo/src"]
    import torch
    from peft import PeftModel
    from scripts.score_sft_generation import summarize_proposal_texts
    from transformers import AutoTokenizer

    result_volume.reload()
    output = _output()
    report_path = output / "report.json"
    state_path = output / "state.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed final holdout: {RUN_ID}")
    output.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("protocol_id") != PROTOCOL_ID
            or state.get("manifest_sha256") != MANIFEST_SHA256
            or state.get("run_id") != RUN_ID
            or state.get("status") not in {"running", "interrupted"}
        ):
            raise RuntimeError("Existing holdout state cannot be resumed")
        state["status"] = "running"
        state["resume_count"] = state.get("resume_count", 0) + 1
        state.pop("error", None)
    else:
        state = {
            "protocol_id": PROTOCOL_ID,
            "manifest_sha256": MANIFEST_SHA256,
            "run_id": RUN_ID,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "resume_count": 0,
        }
    _write_json(state_path, state)
    result_volume.commit()
    try:
        manifest = _manifest()
        cases, review_summary = _load_inputs(manifest)
        adapters = {
            "v3-step-300": _validate_adapter_evidence(manifest["incumbent"]),
            "v4-step-320": _validate_adapter_evidence(manifest["candidate"]),
        }
        expected_ids = [case["id"] for case in cases]
        _set_seed(manifest["contract"]["seed"])
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir="/hf-cache/hub",
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch.cuda.reset_peak_memory_stats()
        results = {}
        for label in manifest["execution"]["configurations"]:
            progress_path = output / f"progress-{label}.json"
            progress = _progress(output, label, expected_ids)
            if len(progress["cases"]) < len(cases):
                base = _load_base()
                model = base
                if label != "base":
                    model = PeftModel.from_pretrained(base, adapters[label], is_trainable=False)
                    expected_hash = (
                        manifest["incumbent"]["adapter_state_sha256"]
                        if label == "v3-step-300"
                        else manifest["candidate"]["adapter_state_sha256"]
                    )
                    if _adapter_state_sha256(model) != expected_hash:
                        raise RuntimeError(f"Adapter state hash mismatch for {label}")
                model.config.use_cache = True
                _generate_missing(model, tokenizer, cases, progress, progress_path)
                del model, base
                gc.collect()
                torch.cuda.empty_cache()
            rows = progress["cases"]
            results[label] = {
                "summary": summarize_proposal_texts(rows),
                "metrics": _metrics(rows),
                "cases": rows,
            }
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        report = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "run_id": RUN_ID,
            "manifest_sha256": MANIFEST_SHA256,
            "dataset": manifest["holdout"],
            "contract": manifest["contract"],
            "human_review": review_summary,
            "models": results,
            "provenance": {
                "runner_sha256": _sha256(Path("/repo/modal_sft_final_holdout_v2.py")),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tools_or_database_executed": False,
                "policy_used_for_gold": False,
                "response_repair": False,
                "generation_retry": False,
            },
            "memory": {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "state": state,
        }
        _write_json(report_path, report)
        _write_json(state_path, state)
        result_volume.commit()
        return {
            "run_id": RUN_ID,
            "report_path": str(report_path),
            "scores": {label: value["summary"] for label, value in results.items()},
        }
    except BaseException as error:
        state.update(
            status="interrupted",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        _write_json(state_path, state)
        result_volume.commit()
        raise


@app.local_entrypoint()
def main(stage: str = "preflight") -> None:
    from scripts.sft_final_holdout_protocol_v2 import validate_protocol

    summary = validate_protocol()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if stage == "preflight":
        print("Offline preflight passed. No Modal function or GPU was called.")
        return
    if stage != "run":
        raise ValueError("stage must be preflight or run")
    call = run_final_holdout.spawn(RUN_ID)
    print(
        json.dumps(
            {
                "stage": stage,
                "run_id": RUN_ID,
                "function_call_id": call.object_id,
                "submission": "asynchronous",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
