"""Run the frozen base and QLoRA step-200 candidate on final holdout once."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any

import modal


APP_NAME = "nexus-sft-final-holdout"
PROTOCOL_ID = "nexus-sft-final-holdout-v1"
MANIFEST_SHA256 = "05a36512e6cffc0f33556969fdecb6057f02ab08aa35ec2a34a4c4c6d9d0d8cc"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
HOLDOUT_SHA256 = "f95f8c836ce93081298b1308a0a9bdd46e86eec688579ce5761d1edd1931344f"
HUMAN_REVIEW_SHA256 = "07e8297f03fc4d3f63749ad21928995136474038a605a67e19cb323e0b0ad925"
MAX_NEW_TOKENS = 128
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


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

protocol_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub==0.36.0")
    .add_local_file(
        "evals/sft_final_holdout_manifest_v1.json",
        "/repo/evals/sft_final_holdout_manifest_v1.json",
    )
    .add_local_file("modal_sft_final_holdout.py", "/repo/modal_sft_final_holdout.py")
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
        "evals/model_proposal_holdout_v1.json",
        "/repo/evals/model_proposal_holdout_v1.json",
    )
    .add_local_file(
        "evals/model_proposal_holdout_human_review_v1.json",
        "/repo/evals/model_proposal_holdout_human_review_v1.json",
    )
    .add_local_file(
        "evals/sft_candidate_development_review_v2.json",
        "/repo/evals/sft_candidate_development_review_v2.json",
    )
    .add_local_file(
        "evals/sft_final_holdout_manifest_v1.json",
        "/repo/evals/sft_final_holdout_manifest_v1.json",
    )
    .add_local_file("modal_sft_final_holdout.py", "/repo/modal_sft_final_holdout.py")
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
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
        digest.update(bytes.fromhex(_file_sha256(candidate)))
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


def _safe_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or not RUN_ID_PATTERN.fullmatch(run_id)
        or run_id in {".", ".."}
    ):
        raise ValueError("run_id must contain 1-128 safe filename characters")
    return run_id


def _output(run_id: str) -> Path:
    return Path("/results") / _safe_run_id(run_id) / "sft-final-holdout-v1"


def _manifest() -> dict[str, Any]:
    path = Path("/repo/evals/sft_final_holdout_manifest_v1.json")
    if _file_sha256(path) != MANIFEST_SHA256:
        raise RuntimeError("Frozen final-holdout manifest hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Frozen final-holdout protocol ID mismatch")
    if value.get("holdout_authorized") is not True:
        raise PermissionError("Final holdout has not been authorized")
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


def _load_quantized_base():
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

    dataset_path = Path("/repo/evals/model_proposal_holdout_v1.json")
    review_path = Path("/repo/evals/model_proposal_holdout_human_review_v1.json")
    selection_path = Path("/repo/evals/sft_candidate_development_review_v2.json")
    if _file_sha256(dataset_path) != HOLDOUT_SHA256:
        raise RuntimeError("Frozen holdout dataset hash mismatch")
    if _file_sha256(review_path) != HUMAN_REVIEW_SHA256:
        raise RuntimeError("Frozen human-review hash mismatch")
    if _file_sha256(selection_path) != manifest["candidate"]["selection_review_sha256"]:
        raise RuntimeError("Frozen candidate-selection review hash mismatch")
    _, cases = load_cases(dataset_path)
    review_summary = validate_review(
        load_review(review_path), dataset_path, require_complete=True
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("decision", {}).get("selected_checkpoint") != "step-200":
        raise RuntimeError("Candidate-selection review no longer selects step-200")
    if len(cases) != manifest["holdout"]["count"]:
        raise RuntimeError("Frozen holdout case count mismatch")
    return cases, review_summary


def _load_train_evidence(manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    from scripts.sft_final_holdout_protocol import validate_train_evidence

    candidate = manifest["candidate"]
    report_path = Path(candidate["train_report_path"])
    adapter_path = Path(candidate["remote_adapter_path"])
    if not report_path.is_file() or not adapter_path.is_dir():
        raise FileNotFoundError("Frozen train report or step-200 adapter is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_train_evidence(
        report,
        candidate,
        report_sha256=_file_sha256(report_path),
        adapter_directory_sha256=_directory_sha256(adapter_path),
    )
    return adapter_path, report


def _progress_path(output: Path, label: str) -> Path:
    return output / f"progress-{label}.json"


def _load_progress(
    path: Path,
    *,
    run_id: str,
    label: str,
    expected_ids: list[str],
) -> dict[str, Any]:
    if not path.exists():
        return {
            "protocol_id": PROTOCOL_ID,
            "manifest_sha256": MANIFEST_SHA256,
            "holdout_sha256": HOLDOUT_SHA256,
            "run_id": run_id,
            "configuration": label,
            "cases": [],
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = {
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": MANIFEST_SHA256,
        "holdout_sha256": HOLDOUT_SHA256,
        "run_id": run_id,
        "configuration": label,
    }
    if any(value.get(key) != expected for key, expected in identity.items()):
        raise RuntimeError(f"Progress identity mismatch for {label}")
    rows = value.get("cases")
    if not isinstance(rows, list) or [row.get("id") for row in rows] != expected_ids[: len(rows)]:
        raise RuntimeError(f"Progress case sequence mismatch for {label}")
    if len(rows) > len(expected_ids):
        raise RuntimeError(f"Progress has too many cases for {label}")
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
        "reply_review": {
            "required": expected["behavior"] != "call",
            "status": "not_evaluated",
        },
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
    progress_path: Path,
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
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
                attention_mask = torch.ones_like(input_ids)
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                elapsed = time.perf_counter() - started
                generated_ids = output[0, input_ids.shape[1] :].detach().cpu().tolist()
                text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                scored = score_proposal_text(case, text)
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
                del input_ids, attention_mask, output
            except Exception as error:
                elapsed = time.perf_counter() - started
                scored = _generation_error(case, error, elapsed)
            progress["cases"].append(scored)
            _write_json(progress_path, progress)
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
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = [row["generation_seconds"] for row in rows if row["generation_error"] is None]
    prompt_tokens = [row["prompt_tokens"] for row in rows if row["prompt_tokens"] is not None]
    generated_tokens = [row["generated_tokens"] for row in rows if row["generated_tokens"] is not None]
    return {
        "generation_errors": sum(row["generation_error"] is not None for row in rows),
        "latency_seconds": {
            "median": statistics.median(times) if times else None,
            "p95": _percentile(times, 0.95),
            "total": sum(times),
        },
        "tokens": {
            "prompt_total": sum(prompt_tokens),
            "generated_total": sum(generated_tokens),
            "generated_per_second": (
                sum(generated_tokens) / sum(times) if times and sum(times) else None
            ),
        },
    }


def _group_summary(rows: list[dict[str, Any]], cases: list[dict[str, Any]], key: str):
    result: dict[str, dict[str, int]] = {}
    for row, case in zip(rows, cases):
        values = case[key] if isinstance(case[key], list) else [case[key]]
        for value in values:
            group = result.setdefault(value, {"cases": 0, "pass": 0, "fail": 0})
            group["cases"] += 1
            group[row["automatic_status"]] += 1
    return dict(sorted(result.items()))


def _provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    import torch
    from importlib.metadata import version

    return {
        "packages": {
            package: version(package)
            for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "jinja2")
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "cuda": torch.version.cuda,
        "runner_sha256": _file_sha256(Path("/repo/modal_sft_final_holdout.py")),
        "manifest_sha256": MANIFEST_SHA256,
        "holdout_sha256": manifest["holdout"]["sha256"],
        "human_review_sha256": manifest["holdout"]["human_review_sha256"],
        "tools_or_database_executed": False,
        "policy_used_for_gold": False,
        "response_repair": False,
    }


@app.function(
    image=protocol_image,
    volumes={"/hf-cache": cache_volume},
    timeout=3600,
    cpu=2,
    memory=4096,
)
def ensure_model() -> dict[str, Any]:
    """Download the frozen checkpoint without allocating a GPU."""

    from huggingface_hub import snapshot_download

    _manifest()
    cache_volume.reload()
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir="/hf-cache/hub",
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.model",
            "*.safetensors",
            "merges.txt",
            "tokenizer.*",
            "vocab.json",
        ],
    )
    cache_volume.commit()
    files = [candidate for candidate in Path(path).rglob("*") if candidate.is_file()]
    return {
        "path": path,
        "revision": MODEL_REVISION,
        "files": len(files),
        "bytes": sum(candidate.stat().st_size for candidate in files),
        "gpu_used": False,
    }


@app.function(
    image=holdout_image,
    gpu="T4",
    cpu=2,
    memory=32768,
    timeout=2400,
    volumes={"/hf-cache": cache_volume, "/results": result_volume},
)
def run_final_holdout(run_id: str) -> dict[str, Any]:
    """Evaluate only missing cases, persisting each raw response atomically."""

    sys.path[:0] = ["/repo", "/repo/src"]
    import torch
    from peft import PeftModel
    from scripts.score_sft_generation import summarize_proposal_texts
    from scripts.sft_final_holdout_protocol import evaluate_automatic_acceptance
    from transformers import AutoTokenizer

    run_id = _safe_run_id(run_id)
    output = _output(run_id)
    report_path = output / "report.json"
    state_path = output / "state.json"
    result_volume.reload()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed final holdout: {run_id}")
    output.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("protocol_id") != PROTOCOL_ID
            or state.get("manifest_sha256") != MANIFEST_SHA256
            or state.get("run_id") != run_id
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
            "run_id": run_id,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "resume_count": 0,
        }
    _write_json(state_path, state)
    result_volume.commit()
    try:
        manifest = _manifest()
        cases, review_summary = _load_inputs(manifest)
        adapter_path, train_report = _load_train_evidence(manifest)
        expected_ids = [case["id"] for case in cases]
        generation = manifest["contract"]
        _set_seed(generation["seed"])
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir="/hf-cache/hub",
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch.cuda.reset_peak_memory_stats()

        progress = {}
        for label in manifest["execution"]["configurations"]:
            path = _progress_path(output, label)
            progress[label] = _load_progress(
                path, run_id=run_id, label=label, expected_ids=expected_ids
            )
            if len(progress[label]["cases"]) == len(cases):
                continue
            base = _load_quantized_base()
            if label == "baseline":
                model = base
            elif label == "step-200":
                model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
                model.config.use_cache = True
                if _adapter_state_sha256(model) != manifest["candidate"]["adapter_state_sha256"]:
                    raise RuntimeError("Reloaded step-200 adapter state hash mismatch")
            else:
                raise RuntimeError(f"Unknown frozen configuration: {label}")
            _generate_missing(model, tokenizer, cases, progress[label], path)
            del model, base
            gc.collect()
            torch.cuda.empty_cache()

        results = {}
        for label, value in progress.items():
            rows = value["cases"]
            if len(rows) != 120:
                raise RuntimeError(f"Incomplete final holdout results for {label}")
            results[label] = {
                "summary": summarize_proposal_texts(rows),
                "metrics": _metrics(rows),
                "by_family": _group_summary(rows, cases, "family"),
                "by_dimension": _group_summary(rows, cases, "dimensions"),
                "cases": rows,
            }
        automatic_decision = evaluate_automatic_acceptance(
            results["baseline"]["cases"],
            results["step-200"]["cases"],
            manifest["acceptance"],
        )
        gates = {
            "human_review_complete": review_summary["complete"],
            "holdout_complete": all(len(value["cases"]) == 120 for value in results.values()),
            "no_generation_errors": all(value["metrics"]["generation_errors"] == 0 for value in results.values()),
            "adapter_evidence_matches": (
                train_report["adapters"]["step-200"]["state_sha256_reloaded"]
                == manifest["candidate"]["adapter_state_sha256"]
            ),
            "no_tools_or_database_executed": True,
            "response_repair_not_used": True,
        }
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        report = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "stage": "final_holdout",
            "model": manifest["model"],
            "candidate": manifest["candidate"],
            "dataset": manifest["holdout"],
            "contract": manifest["contract"],
            "results": results,
            "automatic_decision": automatic_decision,
            "manual_noncall_review": "pending",
            "memory": {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "provenance": _provenance(manifest),
            "gates": gates,
            "state": state,
        }
        _write_json(report_path, report)
        _write_json(state_path, state)
        result_volume.commit()
        return {
            "report": str(report_path),
            "summaries": {label: value["summary"] for label, value in results.items()},
            "automatic_decision": automatic_decision,
            "manual_noncall_review": "pending",
            "gates": gates,
            "state": state,
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
def main(stage: str = "preflight", run_id: str = ""):
    sys.path[:0] = [".", "src"]
    from scripts.sft_final_holdout_protocol import validate_protocol

    protocol = validate_protocol()
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    if stage == "preflight":
        print("Preflight complete. No Modal function, model, GPU or holdout inference was called.")
        return
    if stage != "run":
        raise ValueError("stage must be preflight or run")
    if not protocol["holdout_authorized"]:
        raise PermissionError("Final holdout remains locked")
    if not run_id:
        run_id = "sft-final-holdout-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("Run ID:", run_id)
    ready = ensure_model.remote()
    print("Locked base model ready:", json.dumps(ready, ensure_ascii=False))
    result = run_final_holdout.remote(run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
