"""Compare base and frozen step-300 on exposed DEADLINE/LIST framing pairs."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import modal

sys.path.insert(0, "/repo")
import modal_sft_development_v2 as development_v2


previous = development_v2.previous
APP_NAME = "nexus-sft-framing-diagnostic-v1"
PLAN = Path("/repo/evals/model_proposal_framing_diagnostic_v1_run_plan.json")
DATASET = Path("/repo/evals/model_proposal_framing_diagnostic_v1.json")
RUN_ID = "framing-base-step300-20261021"
SOURCE_TRAIN_RUN = development_v2.SOURCE_TRAIN_RUN

app = modal.App(APP_NAME)
image = (
    development_v2.image
    .add_local_file("evals/model_proposal_framing_diagnostic_v1.json", str(DATASET))
    .add_local_file("evals/model_proposal_framing_diagnostic_v1_run_plan.json", str(PLAN))
    .add_local_file("modal_sft_framing_diagnostic_v1.py", "/repo/modal_sft_framing_diagnostic_v1.py")
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_plan(plan: dict[str, Any], root: Path) -> None:
    if plan.get("protocol_id") != APP_NAME or plan.get("run_id") != RUN_ID:
        raise ValueError("Diagnostic protocol/run changed")
    if plan.get("training_authorized") is not False or plan.get("final_holdout_inference_authorized") is not False:
        raise ValueError("Training/final holdout is outside this diagnostic")
    for name, expected_hash in plan["files"].items():
        path = root / name
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"Locked file changed or missing: {name}")
    if plan["dataset"] != {"path": "evals/model_proposal_framing_diagnostic_v1.json", "sha256": sha256(root / "evals/model_proposal_framing_diagnostic_v1.json"), "count": 20}:
        raise ValueError("Diagnostic dataset changed")
    if plan["model"] != {"id": previous.MODEL_ID, "revision": previous.MODEL_REVISION}:
        raise ValueError("Model revision changed")
    if plan["source_train_run_id"] != SOURCE_TRAIN_RUN or plan["source_adapter_state_sha256"] != "62b5c4f7073fdca99963708bc9c7f428e486046b539371aa1d5349f0131db96e":
        raise ValueError("Adapter source changed")
    if plan["source_train_report_sha256"] != "4f7f82c75a1c394a788a0ed3c4645d17fa1f0e161f33a9391ae7b8f0e05577dd":
        raise ValueError("Training report changed")
    if plan["generation"] != {"prompt_version": "v13", "tool_routing": "all", "temperature": 0.0, "do_sample": False, "seed": 42, "maximum_new_tokens": 128}:
        raise ValueError("Generation settings changed")
    if plan["resources"] != {"gpu": "T4", "cpu": 2, "memory_mib": 16384, "timeout_seconds": 1800, "automatic_retry": False}:
        raise ValueError("Resource limits changed")


def result_path() -> Path:
    return Path("/results") / RUN_ID / "sft-framing-diagnostic-v1"


@app.function(
    image=image, gpu="T4", timeout=1800, cpu=2, memory=16384,
    volumes={"/hf-cache": previous.cache_volume, "/results": previous.result_volume},
)
def run_diagnostic() -> dict[str, Any]:
    """Generate raw proposals only; never execute task tools or open SQLite."""
    sys.path[:0] = ["/repo", "/repo/src"]
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from nexus.agent.prompts import SYSTEM_PROMPTS
    from nexus.agent.tools import TOOL_DEFINITIONS
    from scripts.audit_sft_data import canonical_sha256
    from scripts.eval_proposals import load_cases
    from scripts.score_sft_generation import summarize_proposal_texts
    from scripts.sft_candidate_training_protocol_v3 import validate_train_report

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    validate_plan(plan, Path("/repo"))
    if hashlib.sha256(SYSTEM_PROMPTS["v13"].encode()).hexdigest() != plan["prompt_sha256"]:
        raise RuntimeError("Prompt v13 changed")
    if canonical_sha256(TOOL_DEFINITIONS) != plan["tools_sha256"]:
        raise RuntimeError("Tool schema changed")
    previous.result_volume.reload()
    output = result_path()
    if output.exists():
        raise FileExistsError(f"Diagnostic run already exists: {RUN_ID}")
    output.mkdir(parents=True)
    state_path = output / "state.json"
    report_path = output / "diagnostic.json"
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "modal_input_id": modal.current_input_id()}
    previous._write_json(state_path, state)
    previous.result_volume.commit()
    try:
        _, cases = load_cases(DATASET)
        if len(cases) != 20:
            raise RuntimeError("Diagnostic case count changed")
        train_report_path = previous._output(SOURCE_TRAIN_RUN) / "train.json"
        adapter_paths = {label: previous._output(SOURCE_TRAIN_RUN) / f"adapter-{label}" for label in ("step-150", "step-300")}
        train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
        validate_train_report(train_report, previous._manifest(), expected_run_id=SOURCE_TRAIN_RUN, adapter_paths=adapter_paths)
        if sha256(train_report_path) != plan["source_train_report_sha256"]:
            raise RuntimeError("Training report hash mismatch")
        previous._set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(previous.MODEL_ID, revision=previous.MODEL_REVISION, cache_dir="/hf-cache/hub", local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = previous._load_quantized_base(use_cache=True)
        base_rows = previous._generate_development_cases(base, tokenizer, cases)
        del base
        gc.collect()
        torch.cuda.empty_cache()
        candidate_base = previous._load_quantized_base(use_cache=True)
        candidate = PeftModel.from_pretrained(candidate_base, adapter_paths["step-300"], is_trainable=False)
        candidate.config.use_cache = True
        adapter_hash = previous._adapter_state_sha256(candidate)
        if adapter_hash != plan["source_adapter_state_sha256"]:
            raise RuntimeError("Loaded adapter state changed")
        candidate_rows = previous._generate_development_cases(candidate, tokenizer, cases)
        if len(base_rows) != 20 or len(candidate_rows) != 20:
            raise RuntimeError("Incomplete diagnostic generation")
        state = {**state, "status": "complete", "finished_at": datetime.now(timezone.utc).isoformat()}
        report = {
            "protocol_id": APP_NAME, "run_id": RUN_ID, "stage": "exposed_diagnostic",
            "plan_sha256": sha256(PLAN), "dataset_sha256": sha256(DATASET),
            "source_train_report_sha256": sha256(train_report_path), "adapter_state_sha256": adapter_hash,
            "generation": plan["generation"],
            "baseline": {"summary": summarize_proposal_texts(base_rows), "cases": base_rows},
            "candidate": {"summary": summarize_proposal_texts(candidate_rows), "cases": candidate_rows},
            "state": state, "no_tool_execution": True, "final_holdout_inference": False,
        }
        previous._write_json(report_path, report)
        previous._write_json(state_path, state)
        previous.result_volume.commit()
        return {"run_id": RUN_ID, "report_path": str(report_path), "baseline": report["baseline"]["summary"], "candidate": report["candidate"]["summary"]}
    except BaseException as error:
        state = {**state, "status": "interrupted", "finished_at": datetime.now(timezone.utc).isoformat(), "error": {"type": type(error).__name__, "message": str(error)}}
        previous._write_json(state_path, state)
        previous.result_volume.commit()
        raise


@app.function(image=image, timeout=300, cpu=0.25, memory=512, volumes={"/results": previous.result_volume})
def fetch_report() -> dict[str, Any]:
    previous.result_volume.reload()
    path = result_path() / "diagnostic.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(stage: str = "preflight") -> None:
    from scripts.audit_model_proposal_framing_diagnostic_v1 import audit_dataset

    plan = json.loads(Path("evals/model_proposal_framing_diagnostic_v1_run_plan.json").read_text(encoding="utf-8"))
    validate_plan(plan, Path.cwd())
    audit_dataset()
    if stage == "preflight":
        print("Framing diagnostic preflight passed; no Modal GPU used.")
    elif stage == "run":
        print(run_diagnostic.remote())
    elif stage == "fetch":
        report = fetch_report.remote()
        destination = Path("evals/results") / RUN_ID / "sft-framing-diagnostic-v1/diagnostic.json"
        if destination.exists():
            raise FileExistsError(destination)
        previous._write_json(destination, report)
        print(f"Downloaded {destination}")
    else:
        raise ValueError("stage must be preflight, run or fetch")
