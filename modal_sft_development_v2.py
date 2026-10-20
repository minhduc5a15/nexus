"""Compare the frozen QLoRA step-300 adapter with its base on development v2."""

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
import modal_sft_candidate_v3 as previous


APP_NAME = "nexus-sft-development-v2"
PLAN = Path("/repo/evals/model_proposal_development_v2_run_plan.json")
DATASET = Path("/repo/evals/model_proposal_development_v2.json")
SOURCE_TRAIN_RUN = "sft-candidate-v3-train-20261016d"

app = modal.App(APP_NAME)
image = (
    previous.development_image
    .add_local_file("evals/model_proposal_development_v2.json", str(DATASET))
    .add_local_file("evals/model_proposal_development_v2_run_plan.json", str(PLAN))
    .add_local_file("modal_sft_development_v2.py", "/repo/modal_sft_development_v2.py")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plan(plan: dict[str, Any], root: Path) -> None:
    if plan.get("protocol_id") != "nexus-sft-development-v2":
        raise ValueError("Unexpected development-v2 protocol")
    if plan.get("final_holdout_inference_authorized") is not False:
        raise ValueError("Final holdout must remain closed")
    if plan.get("training_authorized") is not False:
        raise ValueError("This runner must not authorize training")
    for name, spec in plan["files"].items():
        path = root / name
        if not path.is_file() or sha256(path) != spec:
            raise ValueError(f"Locked file changed or missing: {name}")
    if plan["generation"] != {
        "prompt_version": "v13",
        "tool_routing": "all",
        "temperature": 0.0,
        "do_sample": False,
        "seed": 42,
        "maximum_new_tokens": 128,
    }:
        raise ValueError("Generation settings changed")
    if plan["source_train_run_id"] != SOURCE_TRAIN_RUN:
        raise ValueError("Source training run changed")
    if plan["model"] != {"id": previous.MODEL_ID, "revision": previous.MODEL_REVISION}:
        raise ValueError("Base model revision changed")
    if plan["source_adapter_state_sha256"] != "62b5c4f7073fdca99963708bc9c7f428e486046b539371aa1d5349f0131db96e":
        raise ValueError("Selected adapter changed")
    if plan["resources"] != {
        "gpu": "T4", "cpu": 2, "memory_mib": 16384,
        "timeout_seconds": 1800, "automatic_retry": False,
    }:
        raise ValueError("Resource limits changed")
    if plan["dataset"]["count"] != 80 or plan["dataset"]["sha256"] != sha256(root / "evals/model_proposal_development_v2.json"):
        raise ValueError("Development-v2 dataset changed")


def result_path(run_id: str) -> Path:
    return Path("/results") / previous._safe_run_id(run_id) / "sft-development-v2"


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    cpu=2,
    memory=16384,
    volumes={"/hf-cache": previous.cache_volume, "/results": previous.result_volume},
)
def run_development(run_id: str) -> dict[str, Any]:
    """Read frozen artifacts and generate proposals; never execute a task tool."""
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

    run_id = previous._safe_run_id(run_id)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    validate_plan(plan, Path("/repo"))
    if hashlib.sha256(SYSTEM_PROMPTS["v13"].encode()).hexdigest() != plan["prompt_sha256"]:
        raise RuntimeError("Prompt v13 changed")
    if canonical_sha256(TOOL_DEFINITIONS) != plan["tools_sha256"]:
        raise RuntimeError("Tool schema changed")
    previous.result_volume.reload()
    output = result_path(run_id)
    if output.exists():
        raise FileExistsError(f"Run ID already has artifacts: {run_id}")
    output.mkdir(parents=True)
    state_path = output / "state.json"
    report_path = output / "development.json"
    state = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "modal_input_id": modal.current_input_id(),
    }
    previous._write_json(state_path, state)
    previous.result_volume.commit()
    try:
        _, cases = load_cases(DATASET)
        if len(cases) != 80:
            raise RuntimeError("Development-v2 count changed")
        old_manifest = previous._manifest()
        train_report_path = previous._output(SOURCE_TRAIN_RUN) / "train.json"
        adapter_paths = {
            label: previous._output(SOURCE_TRAIN_RUN) / f"adapter-{label}"
            for label in ("step-150", "step-300")
        }
        train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
        validate_train_report(
            train_report,
            old_manifest,
            expected_run_id=SOURCE_TRAIN_RUN,
            adapter_paths=adapter_paths,
        )
        if sha256(train_report_path) != plan["source_train_report_sha256"]:
            raise RuntimeError("Source train report changed")
        previous._set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(
            previous.MODEL_ID,
            revision=previous.MODEL_REVISION,
            cache_dir="/hf-cache/hub",
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch.cuda.reset_peak_memory_stats()

        base = previous._load_quantized_base(use_cache=True)
        base_rows = previous._generate_development_cases(
            base, tokenizer, cases,
            checkpoint=lambda rows: (
                previous._write_json(output / "partial-base.json", rows),
                previous.result_volume.commit(),
            ),
        )
        del base
        gc.collect()
        torch.cuda.empty_cache()

        candidate_base = previous._load_quantized_base(use_cache=True)
        candidate = PeftModel.from_pretrained(
            candidate_base, adapter_paths["step-300"], is_trainable=False
        )
        candidate.config.use_cache = True
        adapter_state_sha256 = previous._adapter_state_sha256(candidate)
        if adapter_state_sha256 != plan["source_adapter_state_sha256"]:
            raise RuntimeError("Loaded adapter state changed")
        candidate_rows = previous._generate_development_cases(
            candidate, tokenizer, cases,
            checkpoint=lambda rows: (
                previous._write_json(output / "partial-step-300.json", rows),
                previous.result_volume.commit(),
            ),
        )
        if len(base_rows) != 80 or len(candidate_rows) != 80:
            raise RuntimeError("Development generation is incomplete")
        state = {**state, "status": "complete", "finished_at": datetime.now(timezone.utc).isoformat()}
        report = {
            "protocol_id": plan["protocol_id"],
            "run_id": run_id,
            "stage": "development",
            "plan_sha256": sha256(PLAN),
            "dataset_sha256": sha256(DATASET),
            "source_train_report_sha256": sha256(train_report_path),
            "adapter_state_sha256": adapter_state_sha256,
            "generation": plan["generation"],
            "baseline": {"summary": summarize_proposal_texts(base_rows), "cases": base_rows},
            "candidate": {"summary": summarize_proposal_texts(candidate_rows), "cases": candidate_rows},
            "memory": {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "state": state,
            "no_tool_execution": True,
            "final_holdout_inference": False,
        }
        previous._write_json(report_path, report)
        previous._write_json(state_path, state)
        previous.result_volume.commit()
        return {"run_id": run_id, "report_path": str(report_path),
                "baseline": report["baseline"]["summary"],
                "candidate": report["candidate"]["summary"]}
    except BaseException as error:
        state = {**state, "status": "interrupted", "finished_at": datetime.now(timezone.utc).isoformat(),
                 "error": {"type": type(error).__name__, "message": str(error)}}
        previous._write_json(state_path, state)
        previous.result_volume.commit()
        raise


@app.function(
    image=image,
    timeout=300,
    cpu=0.25,
    memory=512,
    volumes={"/results": previous.result_volume},
)
def fetch_report(run_id: str) -> dict[str, Any]:
    previous.result_volume.reload()
    path = result_path(run_id) / "development.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(stage: str = "preflight", run_id: str = "") -> None:
    from scripts.audit_model_proposal_development_v2 import audit_dataset

    plan = json.loads((Path("evals/model_proposal_development_v2_run_plan.json")).read_text(encoding="utf-8"))
    validate_plan(plan, Path.cwd())
    audit_dataset()
    if stage == "preflight":
        print("Development v2 preflight passed; no Modal GPU used.")
    elif stage == "run":
        if not run_id:
            raise ValueError("run_id is required")
        print(run_development.remote(run_id))
    elif stage == "fetch":
        if not run_id:
            raise ValueError("run_id is required")
        report = fetch_report.remote(run_id)
        destination = Path("evals/results") / run_id / "sft-development-v2/development.json"
        if destination.exists():
            raise FileExistsError(destination)
        previous._write_json(destination, report)
        print(f"Downloaded {destination}")
    else:
        raise ValueError("stage must be preflight, run or fetch")
