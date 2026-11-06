"""Train one QLoRA v4 adapter and compare model proposals on development v3.

The checked-in execution plan authorizes only the train and development GPU
stages. The local entry point and each remote function independently enforce
that scope; final holdout inference remains disabled.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import modal


APP_NAME = "nexus-sft-candidate-v4"
PLAN_SHA256 = "fda3272cc24491b81db102650f159e1c18fe61976d24164213da010f1b1cf4c6"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TRAIN_DATA_SHA256 = "6b30bb0cebcb253a8d0357c1a1767b38e9d8172385360d192321e038ee09c31b"
DEVELOPMENT_SHA256 = "2121c258d8acf18cf877330fa922d91f61184baf1c7679c128006a3f621bc12d"
MAX_SEQUENCE_TOKENS = 2048
MAX_STEPS = 320
MAX_NEW_TOKENS = 128
TRAIN_RUN_ID = "sft-candidate-v4-train-20261106d"
DEVELOPMENT_RUN_ID = "sft-candidate-v4-development-20261106d"
RESUME_RUN_ID = "sft-candidate-v4-train-20261104c"
RESUME_STEP = 275
RESUME_TRAINER_STATE_SHA256 = "002ae377f6721dd2e04156d73724ec91525e8d7970fad5ddbe5323a8df8b337b"
RESUME_CHECKPOINT_PATH = "sft-candidate-v4-train-20261104c/nexus-sft-candidate-v4/trainer-checkpoints/checkpoint-275"
SOURCE_V3_RUN_ID = "sft-candidate-v3-train-20261016d"
SOURCE_V3_REPORT_SHA256 = "4f7f82c75a1c394a788a0ed3c4645d17fa1f0e161f33a9391ae7b8f0e05577dd"
SOURCE_V3_ADAPTER_SHA256 = "62b5c4f7073fdca99963708bc9c7f428e486046b539371aa1d5349f0131db96e"
PLAN_PATH = Path("/repo/evals/sft_candidate_v4_execution_plan.json")
TRAIN_DATA_PATH = Path("/repo/evals/sft_candidate_v4_draft.json")
DEVELOPMENT_PATH = Path("/repo/evals/model_proposal_development_v3.json")
SUPPORT_FILES = (
    "scripts/__init__.py",
    "scripts/audit_proposal_data.py",
    "scripts/audit_sft_data.py",
    "scripts/eval_proposals.py",
    "scripts/eval_qwen.py",
    "scripts/inspect_sft_tokens.py",
    "scripts/score_sft_generation.py",
)
PACKAGED_FILES = SUPPORT_FILES + ("src", "modal_sft_candidate_v4.py")


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("nexus-hf-cache", create_if_missing=True)
result_volume = modal.Volume.from_name("nexus-eval-results", create_if_missing=True)

packages = (
    "torch==2.8.0", "transformers==4.57.1", "trl==0.27.0", "peft==0.18.0",
    "bitsandbytes==0.48.2", "accelerate==1.12.0", "datasets==4.4.1",
    "jinja2==3.1.6",
)
base_image = modal.Image.debian_slim(python_version="3.12").pip_install(*packages).env({
    "HF_HOME": "/hf-cache", "HF_HUB_CACHE": "/hf-cache/hub",
    "TOKENIZERS_PARALLELISM": "false",
})
for relative in SUPPORT_FILES:
    base_image = base_image.add_local_file(relative, "/repo/" + relative)
base_image = base_image.add_local_dir("src", "/repo/src", ignore=["**/__pycache__/**"])
base_image = base_image.add_local_file("modal_sft_candidate_v4.py", "/repo/modal_sft_candidate_v4.py")
base_image = base_image.add_local_file("evals/sft_candidate_v4_execution_plan.json", str(PLAN_PATH))
training_image = base_image.add_local_file("evals/sft_candidate_v4_draft.json", str(TRAIN_DATA_PATH))
development_image = base_image.add_local_file("evals/model_proposal_development_v3.json", str(DEVELOPMENT_PATH))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(file for file in path.rglob("*") if file.is_file()):
        name = item.relative_to(path).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def _output(run_id: str) -> Path:
    if run_id not in (TRAIN_RUN_ID, DEVELOPMENT_RUN_ID):
        raise ValueError("Unknown v4 run ID")
    return Path("/results") / run_id / APP_NAME


def _plan() -> dict[str, Any]:
    path = PLAN_PATH if PLAN_PATH.is_file() else Path("evals/sft_candidate_v4_execution_plan.json")
    if _sha256(path) != PLAN_SHA256:
        raise RuntimeError("V4 execution plan hash mismatch")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("protocol_id") != APP_NAME:
        raise RuntimeError("V4 execution protocol changed")
    return plan


def _authorize(stage: str, run_id: str) -> dict[str, Any]:
    plan = _plan()
    if stage not in ("train", "development") or run_id != plan["run_ids"][stage]:
        raise PermissionError("Stage or run ID is outside the locked v4 plan")
    if plan["authorization"].get("training_authorized") is not True:
        raise PermissionError("V4 training budget has not been authorized")
    if plan["authorization"].get("development_inference_authorized") is not True:
        raise PermissionError("V4 development inference has not been authorized")
    if plan["authorization"].get("final_holdout_inference_authorized") is not False:
        raise PermissionError("Final holdout must remain closed")
    return plan


def _claim_fresh_output(run_id: str, stage: str, modal_input_id: str) -> tuple[Path, Path, dict]:
    output = _output(run_id)
    if output.exists():
        raise FileExistsError(f"V4 run already has artifacts: {run_id}")
    output.mkdir(parents=True)
    state = {
        "stage": stage, "status": "running", "run_id": run_id,
        "modal_input_id": modal_input_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    state_path = output / "state.json"
    _write_json(state_path, state)
    result_volume.commit()
    return output, state_path, state


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _load_base(*, use_cache: bool):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir="/hf-cache/hub",
        local_files_only=True, quantization_config=config, device_map={"": 0},
        dtype=torch.float16,
    )
    model.config.use_cache = use_cache
    return model


def _load_trainable_adapter(resume_checkpoint: Path | None = None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    base = prepare_model_for_kbit_training(
        _load_base(use_cache=False), use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if resume_checkpoint is None:
        config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules="all-linear",
        )
        model = get_peft_model(base, config)
    else:
        model = PeftModel.from_pretrained(base, resume_checkpoint, is_trainable=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=torch.float32)
    model.config.use_cache = False
    return model


def _resume_checkpoint(plan: dict) -> Path:
    resume = plan.get("resume", {})
    if resume != {
        "source_run_id": RESUME_RUN_ID,
        "source_plan_sha256": "b10c8ec43b0a04e0a776d3989c606a56967cd72f10136950e9afe2bcd0a4c690",
        "checkpoint_step": RESUME_STEP,
        "checkpoint_path": RESUME_CHECKPOINT_PATH,
        "trainer_state_sha256": RESUME_TRAINER_STATE_SHA256,
    }:
        raise RuntimeError("Resume contract changed")
    checkpoint = Path("/results") / RESUME_CHECKPOINT_PATH
    required = (
        ".nexus-complete", "adapter_config.json", "adapter_model.safetensors",
        "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
    )
    if any(not (checkpoint / name).is_file() for name in required):
        raise FileNotFoundError("Resume checkpoint is incomplete")
    trainer_state_path = checkpoint / "trainer_state.json"
    if _sha256(trainer_state_path) != RESUME_TRAINER_STATE_SHA256:
        raise RuntimeError("Resume trainer state hash changed")
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if trainer_state.get("global_step") != RESUME_STEP or trainer_state.get("max_steps") != MAX_STEPS:
        raise RuntimeError("Resume step contract changed")
    return checkpoint


def _adapter_state_sha256(model: Any) -> str:
    import torch
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model)
    if not state:
        raise RuntimeError("Empty adapter state")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().to(dtype=torch.float32, device="cpu")
        digest.update(f"{name}\0{tuple(tensor.shape)}\0{tensor.dtype}\0".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _collate(features: list[dict], pad_token_id: int) -> dict:
    import torch

    length = int(math.ceil(max(len(item["input_ids"]) for item in features) / 8) * 8)
    data = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        padding = length - len(item["input_ids"])
        data["input_ids"].append(item["input_ids"] + [pad_token_id] * padding)
        data["attention_mask"].append(item["attention_mask"] + [0] * padding)
        data["labels"].append(item["labels"] + [-100] * padding)
    return {name: torch.tensor(values, dtype=torch.long) for name, values in data.items()}


def _training_rows(plan: dict) -> tuple[Any, list[dict]]:
    from scripts.audit_sft_data import materialize_case
    from scripts.inspect_sft_tokens import build_assistant_only_labels
    from scripts.eval_proposals import strict_json
    from transformers import AutoTokenizer

    if _sha256(TRAIN_DATA_PATH) != TRAIN_DATA_SHA256:
        raise RuntimeError("V4 training data hash mismatch")
    data = strict_json(TRAIN_DATA_PATH.read_bytes())
    if data.get("dataset_id") != "nexus-sft-candidate-v4-draft" or len(data["cases"]) != 640:
        raise RuntimeError("V4 training data structure mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir="/hf-cache/hub", local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    for case in data["cases"]:
        record = materialize_case(case, prompt_version=plan["generation"]["prompt_version"])
        encoded = build_assistant_only_labels(record, tokenizer)
        if len(encoded["input_ids"]) > MAX_SEQUENCE_TOKENS:
            raise RuntimeError(f"Sequence exceeds limit: {case['id']}")
        rows.append({
            "id": case["id"], "input_ids": encoded["input_ids"],
            "attention_mask": [1] * len(encoded["input_ids"]),
            "labels": encoded["labels"],
        })
    return tokenizer, rows


def _generate(model: Any, tokenizer: Any, cases: list[dict], checkpoint: Any) -> list[dict]:
    import torch
    from nexus.agent.client import build_model_request, default_model_settings
    from nexus.agent.routing import ToolRoutingMode
    from scripts.score_sft_generation import score_proposal_text

    model.eval()
    rows = []
    with torch.inference_mode():
        for case in cases:
            payload, route = build_model_request(
                case["prompt"], prompt_version="v13",
                settings=default_model_settings(model_id="qwen3-4b-instruct-2507", temperature=0.0),
                tool_routing=ToolRoutingMode.ALL,
            )
            prompt_ids = tokenizer.apply_chat_template(
                payload["messages"], tools=payload["tools"],
                add_generation_prompt=True, tokenize=True,
            )
            inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
            started = time.perf_counter()
            output = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True,
            )
            elapsed = time.perf_counter() - started
            generated_ids = output[0, inputs.shape[1]:].detach().cpu().tolist()
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            scored = score_proposal_text(case, raw_text)
            scored.update({
                "prompt_tokens": len(prompt_ids), "generated_tokens": len(generated_ids),
                "generation_seconds": elapsed, "routing": route,
            })
            rows.append(scored)
            checkpoint(rows)
            del inputs, output
    return rows


@app.function(
    image=training_image, gpu="T4", timeout=1800, cpu=2, memory=16384,
    retries=0, volumes={"/hf-cache": cache_volume, "/results": result_volume},
)
def run_train(run_id: str) -> dict:
    """Train from the frozen base; no v3 adapter is used as an initialization."""
    sys.path[:0] = ["/repo", "/repo/src"]
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    plan = _authorize("train", run_id)
    result_volume.reload()
    output, state_path, state = _claim_fresh_output(run_id, "train", modal.current_input_id())
    report_path = output / "train.json"

    class DurableCheckpoint(TrainerCallback):
        def on_save(self, args, trainer_state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{trainer_state.global_step}"
            (checkpoint / ".nexus-complete").write_text(state["modal_input_id"] + "\n", encoding="utf-8")
            result_volume.commit()
            return control

    try:
        _set_seed(42)
        tokenizer, rows = _training_rows(plan)
        resume_checkpoint = _resume_checkpoint(plan)
        torch.cuda.reset_peak_memory_stats()
        model = _load_trainable_adapter(resume_checkpoint)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable or any("lora_" not in name for name in trainable):
            raise RuntimeError("Unexpected trainable parameters")
        before_hash = _adapter_state_sha256(model)
        dataset = Dataset.from_list([{key: row[key] for key in ("input_ids", "attention_mask", "labels")} for row in rows])
        training = plan["training"]
        args = SFTConfig(
            output_dir=str(output / "trainer-checkpoints"), max_steps=MAX_STEPS,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=training["learning_rate"], optim=training["optimizer"],
            lr_scheduler_type=training["scheduler"], warmup_steps=training["warmup_steps"],
            weight_decay=training["weight_decay"], logging_steps=1,
            save_strategy="steps", save_steps=25, save_total_limit=1,
            save_only_model=False, report_to="none", fp16=False, bf16=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=MAX_SEQUENCE_TOKENS, dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False, seed=42, data_seed=42,
        )
        trainer = SFTTrainer(
            model=model, args=args, train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=lambda features: _collate(features, tokenizer.pad_token_id),
            callbacks=[DurableCheckpoint()],
        )
        result = trainer.train(resume_from_checkpoint=str(resume_checkpoint))
        adapter_path = output / "adapter-step-320"
        trainer.save_model(str(adapter_path))
        tokenizer.save_pretrained(adapter_path)
        after_hash = _adapter_state_sha256(model)
        history = [
            {key: row.get(key) for key in ("step", "loss", "grad_norm", "learning_rate")}
            for row in trainer.state.log_history if "loss" in row
        ]
        completed_steps = int(trainer.state.global_step)
        if completed_steps != MAX_STEPS or before_hash == after_hash or not math.isfinite(float(result.training_loss)):
            raise RuntimeError("Training completion or adapter update gate failed")
        del trainer, model, dataset
        gc.collect()
        torch.cuda.empty_cache()
        reloaded_base = _load_base(use_cache=False)
        reloaded = PeftModel.from_pretrained(reloaded_base, adapter_path, is_trainable=False)
        reloaded_hash = _adapter_state_sha256(reloaded)
        if reloaded_hash != after_hash:
            raise RuntimeError("Saved adapter differs after reload")
        del reloaded, reloaded_base
        gc.collect()
        torch.cuda.empty_cache()
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        report = {
            "protocol_id": APP_NAME, "run_id": run_id, "stage": "train",
            "plan_sha256": PLAN_SHA256, "dataset_sha256": TRAIN_DATA_SHA256,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "training": {"completed_steps": completed_steps, "loss": float(result.training_loss), "history": history},
            "resume": {"source_run_id": RESUME_RUN_ID, "checkpoint_step": RESUME_STEP,
                       "trainer_state_sha256": RESUME_TRAINER_STATE_SHA256},
            "adapter": {"path": str(adapter_path), "directory_sha256": _directory_sha256(adapter_path),
                        "state_sha256_before": before_hash, "state_sha256_after": after_hash,
                        "state_sha256_reloaded": reloaded_hash},
            "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved()},
            "state": state, "final_holdout_inference": False,
        }
        _write_json(report_path, report)
        _write_json(state_path, state)
        result_volume.commit()
        return {"run_id": run_id, "report_path": str(report_path), "completed_steps": completed_steps}
    except BaseException as error:
        state.update(status="interrupted", finished_at=datetime.now(timezone.utc).isoformat(),
                     error={"type": type(error).__name__, "message": str(error)})
        _write_json(state_path, state)
        result_volume.commit()
        raise


@app.function(
    image=development_image, gpu="T4", timeout=800, cpu=2, memory=16384,
    retries=0, volumes={"/hf-cache": cache_volume, "/results": result_volume},
)
def run_development(run_id: str) -> dict:
    """Compare base, frozen v3 step 300 and new v4 step 320 without side effects."""
    sys.path[:0] = ["/repo", "/repo/src"]
    import torch
    from peft import PeftModel
    from scripts.eval_proposals import load_cases
    from scripts.score_sft_generation import summarize_proposal_texts
    from transformers import AutoTokenizer

    plan = _authorize("development", run_id)
    result_volume.reload()
    candidate_report_path = _output(TRAIN_RUN_ID) / "train.json"
    old_report_path = Path("/results") / SOURCE_V3_RUN_ID / "sft-candidate-v3" / "train.json"
    if not candidate_report_path.is_file() or not old_report_path.is_file():
        raise FileNotFoundError("Required completed train report is missing")
    if _sha256(old_report_path) != SOURCE_V3_REPORT_SHA256:
        raise RuntimeError("V3 baseline train report hash changed")
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if (candidate_report.get("plan_sha256") != PLAN_SHA256
            or candidate_report.get("dataset_sha256") != TRAIN_DATA_SHA256
            or candidate_report.get("state", {}).get("status") != "complete"
            or candidate_report.get("training", {}).get("completed_steps") != MAX_STEPS):
        raise RuntimeError("V4 candidate train report is invalid")
    candidate_adapter = _output(TRAIN_RUN_ID) / "adapter-step-320"
    old_adapter = Path("/results") / SOURCE_V3_RUN_ID / "sft-candidate-v3" / "adapter-step-300"
    if _directory_sha256(candidate_adapter) != candidate_report["adapter"]["directory_sha256"]:
        raise RuntimeError("V4 adapter directory hash changed")
    if not old_adapter.is_dir():
        raise FileNotFoundError("V3 adapter is missing")
    if _sha256(DEVELOPMENT_PATH) != DEVELOPMENT_SHA256:
        raise RuntimeError("Development dataset hash mismatch")
    _, cases = load_cases(DEVELOPMENT_PATH)
    if len(cases) != 40:
        raise RuntimeError("Development case count changed")
    output, state_path, state = _claim_fresh_output(run_id, "development", modal.current_input_id())
    report_path = output / "development.json"
    try:
        _set_seed(plan["generation"]["seed"])
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
            cache_dir="/hf-cache/hub", local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch.cuda.reset_peak_memory_stats()
        results = {}
        for name, adapter_path, expected_hash in (
            ("base", None, None),
            ("v3-step-300", old_adapter, SOURCE_V3_ADAPTER_SHA256),
            ("v4-step-320", candidate_adapter, candidate_report["adapter"]["state_sha256_after"]),
        ):
            base = _load_base(use_cache=True)
            model = base if adapter_path is None else PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
            model.config.use_cache = True
            if expected_hash is not None and _adapter_state_sha256(model) != expected_hash:
                raise RuntimeError(f"{name} adapter state hash changed")
            partial = output / f"partial-{name}.json"
            rows = _generate(model, tokenizer, cases, lambda value, path=partial: (
                _write_json(path, value), result_volume.commit(),
            ))
            results[name] = {"summary": summarize_proposal_texts(rows), "cases": rows}
            del model, base
            gc.collect()
            torch.cuda.empty_cache()
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        report = {
            "protocol_id": APP_NAME, "run_id": run_id, "stage": "development",
            "plan_sha256": PLAN_SHA256, "dataset_sha256": DEVELOPMENT_SHA256,
            "candidate_train_report_sha256": _sha256(candidate_report_path),
            "source_v3_train_report_sha256": SOURCE_V3_REPORT_SHA256,
            "generation": plan["generation"], "models": results,
            "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved()},
            "state": state, "no_tool_execution": True, "final_holdout_inference": False,
        }
        _write_json(report_path, report)
        _write_json(state_path, state)
        result_volume.commit()
        return {"run_id": run_id, "report_path": str(report_path),
                "scores": {name: value["summary"] for name, value in results.items()}}
    except BaseException as error:
        state.update(status="interrupted", finished_at=datetime.now(timezone.utc).isoformat(),
                     error={"type": type(error).__name__, "message": str(error)})
        _write_json(state_path, state)
        result_volume.commit()
        raise


@app.local_entrypoint()
def main(stage: str = "preflight", run_id: str = "") -> None:
    from scripts.audit_sft_candidate_v4_execution import validate_protocol

    summary = validate_protocol()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if stage == "preflight":
        print("Offline preflight passed. No Modal function or GPU was called.")
        return
    if stage not in ("train", "development"):
        raise ValueError("stage must be preflight, train or development")
    chosen = run_id or (TRAIN_RUN_ID if stage == "train" else DEVELOPMENT_RUN_ID)
    _authorize(stage, chosen)
    call = run_train.spawn(chosen) if stage == "train" else run_development.spawn(chosen)
    print(json.dumps({
        "stage": stage,
        "run_id": chosen,
        "function_call_id": call.object_id,
        "submission": "asynchronous",
    }, ensure_ascii=False, indent=2))

