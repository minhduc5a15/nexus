"""Run the locked 16-example QLoRA sanity experiment on Modal.

The default entrypoint is a local preflight. GPU work is split into an explicit
gradient probe and an overfit stage that refuses to run without a passing probe.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

import modal

APP_NAME = "nexus-sft-sanity"
PROTOCOL_ID = "nexus-sft-qlora-sanity-v1"
GENERATION_PROTOCOL_ID = "nexus-sft-generation-sanity-v1"
GENERATION_MANIFEST_SHA256 = "2fffaf1aeda2223b6ca2275ce03a903a74346ba0bea847de3c65f626c5f1f300"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_SHA256 = "79075d3d1acaece69a56506f6168926b22c6c6ae6e265eb4d0ecca0c85f12e67"
MAX_SEQUENCE_TOKENS = 2048
MAX_STEPS = 64
GENERATION_MAX_NEW_TOKENS = 128
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("nexus-hf-cache", create_if_missing=True)
result_volume = modal.Volume.from_name("nexus-eval-results", create_if_missing=True)

protocol_files = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub==0.36.0")
    .add_local_dir("src", "/repo/src", ignore=["**/__pycache__/**"])
    .add_local_dir("scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
    .add_local_file("evals/sft_sanity_v1.json", "/repo/evals/sft_sanity_v1.json")
    .add_local_file(
        "evals/sft_sanity_manifest_v1.json", "/repo/evals/sft_sanity_manifest_v1.json"
    )
    .add_local_file(
        "evals/sft_sanity_training_manifest_v1.json",
        "/repo/evals/sft_sanity_training_manifest_v1.json",
    )
    .add_local_file(
        "evals/sft_generation_manifest_v1.json",
        "/repo/evals/sft_generation_manifest_v1.json",
    )
    .add_local_file("modal_sft_sanity.py", "/repo/modal_sft_sanity.py")
)

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.1",
        "trl==0.27.0",
        "peft==0.18.0",
        "bitsandbytes==0.48.2",
        "accelerate==1.12.0",
        "datasets==4.4.1",
        "jinja2==3.1.6",
    )
    .env(
        {
            "HF_HOME": "/hf-cache",
            "HF_HUB_CACHE": "/hf-cache/hub",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir("src", "/repo/src", ignore=["**/__pycache__/**"])
    .add_local_dir("scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
    .add_local_file("evals/sft_sanity_v1.json", "/repo/evals/sft_sanity_v1.json")
    .add_local_file(
        "evals/sft_sanity_manifest_v1.json", "/repo/evals/sft_sanity_manifest_v1.json"
    )
    .add_local_file(
        "evals/sft_sanity_training_manifest_v1.json",
        "/repo/evals/sft_sanity_training_manifest_v1.json",
    )
    .add_local_file(
        "evals/sft_generation_manifest_v1.json",
        "/repo/evals/sft_generation_manifest_v1.json",
    )
    .add_local_file("modal_sft_sanity.py", "/repo/modal_sft_sanity.py")
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or not RUN_ID_PATTERN.fullmatch(run_id)
        or run_id in {".", ".."}
    ):
        raise ValueError("run_id must contain 1-128 safe filename characters")
    return run_id


def _output(run_id: str) -> Path:
    return Path("/results") / _safe_run_id(run_id) / "sft-sanity"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
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
        metadata = f"{name}\0{tuple(tensor.shape)}\0{tensor.dtype}\0".encode("utf-8")
        digest.update(metadata)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    sys.path[:0] = ["/repo", "/repo/src"]
    from scripts.sft_sanity_protocol import directory_sha256

    return directory_sha256(path)


def _versions() -> dict[str, str]:
    from importlib.metadata import version

    return {
        package: version(package)
        for package in (
            "torch",
            "transformers",
            "trl",
            "peft",
            "bitsandbytes",
            "accelerate",
            "datasets",
        )
    }


def _remote_context() -> dict[str, Any]:
    sys.path[:0] = ["/repo", "/repo/src"]
    from scripts.audit_sft_data import materialize_case
    from scripts.eval_proposals import strict_json
    from scripts.inspect_sft_tokens import build_assistant_only_labels
    from scripts.sft_sanity_protocol import load_json

    dataset_path = Path("/repo/evals/sft_sanity_v1.json")
    if _file_sha256(dataset_path) != DATASET_SHA256:
        raise RuntimeError("Frozen SFT dataset hash mismatch")
    document = strict_json(dataset_path.read_bytes())
    manifest = load_json(Path("/repo/evals/sft_sanity_training_manifest_v1.json"))
    if (
        document.get("dataset_id") != "nexus-sft-sanity-v1"
        or len(document.get("cases", [])) != 16
    ):
        raise RuntimeError("Frozen SFT dataset structure mismatch")
    if document.get("base_model") != manifest.get("model"):
        raise RuntimeError("Frozen SFT model lock mismatch")
    return {
        "document": document,
        "manifest": manifest,
        "materialize_case": materialize_case,
        "build_labels": build_assistant_only_labels,
    }


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _load_tokenizer_and_rows(context: dict[str, Any]):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir="/hf-cache/hub",
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    for case in context["document"]["cases"]:
        record = context["materialize_case"](case, prompt_version="v13")
        encoded = context["build_labels"](record, tokenizer)
        if len(encoded["input_ids"]) > MAX_SEQUENCE_TOKENS:
            raise RuntimeError(f"{case['id']} exceeds {MAX_SEQUENCE_TOKENS} tokens")
        rows.append(
            {
                "id": case["id"],
                "input_ids": encoded["input_ids"],
                "attention_mask": [1] * len(encoded["input_ids"]),
                "labels": encoded["labels"],
                "prompt_tokens": encoded["prompt_tokens"],
                "supervised_tokens": encoded["supervised_tokens"],
            }
        )
    return tokenizer, rows


def _load_quantized_base(*, use_cache: bool):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir="/hf-cache/hub",
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.float16,
    )
    base.config.use_cache = use_cache
    return base


def _load_qlora_model():
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base = _load_quantized_base(use_cache=False)
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(base, config)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=torch.float32)
    model.config.use_cache = False
    return model


def _through_eos(token_ids: list[int], eos_token_id: int | list[int] | None) -> list[int]:
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    eos_ids.discard(None)
    for index, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            return token_ids[: index + 1]
    return token_ids


def _generate_sanity_cases(
    model: Any,
    tokenizer: Any,
    context: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import time
    import torch
    from scripts.score_sft_generation import score_case

    model.eval()
    results = []
    eos_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        for case, row in zip(context["document"]["cases"], rows, strict=True):
            record = context["materialize_case"](case, prompt_version="v13")
            prompt_ids = tokenizer.apply_chat_template(
                record["messages"][:-1],
                tools=record["tools"],
                add_generation_prompt=True,
                tokenize=True,
            )
            expected = _through_eos(
                row["input_ids"][row["prompt_tokens"] :], eos_token_id
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
            attention_mask = torch.ones_like(input_ids)
            started = time.perf_counter()
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
            )
            elapsed = time.perf_counter() - started
            generated = _through_eos(
                output[0, input_ids.shape[1] :].detach().cpu().tolist(), eos_token_id
            )
            text = tokenizer.decode(generated, skip_special_tokens=False)
            scored = score_case(
                case, text, exact_token_sequence=generated == expected
            )
            scored.update({
                "prompt_tokens": len(prompt_ids),
                "expected_tokens": len(expected),
                "generated_tokens": len(generated),
                "generation_seconds": elapsed,
            })
            results.append(scored)
            del input_ids, attention_mask, output
    return results


def _parameter_summary(model: Any) -> dict[str, Any]:
    total = trainable = 0
    trainable_names = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_fraction": trainable / total,
        "trainable_tensor_count": len(trainable_names),
        "only_lora_trainable": bool(trainable_names)
        and all("lora_" in name for name in trainable_names),
        "trainable_dtypes": sorted(
            {
                str(parameter.dtype)
                for parameter in model.parameters()
                if parameter.requires_grad
            }
        ),
        "trainable_name_sample": trainable_names[:12],
    }


def _tensor_row(row: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        key: torch.tensor([row[key]], dtype=torch.long, device="cuda")
        for key in ("input_ids", "attention_mask", "labels")
    }


def _collate(features: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    maximum = max(len(item["input_ids"]) for item in features)
    maximum = int(math.ceil(maximum / 8) * 8)
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        padding = maximum - len(item["input_ids"])
        result["input_ids"].append(item["input_ids"] + [pad_token_id] * padding)
        result["attention_mask"].append(item["attention_mask"] + [0] * padding)
        result["labels"].append(item["labels"] + [-100] * padding)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def _teacher_forced_metrics(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    supervised = correct = exact = 0
    case_metrics = []
    with torch.no_grad():
        for row in rows:
            batch = _tensor_row(row)
            output = model(**batch)
            loss = float(output.loss.detach().cpu())
            labels = batch["labels"][:, 1:]
            predictions = output.logits[:, :-1].argmax(dim=-1)
            mask = labels != -100
            count = int(mask.sum().item())
            hits = int((predictions[mask] == labels[mask]).sum().item())
            is_exact = hits == count
            weighted_loss += loss * count
            supervised += count
            correct += hits
            exact += int(is_exact)
            case_metrics.append(
                {
                    "id": row["id"],
                    "loss": loss,
                    "supervised_tokens": count,
                    "token_accuracy": hits / count,
                    "exact_teacher_forced_sequence": is_exact,
                }
            )
            del batch, output, labels, predictions, mask
    if was_training:
        model.train()
    return {
        "mean_loss": weighted_loss / supervised,
        "token_accuracy": correct / supervised,
        "exact_sequences": exact,
        "cases": len(rows),
        "supervised_tokens": supervised,
        "case_metrics": case_metrics,
    }


def _provenance() -> dict[str, Any]:
    import torch

    return {
        "packages": _versions(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "cuda": torch.version.cuda,
        "dataset_sha256": DATASET_SHA256,
        "runner_sha256": _file_sha256(Path("/repo/modal_sft_sanity.py")),
        "training_manifest_sha256": _file_sha256(
            Path("/repo/evals/sft_sanity_training_manifest_v1.json")
        ),
        "final_holdout_inference": False,
    }


@app.function(
    image=protocol_files,
    volumes={"/hf-cache": cache_volume},
    timeout=3600,
    cpu=2,
    memory=4096,
)
def ensure_model() -> dict[str, Any]:
    """Download the locked base checkpoint without paying for a GPU."""

    from huggingface_hub import snapshot_download

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
    files = sorted(
        candidate for candidate in Path(path).rglob("*") if candidate.is_file()
    )
    return {
        "path": path,
        "revision": MODEL_REVISION,
        "files": len(files),
        "bytes": sum(candidate.stat().st_size for candidate in files),
        "gpu_used": False,
    }


@app.function(
    image=training_image,
    gpu="T4",
    timeout=1200,
    volumes={"/hf-cache": cache_volume, "/results": result_volume},
)
def run_probe(run_id: str) -> dict[str, Any]:
    """Run one backward/optimizer step and prove which weights changed."""

    import torch
    from bitsandbytes.optim import PagedAdamW8bit

    run_id = _safe_run_id(run_id)
    output = _output(run_id)
    report_path = output / "probe.json"
    state_path = output / "probe-state.json"
    if report_path.exists() or state_path.exists():
        raise FileExistsError(f"Refusing to overwrite probe artifacts for {run_id}")
    output.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    _write_json(state_path, state)
    try:
        context = _remote_context()
        _set_seed(context["manifest"]["training"]["seed"])
        tokenizer, rows = _load_tokenizer_and_rows(context)
        row = max(rows, key=lambda item: len(item["input_ids"]))
        torch.cuda.reset_peak_memory_stats()
        model = _load_qlora_model()
        parameters = _parameter_summary(model)
        before = _adapter_state_sha256(model)
        optimizer = PagedAdamW8bit(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=context["manifest"]["training"]["learning_rate"],
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**_tensor_row(row)).loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite probe loss: {loss}")
        loss.backward()
        adapter_gradients = []
        base_gradient_tensors = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                norm = float(parameter.grad.detach().float().norm().cpu())
                adapter_gradients.append((name, norm))
            elif not parameter.requires_grad and parameter.grad is not None:
                base_gradient_tensors += 1
        nonzero = [
            (name, norm)
            for name, norm in adapter_gradients
            if norm > 0 and math.isfinite(norm)
        ]
        optimizer.step()
        after = _adapter_state_sha256(model)
        adapter_dir = output / "probe-adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        adapter_artifact_hash = _directory_sha256(adapter_dir)
        gates = {
            "base_parameters_frozen": parameters["frozen"] > 0,
            "only_lora_parameters_trainable": parameters["only_lora_trainable"],
            "finite_assistant_only_loss": math.isfinite(float(loss.detach().cpu())),
            "nonzero_adapter_gradient": bool(nonzero),
            "zero_base_gradient": base_gradient_tensors == 0,
            "adapter_hash_changes_after_step": before != after,
            "adapter_artifact_saved": bool(adapter_artifact_hash),
        }
        failed = [name for name, passed in gates.items() if not passed]
        if failed:
            raise RuntimeError(f"Probe gates failed: {failed}")
        state.update(
            status="complete", finished_at=datetime.now(timezone.utc).isoformat()
        )
        report = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "stage": "probe",
            "model": context["manifest"]["model"],
            "dataset_sha256": DATASET_SHA256,
            "configuration": {
                key: context["manifest"][key]
                for key in ("runtime", "quantization", "adapter", "training")
            },
            "provenance": _provenance(),
            "parameters": parameters,
            "gradient_probe": {
                "case_id": row["id"],
                "sequence_tokens": len(row["input_ids"]),
                "supervised_tokens": row["supervised_tokens"],
                "loss": float(loss.detach().cpu()),
                "adapter_gradient_tensors": len(adapter_gradients),
                "nonzero_adapter_gradient_tensors": len(nonzero),
                "gradient_norm_sum": sum(norm for _, norm in nonzero),
                "base_gradient_tensors": base_gradient_tensors,
            },
            "adapter": {
                "state_sha256_before_step": before,
                "state_sha256_after_step": after,
                "path": str(adapter_dir),
                "directory_sha256": adapter_artifact_hash,
            },
            "memory": {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "gates": gates,
            "state": state,
        }
        _write_json(report_path, report)
        _write_json(state_path, state)
        result_volume.commit()
        return {"report": str(report_path), "gates": gates, "state": state}
    except BaseException as error:
        state.update(
            status="interrupted",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        _write_json(state_path, state)
        result_volume.commit()
        raise


@app.function(
    image=training_image,
    gpu="T4",
    timeout=2400,
    volumes={"/hf-cache": cache_volume, "/results": result_volume},
)
def run_overfit(run_id: str, probe_run_id: str) -> dict[str, Any]:
    """Train 64 steps only after a passing probe, then save and reload LoRA."""

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    run_id = _safe_run_id(run_id)
    probe_run_id = _safe_run_id(probe_run_id)
    output = _output(run_id)
    report_path = output / "overfit.json"
    state_path = output / "overfit-state.json"
    if report_path.exists() or state_path.exists():
        raise FileExistsError(f"Refusing to overwrite overfit artifacts for {run_id}")
    output.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    _write_json(state_path, state)
    try:
        context = _remote_context()
        manifest = context["manifest"]
        result_volume.reload()
        probe_path = _output(probe_run_id) / "probe.json"
        if not probe_path.is_file():
            raise FileNotFoundError(f"Required probe report not found: {probe_path}")
        probe_report = json.loads(probe_path.read_text(encoding="utf-8"))
        from scripts.sft_sanity_protocol import validate_probe_gate

        approved_probe = validate_probe_gate(
            probe_report, manifest, expected_run_id=probe_run_id
        )
        _set_seed(manifest["training"]["seed"])
        tokenizer, rows = _load_tokenizer_and_rows(context)
        torch.cuda.reset_peak_memory_stats()
        model = _load_qlora_model()
        parameters = _parameter_summary(model)
        initial_adapter_hash = _adapter_state_sha256(model)
        before = _teacher_forced_metrics(model, rows)

        training_rows = [
            {
                key: value
                for key, value in row.items()
                if key in {"input_ids", "attention_mask", "labels"}
            }
            for row in rows
        ]
        dataset = Dataset.from_list(training_rows)
        args = SFTConfig(
            output_dir=f"/tmp/nexus-sft-{run_id}",
            max_steps=MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=manifest["training"]["learning_rate"],
            optim="paged_adamw_8bit",
            lr_scheduler_type="constant",
            warmup_steps=0,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            fp16=False,
            bf16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=MAX_SEQUENCE_TOKENS,
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            seed=manifest["training"]["seed"],
            data_seed=manifest["training"]["seed"],
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=lambda features: _collate(features, tokenizer.pad_token_id),
        )
        train_result = trainer.train()
        after = _teacher_forced_metrics(model, rows)
        final_adapter_hash = _adapter_state_sha256(model)
        adapter_dir = output / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)
        artifact_hash = _directory_sha256(adapter_dir)
        loss_history = [
            {
                "step": item.get("step"),
                "loss": item["loss"],
                "grad_norm": item.get("grad_norm"),
            }
            for item in trainer.state.log_history
            if "loss" in item
        ]
        del trainer, model, dataset
        gc.collect()
        torch.cuda.empty_cache()

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        reloaded_base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir="/hf-cache/hub",
            local_files_only=True,
            quantization_config=quantization,
            device_map={"": 0},
            dtype=torch.float16,
        )
        reloaded = PeftModel.from_pretrained(
            reloaded_base, adapter_dir, is_trainable=False
        )
        reloaded_hash = _adapter_state_sha256(reloaded)
        gates = {
            "completed_probe_matches_protocol": bool(approved_probe),
            "finite_training_loss": math.isfinite(float(train_result.training_loss)),
            "mean_teacher_forced_loss_decreases": after["mean_loss"]
            < before["mean_loss"],
            "adapter_hash_changes": initial_adapter_hash != final_adapter_hash,
            "adapter_reloads_with_identical_weights": final_adapter_hash
            == reloaded_hash,
        }
        failed = [name for name, passed in gates.items() if not passed]
        if failed:
            raise RuntimeError(f"Overfit gates failed: {failed}")
        state.update(
            status="complete", finished_at=datetime.now(timezone.utc).isoformat()
        )
        report = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "stage": "overfit",
            "model": manifest["model"],
            "dataset_sha256": DATASET_SHA256,
            "approved_probe": approved_probe,
            "provenance": _provenance(),
            "parameters": parameters,
            "training": {
                "maximum_steps": MAX_STEPS,
                "completed_steps": trainer_state_step(loss_history),
                "training_loss": float(train_result.training_loss),
                "runtime_seconds": float(
                    train_result.metrics.get("train_runtime", 0.0)
                ),
                "loss_history": loss_history,
            },
            "teacher_forced_before": before,
            "teacher_forced_after": after,
            "adapter": {
                "state_sha256_before": initial_adapter_hash,
                "state_sha256_after": final_adapter_hash,
                "state_sha256_reloaded": reloaded_hash,
                "path": str(adapter_dir),
                "directory_sha256": artifact_hash,
            },
            "memory": {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "gates": gates,
            "state": state,
        }
        _write_json(report_path, report)
        _write_json(state_path, state)
        result_volume.commit()
        return {"report": str(report_path), "gates": gates, "state": state}
    except BaseException as error:
        state.update(
            status="interrupted",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        _write_json(state_path, state)
        result_volume.commit()
        raise


@app.function(
    image=training_image,
    gpu="T4",
    timeout=1200,
    volumes={"/hf-cache": cache_volume, "/results": result_volume},
)
def run_generation(run_id: str, overfit_run_id: str) -> dict[str, Any]:
    """Compare base and reloaded-adapter free generation on training records."""

    sys.path[:0] = ["/repo", "/repo/src"]
    import torch
    from peft import PeftModel
    from scripts.score_sft_generation import summarize
    from scripts.sft_sanity_protocol import validate_overfit_report

    run_id = _safe_run_id(run_id)
    overfit_run_id = _safe_run_id(overfit_run_id)
    output = _output(run_id)
    report_path = output / "generation.json"
    state_path = output / "generation-state.json"
    if report_path.exists() or state_path.exists():
        raise FileExistsError(f"Refusing to overwrite generation artifacts for {run_id}")
    output.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    _write_json(state_path, state)
    try:
        context = _remote_context()
        generation_manifest_path = Path("/repo/evals/sft_generation_manifest_v1.json")
        if _file_sha256(generation_manifest_path) != GENERATION_MANIFEST_SHA256:
            raise RuntimeError("Frozen generation manifest hash mismatch")
        generation_manifest = json.loads(
            generation_manifest_path.read_text(encoding="utf-8")
        )
        source = generation_manifest["source_overfit"]
        if overfit_run_id != source["run_id"]:
            raise RuntimeError("Generation must use the locked overfit run")
        result_volume.reload()
        source_report_path = Path("/results") / source["report_path"]
        adapter_dir = Path("/results") / source["adapter_path"]
        if not source_report_path.is_file() or not adapter_dir.is_dir():
            raise FileNotFoundError("Locked overfit report or adapter is missing")
        if _file_sha256(source_report_path) != source["report_sha256"]:
            raise RuntimeError("Locked overfit report hash mismatch")
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        approved_overfit = validate_overfit_report(
            source_report,
            context["manifest"],
            expected_run_id=source["run_id"],
            expected_probe_run_id=source["probe_run_id"],
        )
        adapter_directory_hash = _directory_sha256(adapter_dir)
        if adapter_directory_hash != source["adapter_directory_sha256"]:
            raise RuntimeError("Locked adapter directory hash mismatch")

        _set_seed(generation_manifest["generation"]["seed"])
        tokenizer, rows = _load_tokenizer_and_rows(context)
        torch.cuda.reset_peak_memory_stats()
        base = _load_quantized_base(use_cache=True)
        baseline_cases = _generate_sanity_cases(base, tokenizer, context, rows)
        baseline_summary = summarize(baseline_cases)

        candidate = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        candidate.config.use_cache = True
        adapter_state_hash = _adapter_state_sha256(candidate)
        if adapter_state_hash != source["adapter_state_sha256"]:
            raise RuntimeError("Reloaded adapter state hash mismatch")
        candidate_cases = _generate_sanity_cases(candidate, tokenizer, context, rows)
        candidate_summary = summarize(candidate_cases)
        gates = {
            "source_overfit_valid": bool(approved_overfit),
            "adapter_directory_hash_matches": adapter_directory_hash
            == source["adapter_directory_sha256"],
            "adapter_state_hash_matches": adapter_state_hash
            == source["adapter_state_sha256"],
            "baseline_completed": len(baseline_cases) == 16,
            "candidate_completed": len(candidate_cases) == 16,
        }
        failed = [name for name, passed in gates.items() if not passed]
        if failed:
            raise RuntimeError(f"Generation infrastructure gates failed: {failed}")
        state.update(
            status="complete", finished_at=datetime.now(timezone.utc).isoformat()
        )
        report = {
            "protocol_id": GENERATION_PROTOCOL_ID,
            "run_id": run_id,
            "stage": "generation",
            "dataset_sha256": DATASET_SHA256,
            "model": context["manifest"]["model"],
            "generation_manifest_sha256": GENERATION_MANIFEST_SHA256,
            "source_overfit": approved_overfit,
            "adapter": {
                "path": str(adapter_dir),
                "directory_sha256": adapter_directory_hash,
                "state_sha256": adapter_state_hash,
            },
            "generation": generation_manifest["generation"],
            "baseline": {"summary": baseline_summary, "cases": baseline_cases},
            "candidate": {"summary": candidate_summary, "cases": candidate_cases},
            "comparison": {
                "proposal_correct_delta": candidate_summary["proposal_correct"]
                - baseline_summary["proposal_correct"],
                "exact_token_sequence_delta": candidate_summary[
                    "exact_token_sequences"
                ]
                - baseline_summary["exact_token_sequences"],
            },
            "memory": {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "provenance": _provenance(),
            "gates": gates,
            "state": state,
        }
        _write_json(report_path, report)
        _write_json(state_path, state)
        result_volume.commit()
        return {
            "report": str(report_path),
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "comparison": report["comparison"],
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


def trainer_state_step(loss_history: list[dict[str, Any]]) -> int:
    steps = [
        item.get("step")
        for item in loss_history
        if isinstance(item.get("step"), int | float)
    ]
    return int(max(steps, default=0))


@app.local_entrypoint()
def main(
    stage: str = "preflight",
    run_id: str = "",
    probe_run_id: str = "",
    overfit_run_id: str = "",
):
    sys.path.insert(0, "src")
    from scripts.sft_sanity_protocol import validate_protocol
    from scripts.sft_generation_protocol import (
        validate_protocol as validate_generation_protocol,
    )

    summary = validate_protocol()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    generation_summary = validate_generation_protocol()
    print(json.dumps(generation_summary, ensure_ascii=False, indent=2))
    if stage == "preflight":
        print(
            "Preflight complete. No Modal function, model download or GPU was called."
        )
        return
    if stage not in {"probe", "overfit", "generation"}:
        raise ValueError("stage must be preflight, probe, overfit or generation")
    if not run_id:
        run_id = (
            "sft-sanity-"
            + stage
            + "-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
    if stage == "overfit" and not probe_run_id:
        raise ValueError(
            "overfit requires --probe-run-id from a completed passing probe"
        )
    if stage == "generation" and not overfit_run_id:
        raise ValueError(
            "generation requires --overfit-run-id from the locked completed overfit run"
        )
    print("Run ID:", run_id)
    ready = ensure_model.remote()
    print("Locked base model ready:", json.dumps(ready, ensure_ascii=False))
    if stage == "probe":
        result = run_probe.remote(run_id)
    elif stage == "overfit":
        result = run_overfit.remote(run_id, probe_run_id)
    else:
        result = run_generation.remote(run_id, overfit_run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f"Download artifacts: modal volume get nexus-eval-results {run_id} ./evals/results/"
    )
