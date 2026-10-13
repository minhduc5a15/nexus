"""Train and evaluate the locked 600-example NEXUS QLoRA candidate v3 on Modal."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any

import modal

APP_NAME = "nexus-sft-candidate-v3"
PROTOCOL_ID = "nexus-sft-qlora-candidate-v3"
MANIFEST_SHA256 = "9a67e53900fe47eb0ab7fc27c1e9aff598676eb87e83a62829fe687ab71078da"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
CANDIDATE_SHA256 = "a0e54748aed6926d48e7ddc5f6674954bba7a19c9eab9f469cc09db80c7b83e7"
DEVELOPMENT_SHA256 = "c7287c09af77f98ddd02d9888b34bbb8b0aac0f4d9cf4f330990e47ef98d1b27"
MAX_SEQUENCE_TOKENS = 2048
MAX_STEPS = 300
MAX_NEW_TOKENS = 128
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("nexus-hf-cache", create_if_missing=True)
result_volume = modal.Volume.from_name("nexus-eval-results", create_if_missing=True)

packages = (
    "torch==2.8.0",
    "transformers==4.57.1",
    "trl==0.27.0",
    "peft==0.18.0",
    "bitsandbytes==0.48.2",
    "accelerate==1.12.0",
    "datasets==4.4.1",
    "jinja2==3.1.6",
)

protocol_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub==0.36.0")
    .add_local_file(
        "evals/sft_candidate_training_plan_v3.json",
        "/repo/evals/sft_candidate_training_plan_v3.json",
    )
    .add_local_file("modal_sft_candidate_v3.py", "/repo/modal_sft_candidate_v3.py")
)

training_image = (
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
    .add_local_file("evals/sft_candidate_v3.json", "/repo/evals/sft_candidate_v3.json")
    .add_local_file(
        "evals/sft_candidate_training_plan_v3.json",
        "/repo/evals/sft_candidate_training_plan_v3.json",
    )
    .add_local_file("modal_sft_candidate_v3.py", "/repo/modal_sft_candidate_v3.py")
)

development_image = (
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
        "evals/model_proposal_development_v1.json",
        "/repo/evals/model_proposal_development_v1.json",
    )
    .add_local_file(
        "evals/sft_candidate_training_plan_v3.json",
        "/repo/evals/sft_candidate_training_plan_v3.json",
    )
    .add_local_file("modal_sft_candidate_v3.py", "/repo/modal_sft_candidate_v3.py")
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
    return Path("/results") / _safe_run_id(run_id) / "sft-candidate-v3"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
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


def _manifest() -> dict[str, Any]:
    path = Path("/repo/evals/sft_candidate_training_plan_v3.json")
    if _file_sha256(path) != MANIFEST_SHA256:
        raise RuntimeError("Frozen candidate training manifest hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Frozen candidate protocol ID mismatch")
    return value


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
            "jinja2",
        )
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


def _load_quantized_base(*, use_cache: bool):
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
    model.config.use_cache = use_cache
    return model


def _load_qlora_model():
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base = prepare_model_for_kbit_training(
        _load_quantized_base(use_cache=False),
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


def _parameter_summary(model: Any) -> dict[str, Any]:
    total = trainable = 0
    names = []
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
            names.append(name)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_fraction": trainable / total,
        "trainable_tensor_count": len(names),
        "only_lora_trainable": bool(names) and all("lora_" in name for name in names),
        "trainable_dtypes": sorted(
            {
                str(parameter.dtype)
                for parameter in model.parameters()
                if parameter.requires_grad
            }
        ),
    }


def _collate(features: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    maximum = int(math.ceil(max(len(item["input_ids"]) for item in features) / 8) * 8)
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
    with torch.no_grad():
        for row in rows:
            batch = {
                key: torch.tensor([row[key]], dtype=torch.long, device="cuda")
                for key in ("input_ids", "attention_mask", "labels")
            }
            output = model(**batch)
            labels = batch["labels"][:, 1:]
            predictions = output.logits[:, :-1].argmax(dim=-1)
            mask = labels != -100
            count = int(mask.sum().item())
            hits = int((predictions[mask] == labels[mask]).sum().item())
            weighted_loss += float(output.loss.detach().cpu()) * count
            supervised += count
            correct += hits
            exact += int(hits == count)
            del batch, output, labels, predictions, mask
    if was_training:
        model.train()
    return {
        "mean_loss": weighted_loss / supervised,
        "token_accuracy": correct / supervised,
        "exact_sequences": exact,
        "cases": len(rows),
        "supervised_tokens": supervised,
    }


def _load_training_rows(manifest: dict[str, Any]):
    sys.path[:0] = ["/repo", "/repo/src"]
    from scripts.audit_sft_data import materialize_case
    from scripts.inspect_sft_tokens import build_assistant_only_labels
    from scripts.eval_proposals import strict_json
    from transformers import AutoTokenizer

    path = Path("/repo/evals/sft_candidate_v3.json")
    if _file_sha256(path) != CANDIDATE_SHA256:
        raise RuntimeError("Frozen candidate dataset hash mismatch")
    document = strict_json(path.read_bytes())
    if (
        document.get("dataset_id") != "nexus-sft-candidate-v3"
        or len(document["cases"]) != 600
    ):
        raise RuntimeError("Frozen candidate dataset structure mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir="/hf-cache/hub",
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    for case in document["cases"]:
        record = materialize_case(
            case, prompt_version=manifest["contract"]["prompt_version"]
        )
        encoded = build_assistant_only_labels(record, tokenizer)
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


def _provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        "packages": _versions(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "cuda": torch.version.cuda,
        "runner_sha256": _file_sha256(Path("/repo/modal_sft_candidate_v3.py")),
        "manifest_sha256": MANIFEST_SHA256,
        "candidate_sha256": manifest["dataset"]["sha256"],
        "development_sha256": manifest["development"]["sha256"],
        "final_holdout_packaged": False,
        "final_holdout_inference": False,
        "tools_or_database_executed": False,
    }


@app.function(
    image=protocol_image,
    volumes={"/hf-cache": cache_volume},
    timeout=3600,
    cpu=2,
    memory=4096,
)
def ensure_model() -> dict[str, Any]:
    """Download the locked checkpoint without allocating a GPU."""

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
    image=training_image,
    gpu="T4",
    timeout=8400,
    cpu=2,
    memory=16384,
    volumes={"/hf-cache": cache_volume, "/results": result_volume},
)
def run_train(run_id: str) -> dict[str, Any]:
    """Train one locked 300-step adapter and verify save/reload identity."""

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from trl import SFTConfig, SFTTrainer

    run_id = _safe_run_id(run_id)
    output = _output(run_id)
    report_path = output / "train.json"
    state_path = output / "train-state.json"
    if report_path.exists() or state_path.exists():
        raise FileExistsError(f"Refusing to overwrite candidate artifacts for {run_id}")
    output.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    _write_json(state_path, state)
    try:
        manifest = _manifest()
        if manifest.get("training_authorized") is not True:
            raise PermissionError("Candidate training budget has not been authorized")
        _set_seed(manifest["training"]["seed"])
        tokenizer, rows = _load_training_rows(manifest)
        rows_by_id = {row["id"]: row for row in rows}
        probe_rows = [
            rows_by_id[case_id]
            for case_id in manifest["training"]["teacher_forced_probe_ids"]
        ]
        torch.cuda.reset_peak_memory_stats()
        model = _load_qlora_model()
        parameters = _parameter_summary(model)
        initial_hash = _adapter_state_sha256(model)
        before = _teacher_forced_metrics(model, probe_rows)
        dataset = Dataset.from_list(
            [
                {key: row[key] for key in ("input_ids", "attention_mask", "labels")}
                for row in rows
            ]
        )
        training = manifest["training"]
        args = SFTConfig(
            output_dir=f"/tmp/nexus-sft-candidate-v3-{run_id}",
            max_steps=training["maximum_steps"],
            per_device_train_batch_size=training["per_device_batch_size"],
            gradient_accumulation_steps=training["gradient_accumulation_steps"],
            learning_rate=training["learning_rate"],
            optim=training["optimizer"],
            lr_scheduler_type=training["scheduler"],
            warmup_steps=training["warmup_steps"],
            weight_decay=training["weight_decay"],
            logging_steps=1,
            save_strategy="steps",
            save_steps=training["save_steps"][0],
            save_total_limit=len(training["save_steps"]),
            save_only_model=True,
            report_to="none",
            fp16=False,
            bf16=False,
            gradient_checkpointing=training["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=MAX_SEQUENCE_TOKENS,
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            seed=training["seed"],
            data_seed=training["seed"],
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=lambda features: _collate(features, tokenizer.pad_token_id),
        )
        train_result = trainer.train()
        after = _teacher_forced_metrics(model, probe_rows)
        final_hash = _adapter_state_sha256(model)
        temporary_output = Path(args.output_dir)
        epoch_one_source = temporary_output / "checkpoint-150"
        if not epoch_one_source.is_dir():
            raise RuntimeError("Trainer did not save the step-150 adapter")
        epoch_one_dir = output / "adapter-step-150"
        epoch_two_dir = output / "adapter-step-300"
        shutil.copytree(epoch_one_source, epoch_one_dir)
        trainer.save_model(str(epoch_two_dir))
        tokenizer.save_pretrained(epoch_one_dir)
        tokenizer.save_pretrained(epoch_two_dir)
        directory_hashes = {
            "step-150": _directory_sha256(epoch_one_dir),
            "step-300": _directory_sha256(epoch_two_dir),
        }
        loss_history = [
            {
                "step": item.get("step"),
                "loss": item["loss"],
                "grad_norm": item.get("grad_norm"),
                "learning_rate": item.get("learning_rate"),
            }
            for item in trainer.state.log_history
            if "loss" in item
        ]
        completed_steps = int(
            max((item["step"] or 0 for item in loss_history), default=0)
        )
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        del trainer, model, dataset
        gc.collect()
        torch.cuda.empty_cache()

        reloaded_hashes = {}
        for label, adapter_dir in (
            ("step-150", epoch_one_dir),
            ("step-300", epoch_two_dir),
        ):
            reloaded_base = _load_quantized_base(use_cache=False)
            reloaded = PeftModel.from_pretrained(
                reloaded_base, adapter_dir, is_trainable=False
            )
            reloaded_hashes[label] = _adapter_state_sha256(reloaded)
            del reloaded, reloaded_base
            gc.collect()
            torch.cuda.empty_cache()
        gates = {
            "training_dataset_complete": len(rows) == manifest["dataset"]["count"],
            "only_lora_parameters_trainable": parameters["only_lora_trainable"],
            "completed_all_steps": completed_steps == MAX_STEPS,
            "finite_training_loss": math.isfinite(float(train_result.training_loss)),
            "teacher_forced_loss_decreased": after["mean_loss"] < before["mean_loss"],
            "adapter_hash_changed": initial_hash != final_hash,
            "epoch_one_adapter_saved": epoch_one_dir.is_dir(),
            "epoch_two_adapter_saved": epoch_two_dir.is_dir(),
            "final_adapter_reloaded_identically": final_hash
            == reloaded_hashes["step-300"],
        }
        failed = [name for name, passed in gates.items() if not passed]
        if failed:
            raise RuntimeError(f"Candidate train gates failed: {failed}")
        state.update(
            status="complete", finished_at=datetime.now(timezone.utc).isoformat()
        )
        report = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "stage": "train",
            "model": manifest["model"],
            "dataset": manifest["dataset"],
            "configuration": {
                key: manifest[key]
                for key in ("runtime", "quantization", "adapter", "training")
            },
            "parameters": parameters,
            "training": {
                "completed_steps": completed_steps,
                "training_loss": float(train_result.training_loss),
                "runtime_seconds": float(
                    train_result.metrics.get("train_runtime", 0.0)
                ),
                "loss_history": loss_history,
            },
            "teacher_forced_before": before,
            "teacher_forced_after": after,
            "adapters": {
                "step-150": {
                    "path": str(epoch_one_dir),
                    "directory_sha256": directory_hashes["step-150"],
                    "state_sha256_reloaded": reloaded_hashes["step-150"],
                },
                "step-300": {
                    "path": str(epoch_two_dir),
                    "directory_sha256": directory_hashes["step-300"],
                    "state_sha256_before": initial_hash,
                    "state_sha256_after": final_hash,
                    "state_sha256_reloaded": reloaded_hashes["step-300"],
                },
            },
            "memory": {
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
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
            "training": report["training"],
            "teacher_forced_before": before,
            "teacher_forced_after": after,
            "adapters": report["adapters"],
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


def _generate_development_cases(
    model: Any, tokenizer: Any, cases: list[dict[str, Any]]
):
    import torch
    from nexus.agent.client import build_model_request, default_model_settings
    from nexus.agent.routing import ToolRoutingMode
    from scripts.score_sft_generation import score_proposal_text

    model.eval()
    rows = []
    with torch.inference_mode():
        for index, case in enumerate(cases, start=1):
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
            started = time.perf_counter()
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
                    "generation_seconds": elapsed,
                    "routing": routing,
                }
            )
            rows.append(scored)
            print(
                f"[{index:02d}/{len(cases)}] {case['id']}: "
                f"{scored['automatic_status']} {scored['error_tags']}",
                flush=True,
            )
            del input_ids, attention_mask, output
    return rows


@app.function(
    image=development_image,
    gpu="T4",
    timeout=1800,
    cpu=2,
    memory=16384,
    volumes={"/hf-cache": cache_volume, "/results": result_volume},
)
def run_development(run_id: str, train_run_id: str) -> dict[str, Any]:
    """Compare base and the locked candidate adapter on exposed development data."""

    sys.path[:0] = ["/repo", "/repo/src"]
    import torch
    from peft import PeftModel
    from scripts.eval_proposals import load_cases
    from scripts.score_sft_generation import summarize_proposal_texts
    from scripts.sft_candidate_training_protocol_v3 import validate_train_report
    from transformers import AutoTokenizer

    run_id = _safe_run_id(run_id)
    train_run_id = _safe_run_id(train_run_id)
    output = _output(run_id)
    report_path = output / "development.json"
    state_path = output / "development-state.json"
    if report_path.exists() or state_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite development artifacts for {run_id}"
        )
    output.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    _write_json(state_path, state)
    try:
        manifest = _manifest()
        development_path = Path("/repo/evals/model_proposal_development_v1.json")
        if _file_sha256(development_path) != DEVELOPMENT_SHA256:
            raise RuntimeError("Frozen development dataset hash mismatch")
        _, cases = load_cases(development_path)
        if len(cases) != manifest["development"]["count"]:
            raise RuntimeError("Frozen development case count mismatch")

        result_volume.reload()
        train_report_path = _output(train_run_id) / "train.json"
        adapter_dirs = {
            label: _output(train_run_id) / f"adapter-{label}"
            for label in manifest["development"]["candidates"]
        }
        if not train_report_path.is_file() or not all(
            path.is_dir() for path in adapter_dirs.values()
        ):
            raise FileNotFoundError(
                "Completed candidate train report or one of its adapters is missing"
            )
        train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
        approved_train = validate_train_report(
            train_report,
            manifest,
            expected_run_id=train_run_id,
            adapter_paths=adapter_dirs,
        )

        generation = manifest["development"]["generation"]
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
        base = _load_quantized_base(use_cache=True)
        baseline_cases = _generate_development_cases(base, tokenizer, cases)
        baseline_summary = summarize_proposal_texts(baseline_cases)
        del base
        gc.collect()
        torch.cuda.empty_cache()

        candidates = {}
        candidate_hashes = {}
        for label, adapter_dir in adapter_dirs.items():
            candidate_base = _load_quantized_base(use_cache=True)
            candidate = PeftModel.from_pretrained(
                candidate_base, adapter_dir, is_trainable=False
            )
            candidate.config.use_cache = True
            candidate_hash = _adapter_state_sha256(candidate)
            expected_hash = train_report["adapters"][label]["state_sha256_reloaded"]
            if candidate_hash != expected_hash:
                raise RuntimeError(f"Reloaded {label} state hash mismatch")
            candidate_cases = _generate_development_cases(candidate, tokenizer, cases)
            candidates[label] = {
                "summary": summarize_proposal_texts(candidate_cases),
                "cases": candidate_cases,
            }
            candidate_hashes[label] = candidate_hash
            del candidate, candidate_base
            gc.collect()
            torch.cuda.empty_cache()
        gates = {
            "completed_train_valid": bool(approved_train),
            "development_dataset_complete": len(cases) == 80,
            "baseline_completed": len(baseline_cases) == 80,
            "all_candidates_completed": all(
                len(value["cases"]) == 80 for value in candidates.values()
            ),
            "all_adapter_state_hashes_match": all(
                candidate_hashes[label]
                == train_report["adapters"][label]["state_sha256_reloaded"]
                for label in candidates
            ),
            "no_tools_or_database_executed": True,
            "final_holdout_not_used": True,
        }
        failed = [name for name, passed in gates.items() if not passed]
        if failed:
            raise RuntimeError(f"Development infrastructure gates failed: {failed}")
        state.update(
            status="complete", finished_at=datetime.now(timezone.utc).isoformat()
        )
        report = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "stage": "development",
            "train_run_id": train_run_id,
            "model": manifest["model"],
            "dataset": manifest["development"],
            "approved_train": approved_train,
            "generation": generation,
            "baseline": {"summary": baseline_summary, "cases": baseline_cases},
            "candidates": candidates,
            "comparisons": {
                label: {
                    "pass_delta": value["summary"]["pass"]
                    - baseline_summary["pass"],
                    "unexpected_call_delta": value["summary"]["unexpected_calls"]
                    - baseline_summary["unexpected_calls"],
                }
                for label, value in candidates.items()
            },
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
            "baseline": baseline_summary,
            "candidates": {
                label: value["summary"] for label, value in candidates.items()
            },
            "comparisons": report["comparisons"],
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
def main(stage: str = "preflight", run_id: str = "", train_run_id: str = ""):
    sys.path[:0] = [".", "src"]
    from scripts.sft_candidate_training_protocol_v3 import validate_protocol

    protocol = validate_protocol()
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    if stage == "preflight":
        print(
            "Preflight complete. No Modal function, model download or GPU was called."
        )
        return
    if stage not in {"train", "development"}:
        raise ValueError("stage must be preflight, train or development")
    if not protocol["training_authorized"]:
        raise PermissionError(
            "Training is locked until the user approves the stated Modal budget"
        )
    if not run_id:
        run_id = (
            "sft-candidate-v3-"
            + stage
            + "-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
    if stage == "development" and not train_run_id:
        raise ValueError("development requires --train-run-id from completed training")
    print("Run ID:", run_id)
    ready = ensure_model.remote()
    print("Locked base model ready:", json.dumps(ready, ensure_ascii=False))
    result = (
        run_train.remote(run_id)
        if stage == "train"
        else run_development.remote(run_id, train_run_id)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f"Download artifacts: modal volume get nexus-eval-results {run_id} ./evals/results/"
    )
