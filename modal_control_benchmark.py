"""Run the frozen Qwen3-30B-A3B stronger-model control on Modal.

The default local entrypoint is preflight and does not call a Modal function.
Canary and development are separate explicit stages; development additionally
requires the run ID of a completed, error-free canary made with this protocol.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

import modal


APP_NAME = "nexus-qwen30b-control"
PROTOCOL_ID = "qwen3-30b-a3b-control-v1"
MODEL_KEY = "qwen3-30b-a3b-instruct-2507-q8_0"
MODEL_FILENAME = "qwen3-30b-a3b-instruct-2507-q8_0.gguf"
MODEL_URL = (
    "https://huggingface.co/ggml-org/Qwen3-30B-A3B-Instruct-2507-Q8_0-GGUF/resolve/"
    "c9abf4f74f8da42fe93a72a3debcca56c8f363dc/qwen3-30b-a3b-instruct-2507-q8_0.gguf"
)
MODEL_SIZE = 32_483_930_432
MODEL_SHA256 = "66cbfc7624628ca4725a6a0581b95f2155aa6aa073a6ed4ff8d158794ee8fee3"
LLAMA_COMMIT = "5266f24da75dc449bd56cbed7addb9c8e4a6a73e"
CANARY_SHA256 = "9796a10912230d116a5f6207e8f77fc6a4abc7a974b34509b2fbb93c759d5031"
DEVELOPMENT_SHA256 = "c7287c09af77f98ddd02d9888b34bbb8b0aac0f4d9cf4f330990e47ef98d1b27"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DOWNLOAD_PROGRESS_BYTES = 1024 * 1024 * 1024
DOWNLOAD_COMMIT_BYTES = 8 * DOWNLOAD_PROGRESS_BYTES


app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("nexus-qwen-models", create_if_missing=True)
result_volume = modal.Volume.from_name("nexus-eval-results", create_if_missing=True)

protocol_files = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir("scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
    .add_local_file("evals/model_control_manifest_v1.json", "/repo/evals/model_control_manifest_v1.json")
)
download_image = protocol_files
eval_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev")
    .run_commands(
        "git init /llama.cpp && git -C /llama.cpp remote add origin https://github.com/ggml-org/llama.cpp && "
        f"git -C /llama.cpp fetch --depth 1 origin {LLAMA_COMMIT} && git -C /llama.cpp checkout FETCH_HEAD",
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1",
        "cmake -S /llama.cpp -B /llama.cpp/build -DGGML_CUDA=ON -DGGML_NATIVE=OFF "
        "-DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release",
        "LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
        "LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
        "cmake --build /llama.cpp/build --parallel 4 --target llama-server",
    )
    .pip_install("tzdata==2025.2")
    .add_local_dir("src", "/repo/src", ignore=["**/__pycache__/**"])
    .add_local_dir("scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
    .add_local_file("evals/model_control_manifest_v1.json", "/repo/evals/model_control_manifest_v1.json")
    .add_local_file("evals/model_control_canary_v1.json", "/repo/evals/model_control_canary_v1.json")
    .add_local_file("evals/model_proposal_development_v1.json", "/repo/evals/model_proposal_development_v1.json")
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def model_output(run_id: str) -> Path:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be 1-128 safe filename characters")
    return Path("/results") / run_id / MODEL_KEY


@app.function(
    image=download_image,
    volumes={"/models": model_volume},
    timeout=14_400,
    cpu=2,
    memory=4096,
)
def ensure_model() -> dict:
    """Download the locked 32.5 GB artifact without allocating a GPU."""

    target = Path("/models") / MODEL_FILENAME
    if target.exists() and target.stat().st_size == MODEL_SIZE and file_sha256(target) == MODEL_SHA256:
        return {"path": str(target), "size": MODEL_SIZE, "sha256": MODEL_SHA256, "cached": True}

    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists() and partial.stat().st_size > MODEL_SIZE:
        partial.unlink()
    if partial.exists() and partial.stat().st_size == MODEL_SIZE:
        actual = file_sha256(partial)
        if actual == MODEL_SHA256:
            partial.replace(target)
            model_volume.commit()
            return {"path": str(target), "size": MODEL_SIZE, "sha256": actual, "cached": False}
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(MODEL_URL)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    print(f"Downloading model from byte {offset:,} of {MODEL_SIZE:,}", flush=True)

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            append = offset > 0 and response.status == 206
            if offset and not append:
                offset = 0
            mode = "ab" if append else "wb"
            committed_at = offset
            reported_at = offset
            with partial.open(mode) as output:
                while block := response.read(8 * 1024 * 1024):
                    output.write(block)
                    offset += len(block)
                    if offset - reported_at >= DOWNLOAD_PROGRESS_BYTES:
                        print(f"Downloaded {offset / 1_000_000_000:.1f} GB", flush=True)
                        reported_at = offset
                    if offset - committed_at >= DOWNLOAD_COMMIT_BYTES:
                        output.flush()
                        model_volume.commit()
                        committed_at = offset
                        print(f"Committed partial model at {offset / 1_000_000_000:.1f} GB", flush=True)
    except BaseException:
        if partial.exists():
            model_volume.commit()
        raise

    model_volume.commit()
    if partial.stat().st_size != MODEL_SIZE:
        raise RuntimeError(f"Model size mismatch: {partial.stat().st_size} != {MODEL_SIZE}")
    actual = file_sha256(partial)
    if actual != MODEL_SHA256:
        raise RuntimeError(f"Model SHA-256 mismatch: {actual}")
    partial.replace(target)
    model_volume.commit()
    return {"path": str(target), "size": MODEL_SIZE, "sha256": actual, "cached": False}


def _remote_context():
    sys.path[:0] = ["/repo", "/repo/src"]
    from nexus.agent.client import default_model_settings
    from nexus.agent.routing import ToolRoutingMode
    from scripts.eval_proposals import SCORING_VERSION, evaluate_case, load_cases, summarize
    from scripts.model_control_protocol import load_json, validate_canary_gate
    from scripts.eval_qwen import send_chat

    manifest = load_json(Path("/repo/evals/model_control_manifest_v1.json"))
    request_settings = default_model_settings(model_id=MODEL_KEY, temperature=0.0)
    request_settings["seed"] = 42
    canonical = manifest["runtime"]["settings"]
    expected_request_settings = {
        "model": MODEL_KEY,
        "temperature": canonical["temperature"],
        "top_p": canonical["top_p"],
        "top_k": canonical["top_k"],
        "min_p": canonical["min_p"],
        "max_tokens": canonical["max_tokens"],
        "seed": canonical["seed"],
        "chat_template_kwargs": {"enable_thinking": canonical["enable_thinking"]},
    }
    if request_settings != expected_request_settings:
        raise RuntimeError("Runtime request settings differ from the frozen control manifest")
    return {
        "manifest": manifest,
        "request_settings": request_settings,
        "routing": ToolRoutingMode.ALL,
        "scoring_version": SCORING_VERSION,
        "evaluate_case": evaluate_case,
        "load_cases": load_cases,
        "summarize": summarize,
        "send_chat": send_chat,
        "validate_canary_gate": validate_canary_gate,
    }


def _checked_model() -> Path:
    model_volume.reload()
    path = Path("/models") / MODEL_FILENAME
    if not path.exists() or path.stat().st_size != MODEL_SIZE or file_sha256(path) != MODEL_SHA256:
        raise RuntimeError("Locked model artifact is missing or does not match size and SHA-256")
    return path


def _start_server(output: Path, block: str, model_path: Path):
    log = (output / f"{block}-server.log").open("x", encoding="utf-8")
    process = subprocess.Popen(
        [
            "/llama.cpp/build/bin/llama-server",
            "--model", str(model_path),
            "--alias", MODEL_KEY,
            "--host", "127.0.0.1",
            "--port", "8089",
            "--ctx-size", "4096",
            "--parallel", "1",
            "--threads", "8",
            "--gpu-layers", "all",
            "--fit", "off",
            "--jinja",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited {process.returncode}; inspect {log.name}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=2) as response:
                    if json.load(response).get("status") == "ok":
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise TimeoutError("llama-server did not become ready within 300 seconds")
        with urllib.request.urlopen("http://127.0.0.1:8089/props", timeout=5) as response:
            write_json(output / f"{block}-props.json", json.load(response))
        return process, log
    except BaseException:
        _stop_server(process, log)
        raise


def _stop_server(process, log) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)
    log.close()


def _provenance(context: dict, run_id: str, model_path: Path) -> dict:
    manifest = context["manifest"]
    runtime_version = subprocess.check_output(
        ["/llama.cpp/build/bin/llama-server", "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
    ).strip()
    return {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "model": manifest["model"],
        "model_sha256": file_sha256(model_path),
        "runtime_version": runtime_version,
        "llama_commit": LLAMA_COMMIT,
        "gpu": gpu_info,
        "canonical_settings": manifest["runtime"]["settings"],
        "request_settings": context["request_settings"],
        "holdout_inference": False,
        "code_sha256": {
            str(path.relative_to("/repo")): file_sha256(path)
            for root in (Path("/repo/src/nexus"), Path("/repo/scripts"))
            for path in sorted(root.rglob("*.py"))
        },
    }


def _run_block(*, run_id: str, block: str, dataset_path: Path, report_name: str, approved_canary=None):
    context = _remote_context()
    manifest = context["manifest"]
    expected_dataset = manifest["datasets"]["canary" if block == "control-canary" else "development"]
    if file_sha256(dataset_path) != expected_dataset["sha256"]:
        raise RuntimeError(f"Frozen {block} dataset hash mismatch")
    _, cases = context["load_cases"](dataset_path)
    if len(cases) != expected_dataset["count"]:
        raise RuntimeError(f"Frozen {block} dataset count mismatch")

    development_path = Path("/repo/evals/model_proposal_development_v1.json")
    if file_sha256(development_path) != DEVELOPMENT_SHA256:
        raise RuntimeError("Development dataset hash mismatch")
    _, development = context["load_cases"](development_path)
    cold = next(case for case in development if case["id"] == "list_direct")

    model_path = _checked_model()
    output = model_output(run_id)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / report_name
    state_path = output / f"{block}-state.json"
    if report_path.exists() or state_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path} or {state_path}")

    provenance = _provenance(context, run_id, model_path)
    state = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "current_case": "cold_probe",
        "scored_completed": 0,
    }
    write_json(state_path, state)
    process, log = _start_server(output, block, model_path)
    try:
        generate = lambda payload: context["send_chat"](
            "http://127.0.0.1:8089/v1/chat/completions", payload, timeout=180
        )
        probe = context["evaluate_case"](
            cold,
            generate,
            prompt_version="v13",
            settings=context["request_settings"],
            tool_routing=context["routing"],
        )
        write_json(output / f"{block}-cold.json", {"scope": "Unscored first inference", "record": probe})
        if probe["automatic_status"] == "error":
            raise RuntimeError("Cold probe generation failed")

        report = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "scoring_version": context["scoring_version"],
            "mode": "live",
            "block": block,
            "prompt_version": "v13",
            "tool_routing": "all",
            "settings": manifest["runtime"]["settings"],
            "request_settings": context["request_settings"],
            "dataset": {"path": expected_dataset["path"], "sha256": expected_dataset["sha256"]},
            "model_sha256": MODEL_SHA256,
            "provenance": provenance,
            "approved_canary": approved_canary,
            "cases": [],
            "summary": context["summarize"]([]),
            "state": state,
        }
        write_json(report_path, report)
        for index, case in enumerate(cases, 1):
            state["current_case"] = case["id"]
            write_json(state_path, state)
            record = context["evaluate_case"](
                case,
                generate,
                prompt_version="v13",
                settings=context["request_settings"],
                tool_routing=context["routing"],
            )
            report["cases"].append(record)
            report["summary"] = context["summarize"](report["cases"])
            state["scored_completed"] = index
            report["state"] = dict(state)
            write_json(report_path, report)
            write_json(state_path, state)
            print(
                f"{block} {index}/{len(cases)} {case['id']}: "
                f"{record['automatic_status']} {record['generation_seconds']:.3f}s {record['error_tags']}",
                flush=True,
            )
            if record["automatic_status"] == "error":
                raise RuntimeError("Generation error; stopped without retry")
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(), current_case=None)
        report["state"] = dict(state)
        write_json(report_path, report)
        write_json(state_path, state)
        result_volume.commit()
        return {"report": str(report_path), "summary": report["summary"], "state": state}
    except BaseException as error:
        state.update(status="interrupted", error={"type": type(error).__name__, "message": str(error)})
        write_json(state_path, state)
        result_volume.commit()
        raise
    finally:
        _stop_server(process, log)
        result_volume.commit()


@app.function(
    image=eval_image,
    gpu="A100-80GB",
    timeout=1200,
    volumes={"/models": model_volume, "/results": result_volume},
)
def run_canary(run_id: str) -> dict:
    return _run_block(
        run_id=run_id,
        block="control-canary",
        dataset_path=Path("/repo/evals/model_control_canary_v1.json"),
        report_name="canary.json",
    )


@app.function(
    image=eval_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/models": model_volume, "/results": result_volume},
)
def run_development(run_id: str, canary_run_id: str) -> dict:
    context = _remote_context()
    result_volume.reload()
    canary_path = model_output(canary_run_id) / "canary.json"
    if not canary_path.exists():
        raise FileNotFoundError(f"Required canary report not found: {canary_path}")
    canary_report = json.loads(canary_path.read_text(encoding="utf-8"))
    approved = context["validate_canary_gate"](
        canary_report, context["manifest"], expected_run_id=canary_run_id
    )
    return _run_block(
        run_id=run_id,
        block="control-development",
        dataset_path=Path("/repo/evals/model_proposal_development_v1.json"),
        report_name="development-all.json",
        approved_canary=approved,
    )


@app.local_entrypoint()
def main(stage: str = "preflight", run_id: str = "", canary_run_id: str = ""):
    from scripts.model_control_protocol import validate_control_protocol

    summary = validate_control_protocol()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if stage == "preflight":
        print("Preflight complete. No Modal function or GPU was called.")
        return
    if stage not in {"canary", "development"}:
        raise ValueError("stage must be preflight, canary or development")
    if not run_id:
        run_id = "qwen30b-control-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if stage == "development" and not canary_run_id:
        raise ValueError("development requires --canary-run-id from an explicit completed canary")

    print("Run ID:", run_id)
    ready = ensure_model.remote()
    print("Model ready:", ready)
    if stage == "canary":
        result = run_canary.remote(run_id)
    else:
        result = run_development.remote(run_id, canary_run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Artifacts: modal volume get nexus-eval-results {run_id} ./evals/results/")
