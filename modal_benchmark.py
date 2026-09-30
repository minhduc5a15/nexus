"""Run the frozen NEXUS proposal baseline on a Modal T4 GPU."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import modal


APP_NAME = "nexus-qwen-proposal-baseline"
LLAMA_COMMIT = "5266f24da75dc449bd56cbed7addb9c8e4a6a73e"
DEVELOPMENT_SHA256 = "c7287c09af77f98ddd02d9888b34bbb8b0aac0f4d9cf4f330990e47ef98d1b27"
REPEATS_SHA256 = "3a1f45984b014b62d3cdbfdc3ad74ae8bd2aed7a38a67945fe64937c7f8e05fb"
SESSION_SMOKE_SHA256 = "c7258aa019a8138ba41dcd9d6f6ef405372cfc8f87ffb3e2762139ba0f7306f0"

MODEL_SPECS = {
    "q4_k_m": {
        "filename": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/d7f438a3f394c3cefaf521b22d4c581aa240a7e2/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "sha256": "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
        "size": 2_497_281_120,
    },
    "q8_0": {
        "filename": "Qwen3-4B-Instruct-2507-Q8_0.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/d7f438a3f394c3cefaf521b22d4c581aa240a7e2/Qwen3-4B-Instruct-2507-Q8_0.gguf",
        "sha256": "391c1e410fd9f4cf2de2b510273b56a84c19ce18f4fa3bfb3774031dac4ef068",
        "size": 4_280_405_600,
    },
}

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("nexus-qwen-models", create_if_missing=True)
result_volume = modal.Volume.from_name("nexus-eval-results", create_if_missing=True)

download_image = modal.Image.debian_slim(python_version="3.12")
eval_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev")
    .run_commands(
        "git init /llama.cpp && git -C /llama.cpp remote add origin https://github.com/ggml-org/llama.cpp && "
        f"git -C /llama.cpp fetch --depth 1 origin {LLAMA_COMMIT} && git -C /llama.cpp checkout FETCH_HEAD",
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1",
        "cmake -S /llama.cpp -B /llama.cpp/build -DGGML_CUDA=ON -DGGML_NATIVE=OFF "
        "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release",
        "LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
        "LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
        "cmake --build /llama.cpp/build --parallel 4 --target llama-server",
    )
    .pip_install("tzdata==2025.2")
    .add_local_dir("src", "/repo/src", ignore=["**/__pycache__/**"])
    .add_local_dir("scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
    .add_local_file("evals/model_proposal_development_v1.json", "/repo/evals/model_proposal_development_v1.json")
    .add_local_file("evals/model_proposal_repeats_v1.json", "/repo/evals/model_proposal_repeats_v1.json")
    .add_local_file("evals/session_smoke_v2.json", "/repo/evals/session_smoke_v2.json")
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@app.function(image=download_image, volumes={"/models": model_volume}, timeout=3600)
def ensure_model(quantization):
    spec = MODEL_SPECS[quantization]
    target = Path("/models") / spec["filename"]
    if target.exists() and target.stat().st_size == spec["size"] and file_sha256(target) == spec["sha256"]:
        return {"path": str(target), "sha256": spec["sha256"], "cached": True}
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)
    print(f"Downloading {quantization}: {spec['size'] / 1_000_000_000:.2f} GB", flush=True)
    with urllib.request.urlopen(spec["url"], timeout=120) as response, partial.open("wb") as output:
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
    if partial.stat().st_size != spec["size"]:
        raise RuntimeError(f"Size mismatch for {quantization}: {partial.stat().st_size}")
    actual = file_sha256(partial)
    if actual != spec["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {quantization}: {actual}")
    partial.replace(target)
    model_volume.commit()
    return {"path": str(target), "sha256": actual, "cached": False}


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=7200,
    volumes={"/models": model_volume, "/results": result_volume},
)
def run_baseline(quantization, run_id):
    sys.path[:0] = ["/repo", "/repo/src"]
    from nexus.agent.client import default_model_settings
    from nexus.agent.routing import ToolRoutingMode
    from scripts.eval_proposals import SCORING_VERSION, evaluate_case, load_cases, summarize
    from scripts.eval_qwen import send_chat

    model_volume.reload()
    spec = MODEL_SPECS[quantization]
    model_path = Path("/models") / spec["filename"]
    if not model_path.exists() or file_sha256(model_path) != spec["sha256"]:
        raise RuntimeError("Model is missing or does not match its locked SHA-256")

    output = Path("/results") / run_id / quantization
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    development_path = Path("/repo/evals/model_proposal_development_v1.json")
    repeats_path = Path("/repo/evals/model_proposal_repeats_v1.json")
    if file_sha256(development_path) != DEVELOPMENT_SHA256 or file_sha256(repeats_path) != REPEATS_SHA256:
        raise RuntimeError("Frozen dataset hash mismatch")
    _, development = load_cases(development_path)
    _, repeats = load_cases(repeats_path)
    if len(development) != 80 or len(repeats) != 14:
        raise RuntimeError("Unexpected frozen dataset size")

    blocks = [
        ("development-all", "all", development, development_path),
        ("development-classified", "classified", development, development_path),
        ("repeat-1-all", "all", repeats, repeats_path),
        ("repeat-1-classified", "classified", repeats, repeats_path),
        ("repeat-2-classified", "classified", repeats, repeats_path),
        ("repeat-2-all", "all", repeats, repeats_path),
        ("repeat-3-all", "all", repeats, repeats_path),
        ("repeat-3-classified", "classified", repeats, repeats_path),
    ]
    cold = next(case for case in development if case["id"] == "list_direct")
    model_id = f"qwen3-4b-instruct-2507-{quantization}"
    settings = default_model_settings(model_id=model_id, temperature=0.0)
    settings["seed"] = 42
    runtime_version = subprocess.check_output(
        ["/llama.cpp/build/bin/llama-server", "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
    ).strip()
    metadata = {
        "run_id": run_id,
        "quantization": quantization,
        "model": spec,
        "model_sha256": file_sha256(model_path),
        "runtime_version": runtime_version,
        "llama_commit": LLAMA_COMMIT,
        "gpu": gpu_info,
        "settings": settings,
        "development_sha256": DEVELOPMENT_SHA256,
        "repeats_sha256": REPEATS_SHA256,
        "holdout_inference": False,
        "code_sha256": {
            str(path.relative_to("/repo")): file_sha256(path)
            for root in (Path("/repo/src/nexus"), Path("/repo/scripts"))
            for path in sorted(root.rglob("*.py"))
        },
    }
    write_json(output / "metadata.json", metadata)
    state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_blocks": [],
        "scored_completed": 0,
        "status": "running",
    }
    write_json(output / "run-state.json", state)

    def stop_server(process, log):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        log.close()

    def start_server(block_name):
        log = (output / f"{block_name}-server.log").open("x", encoding="utf-8")
        command = [
            "/llama.cpp/build/bin/llama-server",
            "--model", str(model_path),
            "--alias", model_id,
            "--host", "127.0.0.1",
            "--port", "8089",
            "--ctx-size", "4096",
            "--parallel", "1",
            "--threads", "6",
            "--gpu-layers", "all",
            "--fit", "off",
            "--jinja",
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; inspect {block_name}-server.log")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=2) as response:
                        if json.load(response).get("status") == "ok":
                            break
                except Exception:
                    time.sleep(0.5)
            else:
                raise TimeoutError("llama-server did not become ready")
            with urllib.request.urlopen("http://127.0.0.1:8089/props", timeout=5) as response:
                write_json(output / f"{block_name}-props.json", json.load(response))
            return process, log
        except BaseException:
            stop_server(process, log)
            raise

    try:
        for block_name, routing_name, cases, dataset_path in blocks:
            routing = ToolRoutingMode(routing_name)
            state.update(current_block=block_name, current_case="cold_probe")
            write_json(output / "run-state.json", state)
            print(f"START {quantization} {block_name}: {len(cases)} cases", flush=True)
            process, log = start_server(block_name)
            try:
                generate = lambda payload: send_chat(
                    "http://127.0.0.1:8089/v1/chat/completions", payload, timeout=180
                )
                probe = evaluate_case(
                    cold, generate, prompt_version="v13", settings=settings, tool_routing=routing
                )
                write_json(output / f"{block_name}-cold.json", {
                    "scope": "Unscored first inference after a fresh server",
                    "record": probe,
                })
                if probe["automatic_status"] == "error":
                    raise RuntimeError("Cold probe generation failed")
                report = {
                    "scoring_version": SCORING_VERSION,
                    "mode": "live",
                    "block": block_name,
                    "prompt_version": "v13",
                    "tool_routing": routing_name,
                    "settings": settings,
                    "dataset": {"path": str(dataset_path), "sha256": file_sha256(dataset_path)},
                    "model_sha256": metadata["model_sha256"],
                    "runtime_version": runtime_version,
                    "provenance": metadata,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "cache_state": "warm process after fixed cold LIST probe",
                    "cases": [],
                    "summary": summarize([]),
                }
                write_json(output / f"{block_name}.json", report)
                for index, case in enumerate(cases, 1):
                    state["current_case"] = case["id"]
                    write_json(output / "run-state.json", state)
                    record = evaluate_case(
                        case, generate, prompt_version="v13", settings=settings, tool_routing=routing
                    )
                    report["cases"].append(record)
                    report["summary"] = summarize(report["cases"])
                    write_json(output / f"{block_name}.json", report)
                    state["scored_completed"] += 1
                    write_json(output / "run-state.json", state)
                    print(
                        f"{block_name} {index}/{len(cases)} {case['id']}: "
                        f"{record['automatic_status']} {record['generation_seconds']:.3f}s {record['error_tags']}",
                        flush=True,
                    )
                    if record["automatic_status"] == "error":
                        raise RuntimeError("Generation error; stopped without retry")
                state["completed_blocks"].append(block_name)
                write_json(output / "run-state.json", state)
                result_volume.commit()
            finally:
                stop_server(process, log)
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status="interrupted", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write_json(output / "run-state.json", state)
        result_volume.commit()

    summaries = {
        name: json.loads((output / f"{name}.json").read_text(encoding="utf-8"))["summary"]
        for name, _, _, _ in blocks
    }
    result = {"run_id": run_id, "quantization": quantization, "state": state, "summaries": summaries}
    write_json(output / "summary.json", result)
    result_volume.commit()
    return result


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=1800,
    volumes={"/models": model_volume, "/results": result_volume},
)
def run_application_smoke(quantization, run_id):
    model_volume.reload()
    spec = MODEL_SPECS[quantization]
    model_path = Path("/models") / spec["filename"]
    if not model_path.exists() or file_sha256(model_path) != spec["sha256"]:
        raise RuntimeError("Model is missing or does not match its locked SHA-256")

    cases = Path("/repo/evals/session_smoke_v2.json")
    if file_sha256(cases) != SESSION_SMOKE_SHA256:
        raise RuntimeError("Session smoke dataset hash mismatch")
    output = Path("/results") / run_id / quantization
    output.mkdir(parents=True, exist_ok=True)
    report = output / "application-smoke-v9-all.json"
    log_path = output / "application-smoke-v9-all-server.log"
    run_log_path = output / "application-smoke-v9-all-run.log"
    if report.exists() or log_path.exists() or run_log_path.exists():
        raise FileExistsError("Refusing to overwrite application smoke artifacts")

    model_id = f"qwen3-4b-instruct-2507-{quantization}"
    log = log_path.open("x", encoding="utf-8")
    server = subprocess.Popen([
        "/llama.cpp/build/bin/llama-server",
        "--model", str(model_path),
        "--alias", model_id,
        "--host", "127.0.0.1",
        "--port", "8089",
        "--ctx-size", "4096",
        "--parallel", "1",
        "--threads", "6",
        "--gpu-layers", "all",
        "--fit", "off",
        "--jinja",
    ], stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited {server.returncode}; inspect {log_path.name}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=2) as response:
                    if json.load(response).get("status") == "ok":
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise TimeoutError("llama-server did not become ready")
        environment = dict(os.environ, PYTHONPATH="/repo/src:/repo")
        completed = subprocess.run([
            sys.executable,
            "/repo/scripts/eval_session.py",
            "--mode", "live",
            "--cases", str(cases),
            "--output", str(report),
            "--endpoint", "http://127.0.0.1:8089/v1/chat/completions",
            "--model", model_id,
            "--prompt-version", "v9",
            "--tool-routing", "all",
            "--temperature", "0.0",
        ], text=True, capture_output=True, env=environment)
        run_log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        result = json.loads(report.read_text(encoding="utf-8"))
        result_volume.commit()
        if completed.returncode not in (0, 1):
            raise RuntimeError(f"Session evaluator exited {completed.returncode}")
        return result["summary"]
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
        log.close()
        result_volume.commit()


@app.local_entrypoint()
def main(models: str = "q4_k_m,q8_0", run_id: str = "", smoke_only: bool = False):
    selected = [value.strip().lower() for value in models.split(",") if value.strip()]
    unknown = [value for value in selected if value not in MODEL_SPECS]
    if not selected or unknown:
        raise ValueError(f"models must be a comma-separated subset of {sorted(MODEL_SPECS)}; unknown={unknown}")
    if not run_id:
        run_id = "modal-gpu-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"Run ID: {run_id}")
    for quantization in selected:
        cached = ensure_model.remote(quantization)
        print("Model ready:", cached)
        if not smoke_only:
            result = run_baseline.remote(quantization, run_id)
            development = {
                key: value["automatic"]
                for key, value in result["summaries"].items()
                if key.startswith("development-")
            }
            print(quantization, development)
        smoke = run_application_smoke.remote(quantization, run_id)
        print(quantization, "application-smoke-v9-all", smoke)
    print(f"Artifacts: modal volume get nexus-eval-results {run_id} ./evals/results/")
