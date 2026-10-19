"""Benchmark llama-cpp-python optimization profiles and save raw JSON logs."""

import argparse
import gc
import hashlib
import json
import platform
import re
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.profiles import InferenceProfile, get_profile, list_profiles

try:
    import pynvml

    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    pynvml = None
    HAS_PYNVML = False

try:
    import torch

    HAS_CUDA_TORCH = torch.cuda.is_available()
except Exception:
    torch = None
    HAS_CUDA_TORCH = False


def get_current_vram_mb(device_index: int = 0) -> float:
    """Return total used device memory in MiB, or 0 when it cannot be read."""
    if HAS_PYNVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024.0**2)
        except Exception:
            pass
    if HAS_CUDA_TORCH:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            return (total_bytes - free_bytes) / (1024.0**2)
        except Exception:
            pass
    return 0.0


class PeakVRAMTracker:
    """Sample total GPU memory use while a model is loaded and generating."""

    def __init__(self, device_index: int = 0, interval_sec: float = 0.02):
        self.device_index = device_index
        self.interval_sec = interval_sec
        self.peak_vram_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "PeakVRAMTracker":
        self.peak_vram_mb = get_current_vram_mb(self.device_index)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            self.peak_vram_mb = max(
                self.peak_vram_mb, get_current_vram_mb(self.device_index)
            )
            time.sleep(self.interval_sec)

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.peak_vram_mb = max(
            self.peak_vram_mb, get_current_vram_mb(self.device_index)
        )
        return self.peak_vram_mb


def load_llama_model(profile: InferenceProfile, model_path: str) -> Any:
    """Load a Llama model with the given profile. Caller is responsible for del + gc."""
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install a CUDA build for GPU runs."
        ) from exc
    return Llama(model_path=model_path, verbose=True, **profile.to_llama_kwargs())


def run_inference(
    model: Any,
    profile: InferenceProfile,
    prompt: str,
    max_tokens: int,
    device_index: int,
    temperature: float,
    seed: int,
    load_time_sec: float,
    idle_vram_mb: float,
) -> Dict[str, Any]:
    """Run one inference pass on an already-loaded model and return metrics."""
    chunks: List[str] = []
    generated_tokens = 0
    ttft_sec: Optional[float] = None
    tracker = PeakVRAMTracker(device_index).start()
    gen_start = time.perf_counter()
    try:
        stream = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stream=True,
            echo=False,
        )
        for chunk in stream:
            if ttft_sec is None:
                ttft_sec = time.perf_counter() - gen_start
            chunks.append(chunk["choices"][0]["text"])
            generated_tokens += 1
    finally:
        total_time_sec = time.perf_counter() - gen_start
        peak_vram_mb = tracker.stop()

    if ttft_sec is None:
        ttft_sec = total_time_sec
    decode_time_sec = max(total_time_sec - ttft_sec, 1e-9)
    return {
        "profile_name": profile.name,
        "n_ctx": profile.n_ctx,
        "flash_attn": profile.flash_attn,
        "type_k": profile.type_k,
        "type_v": profile.type_v,
        "n_gpu_layers": profile.n_gpu_layers,
        "n_batch": profile.n_batch,
        "idle_vram_mb": idle_vram_mb,
        "peak_vram_mb": peak_vram_mb,
        "load_time_sec": load_time_sec,
        "ttft_sec": ttft_sec,
        "total_time_sec": total_time_sec,
        "tokens_generated": generated_tokens,
        "tokens_per_sec": (generated_tokens - 1) / decode_time_sec
        if generated_tokens > 1
        else 0.0,
        "text": "".join(chunks).strip(),
    }


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(case.get("id"), str) or not isinstance(
                case.get("prompt"), str
            ):
                raise ValueError(
                    f"Each case in {path} needs string fields 'id' and 'prompt'"
                )
            cases.append(case)
    if not cases:
        raise ValueError(f"No test cases found in {path}")
    return cases


def evaluate_case(case: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """Check optional JSON validity, required keys, and expected JSON values."""
    result: Dict[str, Any] = {"json_valid": None, "json_checks_passed": None}
    if not case.get("expect_json", False):
        return result
    # Strip markdown code fences that models often add even when asked for raw JSON.
    # Handles ```json ... ``` and ``` ... ``` wrappers.
    clean = answer.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean).strip()
    try:
        parsed = json.loads(clean)
    except (json.JSONDecodeError, TypeError) as exc:
        return {
            "json_valid": False,
            "json_checks_passed": False,
            "json_error": str(exc),
        }
    if not isinstance(parsed, dict):
        return {
            "json_valid": True,
            "json_checks_passed": False,
            "json_error": "Top-level JSON value must be an object",
        }
    required_keys = case.get("required_keys", [])
    missing_keys = [key for key in required_keys if key not in parsed]
    expected_values = case.get("expected_values", {})
    mismatches = {
        key: {"expected": expected, "actual": parsed.get(key)}
        for key, expected in expected_values.items()
        if parsed.get(key) != expected
    }
    passed = not missing_keys and not mismatches
    return {
        "json_valid": True,
        "json_checks_passed": passed,
        "missing_keys": missing_keys,
        "value_mismatches": mismatches,
    }


def runtime_metadata(model_path: Path, device_index: int) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_monitor": "pynvml" if HAS_PYNVML else "torch" if HAS_CUDA_TORCH else None,
        "device_index": device_index,
        "cuda_available_in_torch": HAS_CUDA_TORCH,
        "model": {
            "path": str(model_path.resolve()),
            "size_bytes": model_path.stat().st_size,
        },
    }
    digest = hashlib.sha256()
    with model_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    metadata["model"]["sha256"] = digest.hexdigest()

    try:
        import llama_cpp

        metadata["llama_cpp_version"] = getattr(llama_cpp, "__version__", "unknown")
    except Exception:
        metadata["llama_cpp_version"] = None
    if HAS_PYNVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            metadata["gpu"] = pynvml.nvmlDeviceGetName(handle)
            metadata["gpu_total_vram_mb"] = (
                pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024.0**2)
            )
            metadata["cuda_driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        except Exception as exc:
            metadata["gpu_metadata_error"] = str(exc)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark llama-cpp-python profiles and save raw JSON logs."
    )
    parser.add_argument("--model-path", required=True, help="Path to the GGUF model")
    parser.add_argument("--prompt", default="Объясни принцип работы квантования нейросетей.")
    parser.add_argument(
        "--cases-jsonl",
        help="Optional JSONL quality cases with id, prompt, and optional JSON expectations",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["baseline", "aggressive", "aggressive_q4"],
        choices=list_profiles(),
    )
    parser.add_argument(
        "--kv-ablation-context",
        type=int,
        help=(
            "Run a matched FP16/Q8_0/Q4_0 KV comparison at this context length "
            "with FlashAttention enabled for all three variants"
        ),
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "raw_logs"))
    parser.add_argument(
        "--mock", action="store_true", help="Dry run only; mock output is never saved"
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not args.mock and not model_path.is_file():
        parser.error(f"Model file does not exist: {model_path}")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.kv_ablation_context is not None and args.kv_ablation_context < 1:
        parser.error("--kv-ablation-context must be at least 1")
    cases = load_cases(Path(args.cases_jsonl)) if args.cases_jsonl else [
        {"id": "single-prompt", "category": "benchmark", "prompt": args.prompt}
    ]

    if args.mock:
        print("Mock mode requested: no inference or raw result file will be produced.")
        return

    metadata = runtime_metadata(model_path, args.device_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    if args.kv_ablation_context is not None:
        context = args.kv_ablation_context
        profiles = [
            replace(get_profile("baseline"), name=f"kv_fp16_ctx{context}_fa_on", n_ctx=context, flash_attn=True),
            replace(get_profile("aggressive"), name=f"kv_q8_ctx{context}_fa_on", n_ctx=context),
            replace(get_profile("aggressive_q4"), name=f"kv_q4_ctx{context}_fa_on", n_ctx=context),
        ]
    else:
        profiles = [get_profile(name) for name in args.profiles]

    for profile in profiles:
        profile_name = profile.name

        # Load model once for this profile
        print(f"\nLoading model for profile '{profile_name}'...")
        idle_vram_mb = get_current_vram_mb(args.device_index)
        load_start = time.perf_counter()
        model = load_llama_model(profile, str(model_path))
        load_time_sec = time.perf_counter() - load_start
        print(f"  Loaded in {load_time_sec:.1f}s. Running warm-up...")

        # Warm-up: one short generation, not saved
        run_inference(
            model=model,
            profile=profile,
            prompt=cases[0]["prompt"],
            max_tokens=min(args.max_tokens, 16),
            device_index=args.device_index,
            temperature=args.temperature,
            seed=args.seed,
            load_time_sec=load_time_sec,
            idle_vram_mb=idle_vram_mb,
        )

        # Measured runs: all repeats × all cases with the same loaded model
        runs = []
        for repeat in range(args.repeats):
            for case in cases:
                metrics = run_inference(
                    model=model,
                    profile=profile,
                    prompt=case["prompt"],
                    max_tokens=args.max_tokens,
                    device_index=args.device_index,
                    temperature=args.temperature,
                    seed=args.seed,
                    load_time_sec=load_time_sec,
                    idle_vram_mb=idle_vram_mb,
                )
                runs.append(
                    {
                        "repeat": repeat + 1,
                        "case_id": case["id"],
                        "category": case.get("category"),
                        "prompt": case["prompt"],
                        "metrics": metrics,
                        "quality_checks": evaluate_case(case, metrics["text"]),
                    }
                )

        # Unload model before next profile
        del model
        gc.collect()
        if HAS_CUDA_TORCH:
            torch.cuda.empty_cache()

        document = {
            "schema_version": 1,
            "run_id": f"{timestamp}_{profile_name}",
            "started_at_utc": timestamp,
            "mode": "real_inference",
            "profile": profile.to_dict(),
            "generation": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "repeats": args.repeats,
            },
            "runtime": metadata,
            "runs": runs,
        }
        output_path = output_dir / f"{timestamp}_{profile_name}.json"
        output_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(runs)} runs for {profile_name}: {output_path}")



if __name__ == "__main__":
    main()
