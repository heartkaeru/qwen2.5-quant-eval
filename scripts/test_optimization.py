"""Benchmark and compare inference optimization profiles for llama-cpp-python.

Designed for Google Colab (Tesla T4) and local CUDA GPU environments.
Measures peak VRAM allocation, TTFT (Time To First Token), and throughput (tokens/sec)
across 'baseline' and 'aggressive' profiles.
"""

import argparse
import gc
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output in Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.profiles import (
    AGGRESSIVE_PROFILE,
    BASELINE_PROFILE,
    InferenceProfile,
    get_profile,
    list_profiles,
)

# Optional GPU monitoring tools
HAS_PYNVML = False
try:
    import pynvml

    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False

HAS_TORCH = False
try:
    import torch

    HAS_TORCH = torch.cuda.is_available()
except Exception:
    HAS_TORCH = False


def get_current_vram_mb(device_index: int = 0) -> float:
    """Get current allocated/used VRAM in Megabytes."""
    if HAS_PYNVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.used) / (1024.0**2)
        except Exception:
            pass

    if HAS_TORCH:
        try:
            # torch.cuda.mem_get_info returns (free, total) from driver
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            used_bytes = total_bytes - free_bytes
            return float(used_bytes) / (1024.0**2)
        except Exception:
            try:
                return float(torch.cuda.memory_allocated(device_index)) / (1024.0**2)
            except Exception:
                pass

    return 0.0


class PeakVRAMTracker:
    """Background monitor thread to sample peak VRAM during execution."""

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
            current = get_current_vram_mb(self.device_index)
            if current > self.peak_vram_mb:
                self.peak_vram_mb = current
            time.sleep(self.interval_sec)

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        final_sample = get_current_vram_mb(self.device_index)
        if final_sample > self.peak_vram_mb:
            self.peak_vram_mb = final_sample
        return self.peak_vram_mb


def run_mock_profile(
    profile: InferenceProfile,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """Simulate profile execution for validation on environments without GPU or weights."""
    print(f"  [MOCK MODE] Simulating run for profile: {profile.name}...")
    time.sleep(0.3)

    # Simulated metrics based on theoretical models:
    # Baseline (4096 ctx, F16 KV): ~5.2 GB total VRAM, slower prefill
    # Aggressive (1024 ctx, Q8_0 KV, FlashAttn): ~4.4 GB total VRAM, faster prefill
    if profile.name == "baseline":
        peak_vram = 5240.0
        ttft = 0.28
        tps = 32.5
    elif profile.name == "aggressive_q4":
        peak_vram = 4180.0
        ttft = 0.14
        tps = 41.0
    else:  # aggressive (Q8_0)
        peak_vram = 4450.0
        ttft = 0.17
        tps = 39.2

    generated_tokens = min(max_tokens, 64)
    gen_time = (generated_tokens - 1) / tps if generated_tokens > 1 else 0.01
    total_time = ttft + gen_time

    return {
        "profile_name": profile.name,
        "n_ctx": profile.n_ctx,
        "flash_attn": profile.flash_attn,
        "type_k": profile.type_k,
        "type_v": profile.type_v,
        "peak_vram_mb": peak_vram,
        "ttft_sec": ttft,
        "total_time_sec": total_time,
        "tokens_generated": generated_tokens,
        "tokens_per_sec": tps,
        "text": f"Mock output for prompt '{prompt[:30]}...' under {profile.name}.",
    }


def run_llama_profile(
    profile: InferenceProfile,
    model_path: str,
    prompt: str,
    max_tokens: int,
    device_index: int = 0,
) -> Dict[str, Any]:
    """Execute real inference using llama_cpp.Llama and profile parameters."""
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Run 'pip install llama-cpp-python' "
            "or use --mock for simulated testing."
        ) from exc

    llama_kwargs = profile.to_llama_kwargs()
    print(f"\n==================================================")
    print(f"Loading model with profile '{profile.name}'...")
    print(f"Parameters: {llama_kwargs}")
    print(f"==================================================")

    # Start VRAM tracking
    tracker = PeakVRAMTracker(device_index=device_index).start()
    idle_vram = get_current_vram_mb(device_index)

    # Initialize model
    t_init_start = time.perf_counter()
    model = Llama(
        model_path=model_path,
        verbose=False,
        **llama_kwargs,
    )
    init_time = time.perf_counter() - t_init_start
    print(f"Model loaded in {init_time:.2f} s. Running generation...")

    # Generation with streaming to measure TTFT
    response_chunks: List[str] = []
    generated_tokens = 0
    ttft_sec: Optional[float] = None

    t_gen_start = time.perf_counter()
    stream = model(
        prompt,
        max_tokens=max_tokens,
        stream=True,
        echo=False,
    )

    for chunk in stream:
        if ttft_sec is None:
            ttft_sec = time.perf_counter() - t_gen_start
        delta_text = chunk["choices"][0]["text"]
        response_chunks.append(delta_text)
        generated_tokens += 1

    total_gen_time = time.perf_counter() - t_gen_start
    peak_vram_mb = tracker.stop()

    if ttft_sec is None:
        ttft_sec = total_gen_time

    decode_time = max(total_gen_time - ttft_sec, 1e-4)
    tokens_per_sec = (generated_tokens - 1) / decode_time if generated_tokens > 1 else 0.0

    # Explicitly release memory
    del model
    gc.collect()
    if HAS_TORCH:
        torch.cuda.empty_cache()
    time.sleep(1.0)

    full_text = "".join(response_chunks).strip()

    return {
        "profile_name": profile.name,
        "n_ctx": profile.n_ctx,
        "flash_attn": profile.flash_attn,
        "type_k": profile.type_k,
        "type_v": profile.type_v,
        "idle_vram_mb": idle_vram,
        "peak_vram_mb": peak_vram_mb,
        "ttft_sec": ttft_sec,
        "total_time_sec": total_gen_time,
        "tokens_generated": generated_tokens,
        "tokens_per_sec": tokens_per_sec,
        "text": full_text,
    }


def print_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """Print structured comparison table and optimization delta."""
    print("\n" + "=" * 80)
    print("                      РЕЗУЛЬТАТЫ СРАВНЕНИЯ ПРОФИЛЕЙ")
    print("=" * 80)

    header = (
        f"{'Профиль':<15} | {'Контекст':<8} | {'KV Type':<8} | "
        f"{'FlashAttn':<9} | {'Peak VRAM (MB)':<14} | {'TTFT (ms)':<10} | {'TPS (tok/s)':<11}"
    )
    print(header)
    print("-" * 80)

    for name, r in results.items():
        kv_str = f"k:{r['type_k']}/v:{r['type_v']}"
        fa_str = "ON" if r["flash_attn"] else "OFF"
        ttft_ms = r["ttft_sec"] * 1000.0
        line = (
            f"{name:<15} | {r['n_ctx']:<8} | {kv_str:<8} | "
            f"{fa_str:<9} | {r['peak_vram_mb']:<14.1f} | {ttft_ms:<10.1f} | {r['tokens_per_sec']:<11.2f}"
        )
        print(line)

    print("=" * 80)

    # Calculate delta if baseline and aggressive are present
    if "baseline" in results and "aggressive" in results:
        base = results["baseline"]
        aggr = results["aggressive"]

        vram_saved_mb = base["peak_vram_mb"] - aggr["peak_vram_mb"]
        vram_saved_pct = (vram_saved_mb / base["peak_vram_mb"] * 100.0) if base["peak_vram_mb"] > 0 else 0.0

        ttft_diff_ms = (base["ttft_sec"] - aggr["ttft_sec"]) * 1000.0
        speedup_pct = (
            ((aggr["tokens_per_sec"] - base["tokens_per_sec"]) / base["tokens_per_sec"] * 100.0)
            if base["tokens_per_sec"] > 0
            else 0.0
        )

        print("\nИТОГОВЫЙ ВЫИГРЫШ ОТ АГРЕССИВНОЙ ОПТИМИЗАЦИИ (Aggressive vs Baseline):")
        print(f"  • Экономия VRAM:       {vram_saved_mb:+.1f} MB ({vram_saved_pct:.1f}% снижения)")
        print(f"  • Ускорение TTFT:      {ttft_diff_ms:+.1f} ms (быстрее отклик на первый токен)")
        print(f"  • Пропускная способность: {aggr['tokens_per_sec']:.2f} tok/s vs {base['tokens_per_sec']:.2f} tok/s ({speedup_pct:+.1f}%)")
        print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение Baseline и Aggressive профилей оптимизации llama-cpp-python."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Путь к файлу модели в формате GGUF (например, models/qwen2.5-7b-q4_k_m.gguf).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Объясни в трех коротких предложениях принцип работы квантования нейросетей.",
        help="Тестовый промпт для генерации.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Максимальное количество токенов для генерации.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["baseline", "aggressive"],
        choices=list_profiles(),
        help="Список профилей для тестирования (по умолчанию: baseline aggressive).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Запустить в режиме симуляции без загрузки реальной модели (для локальной отладки без GPU).",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Индекс GPU для мониторинга VRAM (по умолчанию: 0).",
    )

    args = parser.parse_args()

    # Determine execution mode
    use_mock = args.mock or not args.model_path or not os.path.isfile(args.model_path)
    if not use_mock and not os.path.isfile(args.model_path):
        print(f"Ошибка: Файл модели '{args.model_path}' не найден.", file=sys.stderr)
        sys.exit(1)

    if use_mock:
        if not args.mock:
            print(
                "[ВНИМАНИЕ] Файл модели не указан или не найден. "
                "Скрипт переключен в режим --mock (симуляция) для проверки логики замеров."
            )

    results: Dict[str, Dict[str, Any]] = {}

    for profile_name in args.profiles:
        profile = get_profile(profile_name)
        if use_mock:
            res = run_mock_profile(profile, args.prompt, args.max_tokens)
        else:
            res = run_llama_profile(
                profile=profile,
                model_path=args.model_path,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                device_index=args.device_index,
            )
        results[profile_name] = res

    print_comparison(results)


if __name__ == "__main__":
    main()
