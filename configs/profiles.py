"""Inference profiles configuration for llama-cpp-python benchmarks.

Defines profiles for evaluating the trade-off between resource utilization
(VRAM, TTFT, throughput) and generation quality on consumer and cloud GPUs
(e.g., Google Colab with NVIDIA Tesla T4 16GB).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# GGML quantization type constants.
# We attempt to import them from llama_cpp if installed,
# otherwise fallback to canonical GGML enum integer values.
try:
    import llama_cpp

    GGML_TYPE_F16 = int(llama_cpp.GGML_TYPE_F16)
    GGML_TYPE_Q8_0 = int(llama_cpp.GGML_TYPE_Q8_0)
    GGML_TYPE_Q4_0 = int(llama_cpp.GGML_TYPE_Q4_0)
except (ImportError, AttributeError):
    # Standard ggml_type enum mappings from ggml.h
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q8_0 = 8


@dataclass
class InferenceProfile:
    """Configuration profile for llama-cpp-python model initialization."""

    name: str
    description: str
    n_ctx: int
    flash_attn: bool
    type_k: int
    type_v: int
    n_gpu_layers: int = -1
    n_batch: int = 512
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_llama_kwargs(self) -> Dict[str, Any]:
        """Convert profile fields to keyword arguments accepted by Llama().

        Returns:
            Dictionary ready to be unpacked as Llama(model_path=..., **kwargs).
        """
        kwargs: Dict[str, Any] = {
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_batch": self.n_batch,
            "flash_attn": self.flash_attn,
            "type_k": self.type_k,
            "type_v": self.type_v,
        }
        kwargs.update(self.extra_kwargs)
        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to serializable dictionary."""
        data = asdict(self)
        return data


# --- Profiles Registry ---

BASELINE_PROFILE = InferenceProfile(
    name="baseline",
    description=(
        "Контрольный профиль: стандартный контекст 4096 токенов, "
        "несжатый KV-кэш (FP16), стандартное внимание (flash_attn=False)."
    ),
    n_ctx=4096,
    flash_attn=False,
    type_k=GGML_TYPE_F16,
    type_v=GGML_TYPE_F16,
    n_gpu_layers=-1,
    n_batch=512,
)

AGGRESSIVE_PROFILE = InferenceProfile(
    name="aggressive",
    description=(
        "Агрессивная оптимизация: урезанный контекст 1024 токена, "
        "квантованный KV-кэш Q8_0 для K и V, включенный FlashAttention."
    ),
    n_ctx=1024,
    flash_attn=True,
    type_k=GGML_TYPE_Q8_0,
    type_v=GGML_TYPE_Q8_0,
    n_gpu_layers=-1,
    n_batch=512,
)

AGGRESSIVE_Q4_PROFILE = InferenceProfile(
    name="aggressive_q4",
    description=(
        "Экстремальная оптимизация: урезанный контекст 1024 токена, "
        "глубоко квантованный KV-кэш Q4_0 для K и V, включенный FlashAttention."
    ),
    n_ctx=1024,
    flash_attn=True,
    type_k=GGML_TYPE_Q4_0,
    type_v=GGML_TYPE_Q4_0,
    n_gpu_layers=-1,
    n_batch=512,
)

PROFILES: Dict[str, InferenceProfile] = {
    "baseline": BASELINE_PROFILE,
    "aggressive": AGGRESSIVE_PROFILE,
    "aggressive_q4": AGGRESSIVE_Q4_PROFILE,
}


def get_profile(name: str) -> InferenceProfile:
    """Retrieve an inference profile by name.

    Args:
        name: Name of the profile ('baseline', 'aggressive', 'aggressive_q4').

    Returns:
        The corresponding InferenceProfile instance.

    Raises:
        KeyError: If profile name is not found in registry.
    """
    normalized = name.strip().lower()
    if normalized not in PROFILES:
        valid = ", ".join(PROFILES.keys())
        raise KeyError(f"Unknown profile '{name}'. Available profiles: {valid}")
    return PROFILES[normalized]


def list_profiles() -> List[str]:
    """List available profile names."""
    return list(PROFILES.keys())
