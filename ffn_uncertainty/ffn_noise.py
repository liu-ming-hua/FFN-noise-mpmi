import os
import logging
from types import MethodType
from dataclasses import asdict
from typing import Any, Dict, List

import torch

from .config import FFNUncertaintyConfig

logger = logging.getLogger(__name__)


class FFNNoiseInjector:
    """Stage-1 placeholder for FFN-only noise injection.

    This keeps a stable API so stage-2 can swap in real FFN perturbation
    logic without changing the entry script and serialization path.
    
    Supports both model architectures:
    - Llama-style: gate_proj + up_proj + down_proj (SwiGLU)
    - Qwen-style: gate_up_proj + down_proj (fused FFN)
    """

    def __init__(self, config: FFNUncertaintyConfig):
        self.config = config
        self.patched_modules: List[str] = []
        self.model_type = None

    def apply(self, model: Any) -> None:
        """Attach FFN perturbation to supported HF decoder modules."""

        def _make_noisy_forward(module: Any, prefix: str):
            # Check for Llama-style FFN: separate gate, up, down projections
            if hasattr(module, "gate_proj") and hasattr(module, "up_proj") and hasattr(module, "down_proj"):
                def forward(self, hidden_states, *args, **kwargs):
                    gate = self.gate_proj(hidden_states)
                    up = self.up_proj(hidden_states)
                    gate_up = torch.cat([gate, up], dim=-1)
                    gate_up = apply_ffn_noise(gate_up, prefix)
                    gate, up = gate_up.chunk(2, dim=-1)
                    try:
                        hidden_states = self.act_fn(gate) * up
                    except TypeError:
                        hidden_states = self.act_fn(torch.cat([gate, up], dim=-1))
                    return self.down_proj(hidden_states)

                return MethodType(forward, module), "llama_swiglu"

            # Check for Qwen-style FFN: fused gate_up projection
            if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
                def forward(self, hidden_states, *args, **kwargs):
                    gate_up = self.gate_up_proj(hidden_states)
                    gate_up = apply_ffn_noise(gate_up, prefix)
                    hidden_states = self.act_fn(gate_up)
                    return self.down_proj(hidden_states)

                return MethodType(forward, module), "qwen_fused"

            return None, None

        llama_count = 0
        qwen_count = 0
        
        for module_name, module in model.named_modules():
            if getattr(module, "_tokur_ffn_patched", False):
                continue
            patched_forward, arch_type = _make_noisy_forward(module, module_name)
            if patched_forward is not None:
                module.forward = patched_forward
                module._tokur_ffn_patched = True
                self.patched_modules.append(module_name)
                
                if arch_type == "llama_swiglu":
                    llama_count += 1
                elif arch_type == "qwen_fused":
                    qwen_count += 1
        
        # Determine model type based on patched modules
        if llama_count > 0 and qwen_count == 0:
            self.model_type = "llama"
            logger.info(f"Detected Llama-style architecture: patched {llama_count} FFN modules")
        elif qwen_count > 0 and llama_count == 0:
            self.model_type = "qwen"
            logger.info(f"Detected Qwen-style architecture: patched {qwen_count} FFN modules")
        elif llama_count > 0 and qwen_count > 0:
            logger.warning(f"Mixed architecture detected: {llama_count} Llama-style + {qwen_count} Qwen-style modules")
            self.model_type = "mixed"
        else:
            logger.warning("No FFN modules detected for patching")
            self.model_type = "unknown"

    def metadata(self) -> Dict[str, Any]:
        """Return serializable config metadata for debugging and audit."""
        meta = asdict(self.config)
        meta.update({
            "model_type": self.model_type,
            "num_patched_modules": len(self.patched_modules),
            "patched_modules": self.patched_modules[:10],  # Store first 10 for brevity
        })
        return meta


def _parse_layer_scope(raw_scope: str) -> set[int] | None:
    if not raw_scope or raw_scope == "all":
        return None

    layers: set[int] = set()
    for part in raw_scope.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return layers


def _extract_layer_index(prefix: str) -> int | None:
    parts = prefix.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def ffn_noise_is_enabled() -> bool:
    return os.environ.get("TOKUR_FFN_NOISE_ENABLED", "0") == "1"


def should_apply_ffn_noise(prefix: str) -> bool:
    if not ffn_noise_is_enabled():
        return False

    target_layer = os.environ.get("TOKUR_FFN_NOISE_TARGET_LAYER", "")
    if target_layer:
        layer_index = _extract_layer_index(prefix)
        if layer_index is None or str(layer_index) != target_layer:
            return False

    raw_scope = os.environ.get("TOKUR_FFN_NOISE_LAYERS", "all")
    allowed_layers = _parse_layer_scope(raw_scope)
    if allowed_layers is None:
        return True

    layer_index = _extract_layer_index(prefix)
    return layer_index is not None and layer_index in allowed_layers


def apply_ffn_noise(hidden_states: torch.Tensor, prefix: str) -> torch.Tensor:
    """Apply neuron-level random masking plus Gaussian noise to FFN activations."""
    if not should_apply_ffn_noise(prefix):
        return hidden_states

    sigma = float(os.environ.get("TOKUR_FFN_NOISE_SIGMA", "0.0"))
    prob = float(os.environ.get("TOKUR_FFN_NOISE_PROB", "0.0"))
    mode = os.environ.get("TOKUR_FFN_NOISE_MODE", "gaussian")

    if sigma <= 0.0 or prob <= 0.0:
        return hidden_states

    if mode != "gaussian":
        return hidden_states

    # Sample a neuron-level mask vector for this forward pass.
    # The mask is defined per neuron (last dimension) and applied across
    # all token positions/time steps. This implements the idea of
    # selecting a subset of neurons in the layer to perturb for the
    # current Monte Carlo draw.

    # hidden_states shape can be (..., d_m). We create a mask of shape
    # (d_m,) and then broadcast it to hidden_states.
    d_m = hidden_states.shape[-1]

    # If user provided an explicit per-layer mask via env var,
    # parse it. The env var name uses the extracted layer index.
    layer_index = _extract_layer_index(prefix)
    explicit_mask = None
    if layer_index is not None:
        env_name = f"TOKUR_FFN_NOISE_MASK_{layer_index}"
        raw = os.environ.get(env_name, "")
        if raw:
            try:
                idxs = [int(x) for x in raw.split(",") if x.strip()]
                vec = torch.zeros(d_m, dtype=hidden_states.dtype, device=hidden_states.device)
                for i in idxs:
                    if 0 <= i < d_m:
                        vec[i] = 1.0
                explicit_mask = vec
            except Exception:
                explicit_mask = None

    batch_size = hidden_states.shape[0]
    if explicit_mask is not None:
        neuron_mask = explicit_mask.unsqueeze(0).expand(batch_size, -1)
    else:
        neuron_mask = (torch.rand(batch_size, d_m, device=hidden_states.device) < prob).to(hidden_states.dtype)

    expand_shape = [batch_size] + [1] * (hidden_states.dim() - 2) + [d_m]
    mask = neuron_mask.view(*expand_shape).expand_as(hidden_states)

    noise = torch.randn_like(hidden_states) * sigma
    return hidden_states + noise * mask

