from dataclasses import dataclass
from typing import Optional


@dataclass
class FFNUncertaintyConfig:
    """Configuration for standalone FFN noise uncertainty workflow."""

    noise_sigma: float = 0.05
    noise_prob: float = 0.1
    noise_mode: str = "gaussian"
    noise_layers: str = "all"
    unc_num_samples: int = 4
    seed: Optional[int] = None
