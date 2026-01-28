from dataclasses import dataclass


@dataclass
class NormalizerConfig:
    enable: bool = True
    mode: str = "limits"
    output_min: float = -1.0
    output_max: float = 1.0
    range_eps: float = 1e-4
    fit_offset: bool = True
    last_n_dims: int = 1
    normalize_obs: bool = True
    normalize_actions: bool = True
