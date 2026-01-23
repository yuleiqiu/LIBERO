from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn

from .dict_of_tensor_mixin import DictOfTensorMixin


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.flatten()
    return torch.from_numpy(np.asarray(x)).flatten()


def _normalize(x: Union[torch.Tensor, np.ndarray], params, forward: bool = True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    return x.reshape(src_shape)


class LinearNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fields = nn.ModuleDict()

    def load_state_dict(self, state_dict, strict: bool = True):
        prefix = "fields."
        field_names = set()
        for key in state_dict.keys():
            if key.startswith(prefix):
                parts = key.split(".")
                if len(parts) > 1:
                    field_names.add(parts[1])
        for name in sorted(field_names):
            if name not in self.fields:
                self.fields[name] = SingleFieldLinearNormalizer()
        return super().load_state_dict(state_dict, strict=strict)

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return self.fields[key]

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        if not isinstance(value, SingleFieldLinearNormalizer):
            raise TypeError("LinearNormalizer values must be SingleFieldLinearNormalizer")
        self.fields[key] = value

    def _normalize_impl(self, x, forward: bool):
        if isinstance(x, dict):
            out = {}
            for key, value in x.items():
                field = self.fields[key]
                out[key] = field.normalize(value) if forward else field.unnormalize(value)
            return out
        if "_default" not in self.fields:
            raise RuntimeError("LinearNormalizer is not initialized")
        field = self.fields["_default"]
        return field.normalize(x) if forward else field.unnormalize(x)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ):
        scale = nn.Parameter(_to_tensor(scale), requires_grad=False)
        offset = nn.Parameter(_to_tensor(offset), requires_grad=False)
        stats = {
            name: nn.Parameter(_to_tensor(value), requires_grad=False)
            for name, value in input_stats_dict.items()
        }
        for value in stats.values():
            if value.shape != scale.shape or value.dtype != scale.dtype:
                raise ValueError("input stats must match scale shape/dtype")

        params = {"scale": scale, "offset": offset}
        for name, value in stats.items():
            params[f"input_stats_{name}"] = value
        params_dict = nn.ParameterDict(params)
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)
