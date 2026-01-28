import collections
from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Optional

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform, functional as F

from standalone.configs.data import ImageTransformConfig, ImageTransformsConfig


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations."""

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: Optional[List[float]] = None,
        n_subset: Optional[int] = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1.0] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order
        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video."""

    def __init__(self, sharpness) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int, float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError("sharpness should be a single number or a sequence of length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


class RandomCrop(Transform):
    """Randomly crop an image while optionally keeping the original size."""

    def __init__(
        self,
        size=None,
        padding=0,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
    ) -> None:
        super().__init__()
        self.size = size
        self.padding = padding
        self.pad_if_needed = bool(pad_if_needed)
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def _get_hw(inpt):
        if torch.is_tensor(inpt):
            return int(inpt.shape[-2]), int(inpt.shape[-1])
        raise TypeError("RandomCrop expects torch.Tensor inputs")

    @staticmethod
    def _parse_size(size, inpt_h, inpt_w):
        if size is None:
            return inpt_h, inpt_w
        if isinstance(size, int):
            return int(size), int(size)
        if isinstance(size, Sequence) and len(size) == 2:
            return int(size[0]), int(size[1])
        raise TypeError("size should be int, sequence of length 2, or None")

    @staticmethod
    def _parse_padding(padding):
        if padding is None:
            return 0, 0, 0, 0
        if isinstance(padding, int):
            pad = int(padding)
            return pad, pad, pad, pad
        if isinstance(padding, Sequence):
            padding = list(padding)
            if len(padding) == 2:
                pad_w = int(padding[0])
                pad_h = int(padding[1])
                return pad_w, pad_h, pad_w, pad_h
            if len(padding) == 4:
                return tuple(int(p) for p in padding)
        raise TypeError("padding should be int or sequence of length 2 or 4")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        inpt = flat_inputs[0]
        inpt_h, inpt_w = self._get_hw(inpt)
        crop_h, crop_w = self._parse_size(self.size, inpt_h, inpt_w)
        pad_left, pad_top, pad_right, pad_bottom = self._parse_padding(self.padding)

        padded_h = inpt_h + pad_top + pad_bottom
        padded_w = inpt_w + pad_left + pad_right

        if self.pad_if_needed and padded_w < crop_w:
            diff = crop_w - padded_w
            pad_left += diff // 2
            pad_right += diff - diff // 2
            padded_w = crop_w
        if self.pad_if_needed and padded_h < crop_h:
            diff = crop_h - padded_h
            pad_top += diff // 2
            pad_bottom += diff - diff // 2
            padded_h = crop_h

        if crop_h > padded_h or crop_w > padded_w:
            raise ValueError(
                f"requested crop size {(crop_h, crop_w)} is larger than input size {(padded_h, padded_w)}"
            )

        max_i = padded_h - crop_h
        max_j = padded_w - crop_w
        top = int(torch.randint(0, max_i + 1, (1,)).item()) if max_i > 0 else 0
        left = int(torch.randint(0, max_j + 1, (1,)).item()) if max_j > 0 else 0
        return {
            "top": top,
            "left": left,
            "height": crop_h,
            "width": crop_w,
            "padding": (pad_left, pad_top, pad_right, pad_bottom),
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        padding = params["padding"]
        if any(padding):
            inpt = self._call_kernel(
                F.pad,
                inpt,
                padding=padding,
                fill=self.fill,
                padding_mode=self.padding_mode,
            )
        return self._call_kernel(
            F.crop,
            inpt,
            top=params["top"],
            left=params["left"],
            height=params["height"],
            width=params["width"],
        )


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    if cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    if cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    if cfg.type == "RandomCrop":
        return RandomCrop(**cfg.kwargs)
    if cfg.type == "RandomAffine":
        return v2.RandomAffine(**cfg.kwargs)
    raise ValueError(f"Transform '{cfg.type}' is not valid.")


class ImageTransforms(Transform):
    """Compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self.weights = []
        self.transforms = []
        for _, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue
            self.transforms.append(make_transform_from_config(tf_cfg))
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=self.transforms,
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
