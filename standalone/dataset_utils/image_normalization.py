import numpy as np


def _infer_channel_axis(images):
    if images.ndim < 3:
        raise ValueError(f"expected image dims >= 3, got shape {images.shape}")
    if images.shape[-1] in (1, 3):
        return -1
    if images.shape[-3] in (1, 3):
        return -3
    raise ValueError(f"cannot infer channel axis from shape {images.shape}")


def normalize_images(images, image_norm, input_scale="0_255"):
    norm = str(image_norm or "none").lower()
    if input_scale not in ("0_255", "0_1"):
        raise ValueError(f"input_scale must be '0_255' or '0_1', got {input_scale}")
    arr = images.astype(np.float32, copy=False)

    if norm in ("none", "", "null"):
        return arr * 255.0 if input_scale == "0_1" else arr

    if norm in ("scale_0_1", "0_1", "unit"):
        return arr if input_scale == "0_1" else arr / 255.0

    if norm in ("imagenet", "imagenet_norm"):
        if input_scale == "0_255":
            arr = arr / 255.0
        channel_axis = _infer_channel_axis(arr)
        if channel_axis == -1:
            if arr.shape[-1] != 3:
                raise ValueError(
                    f"imagenet normalization expects 3 channels, got {arr.shape[-1]}"
                )
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            return (arr - mean) / std
        if arr.shape[-3] != 3:
            raise ValueError(
                f"imagenet normalization expects 3 channels, got {arr.shape[-3]}"
            )
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        return (arr - mean) / std

    raise ValueError(f"unknown image_norm setting: {image_norm}")
