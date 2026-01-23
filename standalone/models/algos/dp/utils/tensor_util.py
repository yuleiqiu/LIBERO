"""
Minimal tensor utilities used by DP vision components.
"""
import collections
from typing import Any, Callable, Dict

import numpy as np
import torch


def recursive_dict_list_tuple_apply(x, type_func_dict: Dict[type, Callable[[Any], Any]]):
    if isinstance(x, (dict, collections.OrderedDict)):
        out = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else {}
        for key, value in x.items():
            out[key] = recursive_dict_list_tuple_apply(value, type_func_dict)
        return out
    if isinstance(x, (list, tuple)):
        items = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        return tuple(items) if isinstance(x, tuple) else items
    for t, func in type_func_dict.items():
        if isinstance(x, t):
            return func(x)
    raise NotImplementedError(f"Cannot handle data type {type(x)}")


def unsqueeze(x, dim):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda v: v.unsqueeze(dim=dim),
            np.ndarray: lambda v: np.expand_dims(v, axis=dim),
            type(None): lambda v: v,
        },
    )


def expand_at_single(x, size, dim):
    if isinstance(x, torch.Tensor):
        if dim >= x.ndimension() or x.shape[dim] != 1:
            raise ValueError("expand_at expects a singleton dimension to expand")
        expand_dims = [-1] * x.ndimension()
        expand_dims[dim] = size
        return x.expand(*expand_dims)
    if x.shape[dim] != 1:
        raise ValueError("expand_at expects a singleton dimension to expand")
    shape = list(x.shape)
    shape[dim] = size
    return np.broadcast_to(x, shape)


def expand_at(x, size, dim):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda v, s=size, d=dim: expand_at_single(v, s, d),
            np.ndarray: lambda v, s=size, d=dim: expand_at_single(v, s, d),
            type(None): lambda v: v,
        },
    )


def unsqueeze_expand_at(x, size, dim):
    return expand_at(unsqueeze(x, dim), size, dim)


def flatten_single(x, begin_axis=1):
    fixed_size = x.shape[:begin_axis]
    return x.reshape(*fixed_size, -1)


def flatten(x, begin_axis=1):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda v, b=begin_axis: flatten_single(v, begin_axis=b),
            np.ndarray: lambda v, b=begin_axis: flatten_single(v, begin_axis=b),
            type(None): lambda v: v,
        },
    )


def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    if begin_axis > end_axis or begin_axis < 0 or end_axis >= len(x.shape):
        raise ValueError("invalid reshape range")
    shape = x.shape
    final_shape = []
    for idx in range(len(shape)):
        if idx == begin_axis:
            final_shape.extend(target_dims)
        elif idx < begin_axis or idx > end_axis:
            final_shape.append(shape[idx])
    return x.reshape(*final_shape)


def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda v, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                v, begin_axis=b, end_axis=e, target_dims=t
            ),
            np.ndarray: lambda v, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                v, begin_axis=b, end_axis=e, target_dims=t
            ),
            type(None): lambda v: v,
        },
    )


def join_dimensions(x, begin_axis, end_axis):
    return reshape_dimensions(x, begin_axis, end_axis, target_dims=[-1])
