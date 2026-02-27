#!/usr/bin/env python3
"""Inspect torchvision backbone extraction logic used in standalone/models/encoders/image.py."""

import argparse

import torch
import torch.nn as nn
import torchvision


def _resolve_weights(backbone: str, pretrained: bool):
    if not pretrained:
        return None

    enum_base = backbone
    if backbone.startswith("resnet"):
        enum_base = "ResNet" + backbone[len("resnet") :]
    else:
        enum_base = backbone[0].upper() + backbone[1:]
    enum_name = f"{enum_base}_Weights"
    weights_enum = getattr(torchvision.models, enum_name, None)
    return weights_enum.DEFAULT if weights_enum is not None else None


def build_backbone(
    backbone: str,
    pretrained: bool,
    input_channels: int,
    remove_layer_num: int,
    no_stride: bool,
):
    model_fn = getattr(torchvision.models, backbone, None)
    if model_fn is None:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if remove_layer_num < 0:
        raise ValueError("remove_layer_num must be non-negative")

    weights = _resolve_weights(backbone, pretrained)
    try:
        model = model_fn(weights=weights)
    except TypeError:
        model = model_fn(pretrained=pretrained)

    if input_channels != 3:
        model.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
    if no_stride:
        model.conv1.stride = (1, 1)
        model.maxpool.stride = (1, 1)

    named_layers = list(model.named_children())
    if remove_layer_num >= len(named_layers):
        raise ValueError(
            f"remove_layer_num={remove_layer_num} is too large; model has {len(named_layers)} children"
        )
    if remove_layer_num:
        kept_layers = named_layers[:-remove_layer_num]
        dropped_layers = named_layers[-remove_layer_num:]
    else:
        kept_layers = named_layers
        dropped_layers = []

    seq = nn.Sequential(*(layer for _, layer in kept_layers))
    return model, named_layers, kept_layers, dropped_layers, seq


def _print_layers(title: str, named_layers):
    print(title)
    for i, (name, layer) in enumerate(named_layers):
        print(f"  [{i:02d}] {name:<12s} {layer.__class__.__name__}")
    if not named_layers:
        print("  (none)")
    print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--remove-layer-num", type=int, default=2)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--no-stride", action="store_true")
    parser.add_argument("--input-height", type=int, default=128)
    parser.add_argument("--input-width", type=int, default=128)
    parser.add_argument(
        "--print-model",
        action="store_true",
        help="print full torchvision model repr",
    )
    args = parser.parse_args()

    model, named_layers, kept_layers, dropped_layers, seq = build_backbone(
        backbone=args.backbone,
        pretrained=args.pretrained,
        input_channels=args.input_channels,
        remove_layer_num=args.remove_layer_num,
        no_stride=args.no_stride,
    )

    print(f"Backbone: {args.backbone}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Input channels: {args.input_channels}")
    print(f"no_stride: {args.no_stride}")
    print(f"remove_layer_num: {args.remove_layer_num}")
    print("")

    _print_layers("children() order:", named_layers)
    _print_layers("kept by image.py logic:", kept_layers)
    _print_layers("removed by image.py logic:", dropped_layers)

    with torch.no_grad():
        dummy = torch.zeros(1, args.input_channels, args.input_height, args.input_width)
        out = seq(dummy)
    print(
        f"Dummy forward shape (1, {args.input_channels}, {args.input_height}, {args.input_width}) -> {tuple(out.shape)}"
    )

    if args.print_model:
        print("")
        print("Full model:")
        print(model)


if __name__ == "__main__":
    main()
