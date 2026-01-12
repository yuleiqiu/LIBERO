import torch.nn as nn

try:
    from libero.lifelong.models.modules.rgb_modules import ResnetEncoder as LiberoResnet
except ImportError as exc:
    raise ImportError(
        "libero ResnetEncoder not found; ensure LIBERO is installed in the env."
    ) from exc


class ResnetEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        output_size,
        pretrained=False,
        remove_layer_num=2,
        no_stride=False,
    ):
        super().__init__()
        self.encoder = LiberoResnet(
            input_shape=input_shape,
            output_size=output_size,
            pretrained=pretrained,
            freeze=False,
            remove_layer_num=remove_layer_num,
            no_stride=no_stride,
            language_dim=1,
            language_fusion="none",
        )

    def forward(self, x):
        return self.encoder(x)
