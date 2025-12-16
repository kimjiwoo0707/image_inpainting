import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from .cbam import CBAMBlock

class CBAMUnetPlusPlus(nn.Module):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 1,
        classes: int = 1,
    ):
        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        ch = self.model.encoder.out_channels[-1]
        self.cbam = CBAMBlock(channels=ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.encoder(x)
        features[-1] = self.cbam(features[-1])
        x = self.model.decoder(*features)
        x = self.model.segmentation_head(x)
        return x
