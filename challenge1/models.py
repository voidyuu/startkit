from __future__ import annotations

import torch
from braindecode.models import EEGNeX
from torch import nn


class EEGNeXBackbone(nn.Module):
    def __init__(self, n_chans: int, n_times: int, sfreq: int):
        super().__init__()
        self.backbone = EEGNeX(
            n_chans=n_chans,
            n_outputs=1,
            n_times=n_times,
            sfreq=sfreq,
        )
        self.feature_dim = self.backbone.final_layer.in_features
        self.backbone.final_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EEGNeXTransferModel(nn.Module):
    def __init__(self, n_outputs: int, n_chans: int = 129, n_times: int = 200, sfreq: int = 100):
        super().__init__()
        self.encoder = EEGNeXBackbone(n_chans=n_chans, n_times=n_times, sfreq=sfreq)
        self.head = nn.Linear(self.encoder.feature_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def build_regression_model_from_pretrained(
    pretrained_model: EEGNeXTransferModel,
    *,
    n_chans: int,
    n_times: int,
    sfreq: int,
) -> EEGNeX:
    model = EEGNeX(
        n_chans=n_chans,
        n_outputs=1,
        n_times=n_times,
        sfreq=sfreq,
    )
    model.block_1.load_state_dict(pretrained_model.encoder.backbone.block_1.state_dict())
    model.block_2.load_state_dict(pretrained_model.encoder.backbone.block_2.state_dict())
    model.block_3.load_state_dict(pretrained_model.encoder.backbone.block_3.state_dict())
    model.block_4.load_state_dict(pretrained_model.encoder.backbone.block_4.state_dict())
    model.block_5.load_state_dict(pretrained_model.encoder.backbone.block_5.state_dict())
    return model
