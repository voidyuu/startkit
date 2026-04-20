from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from .config import Challenge1Config
from .models import EEGNeXTransferModel, build_regression_model_from_pretrained
from .training import evaluate, fit_model


@dataclass(frozen=True)
class PretrainingResult:
    model: EEGNeXTransferModel
    best_metric: float
    summary: dict


@dataclass(frozen=True)
class FinetuningResult:
    model: Module
    best_metric: float
    summary: dict
    test_loss: float
    test_rmse: float


def run_pretraining_stage(
    config: Challenge1Config,
    *,
    device: str,
    train_loader: DataLoader,
    valid_loader: DataLoader,
) -> PretrainingResult:
    print("Starting passive-task pretraining...")
    model = EEGNeXTransferModel(
        n_outputs=len(config.passive_tasks),
        n_chans=129,
        n_times=config.window_size_samples,
        sfreq=config.sfreq,
    ).to(device)
    print(model)
    model, best_metric, summary = fit_model(
        model,
        train_loader,
        valid_loader,
        config.pretrain,
        device=device,
        regression=False,
        metric_name="Acc",
    )
    print(f"Best passive-task validation accuracy: {best_metric:.6f}")
    return PretrainingResult(model=model, best_metric=best_metric, summary=summary)


def run_finetuning_stage(
    config: Challenge1Config,
    *,
    device: str,
    pretrained_model: EEGNeXTransferModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
) -> FinetuningResult:
    print("Starting Challenge 1 fine-tuning...")
    model = build_regression_model_from_pretrained(
        pretrained_model,
        n_chans=129,
        n_times=config.window_size_samples,
        sfreq=config.sfreq,
    ).to(device)
    print(model)
    model, best_metric, summary = fit_model(
        model,
        train_loader,
        valid_loader,
        config.finetune,
        device=device,
        regression=True,
        metric_name="RMSE",
    )
    print(f"Best validation RMSE after fine-tuning: {best_metric:.6f}")

    test_loss, test_rmse = evaluate(
        test_loader,
        model,
        torch.nn.MSELoss(),
        device,
        regression=True,
    )
    print(f"Final Test RMSE: {test_rmse:.6f}, Test Loss: {test_loss:.6f}")
    return FinetuningResult(
        model=model,
        best_metric=best_metric,
        summary=summary,
        test_loss=test_loss,
        test_rmse=test_rmse,
    )
