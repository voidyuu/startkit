from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Optional

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig


def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    device: str,
    regression: bool,
    print_batch_stats: bool = True,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    sum_sq_err = 0.0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats)
    for batch_idx, batch in progress_bar:
        X, y = batch[0], batch[1]
        X = X.to(device).float()
        y = y.to(device)
        if regression:
            y = y.float().view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if regression:
            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            total_examples += y_flat.numel()
            metric = (sum_sq_err / max(total_examples, 1)) ** 0.5
            metric_name = "RMSE"
        else:
            pred_labels = preds.detach().argmax(dim=1)
            total_correct += (pred_labels == y).sum().item()
            total_examples += y.numel()
            metric = total_correct / max(total_examples, 1)
            metric_name = "Acc"

        if print_batch_stats:
            progress_bar.set_description(
                f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}, {metric_name}: {metric:.6f}"
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss, metric


@torch.no_grad()
def evaluate(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    device: str,
    regression: bool,
    print_batch_stats: bool = True,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    sum_sq_err = 0.0
    n_batches = len(dataloader)

    iterator = tqdm(enumerate(dataloader), total=n_batches, disable=not print_batch_stats)
    for batch_idx, batch in iterator:
        X, y = batch[0], batch[1]
        X = X.to(device).float()
        y = y.to(device)
        if regression:
            y = y.float().view(-1, 1)

        preds = model(X)
        batch_loss = loss_fn(preds, y).item()
        total_loss += batch_loss

        if regression:
            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            total_examples += y_flat.numel()
            metric = (sum_sq_err / max(total_examples, 1)) ** 0.5
            metric_name = "RMSE"
        else:
            pred_labels = preds.detach().argmax(dim=1)
            total_correct += (pred_labels == y).sum().item()
            total_examples += y.numel()
            metric = total_correct / max(total_examples, 1)
            metric_name = "Acc"

        if print_batch_stats:
            iterator.set_description(
                f"Val Batch {batch_idx + 1}/{n_batches}, "
                f"Loss: {batch_loss:.6f}, {metric_name}: {metric:.6f}"
            )

    avg_loss = total_loss / n_batches if n_batches else float("nan")
    return avg_loss, metric


def fit_model(
    model: Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: TrainConfig,
    *,
    device: str,
    regression: bool,
    metric_name: str,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.n_epochs - 1, 1))
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()

    best_state = None
    best_epoch = None
    best_metric = float("inf") if regression else float("-inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(1, config.n_epochs + 1):
        print(f"Epoch {epoch}/{config.n_epochs}: ", end="")
        train_loss, train_metric = train_one_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            device,
            regression=regression,
        )
        val_loss, val_metric = evaluate(
            valid_loader,
            model,
            loss_fn,
            device,
            regression=regression,
        )

        print(
            f"Train {metric_name}: {train_metric:.6f}, "
            f"Average Train Loss: {train_loss:.6f}, "
            f"Val {metric_name}: {val_metric:.6f}, "
            f"Average Val Loss: {val_loss:.6f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_metric": train_metric,
                "val_loss": val_loss,
                "val_metric": val_metric,
            }
        )

        improved = (
            val_metric < best_metric - config.min_delta
            if regression
            else val_metric > best_metric + config.min_delta
        )
        if improved:
            best_metric = val_metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best Val {metric_name}: {best_metric:.6f} (epoch {best_epoch})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "config": asdict(config),
        "metric_name": metric_name,
        "regression": regression,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "history": history,
    }
    return model, best_metric, summary
