from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from .config import Challenge1Config


def save_training_curves(summary: dict, *, output_dir: Path, prefix: str) -> None:
    history = summary.get("history", [])
    if not history:
        return

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]
    train_metrics = [item["train_metric"] for item in history]
    val_metrics = [item["val_metric"] for item in history]
    metric_name = str(summary.get("metric_name", "metric")).lower()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix.capitalize()} Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_metrics, label=f"train_{metric_name}")
    plt.plot(epochs, val_metrics, label=f"val_{metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(summary.get("metric_name", "Metric"))
    plt.title(f"{prefix.capitalize()} {summary.get('metric_name', 'Metric')} Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_{metric_name}_curve.png")
    plt.close()


def plot_target_distribution(meta_information, output_path: str | Path = "response_time_distribution.png") -> None:
    fig, ax = plt.subplots(figsize=(15, 5))
    ax = meta_information["target"].plot.hist(bins=30, ax=ax, color="lightblue")
    ax.set_xlabel("Response Time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Response Times")
    plt.savefig(output_path)
    plt.close(fig)


def build_metrics_payload(
    *,
    config: Challenge1Config,
    run_dir: Path,
    dataset_sizes: dict[str, int],
    pretrain_summary: dict,
    finetune_summary: dict,
    test_loss: float,
    test_rmse: float,
    device: str,
) -> dict:
    return {
        "device": device,
        "run_dir": str(run_dir),
        "train_releases": list(config.train_releases),
        "valid_release": config.valid_release,
        "use_mini": config.use_mini,
        "standardization": asdict(config.standardization),
        "dataset_sizes": dataset_sizes,
        "pretrain": pretrain_summary,
        "finetune": finetune_summary,
        "final_test": {
            "loss": test_loss,
            "rmse": test_rmse,
        },
    }


def save_run_artifacts(
    *,
    config: Challenge1Config,
    run_dir: Path,
    valid_meta_information,
    model: torch.nn.Module,
    dataset_sizes: dict[str, int],
    pretrain_summary: dict,
    finetune_summary: dict,
    test_loss: float,
    test_rmse: float,
    device: str,
) -> dict:
    plot_target_distribution(
        valid_meta_information,
        output_path=run_dir / "response_time_distribution.png",
    )

    artifacts_weights_path = run_dir / "weights_challenge_1.pt"
    torch.save(model.state_dict(), artifacts_weights_path)
    print(f"Artifact weights saved to '{artifacts_weights_path}'")

    metrics = build_metrics_payload(
        config=config,
        run_dir=run_dir,
        dataset_sizes=dataset_sizes,
        pretrain_summary=pretrain_summary,
        finetune_summary=finetune_summary,
        test_loss=test_loss,
        test_rmse=test_rmse,
        device=device,
    )
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved to '{metrics_path}'")

    save_training_curves(pretrain_summary, output_dir=run_dir, prefix="pretrain")
    save_training_curves(finetune_summary, output_dir=run_dir, prefix="finetune")
    print(f"Training curves saved to '{run_dir}'")
    return metrics
