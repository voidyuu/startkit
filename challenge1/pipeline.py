from __future__ import annotations

import json

import torch
from matplotlib import pyplot as plt

from .config import Challenge1Config
from .data import (
    create_passive_pretraining_datasets,
    create_target_task_windows,
    make_loader,
    plot_target_distribution,
    release_name,
    split_eval_subjects,
    split_train_valid_test_subjects,
    split_window_dataset_by_subject,
)
from .models import EEGNeXTransferModel, build_regression_model_from_pretrained
from .training import evaluate, fit_model


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_device_banner(device: str) -> None:
    if device == "cuda":
        print("CUDA-enabled GPU found. Training should be faster.")
    else:
        print(
            "No GPU found. Training will be carried out on CPU, which might be "
            "slower.\n\nIf running on Google Colab, you can request a GPU runtime by"
            " clicking\n`Runtime/Change runtime type` in the top bar menu, then "
            "selecting 'T4 GPU'\nunder 'Hardware accelerator'."
        )


def save_training_curves(summary: dict, *, output_dir, prefix: str) -> None:
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


def run_training(config: Challenge1Config | None = None, *, device: str | None = None) -> None:
    config = config or Challenge1Config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    device = device or get_default_device()
    print_device_banner(device)

    train_subjects: set[str] | None = None
    normalized_train_releases = {release_name(release) for release in config.train_releases}
    valid_release_tag = release_name(config.valid_release)
    if normalized_train_releases == {valid_release_tag}:
        print(
            f"Train releases and validation release both resolve to {valid_release_tag}; "
            "splitting one target-task dataset into train/valid/test subjects."
        )
        all_target_windows = create_target_task_windows(config, [config.valid_release])
        valid_meta_information = all_target_windows.get_metadata()
        train_subjects, valid_subjects, test_subjects = split_train_valid_test_subjects(
            valid_meta_information,
            random_seed=config.random_seed + 2,
        )
        train_set, valid_set, test_set = split_window_dataset_by_subject(
            all_target_windows,
            train_subjects,
            valid_subjects,
            test_subjects,
        )
    else:
        train_set = create_target_task_windows(config, config.train_releases)
        valid_release_windows = create_target_task_windows(config, [config.valid_release])
        valid_meta_information = valid_release_windows.get_metadata()

        valid_subjects, test_subjects = split_eval_subjects(
            valid_meta_information,
            random_seed=config.random_seed + 2,
        )
        _, valid_set, test_set = split_window_dataset_by_subject(
            valid_release_windows,
            set(),
            valid_subjects,
            test_subjects,
        )
    if train_set is None or len(train_set) == 0:
        raise RuntimeError("Training split is empty after subject partitioning.")
    if valid_set is None or len(valid_set) == 0:
        raise RuntimeError("Validation split is empty after subject partitioning.")
    if test_set is None or len(test_set) == 0:
        raise RuntimeError("Test split is empty after subject partitioning.")

    print("Number of examples in each split")
    print(f"Train:\t{len(train_set)}")
    print(f"Valid:\t{len(valid_set)}")
    print(f"Test:\t{len(test_set)}")

    pretrain_train_set, pretrain_valid_set = create_passive_pretraining_datasets(
        config,
        valid_subjects=valid_subjects,
        train_subjects=train_subjects,
    )
    pretrain_train_loader = make_loader(
        pretrain_train_set,
        batch_size=config.pretrain.batch_size,
        shuffle=True,
        num_workers=config.pretrain.num_workers,
    )
    pretrain_valid_loader = make_loader(
        pretrain_valid_set,
        batch_size=config.pretrain.batch_size,
        shuffle=False,
        num_workers=config.pretrain.num_workers,
    )

    print("Starting passive-task pretraining...")
    pretrain_model = EEGNeXTransferModel(
        n_outputs=len(config.passive_tasks),
        n_chans=129,
        n_times=config.window_size_samples,
        sfreq=config.sfreq,
    ).to(device)
    print(pretrain_model)
    pretrain_model, best_pretrain_acc, pretrain_summary = fit_model(
        pretrain_model,
        pretrain_train_loader,
        pretrain_valid_loader,
        config.pretrain,
        device=device,
        regression=False,
        metric_name="Acc",
    )
    print(f"Best passive-task validation accuracy: {best_pretrain_acc:.6f}")

    print("Starting Challenge 1 fine-tuning...")
    model = build_regression_model_from_pretrained(
        pretrain_model,
        n_chans=129,
        n_times=config.window_size_samples,
        sfreq=config.sfreq,
    ).to(device)
    print(model)

    train_loader = make_loader(
        train_set,
        batch_size=config.finetune.batch_size,
        shuffle=True,
        num_workers=config.finetune.num_workers,
    )
    valid_loader = make_loader(
        valid_set,
        batch_size=config.finetune.batch_size,
        shuffle=False,
        num_workers=config.finetune.num_workers,
    )
    test_loader = make_loader(
        test_set,
        batch_size=config.finetune.batch_size,
        shuffle=False,
        num_workers=config.finetune.num_workers,
    )

    model, best_val_rmse, finetune_summary = fit_model(
        model,
        train_loader,
        valid_loader,
        config.finetune,
        device=device,
        regression=True,
        metric_name="RMSE",
    )
    print(f"Best validation RMSE after fine-tuning: {best_val_rmse:.6f}")

    test_loss, test_rmse = evaluate(
        test_loader,
        model,
        torch.nn.MSELoss(),
        device,
        regression=True,
    )
    print(f"Final Test RMSE: {test_rmse:.6f}, Test Loss: {test_loss:.6f}")

    run_dir = config.make_run_dir()
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving training artifacts to '{run_dir}'")

    plot_target_distribution(
        valid_meta_information,
        output_path=run_dir / "response_time_distribution.png",
    )

    artifacts_weights_path = run_dir / "weights_challenge_1.pt"
    torch.save(model.state_dict(), artifacts_weights_path)
    print(f"Artifact weights saved to '{artifacts_weights_path}'")

    metrics = {
        "device": device,
        "run_dir": str(run_dir),
        "train_releases": list(config.train_releases),
        "valid_release": config.valid_release,
        "use_mini": config.use_mini,
        "dataset_sizes": {
            "train": len(train_set),
            "valid": len(valid_set),
            "test": len(test_set),
            "pretrain_train": len(pretrain_train_set),
            "pretrain_valid": len(pretrain_valid_set),
        },
        "pretrain": pretrain_summary,
        "finetune": finetune_summary,
        "final_test": {
            "loss": test_loss,
            "rmse": test_rmse,
        },
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved to '{metrics_path}'")
    save_training_curves(pretrain_summary, output_dir=run_dir, prefix="pretrain")
    save_training_curves(finetune_summary, output_dir=run_dir, prefix="finetune")
    print(f"Training curves saved to '{run_dir}'")
