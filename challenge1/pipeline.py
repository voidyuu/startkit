from __future__ import annotations

import torch

from .config import Challenge1Config
from .data import (
    create_passive_pretraining_datasets,
    create_target_task_windows,
    make_loader,
    plot_target_distribution,
    split_eval_subjects,
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


def run_training(config: Challenge1Config | None = None, *, device: str | None = None) -> None:
    config = config or Challenge1Config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    device = device or get_default_device()
    print_device_banner(device)

    train_windows = create_target_task_windows(config, config.train_releases)
    valid_release_windows = create_target_task_windows(config, [config.valid_release])
    valid_meta_information = valid_release_windows.get_metadata()
    plot_target_distribution(valid_meta_information)

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
    train_set = train_windows

    print("Number of examples in each split")
    print(f"Train:\t{len(train_set)}")
    print(f"Valid:\t{len(valid_set)}")
    print(f"Test:\t{len(test_set)}")

    pretrain_train_set, pretrain_valid_set = create_passive_pretraining_datasets(config)
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
    pretrain_model, best_pretrain_acc = fit_model(
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

    model, best_val_rmse = fit_model(
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

    torch.save(model.state_dict(), "weights_challenge_1.pt")
    print("Model saved as 'weights_challenge_1.pt'")
