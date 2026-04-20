from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from .config import Challenge1Config
from .data import (
    create_passive_pretraining_datasets,
    create_target_task_windows,
    make_loader,
    release_name,
    split_eval_subjects,
    split_train_valid_test_subjects,
    split_window_dataset_by_subject,
)


@dataclass(frozen=True)
class TargetTaskData:
    train_set: Dataset
    valid_set: Dataset
    test_set: Dataset
    valid_meta_information: object
    train_subjects: set[str] | None
    valid_subjects: set[str]
    test_subjects: set[str]


@dataclass(frozen=True)
class PretrainingData:
    train_set: Dataset
    valid_set: Dataset
    train_loader: DataLoader
    valid_loader: DataLoader


@dataclass(frozen=True)
class FinetuneLoaders:
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader


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


def prepare_target_task_data(config: Challenge1Config) -> TargetTaskData:
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

    return TargetTaskData(
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        valid_meta_information=valid_meta_information,
        train_subjects=train_subjects,
        valid_subjects=valid_subjects,
        test_subjects=test_subjects,
    )


def build_pretraining_data(config: Challenge1Config, target_data: TargetTaskData) -> PretrainingData:
    train_set, valid_set = create_passive_pretraining_datasets(
        config,
        valid_subjects=target_data.valid_subjects,
        train_subjects=target_data.train_subjects,
    )
    train_loader = make_loader(
        train_set,
        batch_size=config.pretrain.batch_size,
        shuffle=True,
        num_workers=config.pretrain.num_workers,
    )
    valid_loader = make_loader(
        valid_set,
        batch_size=config.pretrain.batch_size,
        shuffle=False,
        num_workers=config.pretrain.num_workers,
    )
    return PretrainingData(
        train_set=train_set,
        valid_set=valid_set,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )


def build_finetune_loaders(config: Challenge1Config, target_data: TargetTaskData) -> FinetuneLoaders:
    return FinetuneLoaders(
        train_loader=make_loader(
            target_data.train_set,
            batch_size=config.finetune.batch_size,
            shuffle=True,
            num_workers=config.finetune.num_workers,
        ),
        valid_loader=make_loader(
            target_data.valid_set,
            batch_size=config.finetune.batch_size,
            shuffle=False,
            num_workers=config.finetune.num_workers,
        ),
        test_loader=make_loader(
            target_data.test_set,
            batch_size=config.finetune.batch_size,
            shuffle=False,
            num_workers=config.finetune.num_workers,
        ),
    )


def describe_target_split_sizes(target_data: TargetTaskData) -> dict[str, int]:
    return {
        "train": len(target_data.train_set),
        "valid": len(target_data.valid_set),
        "test": len(target_data.test_set),
    }
