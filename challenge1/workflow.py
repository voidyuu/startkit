from __future__ import annotations

from dataclasses import dataclass

from braindecode.datasets import BaseConcatDataset
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Challenge1Config
from .data import create_passive_pretraining_datasets, make_loader, release_name
from .data.preprocessing import fit_standardization, standardize_recordings
from .data.splits import (
    get_dataset_subjects,
    split_eval_subjects_from_subjects,
    split_recording_dataset_by_subject,
    split_train_valid_test_subjects_from_subjects,
    split_window_dataset_by_subject,
)
from .data.windowing import (
    create_target_task_windows_from_recordings,
    load_target_task_recordings,
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
        all_target_recordings = load_target_task_recordings(config, [config.valid_release])
        train_subjects, valid_subjects, test_subjects = split_train_valid_test_subjects_from_subjects(
            get_dataset_subjects(all_target_recordings),
            random_seed=config.random_seed + 2,
        )
        train_recordings, valid_recordings, test_recordings = split_recording_dataset_by_subject(
            all_target_recordings,
            train_subjects,
            valid_subjects,
            test_subjects,
        )
        if train_recordings is None or valid_recordings is None or test_recordings is None:
            raise RuntimeError("Target-task same-release split produced an empty recording partition.")

        standardization_state = fit_standardization(
            train_recordings,
            config=config,
            context=f"{config.target_task} train recordings",
        )
        train_recordings = standardize_recordings(
            train_recordings,
            config=config,
            context=f"{config.target_task} train recordings",
            global_zscore_stats=standardization_state,
        )
        valid_recordings = standardize_recordings(
            valid_recordings,
            config=config,
            context=f"{config.target_task} valid recordings",
            global_zscore_stats=standardization_state,
        )
        test_recordings = standardize_recordings(
            test_recordings,
            config=config,
            context=f"{config.target_task} test recordings",
            global_zscore_stats=standardization_state,
        )

        train_set = create_target_task_windows_from_recordings(
            config,
            train_recordings,
            context=f"{config.target_task} train recordings",
        )
        valid_set = create_target_task_windows_from_recordings(
            config,
            valid_recordings,
            context=f"{config.target_task} valid recordings",
        )
        test_set = create_target_task_windows_from_recordings(
            config,
            test_recordings,
            context=f"{config.target_task} test recordings",
        )
        valid_meta_information = BaseConcatDataset(
            train_set.datasets + valid_set.datasets + test_set.datasets
        ).get_metadata()
    else:
        train_recordings = load_target_task_recordings(config, config.train_releases)
        standardization_state = fit_standardization(
            train_recordings,
            config=config,
            context=f"{config.target_task} train recordings",
        )
        train_recordings = standardize_recordings(
            train_recordings,
            config=config,
            context=f"{config.target_task} train recordings",
            global_zscore_stats=standardization_state,
        )
        train_set = create_target_task_windows_from_recordings(
            config,
            train_recordings,
            context=f"{config.target_task} train recordings",
        )

        valid_release_recordings = load_target_task_recordings(config, [config.valid_release])
        valid_subjects, test_subjects = split_eval_subjects_from_subjects(
            get_dataset_subjects(valid_release_recordings),
            random_seed=config.random_seed + 2,
        )
        valid_release_recordings = standardize_recordings(
            valid_release_recordings,
            config=config,
            context=f"{config.target_task} eval recordings",
            global_zscore_stats=standardization_state,
        )
        valid_release_windows = create_target_task_windows_from_recordings(
            config,
            valid_release_recordings,
            context=f"{config.target_task} eval recordings",
        )
        _, valid_set, test_set = split_window_dataset_by_subject(
            valid_release_windows,
            set(),
            valid_subjects,
            test_subjects,
        )
        if valid_set is None or test_set is None:
            raise RuntimeError("Evaluation split is empty after subject partitioning.")
        valid_meta_information = BaseConcatDataset(
            valid_set.datasets + test_set.datasets
        ).get_metadata()

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
