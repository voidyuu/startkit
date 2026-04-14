from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Iterable

import torch
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    create_windows_from_events,
    preprocess,
)
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    keep_only_recordings_with,
)
from matplotlib.pylab import plt
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset

from .config import Challenge1Config


def release_name(release: int | str) -> str:
    release_str = str(release)
    return release_str if release_str.startswith("R") else f"R{release_str}"


class LabeledWindowDataset(Dataset):
    def __init__(self, base_dataset: Dataset, label: int):
        self.base_dataset = base_dataset
        self.label = label

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        X = self.base_dataset[index][0]
        return X, torch.tensor(self.label, dtype=torch.long)


class MultiTaskConcatDataset(Dataset):
    def __init__(self, datasets: Iterable[Dataset]):
        self.datasets = [dataset for dataset in datasets if len(dataset) > 0]
        self.cumulative_sizes = []
        running = 0
        for dataset in self.datasets:
            running += len(dataset)
            self.cumulative_sizes.append(running)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, index: int):
        dataset_idx = bisect_right(self.cumulative_sizes, index)
        prev_size = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = index - prev_size
        return self.datasets[dataset_idx][sample_idx]


def make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def split_window_dataset_by_subject(windows, train_subjects, valid_subjects, test_subjects):
    subject_split = windows.split("subject")
    train_set, valid_set, test_set = [], [], []

    for subject in subject_split:
        if subject in train_subjects:
            train_set.append(subject_split[subject])
        elif subject in valid_subjects:
            valid_set.append(subject_split[subject])
        elif subject in test_subjects:
            test_set.append(subject_split[subject])

    def concat_or_none(datasets):
        return BaseConcatDataset(datasets) if datasets else None

    return (
        concat_or_none(train_set),
        concat_or_none(valid_set),
        concat_or_none(test_set),
    )


def split_eval_subjects(meta_information, *, random_seed: int, valid_frac: float = 0.5):
    subjects = meta_information["subject"].unique()
    removed_subjects = {
        "NDARWV769JM7",
        "NDARME789TD2",
        "NDARUA442ZVF",
        "NDARJP304NK1",
        "NDARTY128YLU",
        "NDARDW550GU6",
        "NDARLD243KRE",
        "NDARUJ292JXV",
        "NDARBA381JGH",
    }
    subjects = [subject for subject in subjects if subject not in removed_subjects]

    valid_subjects, test_subjects = train_test_split(
        subjects,
        test_size=(1 - valid_frac),
        random_state=check_random_state(random_seed),
        shuffle=True,
    )
    return set(valid_subjects), set(test_subjects)


def create_target_task_windows(config: Challenge1Config, releases: Iterable[int | str]):
    all_window_datasets = []

    for release in releases:
        release_tag = release_name(release)
        dataset = EEGChallengeDataset(
            task=config.target_task,
            release=release_tag,
            cache_dir=config.data_dir,
            mini=config.use_mini,
        )
        raw = dataset.datasets[0].raw
        print(f"Loaded raw example for {config.target_task} from {release_tag}: {raw}")

        preprocess(
            dataset,
            [
                Preprocessor(
                    annotate_trials_with_target,
                    target_field="rt_from_stimulus",
                    epoch_length=config.epoch_len_s,
                    require_stimulus=True,
                    require_response=True,
                    apply_on_array=False,
                ),
                Preprocessor(add_aux_anchors, apply_on_array=False),
            ],
            n_jobs=1,
        )

        anchor = "stimulus_anchor"
        dataset = keep_only_recordings_with(anchor, dataset)
        windows = create_windows_from_events(
            dataset,
            mapping={anchor: 0},
            trial_start_offset_samples=int(0.5 * config.sfreq),
            trial_stop_offset_samples=int(2.5 * config.sfreq),
            window_size_samples=config.window_size_samples,
            window_stride_samples=config.window_stride_samples,
            preload=True,
        )
        windows = add_extras_columns(
            windows,
            dataset,
            desc=anchor,
            keys=(
                "target",
                "rt_from_stimulus",
                "rt_from_trialstart",
                "stimulus_onset",
                "response_onset",
                "correct",
                "response_type",
            ),
        )
        all_window_datasets.extend(windows.datasets)

    return BaseConcatDataset(all_window_datasets)


def create_passive_pretraining_datasets(config: Challenge1Config, *, valid_subjects: set[str]):
    train_datasets = []
    valid_datasets = []

    for label, task_name in enumerate(config.passive_tasks):
        print(f"Preparing passive-task pretraining windows for {task_name}...")
        task_train_windows = []
        task_valid_windows = []

        for release in config.train_releases:
            dataset = EEGChallengeDataset(
                task=task_name,
                release=release_name(release),
                cache_dir=config.data_dir,
                mini=config.use_mini,
            )
            windows = create_fixed_length_windows(
                dataset,
                start_offset_samples=0,
                stop_offset_samples=None,
                window_size_samples=config.window_size_samples,
                window_stride_samples=config.window_stride_samples,
                drop_last_window=True,
                preload=True,
            )
            task_train_windows.extend(windows.datasets)

        valid_dataset = EEGChallengeDataset(
            task=task_name,
            release=release_name(config.valid_release),
            cache_dir=config.data_dir,
            mini=config.use_mini,
        )
        valid_windows = create_fixed_length_windows(
            valid_dataset,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=config.window_size_samples,
            window_stride_samples=config.window_stride_samples,
            drop_last_window=True,
            preload=True,
        )
        _, subject_valid_windows, _ = split_window_dataset_by_subject(
            valid_windows,
            set(),
            valid_subjects,
            set(),
        )
        if subject_valid_windows is not None:
            task_valid_windows.extend(subject_valid_windows.datasets)

        if task_train_windows:
            train_datasets.append(LabeledWindowDataset(BaseConcatDataset(task_train_windows), label))
        if task_valid_windows:
            valid_datasets.append(LabeledWindowDataset(BaseConcatDataset(task_valid_windows), label))

    train_dataset = MultiTaskConcatDataset(train_datasets)
    valid_dataset = MultiTaskConcatDataset(valid_datasets)
    if len(train_dataset) == 0:
        raise RuntimeError("Passive-task pretraining dataset is empty.")
    if len(valid_dataset) == 0:
        raise RuntimeError("Passive-task pretraining validation dataset is empty.")
    return train_dataset, valid_dataset


def plot_target_distribution(meta_information, output_path: str | Path = "response_time_distribution.png"):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax = meta_information["target"].plot.hist(bins=30, ax=ax, color="lightblue")
    ax.set_xlabel("Response Time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Response Times")
    plt.savefig(output_path)
    plt.close(fig)
