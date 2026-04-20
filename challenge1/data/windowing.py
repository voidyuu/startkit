from __future__ import annotations

from typing import Iterable

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

from ..config import Challenge1Config
from .datasets import LabeledWindowDataset, MultiTaskConcatDataset
from .preprocessing import filter_recordings, filter_windows_with_valid_target, standardize_recordings
from .splits import release_name, split_window_dataset_by_subject


def create_target_task_windows(config: Challenge1Config, releases: Iterable[int | str]):
    all_window_datasets = []
    min_target_recording_samples = int(2.5 * config.sfreq)

    for release in releases:
        release_tag = release_name(release)
        dataset = EEGChallengeDataset(
            task=config.target_task,
            release=release_tag,
            cache_dir=config.data_dir,
            mini=config.use_mini,
        )
        dataset = filter_recordings(
            dataset,
            expected_n_chans=129,
            min_n_times=min_target_recording_samples,
            context=f"{config.target_task} {release_tag}",
        )
        dataset = standardize_recordings(
            dataset,
            config=config,
            context=f"{config.target_task} {release_tag}",
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
        windows = filter_windows_with_valid_target(
            windows,
            target_name="target",
            context=f"{config.target_task} {release_tag}",
        )
        all_window_datasets.extend(windows.datasets)

    return BaseConcatDataset(all_window_datasets)


def create_passive_pretraining_datasets(
    config: Challenge1Config,
    *,
    valid_subjects: set[str],
    train_subjects: set[str] | None = None,
):
    train_datasets = []
    valid_datasets = []
    valid_release_tag = release_name(config.valid_release)

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
            dataset = filter_recordings(
                dataset,
                expected_n_chans=129,
                min_n_times=config.window_size_samples,
                context=f"{task_name} {release_name(release)}",
            )
            dataset = standardize_recordings(
                dataset,
                config=config,
                context=f"{task_name} {release_name(release)}",
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
            if train_subjects is not None and release_name(release) == valid_release_tag:
                subject_train_windows, _, _ = split_window_dataset_by_subject(
                    windows,
                    train_subjects,
                    set(),
                    set(),
                )
                if subject_train_windows is not None:
                    task_train_windows.extend(subject_train_windows.datasets)
            else:
                task_train_windows.extend(windows.datasets)

        valid_dataset = EEGChallengeDataset(
            task=task_name,
            release=valid_release_tag,
            cache_dir=config.data_dir,
            mini=config.use_mini,
        )
        valid_dataset = filter_recordings(
            valid_dataset,
            expected_n_chans=129,
            min_n_times=config.window_size_samples,
            context=f"{task_name} {valid_release_tag}",
        )
        valid_dataset = standardize_recordings(
            valid_dataset,
            config=config,
            context=f"{task_name} {valid_release_tag}",
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
