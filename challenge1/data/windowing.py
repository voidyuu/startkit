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
from .preprocessing import (
    filter_recordings,
    filter_windows_with_valid_target,
    fit_standardization,
    standardize_recordings,
)
from .splits import release_name, split_recording_dataset_by_subject


def load_recordings(
    config: Challenge1Config,
    *,
    task: str,
    release: int | str,
    min_n_times: int,
) -> BaseConcatDataset:
    release_tag = release_name(release)
    dataset = EEGChallengeDataset(
        task=task,
        release=release_tag,
        cache_dir=config.data_dir,
        mini=config.use_mini,
    )
    return filter_recordings(
        dataset,
        expected_n_chans=129,
        min_n_times=min_n_times,
        context=f"{task} {release_tag}",
    )


def load_recordings_for_releases(
    config: Challenge1Config,
    *,
    task: str,
    releases: Iterable[int | str],
    min_n_times: int,
) -> BaseConcatDataset:
    all_recordings = []

    for release in releases:
        dataset = load_recordings(
            config,
            task=task,
            release=release,
            min_n_times=min_n_times,
        )
        all_recordings.extend(dataset.datasets)

    if not all_recordings:
        raise RuntimeError(f"{task} dataset is empty after loading releases.")

    return BaseConcatDataset(all_recordings)


def load_target_task_recordings(
    config: Challenge1Config,
    releases: Iterable[int | str],
) -> BaseConcatDataset:
    return load_recordings_for_releases(
        config,
        task=config.target_task,
        releases=releases,
        min_n_times=int(2.5 * config.sfreq),
    )


def concat_datasets(
    datasets: Iterable[BaseConcatDataset | None],
    *,
    context: str,
) -> BaseConcatDataset:
    combined = []
    for dataset in datasets:
        if dataset is None:
            continue
        combined.extend(dataset.datasets)

    if not combined:
        raise RuntimeError(f"{context} dataset is empty after concatenation.")

    return BaseConcatDataset(combined)


def create_target_task_windows_from_recordings(
    config: Challenge1Config,
    dataset: BaseConcatDataset,
    *,
    context: str,
) -> BaseConcatDataset:
    raw = dataset.datasets[0].raw
    print(f"Loaded raw example for {context}: {raw}")

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
    return filter_windows_with_valid_target(
        windows,
        target_name="target",
        context=context,
    )


def create_target_task_windows(config: Challenge1Config, releases: Iterable[int | str]):
    releases = tuple(releases)
    recordings = load_target_task_recordings(config, releases)
    standardization_state = fit_standardization(
        recordings,
        config=config,
        context=f"{config.target_task} training recordings",
    )
    recordings = standardize_recordings(
        recordings,
        config=config,
        context=f"{config.target_task} releases {tuple(releases)}",
        global_zscore_stats=standardization_state,
    )
    return create_target_task_windows_from_recordings(
        config,
        recordings,
        context=f"{config.target_task} releases {tuple(releases)}",
    )


def create_passive_task_windows_from_recordings(
    config: Challenge1Config,
    dataset: BaseConcatDataset,
) -> BaseConcatDataset:
    return create_fixed_length_windows(
        dataset,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=config.window_size_samples,
        window_stride_samples=config.window_stride_samples,
        drop_last_window=True,
        preload=True,
    )


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
        train_recording_splits = []

        for release in config.train_releases:
            dataset = load_recordings(
                config,
                task=task_name,
                release=release,
                min_n_times=config.window_size_samples,
            )
            if train_subjects is not None and release_name(release) == valid_release_tag:
                subject_train_recordings, _, _ = split_recording_dataset_by_subject(
                    dataset,
                    train_subjects,
                    set(),
                    set(),
                )
                if subject_train_recordings is not None:
                    train_recording_splits.append(subject_train_recordings)
            else:
                train_recording_splits.append(dataset)

        train_recordings = concat_datasets(
            train_recording_splits,
            context=f"{task_name} pretraining train recordings",
        )
        valid_recordings = load_recordings(
            config,
            task=task_name,
            release=valid_release_tag,
            min_n_times=config.window_size_samples,
        )
        _, valid_recordings, _ = split_recording_dataset_by_subject(
            valid_recordings,
            set(),
            valid_subjects,
            set(),
        )
        if valid_recordings is None:
            raise RuntimeError(f"Validation split is empty for passive task {task_name}.")

        standardization_state = fit_standardization(
            train_recordings,
            config=config,
            context=f"{task_name} pretraining train recordings",
        )
        train_recordings = standardize_recordings(
            train_recordings,
            config=config,
            context=f"{task_name} pretraining train recordings",
            global_zscore_stats=standardization_state,
        )
        valid_recordings = standardize_recordings(
            valid_recordings,
            config=config,
            context=f"{task_name} pretraining valid recordings",
            global_zscore_stats=standardization_state,
        )

        train_windows = create_passive_task_windows_from_recordings(
            config,
            train_recordings,
        )
        valid_windows = create_passive_task_windows_from_recordings(
            config,
            valid_recordings,
        )

        if len(train_windows) > 0:
            train_datasets.append(LabeledWindowDataset(train_windows, label))
        if len(valid_windows) > 0:
            valid_datasets.append(LabeledWindowDataset(valid_windows, label))

    train_dataset = MultiTaskConcatDataset(train_datasets)
    valid_dataset = MultiTaskConcatDataset(valid_datasets)
    if len(train_dataset) == 0:
        raise RuntimeError("Passive-task pretraining dataset is empty.")
    if len(valid_dataset) == 0:
        raise RuntimeError("Passive-task pretraining validation dataset is empty.")
    return train_dataset, valid_dataset
