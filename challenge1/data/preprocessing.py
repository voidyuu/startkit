from __future__ import annotations

import math

from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import Preprocessor, exponential_moving_standardize, preprocess

from ..config import Challenge1Config


def filter_recordings(
    dataset: BaseConcatDataset,
    *,
    expected_n_chans: int,
    min_n_times: int,
    context: str,
) -> BaseConcatDataset:
    filtered_datasets = []
    dropped_bad_channels = 0
    dropped_too_short = 0

    for recording in dataset.datasets:
        raw = recording.raw
        if len(raw.ch_names) != expected_n_chans:
            dropped_bad_channels += 1
            continue
        if raw.n_times < min_n_times:
            dropped_too_short += 1
            continue
        filtered_datasets.append(recording)

    if not filtered_datasets:
        raise RuntimeError(
            f"{context} dataset is empty after filtering invalid recordings "
            f"(expected_n_chans={expected_n_chans}, min_n_times={min_n_times})."
        )

    if dropped_bad_channels or dropped_too_short:
        print(
            f"Filtered {context}: kept {len(filtered_datasets)} recordings, "
            f"dropped {dropped_bad_channels} with unexpected channel count and "
            f"{dropped_too_short} that were too short."
        )

    return BaseConcatDataset(filtered_datasets)


def standardize_recordings(
    dataset: BaseConcatDataset,
    *,
    config: Challenge1Config,
    context: str,
) -> BaseConcatDataset:
    if not config.standardization.enabled:
        return dataset

    print(
        f"Applying exponential moving standardization to {context} "
        f"(factor_new={config.standardization.factor_new}, "
        f"init_block_samples={config.standardization_init_block_samples}, "
        f"eps={config.standardization.eps})"
    )
    preprocess(
        dataset,
        [
            Preprocessor(
                exponential_moving_standardize,
                factor_new=config.standardization.factor_new,
                init_block_size=config.standardization_init_block_samples,
                eps=config.standardization.eps,
            ),
        ],
        n_jobs=1,
    )
    return dataset


def filter_windows_with_valid_target(
    windows: BaseConcatDataset,
    *,
    target_name: str,
    context: str,
) -> BaseConcatDataset:
    filtered_datasets = []
    dropped_invalid_target = 0

    for window_dataset in windows.datasets:
        metadata = getattr(window_dataset, "metadata", None)
        if metadata is None or target_name not in metadata:
            dropped_invalid_target += 1
            continue
        target_values = metadata[target_name]
        has_valid_target = any(_is_finite_numeric(value) for value in target_values)
        if not has_valid_target:
            dropped_invalid_target += 1
            continue
        filtered_datasets.append(window_dataset)

    if not filtered_datasets:
        raise RuntimeError(
            f"{context} dataset is empty after filtering windows with invalid '{target_name}'."
        )

    if dropped_invalid_target:
        print(
            f"Filtered {context}: kept {len(filtered_datasets)} windows, "
            f"dropped {dropped_invalid_target} with missing or non-finite '{target_name}'."
        )

    return BaseConcatDataset(filtered_datasets)


def _is_finite_numeric(value) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
