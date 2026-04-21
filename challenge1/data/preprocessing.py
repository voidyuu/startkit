from __future__ import annotations

from dataclasses import dataclass
import math

from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import Preprocessor, exponential_moving_standardize, preprocess
import numpy as np

from ..config import Challenge1Config


@dataclass(frozen=True)
class GlobalZScoreStats:
    mean: np.ndarray
    std: np.ndarray
    total_samples: int


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


def fit_standardization(
    dataset: BaseConcatDataset,
    *,
    config: Challenge1Config,
    context: str,
) -> GlobalZScoreStats | None:
    if not config.standardization.enabled:
        return None

    if config.standardization.mode == "exponential_moving":
        return None
    if config.standardization.mode == "global_zscore":
        return fit_global_zscore_stats(
            dataset,
            eps=config.standardization.eps,
            context=context,
        )

    raise ValueError(f"Unsupported standardization mode {config.standardization.mode!r}.")


def standardize_recordings(
    dataset: BaseConcatDataset,
    *,
    config: Challenge1Config,
    context: str,
    global_zscore_stats: GlobalZScoreStats | None = None,
) -> BaseConcatDataset:
    if not config.standardization.enabled:
        return dataset

    if config.standardization.mode == "exponential_moving":
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

    if config.standardization.mode == "global_zscore":
        if global_zscore_stats is None:
            raise ValueError("Global z-score standardization requires fitted training statistics.")
        apply_global_zscore_stats(
            dataset,
            stats=global_zscore_stats,
            context=context,
        )
        return dataset

    raise ValueError(f"Unsupported standardization mode {config.standardization.mode!r}.")


def fit_global_zscore_stats(
    dataset: BaseConcatDataset,
    *,
    eps: float,
    context: str,
) -> GlobalZScoreStats:
    total_sum = None
    total_sum_sq = None
    total_samples = 0

    for recording in dataset.datasets:
        raw = recording.raw
        raw.load_data()
        data = np.asarray(raw.get_data(), dtype=np.float64)

        if total_sum is None:
            total_sum = np.zeros(data.shape[0], dtype=np.float64)
            total_sum_sq = np.zeros(data.shape[0], dtype=np.float64)
        elif data.shape[0] != total_sum.shape[0]:
            raise RuntimeError(
                f"Inconsistent channel count while fitting global z-score for {context}: "
                f"expected {total_sum.shape[0]}, got {data.shape[0]}."
            )

        total_sum += data.sum(axis=1)
        total_sum_sq += np.square(data).sum(axis=1)
        total_samples += data.shape[1]

    if total_sum is None or total_sum_sq is None or total_samples <= 0:
        raise RuntimeError(f"Cannot fit global z-score for empty dataset {context!r}.")

    mean = total_sum / total_samples
    variance = np.maximum(total_sum_sq / total_samples - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), eps)

    print(
        f"Fitted global z-score stats for {context} "
        f"(samples={total_samples}, channels={mean.shape[0]}, eps={eps})"
    )
    return GlobalZScoreStats(mean=mean, std=std, total_samples=total_samples)


def apply_global_zscore_stats(
    dataset: BaseConcatDataset,
    *,
    stats: GlobalZScoreStats,
    context: str,
) -> None:
    print(
        f"Applying global z-score standardization to {context} "
        f"(fit_samples={stats.total_samples})"
    )
    mean = stats.mean[:, np.newaxis]
    std = stats.std[:, np.newaxis]

    for recording in dataset.datasets:
        raw = recording.raw
        raw.load_data()
        data = raw._data
        if data.shape[0] != stats.mean.shape[0]:
            raise RuntimeError(
                f"Inconsistent channel count while applying global z-score to {context}: "
                f"expected {stats.mean.shape[0]}, got {data.shape[0]}."
            )
        data[...] = (data - mean) / std


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
