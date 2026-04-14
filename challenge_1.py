# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/eeg2025/startkit/blob/main/challenge_1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Challenge 1: Cross-Task Transfer Learning
#
# This script implements a competition-aligned baseline:
# 1. pre-train an EEGNeX encoder on passive HBN tasks with task classification;
# 2. fine-tune the encoder on the Challenge 1 target task
#    (contrast change detection response-time regression).

# %% Imports and setup [code]
from __future__ import annotations

import copy
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from braindecode.datasets import BaseConcatDataset
from braindecode.models import EEGNeX
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
from torch import nn
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    msg = "CUDA-enabled GPU found. Training should be faster."
else:
    msg = (
        "No GPU found. Training will be carried out on CPU, which might be "
        "slower.\n\nIf running on Google Colab, you can request a GPU runtime by"
        " clicking\n`Runtime/Change runtime type` in the top bar menu, then "
        "selecting 'T4 GPU'\nunder 'Hardware accelerator'."
    )
print(msg)


# %% Configuration [code]
DATA_DIR = Path("/mnt/E/zhuyu_data/eeg-challenges")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RELEASES = [1]
VALID_RELEASE = 5
USE_MINI = True
SFREQ = 100
EPOCH_LEN_S = 2.0
WINDOW_SIZE_SAMPLES = int(EPOCH_LEN_S * SFREQ)
WINDOW_STRIDE_SAMPLES = SFREQ
RANDOM_SEED = 2025

PASSIVE_TASKS = [
    "RestingState",
    "surroundSupp",
    "DiaryOfAWimpyKid",
    "DespicableMe",
    "FunwithFractals",
    "ThePresent",
]
TARGET_TASK = "contrastChangeDetection"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    n_epochs: int
    patience: int
    min_delta: float


PRETRAIN_CONFIG = TrainConfig(
    batch_size=128,
    num_workers=0,
    lr=1e-3,
    weight_decay=1e-5,
    n_epochs=12,
    patience=3,
    min_delta=1e-4,
)

FINETUNE_CONFIG = TrainConfig(
    batch_size=128,
    num_workers=0,
    lr=1e-3,
    weight_decay=1e-5,
    n_epochs=100,
    patience=5,
    min_delta=1e-4,
)


class LabeledWindowDataset(Dataset):
    """Wrap a braindecode window dataset and replace its target by a fixed label."""

    def __init__(self, base_dataset: Dataset, label: int):
        self.base_dataset = base_dataset
        self.label = label

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        X = self.base_dataset[index][0]
        return X, torch.tensor(self.label, dtype=torch.long)


class MultiTaskConcatDataset(Dataset):
    """Concat datasets while exposing only (X, y) tuples to the loader."""

    def __init__(self, datasets: Iterable[Dataset]):
        self.datasets = [d for d in datasets if len(d) > 0]
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


class EEGNeXBackbone(nn.Module):
    def __init__(self, n_chans: int, n_times: int, sfreq: int):
        super().__init__()
        self.backbone = EEGNeX(
            n_chans=n_chans,
            n_outputs=1,
            n_times=n_times,
            sfreq=sfreq,
        )
        feature_dim = self.backbone.final_layer.in_features
        self.backbone.final_layer = nn.Identity()
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EEGNeXTransferModel(nn.Module):
    def __init__(self, n_outputs: int, n_chans: int = 129, n_times: int = 200, sfreq: int = 100):
        super().__init__()
        self.encoder = EEGNeXBackbone(n_chans=n_chans, n_times=n_times, sfreq=sfreq)
        self.head = nn.Linear(self.encoder.feature_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


def make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def build_regression_model_from_pretrained(pretrained_model: EEGNeXTransferModel) -> EEGNeX:
    model = EEGNeX(
        n_chans=129,
        n_outputs=1,
        n_times=WINDOW_SIZE_SAMPLES,
        sfreq=SFREQ,
    )
    model.block_1.load_state_dict(pretrained_model.encoder.backbone.block_1.state_dict())
    model.block_2.load_state_dict(pretrained_model.encoder.backbone.block_2.state_dict())
    model.block_3.load_state_dict(pretrained_model.encoder.backbone.block_3.state_dict())
    model.block_4.load_state_dict(pretrained_model.encoder.backbone.block_4.state_dict())
    model.block_5.load_state_dict(pretrained_model.encoder.backbone.block_5.state_dict())
    return model


def release_name(release: int | str) -> str:
    release_str = str(release)
    return release_str if release_str.startswith("R") else f"R{release_str}"


def split_subjects(meta_information, valid_frac: float = 0.1, test_frac: float = 0.1):
    subjects = meta_information["subject"].unique()
    sub_rm = [
        "NDARWV769JM7",
        "NDARME789TD2",
        "NDARUA442ZVF",
        "NDARJP304NK1",
        "NDARTY128YLU",
        "NDARDW550GU6",
        "NDARLD243KRE",
        "NDARUJ292JXV",
        "NDARBA381JGH",
    ]
    subjects = [s for s in subjects if s not in sub_rm]

    train_subj, valid_test_subject = train_test_split(
        subjects,
        test_size=(valid_frac + test_frac),
        random_state=check_random_state(RANDOM_SEED),
        shuffle=True,
    )
    valid_subj, test_subj = train_test_split(
        valid_test_subject,
        test_size=test_frac,
        random_state=check_random_state(RANDOM_SEED + 1),
        shuffle=True,
    )
    assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)
    return set(train_subj), set(valid_subj), set(test_subj)


def split_eval_subjects(meta_information, valid_frac: float = 0.5):
    subjects = meta_information["subject"].unique()
    sub_rm = [
        "NDARWV769JM7",
        "NDARME789TD2",
        "NDARUA442ZVF",
        "NDARJP304NK1",
        "NDARTY128YLU",
        "NDARDW550GU6",
        "NDARLD243KRE",
        "NDARUJ292JXV",
        "NDARBA381JGH",
    ]
    subjects = [s for s in subjects if s not in sub_rm]

    valid_subj, test_subj = train_test_split(
        subjects,
        test_size=(1 - valid_frac),
        random_state=check_random_state(RANDOM_SEED + 2),
        shuffle=True,
    )
    assert set(valid_subj) | set(test_subj) == set(subjects)
    return set(valid_subj), set(test_subj)


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

    def _concat_or_none(datasets):
        return BaseConcatDataset(datasets) if datasets else None

    return (
        _concat_or_none(train_set),
        _concat_or_none(valid_set),
        _concat_or_none(test_set),
    )


def create_target_task_windows(releases: Iterable[int | str]):
    all_window_datasets = []

    for release in releases:
        release_tag = release_name(release)
        dataset_ccd = EEGChallengeDataset(
            task=TARGET_TASK,
            release=release_tag,
            cache_dir=DATA_DIR,
            mini=USE_MINI,
        )

        raw = dataset_ccd.datasets[0].raw
        print(f"Loaded raw example for {TARGET_TASK} from {release_tag}: {raw}")

        transformation_offline = [
            Preprocessor(
                annotate_trials_with_target,
                target_field="rt_from_stimulus",
                epoch_length=EPOCH_LEN_S,
                require_stimulus=True,
                require_response=True,
                apply_on_array=False,
            ),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]
        preprocess(dataset_ccd, transformation_offline, n_jobs=1)

        anchor = "stimulus_anchor"
        shift_after_stim = 0.5
        window_len = 2.0

        dataset = keep_only_recordings_with(anchor, dataset_ccd)
        single_windows = create_windows_from_events(
            dataset,
            mapping={anchor: 0},
            trial_start_offset_samples=int(shift_after_stim * SFREQ),
            trial_stop_offset_samples=int((shift_after_stim + window_len) * SFREQ),
            window_size_samples=WINDOW_SIZE_SAMPLES,
            window_stride_samples=WINDOW_STRIDE_SAMPLES,
            preload=True,
        )

        single_windows = add_extras_columns(
            single_windows,
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
        all_window_datasets.extend(single_windows.datasets)

    return BaseConcatDataset(all_window_datasets)


def plot_target_distribution(meta_information):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax = meta_information["target"].plot.hist(bins=30, ax=ax, color="lightblue")
    ax.set_xlabel("Response Time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Response Times")
    plt.savefig("response_time_distribution.png")
    plt.close(fig)


def create_passive_pretraining_datasets(train_releases, valid_release):
    train_datasets = []
    valid_datasets = []

    for label, task_name in enumerate(PASSIVE_TASKS):
        print(f"Preparing passive-task pretraining windows for {task_name}...")
        task_train_windows = []
        task_valid_windows = []

        for release in train_releases:
            passive_dataset = EEGChallengeDataset(
                task=task_name,
                release=release_name(release),
                cache_dir=DATA_DIR,
                mini=USE_MINI,
            )
            passive_windows = create_fixed_length_windows(
                passive_dataset,
                start_offset_samples=0,
                stop_offset_samples=None,
                window_size_samples=WINDOW_SIZE_SAMPLES,
                window_stride_samples=WINDOW_STRIDE_SAMPLES,
                drop_last_window=True,
                preload=True,
            )
            task_train_windows.extend(passive_windows.datasets)

        passive_valid_dataset = EEGChallengeDataset(
            task=task_name,
            release=release_name(valid_release),
            cache_dir=DATA_DIR,
            mini=USE_MINI,
        )
        passive_valid_windows = create_fixed_length_windows(
            passive_valid_dataset,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=WINDOW_SIZE_SAMPLES,
            window_stride_samples=WINDOW_STRIDE_SAMPLES,
            drop_last_window=True,
            preload=True,
        )
        task_valid_windows.extend(passive_valid_windows.datasets)

        if task_train_windows:
            train_datasets.append(
                LabeledWindowDataset(BaseConcatDataset(task_train_windows), label=label)
            )
        if task_valid_windows:
            valid_datasets.append(
                LabeledWindowDataset(BaseConcatDataset(task_valid_windows), label=label)
            )

    train_dataset = MultiTaskConcatDataset(train_datasets)
    valid_dataset = MultiTaskConcatDataset(valid_datasets)

    if len(train_dataset) == 0:
        raise RuntimeError("Passive-task pretraining dataset is empty.")

    print(f"Passive pretraining windows: train={len(train_dataset)}, valid={len(valid_dataset)}")
    return train_dataset, valid_dataset


def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    device: str,
    regression: bool,
    print_batch_stats: bool = True,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    sum_sq_err = 0.0

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats
    )

    for batch_idx, batch in progress_bar:
        X, y = batch[0], batch[1]
        X = X.to(device).float()
        y = y.to(device)

        if regression:
            y = y.float().view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if regression:
            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            total_examples += y_flat.numel()
            metric = (sum_sq_err / max(total_examples, 1)) ** 0.5
            metric_name = "RMSE"
        else:
            pred_labels = preds.detach().argmax(dim=1)
            total_correct += (pred_labels == y).sum().item()
            total_examples += y.numel()
            metric = total_correct / max(total_examples, 1)
            metric_name = "Acc"

        if print_batch_stats:
            progress_bar.set_description(
                f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}, {metric_name}: {metric:.6f}"
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss, metric


@torch.no_grad()
def evaluate(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    device: str,
    regression: bool,
    print_batch_stats: bool = True,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    sum_sq_err = 0.0
    n_batches = len(dataloader)

    iterator = tqdm(
        enumerate(dataloader),
        total=n_batches,
        disable=not print_batch_stats,
    )

    for batch_idx, batch in iterator:
        X, y = batch[0], batch[1]
        X = X.to(device).float()
        y = y.to(device)

        if regression:
            y = y.float().view(-1, 1)

        preds = model(X)
        batch_loss = loss_fn(preds, y).item()
        total_loss += batch_loss

        if regression:
            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            total_examples += y_flat.numel()
            metric = (sum_sq_err / max(total_examples, 1)) ** 0.5
            metric_name = "RMSE"
        else:
            pred_labels = preds.detach().argmax(dim=1)
            total_correct += (pred_labels == y).sum().item()
            total_examples += y.numel()
            metric = total_correct / max(total_examples, 1)
            metric_name = "Acc"

        if print_batch_stats:
            iterator.set_description(
                f"Val Batch {batch_idx + 1}/{n_batches}, "
                f"Loss: {batch_loss:.6f}, {metric_name}: {metric:.6f}"
            )

    avg_loss = total_loss / n_batches if n_batches else float("nan")
    return avg_loss, metric


def fit_model(
    model: Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: TrainConfig,
    *,
    regression: bool,
    metric_name: str,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.n_epochs - 1, 1),
    )
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()

    best_state = None
    best_epoch = None
    best_metric = float("inf") if regression else float("-inf")
    epochs_no_improve = 0

    for epoch in range(1, config.n_epochs + 1):
        print(f"Epoch {epoch}/{config.n_epochs}: ", end="")
        train_loss, train_metric = train_one_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            device,
            regression=regression,
        )
        val_loss, val_metric = evaluate(
            valid_loader,
            model,
            loss_fn,
            device,
            regression=regression,
        )

        print(
            f"Train {metric_name}: {train_metric:.6f}, "
            f"Average Train Loss: {train_loss:.6f}, "
            f"Val {metric_name}: {val_metric:.6f}, "
            f"Average Val Loss: {val_loss:.6f}"
        )

        improved = (
            val_metric < best_metric - config.min_delta
            if regression
            else val_metric > best_metric + config.min_delta
        )

        if improved:
            best_metric = val_metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best Val {metric_name}: {best_metric:.6f} (epoch {best_epoch})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metric


def main():
    train_windows = create_target_task_windows(TRAIN_RELEASES)
    valid_release_windows = create_target_task_windows([VALID_RELEASE])
    valid_meta_information = valid_release_windows.get_metadata()
    plot_target_distribution(valid_meta_information)

    valid_subjects, test_subjects = split_eval_subjects(valid_meta_information)
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

    pretrain_train_set, pretrain_valid_set = create_passive_pretraining_datasets(
        train_releases=TRAIN_RELEASES,
        valid_release=VALID_RELEASE,
    )

    pretrain_train_loader = make_loader(
        pretrain_train_set,
        batch_size=PRETRAIN_CONFIG.batch_size,
        shuffle=True,
        num_workers=PRETRAIN_CONFIG.num_workers,
    )
    pretrain_valid_loader = make_loader(
        pretrain_valid_set,
        batch_size=PRETRAIN_CONFIG.batch_size,
        shuffle=False,
        num_workers=PRETRAIN_CONFIG.num_workers,
    )

    print("Starting passive-task pretraining...")
    pretrain_model = EEGNeXTransferModel(
        n_outputs=len(PASSIVE_TASKS),
        n_chans=129,
        n_times=WINDOW_SIZE_SAMPLES,
        sfreq=SFREQ,
    ).to(device)
    print(pretrain_model)

    pretrain_model, best_pretrain_acc = fit_model(
        pretrain_model,
        pretrain_train_loader,
        pretrain_valid_loader,
        PRETRAIN_CONFIG,
        regression=False,
        metric_name="Acc",
    )
    print(f"Best passive-task validation accuracy: {best_pretrain_acc:.6f}")

    print("Starting Challenge 1 fine-tuning...")
    model = build_regression_model_from_pretrained(pretrain_model).to(device)
    print(model)

    train_loader = make_loader(
        train_set,
        batch_size=FINETUNE_CONFIG.batch_size,
        shuffle=True,
        num_workers=FINETUNE_CONFIG.num_workers,
    )
    valid_loader = make_loader(
        valid_set,
        batch_size=FINETUNE_CONFIG.batch_size,
        shuffle=False,
        num_workers=FINETUNE_CONFIG.num_workers,
    )
    test_loader = make_loader(
        test_set,
        batch_size=FINETUNE_CONFIG.batch_size,
        shuffle=False,
        num_workers=FINETUNE_CONFIG.num_workers,
    )

    model, best_val_rmse = fit_model(
        model,
        train_loader,
        valid_loader,
        FINETUNE_CONFIG,
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


if __name__ == "__main__":
    main()
