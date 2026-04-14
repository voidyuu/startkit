from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    n_epochs: int
    patience: int
    min_delta: float


@dataclass(frozen=True)
class Challenge1Config:
    data_dir: Path = Path("/mnt/E/zhuyu_data/eeg-challenges")
    artifacts_dir: Path = Path("artifacts/challenge1")
    train_releases: tuple[int, ...] = (1, 2, 3)
    valid_release: int = 5
    use_mini: bool = True
    sfreq: int = 100
    epoch_len_s: float = 2.0
    random_seed: int = 2025
    passive_tasks: tuple[str, ...] = (
        "RestingState",
        "surroundSupp",
        "DiaryOfAWimpyKid",
        "DespicableMe",
        "FunwithFractals",
        "ThePresent",
    )
    target_task: str = "contrastChangeDetection"
    pretrain: TrainConfig = TrainConfig(
        batch_size=128,
        num_workers=0,
        lr=1e-3,
        weight_decay=1e-5,
        n_epochs=12,
        patience=3,
        min_delta=1e-4,
    )
    finetune: TrainConfig = TrainConfig(
        batch_size=128,
        num_workers=0,
        lr=1e-3,
        weight_decay=1e-5,
        n_epochs=100,
        patience=5,
        min_delta=1e-4,
    )

    @property
    def window_size_samples(self) -> int:
        return int(self.epoch_len_s * self.sfreq)

    @property
    def window_stride_samples(self) -> int:
        return self.sfreq
