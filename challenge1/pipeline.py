from __future__ import annotations

from .artifacts import save_run_artifacts
from .config import Challenge1Config
from .stages import run_finetuning_stage, run_pretraining_stage
from .workflow import (
    build_finetune_loaders,
    build_pretraining_data,
    describe_target_split_sizes,
    get_default_device,
    prepare_target_task_data,
    print_device_banner,
)


def run_training(config: Challenge1Config | None = None, *, device: str | None = None) -> None:
    config = config or Challenge1Config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    device = device or get_default_device()
    print_device_banner(device)

    target_data = prepare_target_task_data(config)
    split_sizes = describe_target_split_sizes(target_data)

    print("Number of examples in each split")
    print(f"Train:\t{split_sizes['train']}")
    print(f"Valid:\t{split_sizes['valid']}")
    print(f"Test:\t{split_sizes['test']}")

    pretraining_data = build_pretraining_data(
        config,
        target_data,
    )
    pretraining_result = run_pretraining_stage(
        config,
        device=device,
        train_loader=pretraining_data.train_loader,
        valid_loader=pretraining_data.valid_loader,
    )
    finetune_loaders = build_finetune_loaders(
        config,
        target_data,
    )
    finetuning_result = run_finetuning_stage(
        config,
        device=device,
        pretrained_model=pretraining_result.model,
        train_loader=finetune_loaders.train_loader,
        valid_loader=finetune_loaders.valid_loader,
        test_loader=finetune_loaders.test_loader,
    )

    run_dir = config.make_run_dir()
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving training artifacts to '{run_dir}'")

    save_run_artifacts(
        config=config,
        run_dir=run_dir,
        valid_meta_information=target_data.valid_meta_information,
        model=finetuning_result.model,
        dataset_sizes={
            **split_sizes,
            "pretrain_train": len(pretraining_data.train_set),
            "pretrain_valid": len(pretraining_data.valid_set),
        },
        pretrain_summary=pretraining_result.summary,
        finetune_summary=finetuning_result.summary,
        test_loss=finetuning_result.test_loss,
        test_rmse=finetuning_result.test_rmse,
        device=device,
    )
