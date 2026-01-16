"""
Evaluation script for UniRig skinning model using configuration files.
Follows the same design pattern as run.py with PyTorch Lightning trainer.

Usage:
    python evaluate_skinning.py --task configs/task/evaluate_skin.yaml
    python evaluate_skinning.py --task configs/task/evaluate_skin.yaml --output_file results.txt
"""

import argparse
import os

import lightning as L
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

from box import Box
import yaml

from src.data.dataset import (
    DatasetConfig,
    UniRigDatasetModule,
)
from src.data.transform import TransformConfig
from src.inference.download import download
from src.model.parse import get_model
from src.system.parse import get_system


def load(task: str, path: str) -> Box:
    """Load a YAML configuration file."""
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    print(f"\033[92mload {task} config: {path}\033[0m")
    return Box(yaml.safe_load(open(path, 'r')))


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(
        description="Evaluate UniRig skinning model using config files"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Path to task configuration file"
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=123, help="random seed"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.txt",
        help="Output file to save results",
    )

    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    # Load task configuration
    task_config = load("task", args.task)
    mode = task_config.mode
    assert mode == "validate", f"Expected mode 'validate', got '{mode}'"

    # Load data configuration
    data_config = load(
        "data", os.path.join("configs/data", task_config.components.data)
    )
    transform_config = load(
        "transform", os.path.join("configs/transform", task_config.components.transform)
    )

    # Get data name
    data_name = task_config.components.get("data_name", "raw_data.npz")

    # Get validation dataset configuration
    validate_dataset_config = data_config.get("validate_dataset_config", None)
    if validate_dataset_config is None:
        # Fallback to predict dataset config if val is not available
        validate_dataset_config = data_config.get("predict_dataset_config", None)

    if validate_dataset_config is None:
        raise ValueError("No validation dataset configuration found in data config")

    validate_dataset_config = DatasetConfig.parse(
        config=validate_dataset_config
    ).split_by_cls()

    # Get validation transform configuration
    validate_transform_config = transform_config.get("validate_transform_config", None)
    if validate_transform_config is None:
        # Fallback to predict transform config if val is not available
        validate_transform_config = transform_config.get(
            "predict_transform_config", None
        )

    if validate_transform_config is None:
        raise ValueError(
            "No validation transform configuration found in transform config"
        )

    validate_transform_config = TransformConfig.parse(config=validate_transform_config)

    # Get model
    model_config = task_config.components.get("model", None)
    if model_config is not None:
        model_config = load("model", os.path.join("configs/model", model_config))
        model = get_model(**model_config)
    else:
        model = None

    # Create datamodule (following run.py pattern)
    datamodule = UniRigDatasetModule(
        process_fn=None if model is None else model._process_fn,
        validate_dataset_config=validate_dataset_config,
        validate_transform_config=validate_transform_config,
        tokenizer_config=None,
        debug=False,
        data_name=data_name,
        datapath=None,
        cls=None,
    )

    # Get trainer configuration
    trainer_config = task_config.get("trainer", {})
    if (
        not torch.cuda.is_available()
        and trainer_config.get("accelerator", "gpu") == "gpu"
    ):
        print("\033[93mWarning: No GPU available, switching to CPU.\033[0m")
        trainer_config["accelerator"] = "cpu"
        trainer_config["precision"] = 32  # Force float32 precision for CPU

    # Get loss config
    loss_config = task_config.get("loss", None)

    # Get system
    system_config = task_config.components.get("system", None)
    if system_config is not None:
        system_config = load("system", os.path.join("configs/system", system_config))
        system = get_system(
            **system_config,
            model=model,
            loss_config=loss_config,
            steps_per_epoch=1,
        )
    else:
        raise ValueError("System configuration is required for evaluation")

    # Set checkpoint path
    resume_from_checkpoint = task_config.get("resume_from_checkpoint", None)
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = download(resume_from_checkpoint)

    # Create trainer
    trainer = L.Trainer(
        callbacks=[],
        logger=None,
        **trainer_config,
    )

    # Run evaluation
    assert (
        resume_from_checkpoint is not None
    ), "expect resume_from_checkpoint in task"

    print(f"Starting evaluation with checkpoint: {resume_from_checkpoint}")
    print(f"Using accelerator: {trainer_config.get('accelerator', 'auto')}")
    print(f"Using devices: {trainer_config.get('devices', 'auto')}")
    print(f"System type: {type(system).__name__}")

    # Run validation using trainer
    validation_results = trainer.validate(
        system,
        datamodule=datamodule,
        ckpt_path=resume_from_checkpoint,
    )

    # Extract results from validation output
    results = validation_results[0] if validation_results else {}

    # Convert to our expected format
    evaluation_results = {
        "avg_mae_loss": results.get("val/val_mae_loss", 0.0),
        "avg_ce_loss": results.get("val/val_ce_loss", 0.0),
        "avg_cosine_sim": results.get("val/val_cosine_sim", 0.0),
        "avg_mae_loss_p": results.get("val/val_mae_loss_p", 0.0),
        "avg_precision": results.get("val/val_precision", 0.0),
        "avg_recall": results.get("val/val_recall", 0.0),
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Checkpoint: {resume_from_checkpoint}")
    print(f"Accelerator: {trainer_config.get('accelerator', 'auto')}")
    print(f"Devices: {trainer_config.get('devices', 'auto')}")
    print()
    print("MAE Loss Metrics:")
    print(f"  Average MAE Loss: {evaluation_results['avg_mae_loss']:.6f}")
    print(f"  Average MAE Loss (Pruned): {evaluation_results['avg_mae_loss_p']:.6f}")
    print()
    print("Cross Entropy Loss Metrics:")
    print(f"  Average CE Loss: {evaluation_results['avg_ce_loss']:.6f}")
    print()
    print("Cosine Similarity Metrics:")
    print(
        f"  Average Cosine Similarity: {evaluation_results['avg_cosine_sim']:.6f}"
    )
    print()
    print("Precision & Recall Metrics:")
    print(f"  Average Precision: {evaluation_results['avg_precision']:.6f}")
    print(f"  Average Recall: {evaluation_results['avg_recall']:.6f}")
    print("=" * 60)

    # Save results to file
    with open(args.output_file, "w") as f:
        f.write("UniRig Skinning Model Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Task Config: {args.task}\n")
        f.write(f"Checkpoint: {resume_from_checkpoint}\n")
        f.write(f"Accelerator: {trainer_config.get('accelerator', 'auto')}\n")
        f.write(f"Devices: {trainer_config.get('devices', 'auto')}\n")
        f.write(f"Data Config: configs/data/{task_config.components.data}\n")
        f.write(
            f"Transform Config: configs/transform/{task_config.components.transform}\n"
        )
        f.write(f"Model Config: configs/model/{task_config.components.model}\n\n")
        f.write("MAE Loss Metrics:\n")
        f.write(f"  Average MAE Loss: {evaluation_results['avg_mae_loss']:.6f}\n")
        f.write(f"  Average MAE Loss (Pruned): {evaluation_results['avg_mae_loss_p']:.6f}\n\n")
        f.write("Cross Entropy Loss Metrics:\n")
        f.write(f"  Average CE Loss: {evaluation_results['avg_ce_loss']:.6f}\n\n")
        f.write("Cosine Similarity Metrics:\n")
        f.write(
            f"  Average Cosine Similarity: {evaluation_results['avg_cosine_sim']:.6f}\n\n"
        )
        f.write("Precision & Recall Metrics:\n")
        f.write(f"  Average Precision: {evaluation_results['avg_precision']:.6f}\n")
        f.write(f"  Average Recall: {evaluation_results['avg_recall']:.6f}\n")

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
