#!/usr/bin/env python3
"""Training script for traffic monitoring model."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from omegaconf import OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.traffic_model import (
    TrafficCongestionModel,
    TrafficDataset,
    train_model,
    evaluate_model,
    create_model,
)
from pipelines.data_pipeline import create_synthetic_dataset
from utils.logging_utils import setup_logging


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_model(model: nn.Module, output_path: str, config: Dict[str, Any]) -> None:
    """Save trained model and configuration.
    
    Args:
        model: Trained model
        output_path: Path to save model
        config: Model configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, output_path)
    
    logging.info(f"Model saved to {output_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train traffic monitoring model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logging.info(f"Using device: {device}")
    
    # Create dataset
    logging.info("Creating synthetic dataset...")
    dataset = create_synthetic_dataset(
        n_samples=config["data"]["n_samples"],
        seed=config["data"]["seed"],
        test_size=config["training"]["test_size"],
    )
    
    # Create data loaders
    train_dataset = TrafficDataset(
        dataset["X_train"],
        dataset["y_train"]
    )
    test_dataset = TrafficDataset(
        dataset["X_test"],
        dataset["y_test"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["device"]["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["device"]["num_workers"],
    )
    
    logging.info(f"Created data loaders: {len(train_loader)} train batches, "
                f"{len(test_loader)} test batches")
    
    # Create model
    model = create_model(config["model"])
    logging.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    logging.info("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        device=device,
    )
    
    # Evaluate model
    logging.info("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Log results
    logging.info("Training completed!")
    logging.info(f"Final metrics: {metrics}")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "baseline_model.pth")
    save_model(model, model_path, config)
    
    # Save training history
    import json
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logging.info(f"Training history saved to {history_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {config['model']}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    print(f"Final F1-score: {metrics['f1_score']:.4f}")
    print(f"Model saved to: {model_path}")
    print("="*50)


if __name__ == "__main__":
    main()
