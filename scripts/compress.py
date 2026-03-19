#!/usr/bin/env python3
"""Model compression script for edge deployment."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.traffic_model import TrafficCongestionModel, TrafficDataset
from export.compression import ModelCompressor, EdgeModelExporter, create_compression_config
from pipelines.data_pipeline import create_synthetic_dataset
from utils.logging_utils import setup_logging, Timer


def load_model(model_path: str) -> tuple[TrafficCongestionModel, Dict[str, Any]]:
    """Load trained model and configuration.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model from config
    config = checkpoint['config']
    model = TrafficCongestionModel(**config['model'])
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info(f"Loaded model from {model_path}")
    return model, config


def main():
    """Main compression function."""
    parser = argparse.ArgumentParser(description="Compress traffic monitoring model for edge deployment")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/edge.yaml",
        help="Path to compression configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/compressed",
        help="Output directory for compressed models"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["quantization", "pruning", "both"],
        default="both",
        help="Compression method to apply"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of samples for quantization calibration"
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
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded compression configuration from {args.config}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    with Timer("Model loading"):
        model, model_config = load_model(args.model_path)
    
    # Create calibration data for quantization
    logging.info("Creating calibration dataset...")
    dataset = create_synthetic_dataset(
        n_samples=args.calibration_samples,
        seed=42,
        test_size=0.0,  # No test split needed for calibration
    )
    
    calibration_data = torch.FloatTensor(dataset["X_train"][:args.calibration_samples])
    
    # Initialize compressor and exporter
    compressor = ModelCompressor()
    exporter = EdgeModelExporter()
    
    # Create compression configuration based on method
    if args.method == "quantization":
        compression_config = create_compression_config(
            prune=False,
            quantize=True,
            quant_method=config.get("compression", {}).get("quant_method", "dynamic"),
        )
    elif args.method == "pruning":
        compression_config = create_compression_config(
            prune=True,
            prune_amount=config.get("compression", {}).get("prune_amount", 0.2),
            prune_method=config.get("compression", {}).get("prune_method", "magnitude"),
            quantize=False,
        )
    else:  # both
        compression_config = create_compression_config(
            prune=True,
            prune_amount=config.get("compression", {}).get("prune_amount", 0.2),
            prune_method=config.get("compression", {}).get("prune_method", "magnitude"),
            quantize=True,
            quant_method=config.get("compression", {}).get("quant_method", "dynamic"),
        )
    
    logging.info(f"Compression configuration: {compression_config}")
    
    # Apply compression
    with Timer("Model compression"):
        compressed_model, compression_stats = compressor.compress_model(
            model,
            compression_config,
            calibration_data,
        )
    
    logging.info(f"Compression statistics: {compression_stats}")
    
    # Evaluate compression impact
    logging.info("Evaluating compression impact...")
    
    # Create test dataset
    test_dataset = create_synthetic_dataset(
        n_samples=200,
        seed=123,
        test_size=0.0,
    )
    
    test_data = torch.FloatTensor(test_dataset["X_train"])
    test_labels = torch.LongTensor(test_dataset["y_train"])
    
    # Evaluate original model
    with torch.no_grad():
        orig_outputs = model(test_data)
        orig_predictions = (torch.sigmoid(orig_outputs) > 0.5).long().squeeze()
        orig_accuracy = (orig_predictions == test_labels).float().mean().item()
    
    # Evaluate compressed model
    with torch.no_grad():
        comp_outputs = compressed_model(test_data)
        comp_predictions = (torch.sigmoid(comp_outputs) > 0.5).long().squeeze()
        comp_accuracy = (comp_predictions == test_labels).float().mean().item()
    
    # Calculate agreement
    agreement = (orig_predictions == comp_predictions).float().mean().item()
    
    logging.info(f"Original accuracy: {orig_accuracy:.4f}")
    logging.info(f"Compressed accuracy: {comp_accuracy:.4f}")
    logging.info(f"Prediction agreement: {agreement:.4f}")
    
    # Save compressed model
    compressed_model_path = os.path.join(args.output_dir, "compressed_model.pth")
    torch.save({
        'model_state_dict': compressed_model.state_dict(),
        'config': model_config,
        'compression_config': compression_config,
        'compression_stats': compression_stats,
    }, compressed_model_path)
    
    logging.info(f"Compressed model saved to {compressed_model_path}")
    
    # Export to edge formats
    logging.info("Exporting to edge formats...")
    
    sample_input = calibration_data[:1]  # Single sample for export
    
    exported_files = exporter.export_all_formats(
        compressed_model,
        sample_input,
        args.output_dir,
        "traffic_model_compressed",
    )
    
    logging.info(f"Exported files: {exported_files}")
    
    # Save compression report
    report = {
        "compression_method": args.method,
        "compression_config": compression_config,
        "compression_stats": compression_stats,
        "accuracy_comparison": {
            "original": orig_accuracy,
            "compressed": comp_accuracy,
            "agreement": agreement,
        },
        "exported_files": exported_files,
    }
    
    import json
    report_path = os.path.join(args.output_dir, "compression_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f"Compression report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPRESSION SUMMARY")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Original size: {compression_stats['original_size_mb']:.2f} MB")
    print(f"Compressed size: {compression_stats['final_size_mb']:.2f} MB")
    print(f"Compression ratio: {compression_stats['overall_compression_ratio']:.1%}")
    print(f"Original accuracy: {orig_accuracy:.4f}")
    print(f"Compressed accuracy: {comp_accuracy:.4f}")
    print(f"Prediction agreement: {agreement:.4f}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
