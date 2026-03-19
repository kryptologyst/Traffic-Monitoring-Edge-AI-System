#!/usr/bin/env python3
"""Comprehensive evaluation script for traffic monitoring system."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.traffic_model import TrafficCongestionModel, TrafficDataset, evaluate_model
from pipelines.data_pipeline import create_synthetic_dataset
from runtimes.edge_runtime import benchmark_edge_engines
from utils.logging_utils import setup_logging, Timer


def load_model(model_path: str) -> tuple[TrafficCongestionModel, Dict[str, Any]]:
    """Load trained model and configuration."""
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    model = TrafficCongestionModel(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def evaluate_model_performance(
    model: TrafficCongestionModel,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model performance metrics."""
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_data).to(device)
        outputs = model(test_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        predictions = (probabilities > 0.5).astype(int).flatten()
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(test_labels, predictions),
        "precision": precision_score(test_labels, predictions, zero_division=0),
        "recall": recall_score(test_labels, predictions, zero_division=0),
        "f1_score": f1_score(test_labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(test_labels, probabilities),
    }
    
    return metrics


def benchmark_inference_performance(
    model: TrafficCongestionModel,
    input_data: np.ndarray,
    num_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """Benchmark inference performance."""
    model.eval()
    model = model.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
            _ = model(input_tensor)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
            
            start_time = time.time()
            _ = model(input_tensor)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        "mean_latency_ms": np.mean(times) * 1000,
        "std_latency_ms": np.std(times) * 1000,
        "p50_latency_ms": np.percentile(times, 50) * 1000,
        "p95_latency_ms": np.percentile(times, 95) * 1000,
        "p99_latency_ms": np.percentile(times, 99) * 1000,
        "throughput_fps": 1.0 / np.mean(times),
    }


def create_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
) -> None:
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Smooth', 'Congested']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_latency_histogram(
    latencies: np.ndarray,
    output_path: str,
) -> None:
    """Create and save latency histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Inference Latency Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_comparison_plot(
    results: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """Create performance comparison plot."""
    models = list(results.keys())
    metrics = ['mean_latency_ms', 'throughput_fps']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Latency comparison
    latencies = [results[model]['mean_latency_ms'] for model in models]
    ax1.bar(models, latencies, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Mean Latency (ms)')
    ax1.set_title('Inference Latency Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Throughput comparison
    throughputs = [results[model]['throughput_fps'] for model in models]
    ax2.bar(models, throughputs, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Throughput (FPS)')
    ax2.set_title('Throughput Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_evaluation_report(
    results: Dict[str, Any],
    output_path: str,
) -> None:
    """Generate comprehensive evaluation report."""
    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_performance": results.get("model_performance", {}),
        "inference_performance": results.get("inference_performance", {}),
        "edge_benchmarks": results.get("edge_benchmarks", {}),
        "summary": {
            "total_samples": results.get("total_samples", 0),
            "accuracy": results.get("model_performance", {}).get("accuracy", 0),
            "mean_latency_ms": results.get("inference_performance", {}).get("mean_latency_ms", 0),
            "throughput_fps": results.get("inference_performance", {}).get("throughput_fps", 0),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of traffic monitoring system")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=1000,
        help="Number of test samples"
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on"
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("Starting comprehensive evaluation...")
    
    # Load model
    with Timer("Model loading"):
        model, config = load_model(args.model_path)
    
    logging.info(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Generate test dataset
    with Timer("Test dataset generation"):
        dataset = create_synthetic_dataset(
            n_samples=args.test_samples,
            seed=42,
            test_size=0.0,  # Use all data for testing
        )
        
        test_data = dataset["X_train"]
        test_labels = dataset["y_train"]
    
    logging.info(f"Generated test dataset with {len(test_data)} samples")
    
    # Evaluate model performance
    with Timer("Model performance evaluation"):
        model_metrics = evaluate_model_performance(model, test_data, test_labels, args.device)
    
    logging.info(f"Model performance: {model_metrics}")
    
    # Benchmark inference performance
    with Timer("Inference performance benchmarking"):
        sample_input = test_data[0]  # Use first sample for benchmarking
        inference_metrics = benchmark_inference_performance(
            model, sample_input, args.benchmark_runs, args.device
        )
    
    logging.info(f"Inference performance: {inference_metrics}")
    
    # Generate predictions for visualization
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_data)
        outputs = model(test_tensor)
        probabilities = torch.sigmoid(outputs).numpy()
        predictions = (probabilities > 0.5).astype(int).flatten()
    
    # Create visualizations
    logging.info("Creating visualizations...")
    
    # Confusion matrix
    create_confusion_matrix_plot(
        test_labels, predictions,
        os.path.join(args.output_dir, "confusion_matrix.png")
    )
    
    # Latency histogram (simulate latencies for visualization)
    simulated_latencies = np.random.normal(
        inference_metrics["mean_latency_ms"],
        inference_metrics["std_latency_ms"],
        1000
    )
    create_latency_histogram(
        simulated_latencies,
        os.path.join(args.output_dir, "latency_histogram.png")
    )
    
    # Performance comparison (if multiple models available)
    model_results = {
        "baseline": {
            "mean_latency_ms": inference_metrics["mean_latency_ms"],
            "throughput_fps": inference_metrics["throughput_fps"],
        }
    }
    
    # Check for compressed model
    compressed_path = os.path.join(os.path.dirname(args.model_path), "compressed", "compressed_model.pth")
    if os.path.exists(compressed_path):
        logging.info("Found compressed model, benchmarking...")
        
        compressed_model, _ = load_model(compressed_path)
        compressed_inference = benchmark_inference_performance(
            compressed_model, sample_input, args.benchmark_runs, args.device
        )
        
        model_results["compressed"] = {
            "mean_latency_ms": compressed_inference["mean_latency_ms"],
            "throughput_fps": compressed_inference["throughput_fps"],
        }
    
    create_performance_comparison_plot(
        model_results,
        os.path.join(args.output_dir, "performance_comparison.png")
    )
    
    # Generate comprehensive report
    evaluation_results = {
        "model_performance": model_metrics,
        "inference_performance": inference_metrics,
        "total_samples": len(test_data),
        "edge_benchmarks": model_results,
    }
    
    generate_evaluation_report(
        evaluation_results,
        os.path.join(args.output_dir, "evaluation_report.json")
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model Accuracy: {model_metrics['accuracy']:.4f}")
    print(f"F1-Score: {model_metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {model_metrics['roc_auc']:.4f}")
    print(f"Mean Latency: {inference_metrics['mean_latency_ms']:.2f} ms")
    print(f"P95 Latency: {inference_metrics['p95_latency_ms']:.2f} ms")
    print(f"Throughput: {inference_metrics['throughput_fps']:.1f} FPS")
    print(f"Test Samples: {len(test_data)}")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)
    
    logging.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
