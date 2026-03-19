#!/usr/bin/env python3
"""Edge inference script for traffic monitoring."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.traffic_model import TrafficCongestionModel
from runtimes.edge_runtime import (
    EdgeRuntimeManager,
    create_edge_engine,
    benchmark_edge_engines,
)
from pipelines.data_pipeline import TrafficDataGenerator
from utils.logging_utils import setup_logging, Timer


def load_model_config(model_path: str) -> tuple[TrafficCongestionModel, Dict[str, Any]]:
    """Load model and configuration.
    
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
    
    return model, config


def simulate_traffic_data(
    n_samples: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate traffic sensor data for inference testing.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    generator = TrafficDataGenerator(seed=seed)
    features, labels = generator.generate_dataset(n_samples)
    
    return features, labels


def run_inference_benchmark(
    model_paths: Dict[str, str],
    input_data: np.ndarray,
    device: str = "cpu",
    num_runs: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Run inference benchmark on multiple engines.
    
    Args:
        model_paths: Dictionary mapping engine types to model paths
        input_data: Input data for benchmarking
        device: Device to run on
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    logging.info(f"Running inference benchmark with {num_runs} runs...")
    
    results = benchmark_edge_engines(
        model_paths=model_paths,
        input_data=input_data,
        device=device,
        num_runs=num_runs,
    )
    
    return results


def run_real_time_simulation(
    model_path: str,
    config_path: str,
    duration_seconds: int = 60,
    sample_interval: float = 1.0,
) -> None:
    """Run real-time traffic monitoring simulation.
    
    Args:
        model_path: Path to model file
        config_path: Path to configuration file
        duration_seconds: Simulation duration
        sample_interval: Interval between samples
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model, model_config = load_model_config(model_path)
    
    # Create data generator
    generator = TrafficDataGenerator(seed=42)
    
    # Initialize runtime manager
    manager = EdgeRuntimeManager()
    
    # Create PyTorch engine
    pytorch_engine = create_edge_engine("pytorch", model_path, "cpu")
    if pytorch_engine:
        manager.register_engine("pytorch", pytorch_engine)
        pytorch_engine.load_model()
    
    logging.info(f"Starting real-time simulation for {duration_seconds} seconds...")
    
    start_time = time.time()
    sample_count = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            # Generate new sensor data
            features, _ = generator.generate_dataset(1)
            input_data = features[0]
            
            # Run inference
            if pytorch_engine and pytorch_engine.is_loaded:
                predictions, inference_time = pytorch_engine.predict(input_data)
                prediction = int(predictions[0] > 0.5)
                confidence = float(predictions[0])
                
                logging.info(
                    f"Sample {sample_count}: "
                    f"Prediction: {'Congested' if prediction else 'Smooth'} "
                    f"(confidence: {confidence:.3f}, "
                    f"latency: {inference_time*1000:.2f}ms)"
                )
            
            sample_count += 1
            
            # Wait for next sample
            time.sleep(sample_interval)
            
    except KeyboardInterrupt:
        logging.info("Simulation interrupted by user")
    
    total_time = time.time() - start_time
    avg_fps = sample_count / total_time
    
    logging.info(f"Simulation completed: {sample_count} samples in {total_time:.2f}s "
                f"(avg {avg_fps:.2f} FPS)")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run edge inference for traffic monitoring")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/edge.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["benchmark", "simulation", "single"],
        default="single",
        help="Inference mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Simulation duration in seconds"
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Sample interval in seconds"
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
    
    if args.mode == "single":
        # Single inference test
        logging.info("Running single inference test...")
        
        # Load model
        model, config = load_model_config(args.model_path)
        
        # Generate test data
        features, labels = simulate_traffic_data(n_samples=5, seed=42)
        
        # Run inference
        with torch.no_grad():
            for i, (feature, label) in enumerate(zip(features, labels)):
                input_tensor = torch.FloatTensor(feature).unsqueeze(0)
                
                start_time = time.time()
                output = model(input_tensor)
                inference_time = time.time() - start_time
                
                prediction = int(torch.sigmoid(output) > 0.5)
                confidence = float(torch.sigmoid(output))
                
                actual_status = "Congested" if label == 1 else "Smooth"
                predicted_status = "Congested" if prediction == 1 else "Smooth"
                
                print(f"Sample {i+1}:")
                print(f"  Features: Vehicle Count={feature[0]:.0f}, "
                      f"Speed={feature[1]:.1f} km/h, "
                      f"Weather={'Rain' if feature[2] == 1 else 'Clear'}, "
                      f"Hour={feature[3]:.0f}")
                print(f"  Actual: {actual_status}")
                print(f"  Predicted: {predicted_status} (confidence: {confidence:.3f})")
                print(f"  Inference time: {inference_time*1000:.2f}ms")
                print()
    
    elif args.mode == "benchmark":
        # Benchmark mode
        logging.info("Running inference benchmark...")
        
        # Generate test data
        features, _ = simulate_traffic_data(n_samples=1, seed=42)
        input_data = features[0]
        
        # Create model paths for different engines
        model_paths = {
            "pytorch": args.model_path,
        }
        
        # Check for compressed models
        compressed_dir = os.path.dirname(args.model_path)
        compressed_path = os.path.join(compressed_dir, "compressed", "compressed_model.pth")
        
        if os.path.exists(compressed_path):
            model_paths["pytorch_compressed"] = compressed_path
        
        # Run benchmark
        with Timer("Benchmark"):
            results = run_inference_benchmark(
                model_paths=model_paths,
                input_data=input_data,
                device=args.device,
                num_runs=args.num_runs,
            )
        
        # Print results
        print("\n" + "="*60)
        print("INFERENCE BENCHMARK RESULTS")
        print("="*60)
        
        for engine_name, metrics in results.items():
            print(f"\n{engine_name.upper()}:")
            print(f"  Mean latency: {metrics['mean_latency_ms']:.2f} ms")
            print(f"  P95 latency: {metrics['p95_latency_ms']:.2f} ms")
            print(f"  Throughput: {metrics['throughput_fps']:.1f} FPS")
            print(f"  Std latency: {metrics['std_latency_ms']:.2f} ms")
        
        print("="*60)
    
    elif args.mode == "simulation":
        # Real-time simulation mode
        run_real_time_simulation(
            model_path=args.model_path,
            config_path=args.config,
            duration_seconds=args.duration,
            sample_interval=args.sample_interval,
        )


if __name__ == "__main__":
    main()
