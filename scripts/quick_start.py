#!/usr/bin/env python3
"""Quick start script for traffic monitoring system."""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main quick start function."""
    print("🚦 Traffic Monitoring Edge AI System - Quick Start")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -e .", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["models", "data/raw", "data/processed", "assets", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Created necessary directories")
    
    # Train baseline model
    if not run_command("python scripts/train.py --config configs/baseline.yaml", "Training baseline model"):
        print("❌ Failed to train baseline model")
        sys.exit(1)
    
    # Compress model
    if not run_command("python scripts/compress.py --model-path models/baseline_model.pth --method quantization", "Compressing model"):
        print("❌ Failed to compress model")
        sys.exit(1)
    
    # Run evaluation
    if not run_command("python scripts/evaluate.py --model-path models/baseline_model.pth", "Running evaluation"):
        print("❌ Failed to run evaluation")
        sys.exit(1)
    
    # Test edge inference
    if not run_command("python scripts/edge_inference.py --model-path models/baseline_model.pth --mode single", "Testing edge inference"):
        print("❌ Failed to test edge inference")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Quick start completed successfully!")
    print("\nNext steps:")
    print("1. Launch the demo: streamlit run demo/app.py")
    print("2. Check evaluation results in assets/evaluation/")
    print("3. Explore the trained models in models/")
    print("\nFor more information, see README.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
