# Traffic Monitoring Edge AI System

## DISCLAIMER

**THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. NOT FOR SAFETY-CRITICAL DEPLOYMENT.**

This traffic monitoring system is designed for academic research, educational demonstrations, and proof-of-concept development. It should NOT be used in production environments where traffic safety depends on its accuracy or reliability. The system may contain bugs, inaccuracies, or limitations that could lead to incorrect traffic predictions or system failures.

## Overview

Edge AI traffic monitoring system that uses sensor data and computer vision to predict traffic congestion in real-time. The system is optimized for edge deployment with model compression, IoT integration, and comprehensive evaluation metrics.

## Features

- **Edge-Optimized Models**: Quantized and pruned neural networks for efficient inference
- **IoT Integration**: MQTT messaging, sensor data streaming, and real-time processing
- **Multiple Deployment Targets**: TFLite, ONNX, OpenVINO, TensorRT support
- **Comprehensive Evaluation**: Accuracy and edge performance metrics
- **Interactive Demo**: Streamlit interface for real-time monitoring simulation
- **Production-Ready Structure**: Clean code, type hints, comprehensive testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Traffic-Monitoring-Edge-AI-System.git
cd Traffic-Monitoring-Edge-AI-System

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For edge deployment
pip install -e ".[edge]"
```

### Basic Usage

```bash
# Train the baseline model
python scripts/train.py --config configs/baseline.yaml

# Compress the model for edge deployment
python scripts/compress.py --model-path models/baseline.pth --method quantization

# Run edge inference simulation
python scripts/edge_inference.py --model-path models/quantized.tflite

# Launch interactive demo
streamlit run demo/app.py
```

## Project Structure

```
traffic-monitoring/
├── src/                    # Source code
│   ├── models/            # Neural network models
│   ├── export/            # Model export utilities
│   ├── runtimes/          # Edge runtime implementations
│   ├── pipelines/         # Data processing pipelines
│   ├── comms/             # IoT communication modules
│   └── utils/             # Utility functions
├── data/                  # Data storage
│   ├── raw/               # Raw sensor data
│   └── processed/         # Processed datasets
├── configs/               # Configuration files
├── scripts/               # Training and deployment scripts
├── tests/                 # Test suite
├── assets/                # Generated artifacts and plots
├── demo/                  # Interactive demo application
└── models/                # Trained model checkpoints
```

## Configuration

The system uses YAML configuration files for easy customization:

- `configs/baseline.yaml`: Baseline model configuration
- `configs/edge.yaml`: Edge-optimized model settings
- `configs/device.yaml`: Device-specific deployment settings
- `configs/quantization.yaml`: Quantization parameters

## Model Architecture

### Baseline Model
- Input: 4 features (vehicle count, speed, weather, hour)
- Architecture: 3-layer fully connected network
- Output: Binary classification (congested/smooth)

### Edge-Optimized Model
- Quantized to INT8 for reduced memory footprint
- Pruned for faster inference
- Optimized for single-batch inference

## Edge Deployment

### Supported Targets

- **TensorFlow Lite**: Mobile and embedded devices
- **ONNX Runtime**: Cross-platform inference
- **OpenVINO**: Intel hardware acceleration
- **TensorRT**: NVIDIA GPU acceleration
- **CoreML**: Apple devices

### Device Configurations

- Raspberry Pi 4 (ARM64)
- NVIDIA Jetson Nano/Orin
- Android/iOS devices
- Intel NUC (x86_64)

## Evaluation Metrics

### Model Quality
- Accuracy: Overall classification accuracy
- F1-Score: Balanced precision and recall
- ROC-AUC: Area under ROC curve

### Edge Performance
- **Latency**: Inference time (p50, p95)
- **Throughput**: Samples per second
- **Memory**: Peak RAM usage
- **Model Size**: Compressed model size
- **Energy**: Estimated power consumption

### Robustness
- Noise tolerance testing
- Packet loss simulation
- Offline mode evaluation

## IoT Integration

### MQTT Topics
- `traffic/sensors/vehicle_count`: Vehicle count data
- `traffic/sensors/speed`: Average speed data
- `traffic/sensors/weather`: Weather conditions
- `traffic/predictions/congestion`: Congestion predictions
- `traffic/alerts`: System alerts and notifications

### Sensor Simulation
The system includes realistic sensor data simulation for testing and demonstration purposes.

## Demo Application

Launch the interactive demo:

```bash
streamlit run demo/app.py
```

Features:
- Real-time traffic monitoring simulation
- Model performance visualization
- Edge device metrics dashboard
- Configuration management

## Development

### Code Quality
- Type hints throughout
- Comprehensive docstrings
- Black code formatting
- Ruff linting
- MyPy type checking

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{traffic_monitoring_edge_ai,
  title={Traffic Monitoring Edge AI System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Traffic-Monitoring-Edge-AI-System}
}
```

## Acknowledgments

- Built with PyTorch and TensorFlow
- Edge optimization techniques from various research papers
- IoT integration patterns from industry best practices
# Traffic-Monitoring-Edge-AI-System
