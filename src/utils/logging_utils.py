"""Utility functions for the traffic monitoring system."""

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level
        format_string: Custom format string
        log_file: Optional log file path
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else []),
        ]
    )


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device() -> str:
    """Get the best available device for computation.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb


def create_directory_structure(base_path: str) -> None:
    """Create the standard directory structure for the project.
    
    Args:
        base_path: Base project path
    """
    directories = [
        "src/models",
        "src/export",
        "src/runtimes",
        "src/pipelines",
        "src/comms",
        "src/utils",
        "data/raw",
        "data/processed",
        "configs",
        "scripts",
        "tests",
        "assets",
        "demo",
        "models",
    ]
    
    for directory in directories:
        dir_path = Path(base_path) / directory
        dir_path.mkdir(parents=True, exist_ok=True)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_yaml_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    import yaml
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_bytes(bytes_value: int) -> str:
    """Format byte size in a human-readable format.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f}PB"


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required configuration keys
        
    Returns:
        True if configuration is valid, False otherwise
    """
    missing_keys = []
    
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        logging.error(f"Missing required configuration keys: {missing_keys}")
        return False
    
    return True


def get_git_info() -> Dict[str, str]:
    """Get Git repository information.
    
    Returns:
        Dictionary with Git information
    """
    try:
        import subprocess
        
        # Get current commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Get current branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Get repository URL
        try:
            remote_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except subprocess.CalledProcessError:
            remote_url = "unknown"
        
        return {
            "commit_hash": commit_hash,
            "branch": branch,
            "remote_url": remote_url,
        }
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit_hash": "unknown",
            "branch": "unknown",
            "remote_url": "unknown",
        }


def create_experiment_id() -> str:
    """Create a unique experiment ID.
    
    Returns:
        Unique experiment ID string
    """
    import time
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = random.randint(1000, 9999)
    
    return f"exp_{timestamp}_{random_suffix}"


def save_experiment_info(
    experiment_id: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: str,
) -> None:
    """Save experiment information to files.
    
    Args:
        experiment_id: Unique experiment ID
        config: Experiment configuration
        metrics: Experiment metrics
        output_dir: Output directory
    """
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, f"{experiment_id}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{experiment_id}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save Git info
    git_info = get_git_info()
    git_path = os.path.join(output_dir, f"{experiment_id}_git.json")
    with open(git_path, 'w') as f:
        json.dump(git_info, f, indent=2)
    
    logging.info(f"Experiment info saved: {experiment_id}")


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        """Initialize timer.
        
        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log duration."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logging.info(f"{self.name} completed in {format_time(duration)}")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


def ensure_numpy_array(data: Any) -> np.ndarray:
    """Ensure data is a numpy array.
    
    Args:
        data: Input data (list, tensor, or array)
        
    Returns:
        Numpy array
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    else:
        raise ValueError(f"Cannot convert {type(data)} to numpy array")
