"""Data generation and processing pipeline for traffic monitoring."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficDataGenerator:
    """Generate synthetic traffic sensor data for training and testing.
    
    This class simulates realistic traffic patterns based on:
    - Time of day (rush hour patterns)
    - Weather conditions
    - Vehicle count and speed relationships
    - Seasonal variations
    """
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        logger.info(f"Initialized TrafficDataGenerator with seed {seed}")
    
    def generate_features(
        self,
        n_samples: int = 1000,
        time_range: Tuple[int, int] = (0, 24),
        vehicle_count_range: Tuple[int, int] = (10, 150),
        speed_mean: float = 50.0,
        speed_std: float = 10.0,
        rain_probability: float = 0.3,
    ) -> np.ndarray:
        """Generate traffic sensor features.
        
        Args:
            n_samples: Number of samples to generate
            time_range: Hour range (0-23)
            vehicle_count_range: Vehicle count range per minute
            speed_mean: Average speed in km/h
            speed_std: Speed standard deviation
            rain_probability: Probability of rain
            
        Returns:
            Feature array of shape (n_samples, 4)
            Columns: [vehicle_count, avg_speed, weather, hour]
        """
        # Generate base features
        vehicle_count = np.random.randint(
            vehicle_count_range[0], vehicle_count_range[1], n_samples
        )
        avg_speed = np.random.normal(speed_mean, speed_std, n_samples)
        weather = np.random.choice([0, 1], n_samples, p=[1 - rain_probability, rain_probability])
        hour = np.random.randint(time_range[0], time_range[1], n_samples)
        
        # Apply realistic correlations
        avg_speed = self._apply_speed_correlations(avg_speed, vehicle_count, hour, weather)
        
        features = np.stack([vehicle_count, avg_speed, weather, hour], axis=1)
        logger.info(f"Generated {n_samples} traffic feature samples")
        
        return features
    
    def _apply_speed_correlations(
        self,
        speed: np.ndarray,
        vehicle_count: np.ndarray,
        hour: np.ndarray,
        weather: np.ndarray,
    ) -> np.ndarray:
        """Apply realistic speed correlations based on traffic conditions.
        
        Args:
            speed: Base speed values
            vehicle_count: Vehicle count values
            hour: Hour of day
            weather: Weather condition (0=clear, 1=rain)
            
        Returns:
            Adjusted speed values
        """
        # Reduce speed during high traffic
        high_traffic_mask = vehicle_count > 100
        speed[high_traffic_mask] *= 0.7
        
        # Reduce speed during rush hours
        rush_hour_mask = ((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))
        speed[rush_hour_mask] *= 0.8
        
        # Reduce speed in rain
        speed[weather == 1] *= 0.9
        
        # Ensure minimum speed
        speed = np.maximum(speed, 5.0)
        
        return speed
    
    def generate_labels(
        self,
        features: np.ndarray,
        congestion_threshold: float = 0.6,
    ) -> np.ndarray:
        """Generate congestion labels based on features.
        
        Args:
            features: Feature array of shape (n_samples, 4)
            congestion_threshold: Threshold for congestion classification
            
        Returns:
            Binary labels (1=congested, 0=smooth)
        """
        vehicle_count = features[:, 0]
        avg_speed = features[:, 1]
        weather = features[:, 2]
        hour = features[:, 3]
        
        # Calculate congestion score
        congestion_score = np.zeros(len(features))
        
        # High vehicle count contributes to congestion
        congestion_score += (vehicle_count > 100).astype(float) * 0.4
        
        # Low speed contributes to congestion
        congestion_score += (avg_speed < 40).astype(float) * 0.3
        
        # Rush hours contribute to congestion
        rush_hour_mask = ((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))
        congestion_score += rush_hour_mask.astype(float) * 0.2
        
        # Rain contributes to congestion
        congestion_score += (weather == 1).astype(float) * 0.1
        
        # Generate binary labels
        labels = (congestion_score >= congestion_threshold).astype(int)
        
        logger.info(f"Generated labels: {np.sum(labels)} congested, {len(labels) - np.sum(labels)} smooth")
        
        return labels
    
    def generate_dataset(
        self,
        n_samples: int = 1000,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete dataset with features and labels.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional arguments for feature generation
            
        Returns:
            Tuple of (features, labels)
        """
        features = self.generate_features(n_samples, **kwargs)
        labels = self.generate_labels(features)
        
        return features, labels


class TrafficDataProcessor:
    """Process and prepare traffic data for training."""
    
    def __init__(self) -> None:
        """Initialize the data processor."""
        self.scaler = StandardScaler()
        self.is_fitted = False
        logger.info("Initialized TrafficDataProcessor")
    
    def fit_transform(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit scaler and split data into train/test sets.
        
        Args:
            features: Input features
            labels: Target labels
            test_size: Proportion of data for testing
            random_state: Random seed for splitting
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        self.is_fitted = True
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler.
        
        Args:
            features: Input features
            
        Returns:
            Scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        return self.scaler.transform(features)
    
    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features.
        
        Args:
            features: Scaled features
            
        Returns:
            Original scale features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        return self.scaler.inverse_transform(features)


def create_synthetic_dataset(
    n_samples: int = 1000,
    seed: int = 42,
    test_size: float = 0.2,
) -> Dict[str, np.ndarray]:
    """Create a complete synthetic traffic dataset.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary containing train/test features and labels
    """
    # Generate data
    generator = TrafficDataGenerator(seed=seed)
    features, labels = generator.generate_dataset(n_samples)
    
    # Process data
    processor = TrafficDataProcessor()
    X_train, X_test, y_train, y_test = processor.fit_transform(
        features, labels, test_size=test_size, random_state=seed
    )
    
    logger.info("Created synthetic traffic dataset")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "processor": processor,
    }


def load_real_dataset(file_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Load real traffic dataset from file.
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        Dataset dictionary or None if file not found
    """
    try:
        # Try to load as CSV
        df = pd.read_csv(file_path)
        
        # Assume standard column names
        feature_columns = ["vehicle_count", "avg_speed", "weather", "hour"]
        label_column = "congestion"
        
        if all(col in df.columns for col in feature_columns + [label_column]):
            features = df[feature_columns].values
            labels = df[label_column].values
            
            processor = TrafficDataProcessor()
            X_train, X_test, y_train, y_test = processor.fit_transform(
                features, labels, test_size=0.2, random_state=42
            )
            
            logger.info(f"Loaded real dataset from {file_path}")
            
            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "processor": processor,
            }
        else:
            logger.warning(f"Dataset {file_path} missing required columns")
            return None
            
    except FileNotFoundError:
        logger.warning(f"Dataset file {file_path} not found")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset {file_path}: {e}")
        return None


def save_dataset(
    dataset: Dict[str, np.ndarray],
    output_dir: str,
    filename_prefix: str = "traffic_dataset",
) -> None:
    """Save dataset to files.
    
    Args:
        dataset: Dataset dictionary
        output_dir: Output directory
        filename_prefix: Prefix for output files
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    for key, value in dataset.items():
        if isinstance(value, np.ndarray):
            filepath = os.path.join(output_dir, f"{filename_prefix}_{key}.npy")
            np.save(filepath, value)
            logger.info(f"Saved {key} to {filepath}")
    
    logger.info(f"Dataset saved to {output_dir}")


def get_feature_names() -> List[str]:
    """Get feature names for the traffic dataset.
    
    Returns:
        List of feature names
    """
    return ["vehicle_count", "avg_speed", "weather", "hour"]


def get_label_names() -> List[str]:
    """Get label names for the traffic dataset.
    
    Returns:
        List of label names
    """
    return ["smooth", "congested"]
