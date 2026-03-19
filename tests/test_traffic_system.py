"""Test suite for traffic monitoring system."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.traffic_model import TrafficCongestionModel, TrafficDataset, create_model
from pipelines.data_pipeline import TrafficDataGenerator, TrafficDataProcessor, create_synthetic_dataset
from export.compression import ModelCompressor, create_compression_config


class TestTrafficModel:
    """Test cases for traffic model."""
    
    def test_model_creation(self):
        """Test model creation with default parameters."""
        model = TrafficCongestionModel()
        assert isinstance(model, TrafficCongestionModel)
        assert model.input_dim == 4
        assert model.hidden_dims == [32, 16]
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = TrafficCongestionModel()
        input_tensor = torch.randn(5, 4)
        output = model(input_tensor)
        assert output.shape == (5, 1)
    
    def test_model_predict(self):
        """Test model prediction method."""
        model = TrafficCongestionModel()
        input_tensor = torch.randn(3, 4)
        predictions = model.predict(input_tensor)
        assert predictions.shape == (3,)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_predict_proba(self):
        """Test model probability prediction."""
        model = TrafficCongestionModel()
        input_tensor = torch.randn(3, 4)
        probabilities = model.predict_proba(input_tensor)
        assert probabilities.shape == (3, 2)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(3))
    
    def test_create_model_from_config(self):
        """Test model creation from configuration."""
        config = {
            "input_dim": 4,
            "hidden_dims": [16, 8],
            "dropout_rate": 0.1,
            "activation": "relu"
        }
        model = create_model(config)
        assert isinstance(model, TrafficCongestionModel)
        assert model.input_dim == 4
        assert model.hidden_dims == [16, 8]


class TestTrafficDataset:
    """Test cases for traffic dataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        features = np.random.randn(100, 4)
        labels = np.random.randint(0, 2, 100)
        dataset = TrafficDataset(features, labels)
        
        assert len(dataset) == 100
        assert dataset.features.shape == (100, 4)
        assert dataset.labels.shape == (100,)
    
    def test_dataset_getitem(self):
        """Test dataset item access."""
        features = np.random.randn(10, 4)
        labels = np.random.randint(0, 2, 10)
        dataset = TrafficDataset(features, labels)
        
        feature, label = dataset[0]
        assert isinstance(feature, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert feature.shape == (4,)
        assert label.shape == ()


class TestDataGenerator:
    """Test cases for data generator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = TrafficDataGenerator(seed=42)
        assert generator.seed == 42
    
    def test_generate_features(self):
        """Test feature generation."""
        generator = TrafficDataGenerator(seed=42)
        features = generator.generate_features(n_samples=100)
        
        assert features.shape == (100, 4)
        assert np.all(features[:, 0] >= 10)  # vehicle_count >= 10
        assert np.all(features[:, 0] <= 150)  # vehicle_count <= 150
        assert np.all(features[:, 2] >= 0)  # weather >= 0
        assert np.all(features[:, 2] <= 1)  # weather <= 1
        assert np.all(features[:, 3] >= 0)  # hour >= 0
        assert np.all(features[:, 3] <= 23)  # hour <= 23
    
    def test_generate_labels(self):
        """Test label generation."""
        generator = TrafficDataGenerator(seed=42)
        features = generator.generate_features(n_samples=100)
        labels = generator.generate_labels(features)
        
        assert labels.shape == (100,)
        assert np.all(np.isin(labels, [0, 1]))
    
    def test_generate_dataset(self):
        """Test complete dataset generation."""
        generator = TrafficDataGenerator(seed=42)
        features, labels = generator.generate_dataset(n_samples=100)
        
        assert features.shape == (100, 4)
        assert labels.shape == (100,)
        assert np.all(np.isin(labels, [0, 1]))


class TestDataProcessor:
    """Test cases for data processor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = TrafficDataProcessor()
        assert not processor.is_fitted
    
    def test_fit_transform(self):
        """Test fit and transform."""
        processor = TrafficDataProcessor()
        features = np.random.randn(100, 4)
        labels = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = processor.fit_transform(features, labels)
        
        assert X_train.shape[0] + X_test.shape[0] == 100
        assert y_train.shape[0] + y_test.shape[0] == 100
        assert processor.is_fitted
    
    def test_transform_without_fit(self):
        """Test transform without fitting."""
        processor = TrafficDataProcessor()
        features = np.random.randn(10, 4)
        
        with pytest.raises(ValueError):
            processor.transform(features)


class TestModelCompressor:
    """Test cases for model compressor."""
    
    def test_compressor_initialization(self):
        """Test compressor initialization."""
        compressor = ModelCompressor()
        assert compressor is not None
    
    def test_get_model_size(self):
        """Test model size calculation."""
        compressor = ModelCompressor()
        model = TrafficCongestionModel()
        size = compressor._get_model_size(model)
        
        assert isinstance(size, float)
        assert size > 0
    
    def test_create_compression_config(self):
        """Test compression configuration creation."""
        config = create_compression_config(
            prune=True,
            prune_amount=0.2,
            quantize=True,
            quant_method="dynamic"
        )
        
        assert config["prune"] is True
        assert config["prune_amount"] == 0.2
        assert config["quantize"] is True
        assert config["quant_method"] == "dynamic"


class TestSyntheticDataset:
    """Test cases for synthetic dataset creation."""
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset(n_samples=100, seed=42)
        
        assert "X_train" in dataset
        assert "X_test" in dataset
        assert "y_train" in dataset
        assert "y_test" in dataset
        assert "processor" in dataset
        
        assert dataset["X_train"].shape[0] + dataset["X_test"].shape[0] == 100
        assert dataset["y_train"].shape[0] + dataset["y_test"].shape[0] == 100


class TestIntegration:
    """Integration test cases."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create dataset
        dataset = create_synthetic_dataset(n_samples=100, seed=42)
        
        # Create model
        model = TrafficCongestionModel()
        
        # Create dataset objects
        train_dataset = TrafficDataset(dataset["X_train"], dataset["y_train"])
        test_dataset = TrafficDataset(dataset["X_test"], dataset["y_test"])
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Train model (just a few epochs for testing)
        from models.traffic_model import train_model
        history = train_model(model, train_loader, test_loader, num_epochs=2, device="cpu")
        
        assert "train_loss" in history
        assert "val_loss" in history
        assert "train_acc" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) == 2
    
    def test_model_compression_pipeline(self):
        """Test model compression pipeline."""
        # Create and train a simple model
        dataset = create_synthetic_dataset(n_samples=50, seed=42)
        model = TrafficCongestionModel()
        
        # Train briefly
        train_dataset = TrafficDataset(dataset["X_train"], dataset["y_train"])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(2):
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
        
        # Test compression
        compressor = ModelCompressor()
        compression_config = create_compression_config(prune=True, prune_amount=0.1)
        
        calibration_data = torch.FloatTensor(dataset["X_train"][:10])
        compressed_model, stats = compressor.compress_model(model, compression_config, calibration_data)
        
        assert isinstance(compressed_model, torch.nn.Module)
        assert "original_size_mb" in stats
        assert "final_size_mb" in stats
        assert stats["final_size_mb"] <= stats["original_size_mb"]


if __name__ == "__main__":
    pytest.main([__file__])
