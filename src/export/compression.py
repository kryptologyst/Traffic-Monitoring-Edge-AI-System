"""Model compression utilities for edge deployment."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, quantize_static


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCompressor:
    """Compress neural network models for edge deployment.
    
    This class provides methods for quantization, pruning, and other
    compression techniques to reduce model size and improve inference speed.
    """
    
    def __init__(self) -> None:
        """Initialize the model compressor."""
        logger.info("Initialized ModelCompressor")
    
    def quantize_model(
        self,
        model: nn.Module,
        method: str = "dynamic",
        calibration_data: Optional[torch.Tensor] = None,
        backend: str = "qnnpack",
    ) -> nn.Module:
        """Quantize a model to reduce precision.
        
        Args:
            model: Model to quantize
            method: Quantization method ("dynamic" or "static")
            calibration_data: Data for static quantization calibration
            backend: Quantization backend
            
        Returns:
            Quantized model
        """
        model.eval()
        
        if method == "dynamic":
            # Dynamic quantization - weights quantized, activations in float
            quantized_model = quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization")
            
        elif method == "static":
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            
            # Static quantization - both weights and activations quantized
            model.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate with sample data
            with torch.no_grad():
                for sample in calibration_data:
                    model(sample.unsqueeze(0))
            
            quantized_model = torch.quantization.convert(model, inplace=False)
            logger.info("Applied static quantization")
            
        else:
            raise ValueError(f"Unsupported quantization method: {method}")
        
        return quantized_model
    
    def prune_model(
        self,
        model: nn.Module,
        method: str = "magnitude",
        amount: float = 0.2,
        structured: bool = False,
    ) -> nn.Module:
        """Prune a model to remove unnecessary weights.
        
        Args:
            model: Model to prune
            method: Pruning method ("magnitude", "random", "gradient")
            amount: Fraction of weights to prune (0.0 to 1.0)
            structured: Whether to use structured pruning
            
        Returns:
            Pruned model
        """
        model.eval()
        
        if structured:
            # Structured pruning - removes entire channels/filters
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.ln_structured(
                        module, name="weight", amount=amount, n=2, dim=0
                    )
                    prune.remove(module, "weight")
        else:
            # Unstructured pruning - removes individual weights
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    parameters_to_prune.append((module, "weight"))
            
            if method == "magnitude":
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=amount,
                )
            elif method == "random":
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.RandomUnstructured,
                    amount=amount,
                )
            else:
                raise ValueError(f"Unsupported pruning method: {method}")
            
            # Remove pruning reparameterization
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
        
        logger.info(f"Applied {method} pruning with {amount:.1%} sparsity")
        
        return model
    
    def compress_model(
        self,
        model: nn.Module,
        compression_config: Dict[str, Union[str, float, bool]],
        calibration_data: Optional[torch.Tensor] = None,
    ) -> Tuple[nn.Module, Dict[str, Union[int, float]]]:
        """Apply multiple compression techniques to a model.
        
        Args:
            model: Model to compress
            compression_config: Compression configuration
            calibration_data: Data for quantization calibration
            
        Returns:
            Tuple of (compressed_model, compression_stats)
        """
        original_model = model
        stats = {}
        
        # Get original model size
        original_size = self._get_model_size(original_model)
        stats["original_size_mb"] = original_size
        
        # Apply pruning if specified
        if compression_config.get("prune", False):
            prune_amount = compression_config.get("prune_amount", 0.2)
            prune_method = compression_config.get("prune_method", "magnitude")
            structured = compression_config.get("structured_pruning", False)
            
            model = self.prune_model(model, prune_method, prune_amount, structured)
            
            pruned_size = self._get_model_size(model)
            stats["pruned_size_mb"] = pruned_size
            stats["pruning_ratio"] = (original_size - pruned_size) / original_size
        
        # Apply quantization if specified
        if compression_config.get("quantize", False):
            quant_method = compression_config.get("quant_method", "dynamic")
            backend = compression_config.get("quant_backend", "qnnpack")
            
            model = self.quantize_model(model, quant_method, calibration_data, backend)
            
            quantized_size = self._get_model_size(model)
            stats["quantized_size_mb"] = quantized_size
            stats["quantization_ratio"] = (stats.get("pruned_size_mb", original_size) - quantized_size) / stats.get("pruned_size_mb", original_size)
        
        # Calculate overall compression ratio
        final_size = self._get_model_size(model)
        stats["final_size_mb"] = final_size
        stats["overall_compression_ratio"] = (original_size - final_size) / original_size
        
        logger.info(f"Model compressed: {original_size:.2f}MB -> {final_size:.2f}MB "
                   f"({stats['overall_compression_ratio']:.1%} reduction)")
        
        return model, stats
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB.
        
        Args:
            model: Model to measure
            
        Returns:
            Model size in megabytes
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def evaluate_compression_impact(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate the impact of compression on model performance.
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            test_data: Test input data
            test_labels: Test labels
            
        Returns:
            Dictionary with performance comparison metrics
        """
        original_model.eval()
        compressed_model.eval()
        
        with torch.no_grad():
            # Original model predictions
            orig_outputs = original_model(test_data)
            orig_predictions = (torch.sigmoid(orig_outputs) > 0.5).long().squeeze()
            
            # Compressed model predictions
            comp_outputs = compressed_model(test_data)
            comp_predictions = (torch.sigmoid(comp_outputs) > 0.5).long().squeeze()
        
        # Calculate metrics
        orig_accuracy = (orig_predictions == test_labels).float().mean().item()
        comp_accuracy = (comp_predictions == test_labels).float().mean().item()
        
        # Calculate prediction agreement
        agreement = (orig_predictions == comp_predictions).float().mean().item()
        
        # Calculate output correlation
        output_correlation = torch.corrcoef(torch.stack([
            torch.sigmoid(orig_outputs).squeeze(),
            torch.sigmoid(comp_outputs).squeeze()
        ]))[0, 1].item()
        
        results = {
            "original_accuracy": orig_accuracy,
            "compressed_accuracy": comp_accuracy,
            "accuracy_drop": orig_accuracy - comp_accuracy,
            "prediction_agreement": agreement,
            "output_correlation": output_correlation,
        }
        
        logger.info(f"Compression impact: {comp_accuracy:.4f} accuracy "
                   f"({orig_accuracy - comp_accuracy:+.4f} change), "
                   f"{agreement:.4f} agreement")
        
        return results


class EdgeModelExporter:
    """Export models for edge deployment."""
    
    def __init__(self) -> None:
        """Initialize the model exporter."""
        logger.info("Initialized EdgeModelExporter")
    
    def export_to_tflite(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        optimize: bool = True,
    ) -> None:
        """Export PyTorch model to TensorFlow Lite format.
        
        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            output_path: Output file path
            optimize: Whether to optimize the model
        """
        try:
            import torch_tflite
            
            model.eval()
            traced_model = torch.jit.trace(model, sample_input)
            
            converter = torch_tflite.Converter(
                traced_model,
                [sample_input],
                optimize=optimize,
            )
            
            converter.convert()
            converter.save(output_path)
            
            logger.info(f"Model exported to TFLite: {output_path}")
            
        except ImportError:
            logger.warning("torch_tflite not available, skipping TFLite export")
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
    
    def export_to_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        opset_version: int = 11,
    ) -> None:
        """Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            output_path: Output file path
            opset_version: ONNX opset version
        """
        try:
            model.eval()
            
            torch.onnx.export(
                model,
                sample_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            
            logger.info(f"Model exported to ONNX: {output_path}")
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
    
    def export_to_openvino(
        self,
        onnx_path: str,
        output_dir: str,
        precision: str = "FP16",
    ) -> None:
        """Convert ONNX model to OpenVINO format.
        
        Args:
            onnx_path: Path to ONNX model
            output_dir: Output directory
            precision: Model precision (FP32, FP16, INT8)
        """
        try:
            from openvino.tools import mo
            from openvino.runtime import Core
            
            # Convert ONNX to OpenVINO IR
            mo.convert_model(
                onnx_path,
                output_dir=output_dir,
                compress_to_fp16=(precision == "FP16"),
            )
            
            logger.info(f"Model converted to OpenVINO: {output_dir}")
            
        except ImportError:
            logger.warning("OpenVINO not available, skipping OpenVINO export")
        except Exception as e:
            logger.error(f"OpenVINO export failed: {e}")
    
    def export_all_formats(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_dir: str,
        model_name: str = "traffic_model",
    ) -> Dict[str, str]:
        """Export model to all supported formats.
        
        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            output_dir: Output directory
            model_name: Base name for output files
            
        Returns:
            Dictionary mapping format names to file paths
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        exported_files = {}
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        self.export_to_onnx(model, sample_input, onnx_path)
        exported_files["onnx"] = onnx_path
        
        # Export to TFLite
        tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
        self.export_to_tflite(model, sample_input, tflite_path)
        exported_files["tflite"] = tflite_path
        
        # Export to OpenVINO
        openvino_dir = os.path.join(output_dir, f"{model_name}_openvino")
        self.export_to_openvino(onnx_path, openvino_dir)
        exported_files["openvino"] = openvino_dir
        
        logger.info(f"Exported model to {len(exported_files)} formats")
        
        return exported_files


def create_compression_config(
    prune: bool = True,
    prune_amount: float = 0.2,
    prune_method: str = "magnitude",
    structured_pruning: bool = False,
    quantize: bool = True,
    quant_method: str = "dynamic",
    quant_backend: str = "qnnpack",
) -> Dict[str, Union[str, float, bool]]:
    """Create a compression configuration dictionary.
    
    Args:
        prune: Whether to apply pruning
        prune_amount: Fraction of weights to prune
        prune_method: Pruning method to use
        structured_pruning: Whether to use structured pruning
        quantize: Whether to apply quantization
        quant_method: Quantization method to use
        quant_backend: Quantization backend
        
    Returns:
        Compression configuration dictionary
    """
    return {
        "prune": prune,
        "prune_amount": prune_amount,
        "prune_method": prune_method,
        "structured_pruning": structured_pruning,
        "quantize": quantize,
        "quant_method": quant_method,
        "quant_backend": quant_backend,
    }
