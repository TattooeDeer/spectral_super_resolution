"""Inference/reconstruction for spectral super-resolution models."""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ModelConfig, InferenceConfig
from .metrics import SSIM, SAM, MSE, RMSE
from .models import Hourglass, Koundinya2D_CNN


class SpectralReconstructor:
    """
    Reconstruct hyperspectral images from multispectral inputs.
    
    Args:
        model_config: Model architecture configuration
        inference_config: Inference configuration (paths, device, etc.)
    """
    
    def __init__(self,
                 model_config: ModelConfig,
                 inference_config: InferenceConfig):
        self.model_config = model_config
        self.inference_config = inference_config
        
        # Set device
        if inference_config.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(inference_config.device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics if ground truth is available
        if inference_config.ground_truth_dir is not None:
            self.ssim = SSIM()
            self.sam = SAM()
            self.mse_metric = MSE()
            self.rmse_metric = RMSE()
    
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint."""
        if self.model_config.model_type == 'hourglass':
            model = Hourglass(
                in_channels=self.model_config.in_channels,
                out_channels=self.model_config.out_channels,
                ub_out_channels=self.model_config.encoder_channels[0],
                mb_out_channels=self.model_config.encoder_channels[1],
                lb_out_channels=self.model_config.encoder_channels[2]
            )
        elif self.model_config.model_type == 'koundinya':
            model = Koundinya2D_CNN()
        else:
            raise ValueError(f"Unknown model type: {self.model_config.model_type}")
        
        # Load weights
        checkpoint = torch.load(self.inference_config.checkpoint, 
                               map_location=self.device)
        model.load_state_dict(checkpoint)
        
        print(f"Loaded model from {self.inference_config.checkpoint}")
        return model
    
    def reconstruct(self) -> Optional[Dict[str, float]]:
        """
        Reconstruct hyperspectral images from multispectral inputs.
        
        Returns:
            Dictionary of metrics if ground truth is available, else None
        """
        input_dir = Path(self.inference_config.input_dir)
        output_dir = Path(self.inference_config.output_dir)
        
        # Get list of input files
        input_files = sorted(list(input_dir.glob('*.npy')))
        
        if len(input_files) == 0:
            raise ValueError(f"No .npy files found in {input_dir}")
        
        print(f"\nReconstructing {len(input_files)} patches...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print("=" * 80)
        
        # Initialize metrics storage
        metrics_per_patch = {
            'ssim': [],
            'sam': [],
            'mse': [],
            'rmse': []
        } if self.inference_config.ground_truth_dir is not None else None
        
        # Process each file
        with torch.no_grad():
            for i, input_file in enumerate(input_files):
                if i % 100 == 0:
                    print(f"Processing patch {i}/{len(input_files)}...")
                
                # Load input
                input_patch = np.load(input_file)
                input_tensor = torch.from_numpy(input_patch).float()
                
                # Add batch dimension and permute to (1, C, H, W)
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
                input_tensor = input_tensor.to(self.device)
                
                # Reconstruct
                output_tensor = self.model(input_tensor)
                
                # Reshape to (C, H, W)
                output_tensor = output_tensor.reshape(
                    self.model_config.out_channels, 64, 64
                )
                
                # Convert back to numpy and permute to (H, W, C)
                output_patch = output_tensor.cpu().numpy()
                output_patch = np.transpose(output_patch, (1, 2, 0))
                
                # Save reconstructed patch
                output_file = output_dir / input_file.name
                np.save(output_file, output_patch)
                
                # Compute metrics if ground truth is available
                if self.inference_config.ground_truth_dir is not None:
                    gt_file = Path(self.inference_config.ground_truth_dir) / input_file.name
                    
                    if gt_file.exists():
                        gt_patch = np.load(gt_file)
                        gt_tensor = torch.from_numpy(gt_patch).float()
                        gt_tensor = gt_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                        output_tensor_4d = output_tensor.unsqueeze(0)
                        
                        # Compute metrics
                        ssim_val = self.ssim(output_tensor_4d, gt_tensor).item()
                        sam_val = self.sam(output_tensor_4d, gt_tensor).item()
                        mse_val = self.mse_metric(output_tensor_4d, gt_tensor).item()
                        rmse_val = self.rmse_metric(output_tensor_4d, gt_tensor).item()
                        
                        metrics_per_patch['ssim'].append(ssim_val)
                        metrics_per_patch['sam'].append(sam_val)
                        metrics_per_patch['mse'].append(mse_val)
                        metrics_per_patch['rmse'].append(rmse_val)
        
        print(f"\nReconstruction complete! Saved {len(input_files)} patches to {output_dir}")
        
        # Compute and save average metrics
        if metrics_per_patch is not None:
            avg_metrics = {
                'ssim': float(np.mean(metrics_per_patch['ssim'])),
                'sam': float(np.mean(metrics_per_patch['sam'])),
                'mse': float(np.mean(metrics_per_patch['mse'])),
                'rmse': float(np.mean(metrics_per_patch['rmse'])),
                'rmse_reflectance': float(np.mean(metrics_per_patch['rmse']) / 10000),
                'num_patches': len(input_files)
            }
            
            # Print metrics
            print("\n" + "=" * 80)
            print("Reconstruction Metrics:")
            print(f"  SSIM:             {avg_metrics['ssim']:.4f}")
            print(f"  SAM:              {avg_metrics['sam']:.4f}°")
            print(f"  MSE:              {avg_metrics['mse']:.2f}")
            print(f"  RMSE:             {avg_metrics['rmse']:.2f}")
            print(f"  RMSE (reflectance): {avg_metrics['rmse_reflectance']:.6f}")
            print(f"  Number of patches: {avg_metrics['num_patches']}")
            print("=" * 80)
            
            # Save metrics to JSON
            metrics_file = output_dir / 'reconstruction_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            
            print(f"\nSaved metrics to {metrics_file}")
            
            return avg_metrics
        
        return None


def reconstruct_from_config(model_config: ModelConfig, 
                            inference_config: InferenceConfig) -> Optional[Dict[str, float]]:
    """
    Convenience function to run reconstruction from configs.
    
    Args:
        model_config: Model architecture configuration
        inference_config: Inference configuration
        
    Returns:
        Dictionary of metrics if ground truth is available, else None
    """
    reconstructor = SpectralReconstructor(model_config, inference_config)
    return reconstructor.reconstruct()
