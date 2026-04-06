"""Evaluation metrics for spectral super-resolution."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


eps = 1e-6


class SSIM:
    """
    Structural Similarity Index (SSIM) metric.
    
    Modified from https://github.com/jorge-pessoa/pytorch-msssim
    
    Measures perceptual similarity between images based on luminance, contrast, 
    and structure. Higher values indicate better similarity.
    """
    
    def __init__(self, description: str = "Structural Similarity Index"):
        self.des = description
    
    def __repr__(self):
        return "SSIM"
    
    def gaussian(self, w_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        gauss = torch.Tensor([
            math.exp(-(x - w_size//2)**2 / float(2*sigma**2)) 
            for x in range(w_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, w_size: int, channel: int = 1) -> torch.Tensor:
        """Create 2D Gaussian window."""
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window
    
    def __call__(self, 
                 y_pred: torch.Tensor, 
                 y_true: torch.Tensor,
                 w_size: int = 11,
                 size_average: bool = True,
                 full: bool = False) -> torch.Tensor:
        """
        Compute SSIM between predicted and true images.
        
        Args:
            y_true: Ground truth tensor (batch, channels, height, width)
            y_pred: Predicted tensor (batch, channels, height, width)
            w_size: Window size for Gaussian kernel (default 11)
            size_average: If True, return mean SSIM; else per-sample SSIM
            full: If True, also return contrast sensitivity
            
        Returns:
            SSIM value (higher is better, range [-1, 1], typically [0, 1])
        """
        # Determine value range
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1
        
        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        
        L = max_val - min_val
        
        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)
        
        # Compute local means
        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2
        
        # SSIM constants
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        
        # Contrast sensitivity
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)
        
        # SSIM map
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        
        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        
        if full:
            return ret, cs
        return ret


class SAM:
    """
    Spectral Angle Mapper (SAM) metric.
    
    Also known as Angular Error (AE) in some papers.
    Measures the angle between spectral vectors. Lower values indicate better 
    spectral similarity.
    
    Modified from matlab: colorangle.m, MATLAB V2019b
    angle = acos(RGB1' * RGB2 / (norm(RGB1) * norm(RGB2)))
    angle = 180 / pi * angle
    """
    
    def __init__(self, description: str = "Spectral Angle Mapper / Angular Error"):
        self.des = description
    
    def __repr__(self):
        return "SAM"
    
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute SAM/Angular Error between predicted and true spectral images.
        
        Args:
            y_true: Ground truth tensor (batch, channels, height, width)
            y_pred: Predicted tensor (batch, channels, height, width)
            
        Returns:
            Mean angular error in degrees (lower is better)
        """
        # Compute dot product along channel dimension
        dot_product = torch.sum(y_pred * y_true, dim=1)
        
        # Compute norms
        norm_pred = torch.sqrt(torch.sum(y_pred * y_pred, dim=1))
        norm_true = torch.sqrt(torch.sum(y_true * y_true, dim=1))
        
        # Compute angle in degrees
        cos_angle = dot_product / (norm_pred * norm_true + eps)
        
        # Clamp to avoid numerical issues with acos
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        angle = torch.acos(cos_angle) * 180.0 / math.pi
        
        # Return mean across spatial dimensions and batch
        return angle.mean()


class MSE:
    """Mean Squared Error metric."""
    
    def __init__(self, description: str = "Mean Squared Error"):
        self.des = description
    
    def __repr__(self):
        return "MSE"
    
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE between predicted and true images.
        
        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor
            
        Returns:
            Mean squared error (lower is better)
        """
        return torch.mean((y_pred - y_true) ** 2)


class RMSE:
    """Root Mean Squared Error metric."""
    
    def __init__(self, description: str = "Root Mean Squared Error"):
        self.des = description
    
    def __repr__(self):
        return "RMSE"
    
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSE between predicted and true images.
        
        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor
            
        Returns:
            Root mean squared error (lower is better)
        """
        mse = torch.mean((y_pred - y_true) ** 2)
        return torch.sqrt(mse)
