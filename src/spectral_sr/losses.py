"""Loss functions for spectral super-resolution with perceptual/style components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


def gram_matrix(input: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix for style/perceptual loss.
    
    The Gram matrix captures texture/style by computing correlations between feature maps.
    This implementation correctly handles batched inputs.
    
    Args:
        input: Feature tensor of shape (batch, channels, height, width)
        
    Returns:
        Gram matrix of shape (batch, channels, channels)
        Normalized by the number of elements in each feature map.
    """
    batch, channels, height, width = input.size()
    
    # Reshape to (batch, channels, height*width)
    features = input.view(batch, channels, height * width)
    
    # Compute Gram matrix per sample: (batch, channels, height*width) @ (batch, height*width, channels)
    # Result: (batch, channels, channels)
    G = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by number of elements in each feature map
    return G.div(channels * height * width)


class StyleLoss(nn.Module):
    """
    Style loss based on Gram matrix comparison.
    
    Compares the Gram matrices of two feature maps to measure style/texture similarity.
    Used as a component in the perceptual loss.
    """
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.loss = None
    
    def forward(self, input: tuple) -> torch.Tensor:
        """
        Compute style loss between ground truth and reconstructed features.
        
        Args:
            input: Tuple of (ground_truth_features, reconstructed_features)
                   Each tensor has shape (batch, channels, height, width)
        
        Returns:
            The input tuple (pass-through for compatibility)
        """
        gt_features, recon_features = input
        
        G_gt = gram_matrix(gt_features)
        G_recon = gram_matrix(recon_features)
        
        self.loss = F.mse_loss(G_gt, G_recon)
        
        return input


def get_activation(name: str, activation_dict: Dict[str, torch.Tensor]):
    """
    Create a forward hook to capture intermediate activations.
    
    Args:
        name: Name to store activation under in the dictionary
        activation_dict: Dictionary to store activations
        
    Returns:
        Hook function
    """
    def hook(model, input, output):
        activation_dict[name] = output.detach()
    return hook


class GramLoss(nn.Module):
    """
    Combined perceptual loss using Gram matrices from a pretrained autoencoder.
    
    This loss combines:
    1. Content loss: MSE between reconstructed and ground truth images
    2. Style/perceptual loss: MSE between Gram matrices of features from multiple layers
    
    The perceptual model is a pretrained autoencoder trained on Hyperion images.
    Features are extracted from the encoder blocks to capture spectral characteristics.
    
    Args:
        perceptual_model: Pretrained Hourglass autoencoder (frozen during training)
        style_loss: StyleLoss module for computing Gram matrix loss
        content_loss: Loss function for content (typically MSE)
        content_loss_coeff: Weight for content loss (alpha in paper, default 1.0)
        style_loss_coeff: Weight for style loss (beta in paper, default 1e-3)
        block_coeff: Weights for each encoder block's contribution [upper, middle, lower]
        device: Device to run computations on
    """
    
    def __init__(self, 
                 perceptual_model: nn.Module,
                 style_loss: nn.Module = None,
                 content_loss: nn.Module = None,
                 content_loss_coeff: float = 1.0,
                 style_loss_coeff: float = 1e-3,
                 block_coeff: List[float] = None,
                 device: str = 'cpu'):
        super(GramLoss, self).__init__()
        
        self.perceptual_model = perceptual_model
        self.perceptual_model.eval()  # Frozen
        
        self.style_loss = style_loss if style_loss is not None else StyleLoss()
        self.content_loss = content_loss if content_loss is not None else nn.MSELoss(reduction='mean')
        
        self.device = device
        self.content_loss_coeff = content_loss_coeff
        self.style_loss_coeff = style_loss_coeff
        self.block_coeff = block_coeff if block_coeff is not None else [1.0, 1.0, 1.0]
        
        # Move to device
        self.style_loss.to(device)
        self.content_loss.to(device)
    
    def forward(self, img_recon: torch.Tensor, img_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute combined content + style loss.
        
        Args:
            img_recon: Reconstructed image (batch, channels, height, width)
            img_gt: Ground truth image (batch, channels, height, width)
            
        Returns:
            Combined loss value
        """
        perceptual_feat_gt = {}
        perceptual_feat_recon = {}
        blocks_list = ['encoder_upperBlock', 'encoder_middleBlock', 'encoder_lowerBlock']
        
        # Extract features from ground truth image
        hook_ub = self.perceptual_model.encoder_upperBlock.register_forward_hook(
            get_activation('encoder_upperBlock', perceptual_feat_gt))
        hook_mb = self.perceptual_model.encoder_middleBlock.register_forward_hook(
            get_activation('encoder_middleBlock', perceptual_feat_gt))
        hook_lb = self.perceptual_model.encoder_lowerBlock.register_forward_hook(
            get_activation('encoder_lowerBlock', perceptual_feat_gt))
        
        with torch.no_grad():
            self.perceptual_model(img_gt)
        
        hook_ub.remove()
        hook_mb.remove()
        hook_lb.remove()
        
        # Extract features from reconstructed image
        hook_ub = self.perceptual_model.encoder_upperBlock.register_forward_hook(
            get_activation('encoder_upperBlock', perceptual_feat_recon))
        hook_mb = self.perceptual_model.encoder_middleBlock.register_forward_hook(
            get_activation('encoder_middleBlock', perceptual_feat_recon))
        hook_lb = self.perceptual_model.encoder_lowerBlock.register_forward_hook(
            get_activation('encoder_lowerBlock', perceptual_feat_recon))
        
        with torch.no_grad():
            self.perceptual_model(img_recon)
        
        hook_ub.remove()
        hook_mb.remove()
        hook_lb.remove()
        
        # Compute style loss for each block
        style_loss_total = torch.zeros(1, device=self.device, requires_grad=True)
        
        for block, coeff in zip(blocks_list, self.block_coeff):
            self.style_loss((perceptual_feat_gt[block], perceptual_feat_recon[block]))
            style_loss_total = style_loss_total + coeff * self.style_loss.loss
        
        # Compute content loss
        content_loss = self.content_loss(img_recon, img_gt)
        
        # Combine losses
        total_loss = self.content_loss_coeff * content_loss + self.style_loss_coeff * style_loss_total
        
        return total_loss
