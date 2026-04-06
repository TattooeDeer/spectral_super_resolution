"""Training logic for spectral super-resolution models."""

import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from .config import ModelConfig, DataConfig, TrainConfig
from .datasets import SpectralDataset_npy, HyperspectralDataset_npy
from .losses import GramLoss
from .metrics import SSIM, SAM, MSE, RMSE
from .models import Hourglass, Koundinya2D_CNN


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Unified trainer for spectral super-resolution models.
    
    Supports:
    - Autoencoder training (Hyperion -> Hyperion)
    - Spectral reconstruction (Landsat -> Hyperion)
    - MSE loss
    - Perceptual loss (MSE + Gram loss)
    
    Args:
        model_config: Model architecture configuration
        data_config: Data loading configuration
        train_config: Training hyperparameters and settings
    """
    
    def __init__(self,
                 model_config: ModelConfig,
                 data_config: DataConfig,
                 train_config: TrainConfig):
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        
        # Set device
        if train_config.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(train_config.device)
        
        print(f"Using device: {self.device}")
        
        # Set seed for reproducibility
        set_seed(train_config.seed)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = self._create_loss_function()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_config.lr)
        
        # Initialize metrics
        self.ssim = SSIM()
        self.sam = SAM()
        self.mse_metric = MSE()
        self.rmse_metric = RMSE()
        
        # Create dataloaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Training history
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'val_ssim': [],
            'val_sam': [],
        }
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
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
        
        print(f"Created {self.model_config.model_type} model")
        return model
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        if self.train_config.loss_type == 'mse':
            return nn.MSELoss(reduction='mean')
        
        elif self.train_config.loss_type == 'perceptual':
            # Load pretrained autoencoder for perceptual loss
            ae_model = Hourglass(
                in_channels=self.model_config.out_channels,  # Hyperion -> Hyperion
                out_channels=self.model_config.out_channels,
                ub_out_channels=self.model_config.encoder_channels[0],
                mb_out_channels=self.model_config.encoder_channels[1],
                lb_out_channels=self.model_config.encoder_channels[2]
            )
            
            checkpoint = torch.load(self.train_config.ae_checkpoint, 
                                   map_location=self.device)
            ae_model.load_state_dict(checkpoint)
            ae_model.to(self.device)
            ae_model.eval()
            
            # Freeze autoencoder
            for param in ae_model.parameters():
                param.requires_grad = False
            
            print(f"Loaded pretrained autoencoder from {self.train_config.ae_checkpoint}")
            
            return GramLoss(
                perceptual_model=ae_model,
                content_loss_coeff=self.train_config.content_loss_coeff,
                style_loss_coeff=self.train_config.style_loss_coeff,
                block_coeff=self.train_config.block_coeff,
                device=str(self.device)
            )
        
        else:
            raise ValueError(f"Unknown loss type: {self.train_config.loss_type}")
    
    def _create_dataloaders(self):
        """Create training and validation dataloaders."""
        if self.train_config.mode == 'autoencoder':
            # Autoencoder mode: input = target (Hyperion only)
            train_dataset = HyperspectralDataset_npy(
                root_dir=self.data_config.hyperion_train_dir
            )
            val_dataset = HyperspectralDataset_npy(
                root_dir=self.data_config.hyperion_val_dir
            )
        
        elif self.train_config.mode == 'sr':
            # Spectral reconstruction mode: Landsat -> Hyperion
            if self.data_config.landsat_train_dir is None:
                raise ValueError("landsat_train_dir required for SR mode")
            if self.data_config.landsat_val_dir is None:
                raise ValueError("landsat_val_dir required for SR mode")
            
            train_dataset = SpectralDataset_npy(
                root_dir_multi=self.data_config.landsat_train_dir,
                root_dir_hyp=self.data_config.hyperion_train_dir
            )
            val_dataset = SpectralDataset_npy(
                root_dir_multi=self.data_config.landsat_val_dir,
                root_dir_hyp=self.data_config.hyperion_val_dir
            )
        
        else:
            raise ValueError(f"Unknown mode: {self.train_config.mode}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers
        )
        
        print(f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, (x_batch, y_batch) in enumerate(self.train_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            
            # Reshape output to (batch, channels, height, width)
            batch_size = y_pred.shape[0]
            y_pred = y_pred.reshape(batch_size, self.model_config.out_channels, 64, 64)
            
            # Compute loss
            loss = self.loss_fn(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = np.mean(epoch_losses)
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        val_losses = []
        val_ssims = []
        val_sams = []
        
        with torch.no_grad():
            for x_val, y_val in self.val_loader:
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)
                
                # Forward pass
                y_pred = self.model(x_val)
                
                # Reshape output
                batch_size = y_pred.shape[0]
                y_pred = y_pred.reshape(batch_size, self.model_config.out_channels, 64, 64)
                
                # Compute loss
                loss = self.loss_fn(y_pred, y_val)
                val_losses.append(loss.item())
                
                # Compute metrics
                ssim_val = self.ssim(y_pred, y_val).item()
                sam_val = self.sam(y_pred, y_val).item()
                
                val_ssims.append(ssim_val)
                val_sams.append(sam_val)
        
        return {
            'loss': np.mean(val_losses),
            'ssim': np.mean(val_ssims),
            'sam': np.mean(val_sams)
        }
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.train_config.epochs} epochs")
        print(f"Mode: {self.train_config.mode}, Loss: {self.train_config.loss_type}")
        print("=" * 80)
        
        for epoch in range(self.train_config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.train_config.epochs}")
            print("-" * 80)
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_losses'].append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.history['val_losses'].append(val_metrics['loss'])
            self.history['val_ssim'].append(val_metrics['ssim'])
            self.history['val_sam'].append(val_metrics['sam'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val SSIM:   {val_metrics['ssim']:.4f}")
            print(f"  Val SAM:    {val_metrics['sam']:.4f}°")
            
            # Save checkpoint
            if (epoch + 1) % self.train_config.checkpoint_every == 0:
                self.save_checkpoint(epoch + 1)
        
        print("\n" + "=" * 80)
        print("Training completed!")
        
        # Save final model and history
        self.save_checkpoint('final')
        self.save_history()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        output_dir = Path(self.train_config.output_dir)
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    def save_history(self):
        """Save training history as JSON."""
        output_dir = Path(self.train_config.output_dir)
        history_path = output_dir / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"  Saved training history: {history_path}")
