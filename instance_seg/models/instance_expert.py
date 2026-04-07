"""
Instance Expert Model for Instance Segmentation.

This module provides the 4th expert network that produces pixel embeddings
for discriminative instance segmentation.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add Semi-MoE to path for imports
sys.path.insert(0, '/home/edmund/Desktop/mouse_tubules_MoE_instance_segmentation/Semi-MoE')
from models.getnetwork import get_network


def get_instance_expert(
    in_channels: int = 3,
    embedding_dim: int = 16,
    base_features: int = 64,
    normalize_embeddings: bool = True
) -> nn.Module:
    """
    Create instance embedding expert network.
    
    Parameters
    ----------
    in_channels : int, default=3
        Number of input channels (RGB).
    embedding_dim : int, default=16
        Dimension of output embeddings per pixel.
    base_features : int, default=64
        Number of features in the first encoder layer.
    normalize_embeddings : bool, default=True
        Whether to L2-normalize embeddings.
    
    Returns
    -------
    torch.nn.Module
        Network that returns (features, embeddings).
        features: Shape (B, base_features, H, W) - for gating network
        embeddings: Shape (B, embedding_dim, H, W) - instance embeddings
    """
    base_unet = get_network('unet', in_channels, embedding_dim)
    return InstanceExpertWrapper(base_unet, embedding_dim, normalize_embeddings)


class InstanceExpertWrapper(nn.Module):
    """
    Wrapper to create instance expert from existing U-Net architecture.
    
    This approach reuses the existing U-Net implementation from Semi-MoE
    and only modifies the final output layer to produce embeddings.
    
    Parameters
    ----------
    base_unet : torch.nn.Module
        Base U-Net model (from Semi-MoE/models/unet.py).
    embedding_dim : int, default=16
        Output embedding dimension.
    normalize_embeddings : bool, default=True
        Whether to L2-normalize embeddings to unit sphere.
    """
    
    def __init__(
        self,
        base_unet: nn.Module,
        embedding_dim: int = 16,
        normalize_embeddings: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        
        # Extract encoder layers
        self.Maxpool = base_unet.Maxpool
        self.Conv1 = base_unet.Conv1
        self.Conv2 = base_unet.Conv2
        self.Conv3 = base_unet.Conv3
        self.Conv4 = base_unet.Conv4
        self.Conv5 = base_unet.Conv5
        
        # Extract decoder layers
        self.Up5 = base_unet.Up5
        self.Up_conv5 = base_unet.Up_conv5
        self.Up4 = base_unet.Up4
        self.Up_conv4 = base_unet.Up_conv4
        self.Up3 = base_unet.Up3
        self.Up_conv3 = base_unet.Up_conv3
        self.Up2 = base_unet.Up2
        self.Up_conv2 = base_unet.Up_conv2
        
        # Create embedding head (replaces Conv_1x1)
        self.embedding_head = nn.Conv2d(64, embedding_dim, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, 3, H, W).
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            features: Shape (B, 64, H, W) - for gating
            embeddings: Shape (B, embedding_dim, H, W)
        """
        # Encoding path
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # Decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        # Generate embeddings
        embeddings = self.embedding_head(d2)
        
        # Optionally normalize embeddings to unit sphere
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Return features (d2) for gating and embeddings
        return d2, embeddings


def create_all_experts(
    network_name: str,
    in_channels: int,
    num_classes: int,
    embedding_dim: int = 16
) -> dict:
    """
    Create all four expert networks.
    
    Parameters
    ----------
    network_name : str
        Name of base network (e.g., "unet").
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of segmentation classes.
    embedding_dim : int
        Embedding dimension for instance expert.
    
    Returns
    -------
    dict[str, torch.nn.Module]
        Dictionary with keys: "segment", "sdf", "boundary", "instance"
    """
    experts = {
        'segment': get_network(network_name, in_channels, num_classes),
        'sdf': get_network(network_name, in_channels, 1),
        'boundary': get_network(network_name, in_channels, num_classes),
        'instance': get_instance_expert(in_channels, embedding_dim)
    }
    
    return experts
