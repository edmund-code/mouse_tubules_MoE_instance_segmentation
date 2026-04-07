"""
4-Expert Gating Network for Multi-Task Instance Segmentation.

This module extends the original 3-expert gating network to handle
4 experts including the instance embedding expert.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    """Initialize network weights."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class OutputHead(nn.Module):
    """Simple 1x1 conv output head."""
    def __init__(self, input_channels, output_channels):
        super(OutputHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv1(x)


class GatingModule(nn.Module):
    """
    Gating module to compute attention weights for experts.
    
    Uses global average pooling followed by FC layers to compute
    per-expert attention weights.
    """
    def __init__(self, in_channel=64*4):
        super().__init__()
        self.dim = in_channel
        self.linear1 = nn.Linear(self.dim, int(self.dim * 0.5))
        
        # Attention mechanism
        self.attention = nn.Linear(int(self.dim * 0.5), int(self.dim * 0.5))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(int(self.dim * 0.5), in_channel // 64)

    def forward(self, x):
        # Global average pooling
        x = torch.mean(x, dim=2, keepdim=True).squeeze(dim=2)
        x = torch.mean(x, dim=2, keepdim=True).squeeze(dim=2)
        
        x = self.linear1(x)
        x = self.relu(x)
        
        attention_scores = self.attention(x)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        x = x * attention_weights
        
        x = self.dropout(x)
        x = self.linear2(x)
        prob = F.softmax(x, dim=-1)

        prob = prob.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # B, num_expert, 1, 1, 1
        
        return prob


class MultiGatingAttention4Experts(nn.Module):
    """
    Multi-gate attention network for 4 experts.
    
    Extends the original 3-expert gating to handle the additional instance expert.
    
    Architecture:
    1. Receive concatenated features from 4 experts: (B, 64*4, H, W) = (B, 256, H, W)
    2. Compute attention weights for each expert per task
    3. Produce 4 outputs: semantic, sdf, boundary, embeddings
    
    Parameters
    ----------
    in_channels : int
        Total input channels (64 * 4 = 256 for 4 experts with 64 features each).
    num_classes : int
        Number of segmentation classes for semantic and boundary outputs.
    embedding_dim : int, default=16
        Dimension of embedding output.
    feature_channels : int, default=64
        Number of feature channels per expert.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embedding_dim: int = 16,
        feature_channels: int = 64
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_channels = feature_channels
        self.num_experts = 4
        
        # Verify input channels
        assert in_channels == feature_channels * self.num_experts, \
            f"in_channels ({in_channels}) must equal feature_channels ({feature_channels}) * num_experts ({self.num_experts})"
        
        # Create separate gating modules for each task
        self.gating_seg = GatingModule(in_channels)
        self.gating_sdf = GatingModule(in_channels)
        self.gating_bnd = GatingModule(in_channels)
        self.gating_embed = GatingModule(in_channels)
        
        # Create task-specific output heads
        self.seg_head = OutputHead(feature_channels, num_classes)
        self.sdf_head = OutputHead(feature_channels, 1)
        self.bnd_head = OutputHead(feature_channels, num_classes)
        self.embed_head = OutputHead(feature_channels, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Concatenated expert features of shape (B, 256, H, W).
            Order: [segment_feat, sdf_feat, boundary_feat, instance_feat]
        
        Returns
        -------
        tuple of 4 torch.Tensor
            seg_out: Shape (B, num_classes, H, W) - semantic segmentation logits
            sdf_out: Shape (B, 1, H, W) - signed distance field
            bnd_out: Shape (B, num_classes, H, W) - boundary logits
            embed_out: Shape (B, embedding_dim, H, W) - instance embeddings
        """
        B, C, H, W = x.shape
        
        # Compute gating weights for each task
        w_seg = self.gating_seg(x)    # (B, 4, 1, 1, 1)
        w_sdf = self.gating_sdf(x)    # (B, 4, 1, 1, 1)
        w_bnd = self.gating_bnd(x)    # (B, 4, 1, 1, 1)
        w_embed = self.gating_embed(x) # (B, 4, 1, 1, 1)
        
        # Reshape input to (B, num_experts, feature_channels, H, W)
        x = x.view(B, self.num_experts, self.feature_channels, H, W)
        
        # Compute weighted combinations for each task
        seg_f = (x * w_seg).sum(dim=1)      # (B, 64, H, W)
        sdf_f = (x * w_sdf).sum(dim=1)      # (B, 64, H, W)
        bnd_f = (x * w_bnd).sum(dim=1)      # (B, 64, H, W)
        embed_f = (x * w_embed).sum(dim=1)  # (B, 64, H, W)
        
        # Apply task-specific heads
        seg_out = self.seg_head(seg_f)      # (B, num_classes, H, W)
        sdf_out = self.sdf_head(sdf_f)      # (B, 1, H, W)
        bnd_out = self.bnd_head(bnd_f)      # (B, num_classes, H, W)
        embed_out = self.embed_head(embed_f) # (B, embedding_dim, H, W)
        
        # Optionally normalize embeddings
        embed_out = F.normalize(embed_out, p=2, dim=1)
        
        return seg_out, sdf_out, bnd_out, embed_out


def get_gating_network_4experts(
    feature_channels: int = 64,
    num_experts: int = 4,
    num_classes: int = 2,
    embedding_dim: int = 16
) -> nn.Module:
    """
    Factory function to create 4-expert gating network.
    
    Parameters
    ----------
    feature_channels : int
        Features per expert.
    num_experts : int
        Number of experts (should be 4).
    num_classes : int
        Segmentation classes.
    embedding_dim : int
        Embedding dimension.
    
    Returns
    -------
    torch.nn.Module
        Initialized gating network.
    """
    if num_experts != 4:
        raise ValueError(f"This gating network is designed for 4 experts, got {num_experts}")
    
    in_channels = feature_channels * num_experts
    net = MultiGatingAttention4Experts(
        in_channels=in_channels,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        feature_channels=feature_channels
    )
    
    init_weights(net, 'kaiming')
    
    return net
