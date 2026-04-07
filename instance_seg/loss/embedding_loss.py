"""
Discriminative Loss for Instance Segmentation Embeddings.

This module implements the discriminative loss function from:
"Semantic Instance Segmentation with a Discriminative Loss Function"
(De Brabandere et al., 2017)
"""

from typing import Dict
import torch
import torch.nn as nn


def calculate_instance_means(
    embeddings: torch.Tensor,
    instance_mask: torch.Tensor,
    num_instances: int
) -> torch.Tensor:
    """
    Calculate mean embedding for each instance.
    
    Parameters
    ----------
    embeddings : torch.Tensor
        Embedding tensor of shape (E, H, W) where E is embedding dimension.
    instance_mask : torch.Tensor
        Instance mask of shape (H, W) with values 0, 1, 2, ..., num_instances.
        0 is background (ignored).
    num_instances : int
        Number of instances in the mask (excluding background).
    
    Returns
    -------
    torch.Tensor
        Instance means of shape (num_instances, E).
        means[i] is the mean embedding for instance (i+1).
    """
    E, H, W = embeddings.shape
    embeddings_flat = embeddings.reshape(E, H * W)
    instance_mask_flat = instance_mask.reshape(H * W)
    
    means = []
    for instance_id in range(1, num_instances + 1):
        mask = (instance_mask_flat == instance_id)
        if mask.sum() == 0:
            means.append(torch.zeros(E, device=embeddings.device))
        else:
            instance_embeddings = embeddings_flat[:, mask]
            mean = instance_embeddings.mean(dim=1)
            means.append(mean)
    
    return torch.stack(means) if means else torch.empty((0, E), device=embeddings.device)


def variance_loss(
    embeddings: torch.Tensor,
    instance_mask: torch.Tensor,
    instance_means: torch.Tensor,
    num_instances: int,
    delta_v: float = 0.5
) -> torch.Tensor:
    """
    Compute variance loss that pulls embeddings toward their instance mean.
    
    L_var = (1/C) * Σ_c [ (1/N_c) * Σ_i [ max(0, ||μ_c - x_i|| - δ_v)² ] ]
    
    Parameters
    ----------
    embeddings : torch.Tensor
        Embedding tensor of shape (E, H, W).
    instance_mask : torch.Tensor
        Instance mask of shape (H, W).
    instance_means : torch.Tensor
        Instance means of shape (num_instances, E).
    num_instances : int
        Number of instances.
    delta_v : float, default=0.5
        Variance margin. Embeddings within this distance of mean incur no loss.
    
    Returns
    -------
    torch.Tensor
        Scalar variance loss.
    """
    if num_instances == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    E, H, W = embeddings.shape
    embeddings_flat = embeddings.reshape(E, H * W).t()
    instance_mask_flat = instance_mask.reshape(H * W)
    
    var_losses = []
    for instance_id in range(1, num_instances + 1):
        mask = (instance_mask_flat == instance_id)
        count = mask.sum()
        
        if count == 0:
            continue
        
        instance_embeddings = embeddings_flat[mask]
        mean = instance_means[instance_id - 1]
        
        distances = torch.norm(instance_embeddings - mean.unsqueeze(0), dim=1)
        
        hinged = torch.clamp(distances - delta_v, min=0)
        var_loss = (hinged ** 2).mean()
        var_losses.append(var_loss)
    
    if len(var_losses) == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    return torch.stack(var_losses).mean()


def distance_loss(
    instance_means: torch.Tensor,
    num_instances: int,
    delta_d: float = 1.5
) -> torch.Tensor:
    """
    Compute distance loss that pushes instance means apart.
    
    L_dist = (1 / C(C-1)) * Σ_cA Σ_cB [ max(0, 2*δ_d - ||μ_cA - μ_cB||)² ]
    
    Parameters
    ----------
    instance_means : torch.Tensor
        Instance means of shape (num_instances, E).
    num_instances : int
        Number of instances.
    delta_d : float, default=1.5
        Distance margin. Means farther than 2*delta_d incur no loss.
    
    Returns
    -------
    torch.Tensor
        Scalar distance loss.
    """
    if num_instances <= 1:
        return torch.tensor(0.0, device=instance_means.device)
    
    pairwise_dists = torch.cdist(instance_means, instance_means)
    
    hinged = torch.clamp(2 * delta_d - pairwise_dists, min=0)
    hinged_sq = hinged ** 2
    
    triu_indices = torch.triu_indices(num_instances, num_instances, offset=1)
    dist_loss = hinged_sq[triu_indices[0], triu_indices[1]].mean()
    
    return dist_loss


def regularization_loss(instance_means: torch.Tensor) -> torch.Tensor:
    """
    Compute regularization loss to keep instance means small.
    
    L_reg = (1/C) * Σ_c ||μ_c||
    
    Parameters
    ----------
    instance_means : torch.Tensor
        Instance means of shape (num_instances, E).
    
    Returns
    -------
    torch.Tensor
        Scalar regularization loss.
    """
    if instance_means.shape[0] == 0:
        return torch.tensor(0.0, device=instance_means.device)
    
    norms = torch.norm(instance_means, dim=1)
    return norms.mean()


class DiscriminativeLoss(nn.Module):
    """
    Discriminative loss for instance segmentation embeddings.
    
    Combines variance loss, distance loss, and regularization loss.
    
    L_total = α * L_var + β * L_dist + γ * L_reg
    
    Parameters
    ----------
    delta_v : float, default=0.5
        Variance margin.
    delta_d : float, default=1.5
        Distance margin.
    alpha : float, default=1.0
        Weight for variance loss.
    beta : float, default=1.0
        Weight for distance loss.
    gamma : float, default=0.001
        Weight for regularization loss.
    """
    
    def __init__(
        self,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001
    ):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(
        self,
        embeddings: torch.Tensor,
        instance_masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminative loss for a batch.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Batch of embeddings of shape (B, E, H, W).
        instance_masks : torch.Tensor
            Batch of instance masks of shape (B, H, W).
            Values: 0 = background, 1, 2, 3, ... = instance IDs.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - "loss": Total combined loss (scalar)
            - "var_loss": Variance loss component (scalar)
            - "dist_loss": Distance loss component (scalar)
            - "reg_loss": Regularization loss component (scalar)
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        total_var_loss = torch.tensor(0.0, device=device)
        total_dist_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        for b in range(batch_size):
            emb = embeddings[b]
            inst_mask = instance_masks[b]
            
            num_instances = inst_mask.max().item()
            if num_instances == 0:
                continue
            
            num_instances = int(num_instances)
            valid_samples += 1
            
            means = calculate_instance_means(emb, inst_mask, num_instances)
            
            var_loss_val = variance_loss(emb, inst_mask, means, num_instances, self.delta_v)
            dist_loss_val = distance_loss(means, num_instances, self.delta_d)
            reg_loss_val = regularization_loss(means)
            
            total_var_loss += var_loss_val
            total_dist_loss += dist_loss_val
            total_reg_loss += reg_loss_val
        
        if valid_samples > 0:
            total_var_loss /= valid_samples
            total_dist_loss /= valid_samples
            total_reg_loss /= valid_samples
        
        total_loss = (
            self.alpha * total_var_loss +
            self.beta * total_dist_loss +
            self.gamma * total_reg_loss
        )
        
        return {
            "loss": total_loss,
            "var_loss": total_var_loss,
            "dist_loss": total_dist_loss,
            "reg_loss": total_reg_loss
        }


def compute_embedding_loss_for_batch(
    embeddings: torch.Tensor,
    instance_masks: torch.Tensor,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.001
) -> torch.Tensor:
    """
    Functional interface for computing discriminative loss.
    
    This is a convenience function that wraps DiscriminativeLoss.
    
    Parameters
    ----------
    embeddings : torch.Tensor
        Shape (B, E, H, W).
    instance_masks : torch.Tensor
        Shape (B, H, W).
    delta_v, delta_d, alpha, beta, gamma : float
        Loss hyperparameters.
    
    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    loss_fn = DiscriminativeLoss(delta_v, delta_d, alpha, beta, gamma)
    loss_dict = loss_fn(embeddings, instance_masks)
    return loss_dict["loss"]
