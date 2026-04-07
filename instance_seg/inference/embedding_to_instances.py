"""
Embedding to Instance Conversion.

This module provides functions to convert pixel embeddings to instance
segmentation masks using clustering algorithms.
"""

from typing import Optional
import numpy as np
import torch
from sklearn.cluster import MeanShift
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


def cluster_embeddings_meanshift(
    embeddings: np.ndarray,
    bandwidth: float = 0.5,
    min_bin_freq: int = 5
) -> np.ndarray:
    """
    Cluster embeddings using Mean Shift algorithm.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors of shape (N, E) where N is number of pixels and E is embedding dim.
    bandwidth : float, default=0.5
        Bandwidth parameter for mean shift. Controls cluster size.
    min_bin_freq : int, default=5
        Minimum number of points in a cluster.
    
    Returns
    -------
    np.ndarray
        Cluster labels of shape (N,). Each unique label is an instance.
    """
    if embeddings.shape[0] == 0:
        return np.array([])
    
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=min_bin_freq, n_jobs=-1)
    labels = ms.fit_predict(embeddings)
    
    return labels


def cluster_embeddings_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10
) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN algorithm.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors of shape (N, E).
    min_cluster_size : int, default=50
        Minimum number of points to form a cluster.
    min_samples : int, default=10
        Minimum samples in neighborhood for core points.
    
    Returns
    -------
    np.ndarray
        Cluster labels of shape (N,). -1 indicates noise (unassigned).
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is not installed. Install it with: pip install hdbscan")
    
    if embeddings.shape[0] == 0:
        return np.array([])
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(embeddings)
    
    return labels


def remove_small_instances(
    instance_mask: np.ndarray,
    min_area: int = 20
) -> np.ndarray:
    """
    Remove instances smaller than minimum area.
    
    Parameters
    ----------
    instance_mask : np.ndarray
        Instance mask of shape (H, W).
    min_area : int
        Minimum number of pixels for an instance.
    
    Returns
    -------
    np.ndarray
        Filtered instance mask with small instances set to 0.
    """
    instance_mask = instance_mask.copy()
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]
    
    for inst_id in unique_ids:
        area = np.sum(instance_mask == inst_id)
        if area < min_area:
            instance_mask[instance_mask == inst_id] = 0
    
    return renumber_instances(instance_mask)


def renumber_instances(instance_mask: np.ndarray) -> np.ndarray:
    """
    Renumber instances to be consecutive starting from 1.
    
    Parameters
    ----------
    instance_mask : np.ndarray
        Instance mask that may have gaps in numbering.
    
    Returns
    -------
    np.ndarray
        Instance mask with consecutive numbering: 0, 1, 2, 3, ...
    """
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]
    
    if len(unique_ids) == 0:
        return instance_mask
    
    new_mask = np.zeros_like(instance_mask)
    for new_id, old_id in enumerate(unique_ids, start=1):
        new_mask[instance_mask == old_id] = new_id
    
    return new_mask


def embeddings_to_instances(
    embeddings: torch.Tensor,
    semantic_mask: torch.Tensor,
    method: str = "meanshift",
    bandwidth: float = 0.5,
    min_cluster_size: int = 50,
    min_instance_area: int = 20
) -> torch.Tensor:
    """
    Convert embedding predictions to instance mask.
    
    Parameters
    ----------
    embeddings : torch.Tensor
        Embedding tensor of shape (B, E, H, W) or (E, H, W).
    semantic_mask : torch.Tensor
        Binary semantic mask of shape (B, H, W) or (H, W).
        Only foreground pixels are clustered.
    method : str, default="meanshift"
        Clustering method: "meanshift" or "hdbscan".
    bandwidth : float
        Bandwidth for mean shift.
    min_cluster_size : int
        Min cluster size for HDBSCAN.
    min_instance_area : int
        Minimum area (pixels) for an instance to be kept.
    
    Returns
    -------
    torch.Tensor
        Instance mask of shape (B, H, W) or (H, W).
        Background = 0, instances = 1, 2, 3, ...
    """
    # Handle single sample vs batch
    single_sample = False
    if embeddings.ndim == 3:
        embeddings = embeddings.unsqueeze(0)
        semantic_mask = semantic_mask.unsqueeze(0)
        single_sample = True
    
    device = embeddings.device
    B, E, H, W = embeddings.shape
    
    instance_masks = []
    
    for b in range(B):
        emb = embeddings[b].cpu().numpy()  # (E, H, W)
        sem = semantic_mask[b].cpu().numpy()  # (H, W)
        
        # Get foreground pixels
        fg_mask = (sem > 0)
        fg_indices = np.where(fg_mask)
        
        if len(fg_indices[0]) == 0:
            # No foreground, return zeros
            instance_mask = np.zeros((H, W), dtype=np.int32)
        else:
            # Extract foreground embeddings
            fg_embeddings = emb[:, fg_indices[0], fg_indices[1]].T  # (N_fg, E)
            
            # Cluster embeddings
            if method == "meanshift":
                labels = cluster_embeddings_meanshift(fg_embeddings, bandwidth)
            elif method == "hdbscan":
                labels = cluster_embeddings_hdbscan(fg_embeddings, min_cluster_size)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Handle noise labels from HDBSCAN (-1)
            if np.any(labels == -1):
                # Assign noise to nearest cluster or background
                labels[labels == -1] = 0
            
            # Shift labels to start from 1 (0 is background)
            if labels.max() >= 0:
                labels = labels + 1
            
            # Create instance mask
            instance_mask = np.zeros((H, W), dtype=np.int32)
            instance_mask[fg_indices] = labels
            
            # Remove small instances
            instance_mask = remove_small_instances(instance_mask, min_instance_area)
        
        instance_masks.append(instance_mask)
    
    # Convert to tensor
    instance_masks = np.stack(instance_masks, axis=0)
    instance_masks = torch.from_numpy(instance_masks).to(device)
    
    if single_sample:
        instance_masks = instance_masks.squeeze(0)
    
    return instance_masks


class EmbeddingClusterer:
    """
    Class to handle embedding clustering with configurable parameters.
    
    Parameters
    ----------
    method : str
        Clustering method: "meanshift" or "hdbscan".
    bandwidth : float
        Mean shift bandwidth.
    min_cluster_size : int
        HDBSCAN min cluster size.
    min_instance_area : int
        Minimum instance area in pixels.
    device : str
        Device for torch tensors.
    """
    
    def __init__(
        self,
        method: str = "meanshift",
        bandwidth: float = 0.5,
        min_cluster_size: int = 50,
        min_instance_area: int = 20,
        device: str = "cuda"
    ):
        self.method = method
        self.bandwidth = bandwidth
        self.min_cluster_size = min_cluster_size
        self.min_instance_area = min_instance_area
        self.device = device
    
    def __call__(
        self,
        embeddings: torch.Tensor,
        semantic_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert embeddings to instances.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Embeddings of shape (B, E, H, W) or (E, H, W).
        semantic_mask : torch.Tensor
            Semantic mask of shape (B, H, W) or (H, W).
        
        Returns
        -------
        torch.Tensor
            Instance mask of shape (B, H, W) or (H, W).
        """
        return embeddings_to_instances(
            embeddings=embeddings,
            semantic_mask=semantic_mask,
            method=self.method,
            bandwidth=self.bandwidth,
            min_cluster_size=self.min_cluster_size,
            min_instance_area=self.min_instance_area
        )
