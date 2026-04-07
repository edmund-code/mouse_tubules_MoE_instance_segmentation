"""
Label Generation for Instance Segmentation.

This module generates semantic masks, signed distance fields (SDF), and boundary maps
from instance masks.
"""

from typing import Dict, Optional
import numpy as np
from scipy import ndimage
import cv2
from skimage import morphology
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def instance_to_semantic(instance_mask: np.ndarray) -> np.ndarray:
    """
    Convert instance mask to binary semantic mask.
    
    Parameters
    ----------
    instance_mask : np.ndarray
        Instance mask of shape (H, W) with dtype int.
        Background = 0, instances = 1, 2, 3, ...
    
    Returns
    -------
    np.ndarray
        Binary semantic mask of shape (H, W) with dtype np.int64.
        Background = 0, foreground = 1.
    """
    return (instance_mask > 0).astype(np.int64)


def instance_to_boundary(
    instance_mask: np.ndarray,
    boundary_width: int = 2,
    include_inner_boundaries: bool = True
) -> np.ndarray:
    """
    Generate boundary map from instance mask.
    
    Parameters
    ----------
    instance_mask : np.ndarray
        Instance mask of shape (H, W).
    boundary_width : int, default=2
        Width of boundary in pixels.
    include_inner_boundaries : bool, default=True
        If True, include boundaries between touching instances.
        If False, only include outer boundaries of foreground.
    
    Returns
    -------
    np.ndarray
        Binary boundary mask of shape (H, W) with dtype np.int64.
        Non-boundary = 0, boundary = 1.
    """
    selem = morphology.disk(boundary_width // 2)
    
    if include_inner_boundaries:
        boundary = np.zeros_like(instance_mask, dtype=np.int64)
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances > 0]
        
        for instance_id in unique_instances:
            instance_binary = (instance_mask == instance_id).astype(np.uint8)
            dilated = morphology.dilation(instance_binary, selem)
            eroded = morphology.erosion(instance_binary, selem)
            gradient = dilated.astype(np.int64) - eroded.astype(np.int64)
            boundary = np.maximum(boundary, (gradient > 0).astype(np.int64))
    else:
        semantic = instance_to_semantic(instance_mask).astype(np.uint8)
        dilated = morphology.dilation(semantic, selem)
        eroded = morphology.erosion(semantic, selem)
        boundary = ((dilated - eroded) > 0).astype(np.int64)
    
    return np.clip(boundary, 0, 1).astype(np.int64)


def instance_to_sdf(
    instance_mask: np.ndarray,
    normalize: bool = True,
    truncate: Optional[float] = None
) -> np.ndarray:
    """
    Generate signed distance field from instance mask.
    
    The SDF is computed such that:
    - Pixels inside instances have NEGATIVE values (distance to nearest boundary)
    - Pixels outside instances have POSITIVE values (distance to nearest boundary)
    - Boundary pixels have value close to 0
    
    Parameters
    ----------
    instance_mask : np.ndarray
        Instance mask of shape (H, W).
    normalize : bool, default=True
        If True, normalize SDF to range [-1, 1].
    truncate : float, optional
        If provided, truncate absolute SDF values to this maximum.
        Applied before normalization.
    
    Returns
    -------
    np.ndarray
        SDF of shape (H, W) with dtype np.float32.
    """
    h, w = instance_mask.shape
    sdf = np.zeros((h, w), dtype=np.float32)
    
    unique_instances = np.unique(instance_mask)
    unique_instances = unique_instances[unique_instances > 0]
    
    if len(unique_instances) == 0:
        return sdf
    
    for instance_id in unique_instances:
        instance_binary = (instance_mask == instance_id)
        
        inside_dist = ndimage.distance_transform_edt(instance_binary)
        outside_dist = ndimage.distance_transform_edt(~instance_binary)
        
        instance_sdf = outside_dist - inside_dist
        
        sdf[instance_binary] = instance_sdf[instance_binary]
    
    background_mask = (instance_mask == 0)
    if np.any(background_mask):
        semantic = instance_to_semantic(instance_mask).astype(bool)
        bg_dist = ndimage.distance_transform_edt(~semantic)
        sdf[background_mask] = bg_dist[background_mask]
    
    if truncate is not None:
        sdf = np.clip(sdf, -truncate, truncate)
    
    if normalize:
        max_abs = np.abs(sdf).max()
        if max_abs > 1e-8:
            sdf = sdf / max_abs
    
    return sdf.astype(np.float32)


def instance_to_center_heatmap(
    instance_mask: np.ndarray,
    sigma: float = 8.0
) -> np.ndarray:
    """
    Generate Gaussian heatmap centered at each instance centroid.
    
    This is an optional auxiliary target that can help with instance detection.
    
    Parameters
    ----------
    instance_mask : np.ndarray
        Instance mask of shape (H, W).
    sigma : float, default=8.0
        Standard deviation of Gaussian kernel in pixels.
    
    Returns
    -------
    np.ndarray
        Heatmap of shape (H, W) with dtype np.float32.
        Values in range [0, 1].
    """
    h, w = instance_mask.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    unique_instances = np.unique(instance_mask)
    unique_instances = unique_instances[unique_instances > 0]
    
    y_coords, x_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    for instance_id in unique_instances:
        y_idx, x_idx = np.where(instance_mask == instance_id)
        if len(y_idx) == 0:
            continue
        
        cy = y_idx.mean()
        cx = x_idx.mean()
        
        gaussian = np.exp(-((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (2 * sigma ** 2))
        heatmap = np.maximum(heatmap, gaussian)
    
    return np.clip(heatmap, 0, 1).astype(np.float32)


def generate_all_labels(
    instance_mask: np.ndarray,
    boundary_width: int = 2,
    normalize_sdf: bool = True,
    compute_centers: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate all training labels from an instance mask.
    
    Parameters
    ----------
    instance_mask : np.ndarray
        Instance mask of shape (H, W).
    boundary_width : int, default=2
        Width of boundaries in pixels.
    normalize_sdf : bool, default=True
        Whether to normalize SDF to [-1, 1].
    compute_centers : bool, default=False
        Whether to compute center heatmap (optional auxiliary target).
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - "instance_mask": Original instance mask (H, W), int32
        - "semantic_mask": Binary mask (H, W), int64
        - "boundary": Boundary map (H, W), int64
        - "sdf": Signed distance field (H, W), float32
        - "center_heatmap": (optional) Center heatmap (H, W), float32
    """
    labels = {
        "instance_mask": instance_mask.astype(np.int32),
        "semantic_mask": instance_to_semantic(instance_mask),
        "boundary": instance_to_boundary(instance_mask, boundary_width),
        "sdf": instance_to_sdf(instance_mask, normalize_sdf)
    }
    
    if compute_centers:
        labels["center_heatmap"] = instance_to_center_heatmap(instance_mask)
    
    return labels


def visualize_labels(
    image: np.ndarray,
    labels: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create visualization of image and all generated labels.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3).
    labels : Dict[str, np.ndarray]
        Dictionary from generate_all_labels().
    save_path : str, optional
        If provided, save visualization to this path.
    
    Returns
    -------
    np.ndarray
        Visualization image showing original and all label maps.
    """
    n_plots = 5 if "center_heatmap" not in labels else 6
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    instance_mask = labels["instance_mask"]
    instance_colored = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)
    unique_instances = np.unique(instance_mask)
    unique_instances = unique_instances[unique_instances > 0]
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(unique_instances) + 1, 3), dtype=np.uint8)
    for idx, inst_id in enumerate(unique_instances):
        instance_colored[instance_mask == inst_id] = colors[idx + 1]
    
    axes[1].imshow(instance_colored)
    axes[1].set_title(f"Instance Mask ({len(unique_instances)} instances)")
    axes[1].axis("off")
    
    axes[2].imshow(labels["semantic_mask"], cmap="gray")
    axes[2].set_title("Semantic Mask")
    axes[2].axis("off")
    
    axes[3].imshow(labels["boundary"], cmap="gray")
    axes[3].set_title("Boundary")
    axes[3].axis("off")
    
    sdf = labels["sdf"]
    sdf_vis = axes[4].imshow(sdf, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[4].set_title("Signed Distance Field")
    axes[4].axis("off")
    plt.colorbar(sdf_vis, ax=axes[4])
    
    if "center_heatmap" in labels:
        heatmap_vis = axes[5].imshow(labels["center_heatmap"], cmap="hot")
        axes[5].set_title("Center Heatmap")
        axes[5].axis("off")
        plt.colorbar(heatmap_vis, ax=axes[5])
    else:
        axes[5].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    vis_array = np.frombuffer(buf, dtype=np.uint8)
    vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    vis_array = vis_array[:, :, :3]
    
    plt.close(fig)
    
    return vis_array
