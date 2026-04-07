#!/usr/bin/env python3
"""Pre-compute SDF and boundary labels for all patches."""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage

def compute_sdf(instance_mask: np.ndarray) -> np.ndarray:
    """Compute signed distance function from instance mask."""
    binary_mask = (instance_mask > 0).astype(np.float32)
    
    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask)
    if binary_mask.sum() == binary_mask.size:
        return np.ones_like(binary_mask)
    
    # Distance transform
    pos_dist = ndimage.distance_transform_edt(binary_mask)
    neg_dist = ndimage.distance_transform_edt(1 - binary_mask)
    
    sdf = pos_dist - neg_dist
    
    # Normalize to [-1, 1]
    max_dist = max(pos_dist.max(), neg_dist.max(), 1)
    sdf = np.clip(sdf / max_dist, -1, 1)
    
    return sdf.astype(np.float32)


def compute_boundary(instance_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Compute boundary mask from instance mask."""
    binary_mask = (instance_mask > 0).astype(np.uint8)
    
    # Dilate and erode to find boundary
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated = ndimage.binary_dilation(binary_mask, kernel)
    eroded = ndimage.binary_erosion(binary_mask, kernel)
    
    boundary = (dilated ^ eroded).astype(np.float32)
    return boundary


def precompute_labels(data_dir: str):
    """Pre-compute SDF and boundary for all splits."""
    data_dir = Path(data_dir)
    
    for split in ['train_sup', 'train_unsup', 'val']:
        split_dir = data_dir / split
        masks_dir = split_dir / 'instance_masks'
        
        if not masks_dir.exists():
            print(f"Skipping {split} - not found")
            continue
        
        # Create output directories
        sdf_dir = split_dir / 'sdf'
        bnd_dir = split_dir / 'boundary'
        sdf_dir.mkdir(exist_ok=True)
        bnd_dir.mkdir(exist_ok=True)
        
        mask_files = sorted(masks_dir.glob('*.npy'))
        print(f"\nProcessing {split}: {len(mask_files)} files")
        
        for mask_path in tqdm(mask_files):
            # Load instance mask
            instance_mask = np.load(mask_path)
            
            # Compute labels
            sdf = compute_sdf(instance_mask)
            boundary = compute_boundary(instance_mask)
            
            # Save
            np.save(sdf_dir / mask_path.name, sdf)
            np.save(bnd_dir / mask_path.name, boundary)
    
    print("\nDone! Labels saved to sdf/ and boundary/ directories.")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    precompute_labels(data_dir)