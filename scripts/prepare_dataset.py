#!/usr/bin/env python3
"""
Dataset Preparation Script.

This script splits extracted patches into train_sup, train_unsup, and val sets.
"""

import argparse
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def split_dataset(
    source_dir: str,
    output_base_dir: str,
    val_ratio: float = 0.2,
    sup_ratio: float = 0.2,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Split extracted patches into train_sup, train_unsup, and val sets.
    
    Parameters
    ----------
    source_dir : str
        Directory containing all extracted patches.
    output_base_dir : str
        Base directory for output splits.
    val_ratio : float
        Ratio of data for validation.
    sup_ratio : float
        Ratio of training data to use as labeled (supervised).
    random_seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Split information with patch names for each set.
    """
    source_dir = Path(source_dir)
    output_base_dir = Path(output_base_dir)
    
    # Get all image files
    images_dir = source_dir / "images"
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    image_files = sorted(list(images_dir.glob("*.png")))
    patch_names = [f.stem for f in image_files]
    
    print(f"Found {len(patch_names)} patches")
    
    # Shuffle
    random.seed(random_seed)
    random.shuffle(patch_names)
    
    # Split into val and train
    num_val = int(len(patch_names) * val_ratio)
    val_names = patch_names[:num_val]
    train_names = patch_names[num_val:]
    
    # Split train into supervised and unsupervised
    num_sup = int(len(train_names) * sup_ratio)
    train_sup_names = train_names[:num_sup]
    train_unsup_names = train_names[num_sup:]
    
    print(f"Split: {len(train_sup_names)} train_sup, {len(train_unsup_names)} train_unsup, {len(val_names)} val")
    
    # Create output directories
    splits = {
        'train_sup': train_sup_names,
        'train_unsup': train_unsup_names,
        'val': val_names
    }
    
    for split_name, patch_list in splits.items():
        split_dir = output_base_dir / split_name
        split_images_dir = split_dir / "images"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create masks directory for supervised splits
        if split_name != 'train_unsup':
            split_masks_dir = split_dir / "instance_masks"
            split_masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        print(f"\nCopying {split_name}...")
        for patch_name in tqdm(patch_list):
            # Copy image
            src_image = source_dir / "images" / f"{patch_name}.png"
            dst_image = split_images_dir / f"{patch_name}.png"
            shutil.copy2(src_image, dst_image)
            
            # Copy mask (only for supervised splits)
            if split_name != 'train_unsup':
                src_mask = source_dir / "instance_masks" / f"{patch_name}.npy"
                dst_mask = split_masks_dir / f"{patch_name}.npy"
                if src_mask.exists():
                    shutil.copy2(src_mask, dst_mask)
    
    # Save split information
    split_info_path = output_base_dir / "split_info.json"
    with open(split_info_path, 'w') as f:
        json.dump({
            'train_sup': train_sup_names,
            'train_unsup': train_unsup_names,
            'val': val_names,
            'total': len(patch_names),
            'val_ratio': val_ratio,
            'sup_ratio': sup_ratio,
            'random_seed': random_seed
        }, f, indent=2)
    
    print(f"\nSplit information saved to: {split_info_path}")
    
    return splits


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train_sup, train_unsup, and val sets")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory with all patches")
    parser.add_argument("--output_dir", type=str, required=True, help="Output base directory")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--sup_ratio", type=float, default=0.2, help="Supervised training ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    splits = split_dataset(
        source_dir=args.source_dir,
        output_base_dir=args.output_dir,
        val_ratio=args.val_ratio,
        sup_ratio=args.sup_ratio,
        random_seed=args.seed
    )
    
    print("\nDataset preparation complete!")
    print(f"  Train supervised: {len(splits['train_sup'])} patches")
    print(f"  Train unsupervised: {len(splits['train_unsup'])} patches")
    print(f"  Validation: {len(splits['val'])} patches")


if __name__ == "__main__":
    main()
