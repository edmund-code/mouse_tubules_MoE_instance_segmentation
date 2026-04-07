#!/usr/bin/env python3
"""
Patch Extraction Script for WSI and QuPath Annotations.

This script extracts patches from TIF whole slide images using QuPath GeoJSON
annotations for instance segmentation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from instance_seg.utils.geojson_parser import QuPathAnnotationHandler
from instance_seg.utils.label_generation import generate_all_labels, visualize_labels


def load_tif_wsi(tif_path: str) -> np.ndarray:
    """
    Load a TIF whole slide image.
    
    Parameters
    ----------
    tif_path : str
        Path to .tif file.
    
    Returns
    -------
    np.ndarray
        Image array of shape (H, W, 3) for RGB.
    """
    image = tifffile.imread(tif_path)
    
    # Handle different formats
    if image.ndim == 2:
        # Grayscale -> RGB
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            # RGBA -> RGB
            image = image[:, :, :3]
        elif image.shape[2] > 3:
            # Take first 3 channels
            image = image[:, :, :3]
    
    return image.astype(np.uint8)


def extract_patches_from_wsi(
    wsi_image: np.ndarray,
    annotation_handler: Optional[QuPathAnnotationHandler],
    patch_size: int = 256,
    stride: int = 256,
    min_tissue_ratio: float = 0.1,
    min_annotation_ratio: float = 0.0
) -> List[Dict]:
    """
    Extract patches from a WSI with corresponding instance masks.
    
    Parameters
    ----------
    wsi_image : np.ndarray
        Full WSI image of shape (H, W, 3).
    annotation_handler : QuPathAnnotationHandler or None
        Handler for QuPath annotations. None for unsupervised extraction.
    patch_size : int
        Size of patches to extract.
    stride : int
        Stride between patches.
    min_tissue_ratio : float
        Minimum ratio of patch that must be tissue (non-white).
    min_annotation_ratio : float
        Minimum ratio of patch that must have annotations.
    
    Returns
    -------
    list of dict
        List of patch dictionaries with image, instance_mask, coordinates.
    """
    H, W, _ = wsi_image.shape
    patches = []
    
    # Calculate grid
    y_positions = range(0, H - patch_size + 1, stride)
    x_positions = range(0, W - patch_size + 1, stride)
    
    total_positions = len(list(y_positions)) * len(list(x_positions))
    
    with tqdm(total=total_positions, desc="Extracting patches") as pbar:
        for y in y_positions:
            for x in x_positions:
                # Extract image patch
                image_patch = wsi_image[y:y+patch_size, x:x+patch_size]
                
                # Check if patch has correct size (edge cases)
                if image_patch.shape[:2] != (patch_size, patch_size):
                    pbar.update(1)
                    continue
                
                # Check tissue ratio (avoid white background)
                gray = np.mean(image_patch, axis=2).astype(np.uint8)
                tissue_mask = gray < 240
                tissue_ratio = tissue_mask.sum() / (patch_size ** 2)
                
                if tissue_ratio < min_tissue_ratio:
                    pbar.update(1)
                    continue
                
                # Get instance mask if annotations available
                if annotation_handler is not None:
                    bounds = (x, y, x + patch_size, y + patch_size)
                    instance_mask = annotation_handler.create_mask_for_region(
                        bounds=bounds,
                        mask_shape=(patch_size, patch_size),
                        min_overlap=0.1
                    )
                    
                    # Check annotation ratio
                    annotation_ratio = (instance_mask > 0).sum() / (patch_size ** 2)
                    if annotation_ratio < min_annotation_ratio:
                        pbar.update(1)
                        continue
                    
                    num_instances = len(np.unique(instance_mask)) - 1
                else:
                    instance_mask = np.zeros((patch_size, patch_size), dtype=np.int32)
                    num_instances = 0
                
                patches.append({
                    'image': image_patch,
                    'instance_mask': instance_mask,
                    'x': x,
                    'y': y,
                    'num_instances': num_instances
                })
                
                pbar.update(1)
    
    return patches


def save_patches(
    patches: List[Dict],
    output_dir: str,
    wsi_name: str,
    save_visualization: bool = False
) -> None:
    """
    Save extracted patches to disk.
    
    Parameters
    ----------
    patches : list of dict
        List of patch dictionaries.
    output_dir : str
        Base output directory.
    wsi_name : str
        Name of source WSI.
    save_visualization : bool
        If True, save visualization images.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "instance_masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if save_visualization:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    for i, patch in enumerate(tqdm(patches, desc="Saving patches")):
        x, y = patch['x'], patch['y']
        patch_name = f"{wsi_name}_x{x}_y{y}"
        
        # Save image
        image_path = images_dir / f"{patch_name}.png"
        Image.fromarray(patch['image']).save(image_path)
        
        # Save instance mask
        mask_path = masks_dir / f"{patch_name}.npy"
        np.save(mask_path, patch['instance_mask'])
        
        # Save visualization
        if save_visualization and patch['num_instances'] > 0:
            labels = generate_all_labels(patch['instance_mask'])
            vis = visualize_labels(patch['image'], labels)
            vis_path = vis_dir / f"{patch_name}.png"
            Image.fromarray(vis).save(vis_path)


def compute_dataset_statistics(output_dir: str) -> Dict:
    """
    Compute statistics for extracted dataset.
    
    Parameters
    ----------
    output_dir : str
        Directory containing extracted patches.
    
    Returns
    -------
    dict
        Statistics including num_patches, instance counts, pixel mean/std.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "instance_masks"
    
    if not masks_dir.exists():
        return {}
    
    mask_files = sorted(list(masks_dir.glob("*.npy")))
    image_files = sorted(list(images_dir.glob("*.png")))
    
    instance_counts = []
    pixel_values = []
    
    print("Computing dataset statistics...")
    for mask_file in tqdm(mask_files):
        mask = np.load(mask_file)
        unique_instances = np.unique(mask)
        unique_instances = unique_instances[unique_instances > 0]
        instance_counts.append(len(unique_instances))
    
    # Sample images for mean/std (can be slow for large datasets)
    sample_size = min(100, len(image_files))
    for i in tqdm(range(sample_size), desc="Computing pixel statistics"):
        image = np.array(Image.open(image_files[i]))
        pixel_values.append(image.reshape(-1, 3))
    
    if pixel_values:
        pixel_values = np.concatenate(pixel_values, axis=0)
        mean_rgb = pixel_values.mean(axis=0) / 255.0
        std_rgb = pixel_values.std(axis=0) / 255.0
    else:
        mean_rgb = [0.5, 0.5, 0.5]
        std_rgb = [0.5, 0.5, 0.5]
    
    instance_counts = np.array(instance_counts)
    
    stats = {
        'num_patches': len(mask_files),
        'num_instances_total': int(instance_counts.sum()),
        'instances_per_patch_mean': float(instance_counts.mean()),
        'instances_per_patch_std': float(instance_counts.std()),
        'instances_per_patch_min': int(instance_counts.min()) if len(instance_counts) > 0 else 0,
        'instances_per_patch_max': int(instance_counts.max()) if len(instance_counts) > 0 else 0,
        'mean_rgb': mean_rgb.tolist(),
        'std_rgb': std_rgb.tolist()
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract patches from TIF WSIs")
    parser.add_argument("--wsi_dir", type=str, required=True, help="Directory containing .tif files")
    parser.add_argument("--geojson_dir", type=str, default=None, help="Directory containing .geojson files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for patches")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size in pixels")
    parser.add_argument("--stride", type=int, default=256, help="Stride between patches")
    parser.add_argument("--min_tissue_ratio", type=float, default=0.3, help="Minimum tissue ratio")
    parser.add_argument("--min_annotation_ratio", type=float, default=0.0, help="Minimum annotation ratio")
    parser.add_argument("--class_names", nargs="+", default=["Tubule"], help="Annotation class names")
    parser.add_argument("--save_visualizations", action="store_true", help="Save visualization images")
    parser.add_argument("--no_annotations", action="store_true", help="Extract without annotations (unsupervised)")
    
    args = parser.parse_args()
    
    wsi_dir = Path(args.wsi_dir)
    output_dir = Path(args.output_dir)
    
    if not wsi_dir.exists():
        print(f"Error: WSI directory not found: {wsi_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get WSI files
    wsi_files = sorted(list(wsi_dir.glob("*.tif")) + list(wsi_dir.glob("*.tiff")))
    
    if len(wsi_files) == 0:
        print(f"Error: No .tif files found in {wsi_dir}")
        return
    
    print(f"Found {len(wsi_files)} WSI files")
    
    total_patches = 0
    
    for wsi_path in wsi_files:
        print(f"\nProcessing: {wsi_path.name}")
        
        # Load WSI
        wsi_image = load_tif_wsi(str(wsi_path))
        print(f"  Image size: {wsi_image.shape}")
        
        # Load annotations if available
        annotation_handler = None
        if not args.no_annotations:
            if args.geojson_dir is None:
                print("Error: --geojson_dir required when not using --no_annotations")
                return
            
            geojson_dir = Path(args.geojson_dir)
            geojson_path = geojson_dir / f"{wsi_path.stem}.geojson"
            
            if not geojson_path.exists():
                print(f"  Warning: GeoJSON not found: {geojson_path.name}, skipping...")
                continue
            
            annotation_handler = QuPathAnnotationHandler(
                str(geojson_path),
                class_names=args.class_names
            )
            print(f"  Loaded {len(annotation_handler)} annotations")
        
        # Extract patches
        patches = extract_patches_from_wsi(
            wsi_image=wsi_image,
            annotation_handler=annotation_handler,
            patch_size=args.patch_size,
            stride=args.stride,
            min_tissue_ratio=args.min_tissue_ratio,
            min_annotation_ratio=args.min_annotation_ratio
        )
        
        print(f"  Extracted {len(patches)} patches")
        
        if len(patches) > 0:
            # Save patches
            save_patches(
                patches=patches,
                output_dir=str(output_dir),
                wsi_name=wsi_path.stem,
                save_visualization=args.save_visualizations
            )
            
            total_patches += len(patches)
    
    print(f"\nTotal patches extracted: {total_patches}")
    
    # Compute and save statistics
    if total_patches > 0:
        stats = compute_dataset_statistics(str(output_dir))
        stats_path = output_dir / "statistics.json"
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print(f"\nStatistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
