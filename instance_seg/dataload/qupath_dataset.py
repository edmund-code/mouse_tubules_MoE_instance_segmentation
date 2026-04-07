"""
PyTorch Dataset for QuPath Instance Segmentation.

This module provides Dataset classes for loading pre-extracted patches
with instance segmentation labels.
"""

import os
from typing import Optional, Callable, Dict, List
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A

from ..utils.label_generation import generate_all_labels


class QuPathPatchDataset(Dataset):
    """
    PyTorch Dataset for pre-extracted patches with instance segmentation labels.
    
    Expected directory structure:
    data_dir/
    ├── images/
    │   ├── patch_001.png
    │   ├── patch_002.png
    │   └── ...
    ├── instance_masks/
    │   ├── patch_001.npy
    │   ├── patch_002.npy
    │   └── ...
    └── metadata.json (optional)
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        normalize: Optional[Callable] = None,
        supervised: bool = True,
        generate_labels_online: bool = True,
        use_precomputed_labels: bool = True,
        boundary_width: int = 2,
        normalize_sdf: bool = True
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        data_dir : str
            Path to directory containing images/ and instance_masks/ subdirectories.
        transform : callable, optional
            Albumentations transform to apply to image AND masks together.
        normalize : callable, optional
            Normalization transform to apply to image only.
        supervised : bool, default=True
            If True, load and return instance masks and derived labels.
        generate_labels_online : bool, default=True
            If True, generate semantic/boundary/SDF labels on-the-fly.
        use_precomputed_labels : bool, default=True
            If True, use precomputed labels from data_dir/boundary and data_dir/sdf
            when available.
        boundary_width : int, default=2
            Width of boundaries when generating labels.
        normalize_sdf : bool, default=True
            Whether to normalize SDF to [-1, 1].
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.normalize = normalize
        self.supervised = supervised
        self.generate_labels_online = generate_labels_online
        self.use_precomputed_labels = use_precomputed_labels
        self.boundary_width = boundary_width
        self.normalize_sdf = normalize_sdf
        self.has_precomputed_labels = False
        
        image_dir = self.data_dir / "images"
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        self.image_paths = sorted([
            str(p) for p in image_dir.iterdir()
            if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        ])
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        if self.supervised:
            mask_dir = self.data_dir / "instance_masks"
            if not mask_dir.exists():
                raise ValueError(f"Instance mask directory not found: {mask_dir}")
            
            self.mask_paths = []
            for img_path in self.image_paths:
                img_name = Path(img_path).stem
                mask_path = mask_dir / f"{img_name}.npy"
                if not mask_path.exists():
                    raise ValueError(f"Mask not found for image {img_name}: {mask_path}")
                self.mask_paths.append(str(mask_path))

            self.boundary_paths = []
            self.sdf_paths = []

            if self.use_precomputed_labels:
                boundary_dir = self.data_dir / "boundary"
                sdf_dir = self.data_dir / "sdf"

                if boundary_dir.exists() and sdf_dir.exists():
                    missing_precomputed = []
                    for img_path in self.image_paths:
                        img_name = Path(img_path).stem
                        boundary_path = boundary_dir / f"{img_name}.npy"
                        sdf_path = sdf_dir / f"{img_name}.npy"

                        if not boundary_path.exists() or not sdf_path.exists():
                            missing_precomputed.append(img_name)
                            self.boundary_paths.append("")
                            self.sdf_paths.append("")
                        else:
                            self.boundary_paths.append(str(boundary_path))
                            self.sdf_paths.append(str(sdf_path))

                    self.has_precomputed_labels = len(missing_precomputed) == 0

                    if missing_precomputed and not self.generate_labels_online:
                        preview = ", ".join(missing_precomputed[:5])
                        if len(missing_precomputed) > 5:
                            preview += ", ..."
                        raise ValueError(
                            "Missing precomputed labels for images: "
                            f"{preview}. Set generate_labels_online=True to fall back."
                        )
                elif not self.generate_labels_online:
                    raise ValueError(
                        f"Precomputed labels requested but directories are missing: "
                        f"{boundary_dir} and/or {sdf_dir}"
                    )
        else:
            self.mask_paths = []
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        img_id = Path(img_path).stem
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.uint8)
        
        if self.supervised:
            instance_mask = np.load(self.mask_paths[idx])
            instance_mask = instance_mask.astype(np.int32)

            if self.has_precomputed_labels:
                boundary = np.load(self.boundary_paths[idx]).astype(np.float32)
                sdf = np.load(self.sdf_paths[idx]).astype(np.float32)
            else:
                boundary = None
                sdf = None
            
            if self.transform is not None:
                if self.has_precomputed_labels:
                    transformed = self.transform(
                        image=image,
                        masks=[instance_mask, boundary, sdf]
                    )
                    image = transformed['image']
                    instance_mask = transformed['masks'][0]
                    boundary = transformed['masks'][1]
                    sdf = transformed['masks'][2]
                else:
                    transformed = self.transform(image=image, mask=instance_mask)
                    image = transformed['image']
                    instance_mask = transformed['mask']

            if self.has_precomputed_labels:
                semantic_mask = (instance_mask > 0).astype(np.int64)
                boundary = (boundary > 0.5).astype(np.int64)
                sdf = sdf.astype(np.float32)
            elif self.generate_labels_online:
                labels = generate_all_labels(
                    instance_mask,
                    boundary_width=self.boundary_width,
                    normalize_sdf=self.normalize_sdf,
                    compute_centers=False
                )
                semantic_mask = labels['semantic_mask']
                boundary = labels['boundary']
                sdf = labels['sdf']
            else:
                semantic_mask = (instance_mask > 0).astype(np.int64)
                boundary = np.zeros_like(instance_mask, dtype=np.int64)
                sdf = np.zeros_like(instance_mask, dtype=np.float32)
            
            if self.normalize is not None:
                image = self.normalize(image)
            
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            instance_mask = torch.from_numpy(instance_mask).long()
            semantic_mask = torch.from_numpy(semantic_mask).long()
            boundary = torch.from_numpy(boundary).long()
            sdf = torch.from_numpy(sdf).float()
            
            return {
                'image': image,
                'instance_mask': instance_mask,
                'semantic_mask': semantic_mask,
                'mask': semantic_mask,
                'boundary': boundary,
                'SDF': sdf,
                'ID': img_id
            }
        else:
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            if self.normalize is not None:
                image = self.normalize(image)
            
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            return {
                'image': image,
                'ID': img_id
            }
    
    def get_num_instances_stats(self) -> Dict[str, float]:
        """
        Compute statistics about number of instances per image.
        
        Returns
        -------
        Dict
            Statistics including min, max, mean, std, total_instances, num_images.
        """
        if not self.supervised:
            raise ValueError("Statistics only available for supervised dataset")
        
        instance_counts = []
        for mask_path in self.mask_paths:
            mask = np.load(mask_path)
            unique_instances = np.unique(mask)
            unique_instances = unique_instances[unique_instances > 0]
            instance_counts.append(len(unique_instances))
        
        instance_counts = np.array(instance_counts)
        
        return {
            'min': int(instance_counts.min()),
            'max': int(instance_counts.max()),
            'mean': float(instance_counts.mean()),
            'std': float(instance_counts.std()),
            'total_instances': int(instance_counts.sum()),
            'num_images': len(instance_counts)
        }


class QuPathUnsupervisedDataset(Dataset):
    """
    Simplified dataset for unsupervised (unlabeled) patches.
    Only loads images, no masks.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        normalize: Optional[Callable] = None,
        num_repeats: int = 1
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        data_dir : str
            Path to directory containing images/ subdirectory.
        transform : callable, optional
            Albumentations transform for augmentation.
        normalize : callable, optional
            Normalization transform.
        num_repeats : int, default=1
            Number of times to repeat the dataset.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.normalize = normalize
        self.num_repeats = num_repeats
        
        image_dir = self.data_dir / "images"
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        image_paths = sorted([
            str(p) for p in image_dir.iterdir()
            if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        ])
        
        self.image_paths = image_paths * num_repeats
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        img_id = Path(img_path).stem
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.uint8)
        
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        if self.normalize is not None:
            image = self.normalize(image)
        
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return {
            'image': image,
            'ID': img_id
        }


def get_train_transforms(patch_size: int = 256) -> A.Compose:
    """
    Create training augmentation pipeline.
    
    Parameters
    ----------
    patch_size : int, default=256
        Size of training patches.
    
    Returns
    -------
    albumentations.Compose
        Augmentation pipeline.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def get_val_transforms(patch_size: int = 256) -> A.Compose:
    """
    Create validation transform pipeline.
    
    Parameters
    ----------
    patch_size : int
        Size of validation patches.
    
    Returns
    -------
    albumentations.Compose
        Minimal transforms for validation.
    """
    return A.Compose([])


def get_normalize_transform(
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225)
) -> Callable:
    """
    Create normalization function.
    
    Parameters
    ----------
    mean : tuple
        RGB mean values (ImageNet defaults).
    std : tuple
        RGB std values (ImageNet defaults).
    
    Returns
    -------
    callable
        Function that normalizes images.
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    def normalize(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        return image.astype(np.float32)
    
    return normalize
