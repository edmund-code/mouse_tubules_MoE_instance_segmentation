"""
Configuration for Kidney Tubule Instance Segmentation.

This module provides configuration settings for the kidney tubule
instance segmentation task.
"""

import os
from pathlib import Path
from typing import Dict, Any


def get_kidney_config() -> Dict[str, Any]:
    """
    Get configuration for kidney tubule segmentation dataset.
    
    Returns
    -------
    dict
        Configuration dictionary with dataset, model, and training parameters.
    """
    # Get project root (parent of instance_seg directory)
    project_root = Path(__file__).parent.parent.parent
    
    config = {
        # Dataset paths
        "PATH_DATASET": str(project_root / "data" / "processed"),
        "PATH_TRAINED_MODEL": str(project_root / "trained_models"),
        "PATH_SEG_RESULT": str(project_root / "results"),
        "PATH_TENSORBOARD": str(project_root / "runs"),
        
        # Data directories
        "TRAIN_SUP_DIR": "train_sup",
        "TRAIN_UNSUP_DIR": "train_unsup",
        "VAL_DIR": "val",
        
        # Image properties
        "IN_CHANNELS": 3,
        "NUM_CLASSES": 2,
        "PATCH_SIZE": 256,
        
        # Normalization (ImageNet defaults - can be updated with dataset stats)
        "MEAN": [0.485, 0.456, 0.406],
        "STD": [0.229, 0.224, 0.225],
        
        # Model architecture
        "NETWORK": "unet",
        "EMBEDDING_DIM": 16,
        "FEATURE_CHANNELS": 64,
        "NUM_EXPERTS": 4,
        
        # Instance embedding loss
        "DELTA_V": 0.5,
        "DELTA_D": 1.5,
        "ALPHA_VAR": 1.0,
        "BETA_DIST": 1.0,
        "GAMMA_REG": 0.001,
        
        # Multi-task loss weights
        "WEIGHT_SEMANTIC": 1.0,
        "WEIGHT_SDF": 1.0,
        "WEIGHT_BOUNDARY": 1.0,
        "WEIGHT_EMBEDDING": 1.0,
        
        # Clustering parameters for inference
        "CLUSTER_METHOD": "meanshift",
        "CLUSTER_BANDWIDTH": 0.5,
        "MIN_CLUSTER_SIZE": 50,
        "MIN_INSTANCE_AREA": 50,
        
        # Training hyperparameters
        "BATCH_SIZE": 8,
        "NUM_EPOCHS": 200,
        "LEARNING_RATE": 1e-4,
        "WEIGHT_DECAY": 1e-5,
        "OPTIMIZER": "adam",
        
        # Learning rate scheduler
        "LR_SCHEDULER": "cosine",
        "LR_WARMUP_EPOCHS": 10,
        "LR_MIN": 1e-6,
        
        # Data augmentation
        "AUGMENTATION": True,
        "BOUNDARY_WIDTH": 2,
        "NORMALIZE_SDF": True,
        
        # Validation
        "VAL_INTERVAL": 5,
        "SAVE_CHECKPOINT_INTERVAL": 10,
        
        # Class information
        "CLASS_NAMES": ["background", "tubule"],
        "PALETTE": [[0, 0, 0], [255, 0, 0]],
        
        # Hardware
        "NUM_WORKERS": 4,
        "PIN_MEMORY": True,
        
        # Logging
        "LOG_INTERVAL": 50,
        "SAVE_BEST_ONLY": False,
        
        # Reproducibility
        "SEED": 42,
        
        # WSI extraction (for patch extraction scripts)
        "WSI_LEVEL": 0,
        "WSI_OVERLAP": 0,
        "MIN_TISSUE_RATIO": 0.1,
    }
    
    return config


def update_config(base_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Update configuration with custom values.
    
    Parameters
    ----------
    base_config : dict
        Base configuration from get_kidney_config().
    **kwargs
        Key-value pairs to update.
    
    Returns
    -------
    dict
        Updated configuration.
    
    Examples
    --------
    >>> config = get_kidney_config()
    >>> config = update_config(config, BATCH_SIZE=16, LEARNING_RATE=1e-3)
    """
    config = base_config.copy()
    config.update(kwargs)
    return config


def print_config(config: Dict[str, Any]) -> None:
    """
    Print configuration in a readable format.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    """
    print("=" * 50)
    print("Configuration:")
    print("=" * 50)
    
    # Group related settings
    groups = {
        "Paths": ["PATH_DATASET", "PATH_TRAINED_MODEL", "PATH_SEG_RESULT", "PATH_TENSORBOARD"],
        "Data": ["TRAIN_SUP_DIR", "TRAIN_UNSUP_DIR", "VAL_DIR", "IN_CHANNELS", "NUM_CLASSES", "PATCH_SIZE"],
        "Model": ["NETWORK", "EMBEDDING_DIM", "FEATURE_CHANNELS", "NUM_EXPERTS"],
        "Loss": ["DELTA_V", "DELTA_D", "WEIGHT_SEMANTIC", "WEIGHT_SDF", "WEIGHT_BOUNDARY", "WEIGHT_EMBEDDING"],
        "Training": ["BATCH_SIZE", "NUM_EPOCHS", "LEARNING_RATE", "OPTIMIZER", "LR_SCHEDULER"],
        "Inference": ["CLUSTER_METHOD", "CLUSTER_BANDWIDTH", "MIN_INSTANCE_AREA"],
    }
    
    for group_name, keys in groups.items():
        print(f"\n{group_name}:")
        for key in keys:
            if key in config:
                value = config[key]
                if isinstance(value, (int, float, str, bool)):
                    print(f"  {key}: {value}")
                elif isinstance(value, list) and len(value) <= 10:
                    print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for required keys and reasonable values.
    
    Parameters
    ----------
    config : dict
        Configuration to validate.
    
    Returns
    -------
    bool
        True if valid, raises ValueError otherwise.
    """
    required_keys = [
        "PATH_DATASET", "IN_CHANNELS", "NUM_CLASSES", "PATCH_SIZE",
        "EMBEDDING_DIM", "BATCH_SIZE", "NUM_EPOCHS", "LEARNING_RATE"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate values
    if config["IN_CHANNELS"] <= 0:
        raise ValueError("IN_CHANNELS must be positive")
    
    if config["NUM_CLASSES"] <= 0:
        raise ValueError("NUM_CLASSES must be positive")
    
    if config["PATCH_SIZE"] <= 0:
        raise ValueError("PATCH_SIZE must be positive")
    
    if config["EMBEDDING_DIM"] <= 0:
        raise ValueError("EMBEDDING_DIM must be positive")
    
    if config["BATCH_SIZE"] <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    
    if config["LEARNING_RATE"] <= 0:
        raise ValueError("LEARNING_RATE must be positive")
    
    return True


# Convenience function to get config with overrides
def get_config(**kwargs) -> Dict[str, Any]:
    """
    Get kidney config with optional overrides.
    
    Parameters
    ----------
    **kwargs
        Configuration overrides.
    
    Returns
    -------
    dict
        Configuration dictionary.
    
    Examples
    --------
    >>> config = get_config(BATCH_SIZE=16, NUM_EPOCHS=100)
    """
    config = get_kidney_config()
    if kwargs:
        config = update_config(config, **kwargs)
    validate_config(config)
    return config
