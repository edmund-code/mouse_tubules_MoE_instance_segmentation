# Semi-MoE Instance Segmentation for Kidney Tubules

This project extends the Semi-MoE framework to support instance segmentation of kidney tubules using QuPath GeoJSON annotations and TIF whole slide images.

## ✅ Implementation Status: COMPLETE

All 12 modules have been successfully implemented:

- ✅ Module 1: GeoJSON Parser
- ✅ Module 2: Label Generation
- ✅ Module 3: Embedding Loss
- ✅ Module 4: QuPath Dataset
- ✅ Module 5: Instance Expert Model
- ✅ Module 6: 4-Expert Gating Network
- ✅ Module 7: Instance Metrics
- ✅ Module 8: Embedding to Instances
- ✅ Module 9: Configuration
- ✅ Module 10: Patch Extraction Script
- ✅ Module 11: Dataset Preparation Script
- ✅ Module 12: Training Script

## Overview

The implementation adds a 4th expert (instance embedding head) to the original 3-expert Semi-MoE architecture, enabling pixel-level instance discrimination through discriminative embedding learning.

### Key Features

- **QuPath GeoJSON Integration**: Parse and process QuPath polygon annotations
- **Instance Embedding Expert**: 4th expert network that produces pixel embeddings
- **Discriminative Loss**: Train embeddings to cluster by instance
- **Multi-task Learning**: Joint training of semantic, boundary, SDF, and instance tasks
- **Instance Metrics**: AJI, Panoptic Quality (PQ), F1, Dice evaluation
- **TensorBoard Logging**: Modern visualization replacing Visdom

## Quick Start

```bash
# 1. Extract patches from WSI + GeoJSON
python scripts/extract_patches.py \
    --wsi_dir data/raw/wsi \
    --geojson_dir data/raw/geojson \
    --output_dir data/processed/all_patches

# 2. Split into train/val
python scripts/prepare_dataset.py \
    --source_dir data/processed/all_patches \
    --output_dir data/processed

# 3. Train
python train_instance.py \
    --data_dir data/processed \
    --batch_size 8 \
    --num_epochs 200

# 4. Monitor
tensorboard --logdir runs
```

## Architecture

**4-Expert System:**
1. Semantic Expert → foreground/background
2. SDF Expert → signed distance field
3. Boundary Expert → instance boundaries
4. **Instance Expert (NEW)** → pixel embeddings

**Multi-Gate Attention:** Task-specific routing of expert features

**Discriminative Loss:** L = α·L_var + β·L_dist + γ·L_reg

## Next Steps

1. **Prepare your data**: Place .tif files in `data/raw/wsi/` and .geojson in `data/raw/geojson/`
2. **Install dependencies**: See `environment.yaml`
3. **Run the pipeline**: Follow Quick Start above
4. **Tune hyperparameters**: Adjust embedding_dim, delta_v, delta_d, cluster_bandwidth

## Documentation

- Full implementation details: `implementation_plan.md`
- Configuration: `instance_seg/config/kidney_config.py`
- Each module includes detailed docstrings

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size or patch_size |
| Over-segmentation | Increase cluster_bandwidth, min_instance_area |
| Under-segmentation | Decrease cluster_bandwidth, increase weight_embed |
| Poor separation | Increase delta_v, embedding_dim |

## Acknowledgments

- Original Semi-MoE: [vnlvi2k3/Semi-MoE](https://github.com/vnlvi2k3/Semi-MoE)
- Discriminative Loss: De Brabandere et al., 2017
