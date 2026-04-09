# Semi-MoE Instance Segmentation for Kidney Tubules

This project extends the Semi-MoE framework to support kidney tubule instance segmentation from QuPath GeoJSON annotations and TIF whole slide images. The current training pipeline keeps the original Semi-MoE ideas (semantic, SDF, boundary, multi-gating, semi-supervision) and adds a 4th instance embedding head for instance-level prediction.

## Overview

The implementation adds a 4th expert (instance embedding head) to the original 3-expert Semi-MoE architecture, enabling pixel-level instance discrimination through discriminative embedding learning.

### Key Features

- **QuPath GeoJSON Integration**: Parse and process QuPath polygon annotations
- **Semi-Supervised Training**: Train on `train_sup` and `train_unsup` with pseudo-labeling from the gating module
- **4-Expert System**: Semantic, SDF, boundary, and instance embedding experts
- **Adaptive Multi-Objective Loss**: Learnable uncertainty weighting across tasks
- **Discriminative Instance Loss**: Train embeddings to cluster by instance
- **Dual Validation**: Semantic Dice/Jaccard plus instance AJI, PQ, F1, Dice
- **TensorBoard Logging**: Modern visualization replacing Visdom
- **WSI Inference**: Run a trained checkpoint on a whole-slide TIFF and export GeoJSON

## Quick Start

```bash
# 1. Extract patches from WSI + GeoJSON
python scripts/extract_patches.py \
    --wsi_dir data/raw/wsi \
    --geojson_dir data/raw/geojson \
    --output_dir data/processed/all_patches

# 2. Split into train_sup/train_unsup/val
python scripts/prepare_dataset.py \
    --source_dir data/processed/all_patches \
    --output_dir data/processed

# 3. Precompute SDF + boundary labels
python scripts/precompute_labels.py data/processed

# 4. Train
python train_instance.py \
    --data_dir data/processed \
    --semi_supervised \
    --use_precomputed_labels \
    --optimizer adamw \
    --amp \
    --selection_metric aji

# 5. Monitor
tensorboard --logdir runs

# 6. Run WSI inference and export GeoJSON
python scripts/infer_wsi_to_geojson.py \
    --wsi_path /path/to/slide.tif \
    --checkpoint trained_models/best_model.pth \
    --output_geojson outputs/slide_segmentations.geojson
```

## Architecture

**4-Expert System:**
1. Semantic Expert → foreground/background
2. SDF Expert → signed distance field
3. Boundary Expert → instance boundaries
4. Instance Expert → pixel embeddings

**Multi-Gate Attention:** Task-specific routing of expert features

**Semi-Supervision:** unlabeled pseudo-labels are derived from gated semantic / SDF / boundary predictions and supervise the expert heads on `train_unsup`

**Adaptive Weighting:** learnable uncertainty weighting is applied across semantic, SDF, boundary, and instance losses

**Instance Objective:** discriminative embedding loss is kept for labeled instance masks

## Next Steps

1. **Prepare your data**: Place .tif files in `data/raw/wsi/` and .geojson in `data/raw/geojson/`
2. **Install dependencies**: See `environment.yaml`
3. **Run preprocessing**: extract patches, split data, precompute labels
4. **Train**: use `train_instance.py` on `data/processed`
5. **Tune hyperparameters**: adjust `lambda_unsup_max`, `unsup_warmup_epochs`, `embedding_dim`, `delta_v`, `delta_d`, `cluster_bandwidth`

## Training Notes

- `train_instance.py` uses precomputed `sdf/` and `boundary/` labels by default when present.
- Semi-supervision is enabled by default and uses `train_unsup/`.
- Current best-model selection defaults to `AJI`, but you can switch to `semantic_dice` or `semantic_jaccard` with `--selection_metric`.
- AMP, warmup + cosine LR scheduling, gradient clipping, EMA, early stopping, and persistent workers are built in.

## Inference

Use [`scripts/infer_wsi_to_geojson.py`](/home/edmund/Desktop/mouse_tubules_MoE_instance_segmentation/scripts/infer_wsi_to_geojson.py) to run a trained model on a full TIFF and export QuPath-friendly GeoJSON.

Example:

```bash
python scripts/infer_wsi_to_geojson.py \
    --wsi_path /path/to/slide.tif \
    --checkpoint trained_models/best_model.pth \
    --output_geojson outputs/slide_segmentations.geojson \
    --patch_size 256 \
    --stride 256 \
    --batch_size 8
```

## Documentation

- Full implementation details: `implementation_plan.md`
- Configuration: `instance_seg/config/kidney_config.py`
- Each module includes detailed docstrings

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size or patch_size |
| Low GPU utilization | Increase `num_workers`, keep `persistent_workers` enabled, and tune batch size |
| Over-segmentation | Increase cluster_bandwidth, min_instance_area |
| Under-segmentation | Decrease cluster_bandwidth, increase weight_embed |
| Poor separation | Increase delta_v, embedding_dim |
| Unstable unlabeled training | Reduce `lambda_unsup_max` or increase `unsup_warmup_epochs` |

## Acknowledgments

- Original Semi-MoE: [vnlvi2k3/Semi-MoE](https://github.com/vnlvi2k3/Semi-MoE)
- Discriminative Loss: De Brabandere et al., 2017
