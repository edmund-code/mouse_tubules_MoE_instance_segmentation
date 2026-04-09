#!/usr/bin/env python3
"""
Training script for Semi-MoE with Instance Embedding Head.

This script extends the original Semi-MoE training to support instance segmentation
by adding a 4th expert for instance embeddings.
"""

import os
# Set GPU device before any CUDA imports
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import os
import time
import math
import random
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import from instance_seg
from instance_seg.config.kidney_config import get_kidney_config, update_config
from instance_seg.dataload.qupath_dataset import (
    QuPathPatchDataset,
    QuPathUnsupervisedDataset,
    get_train_transforms,
    get_val_transforms,
    get_normalize_transform
)
from instance_seg.models.instance_expert import create_all_experts
from instance_seg.models.gating_network_4experts import get_gating_network_4experts
from instance_seg.loss.embedding_loss import DiscriminativeLoss
from instance_seg.loss.loss_function import (
    segmentation_loss,
    AdaptiveTaskWeighting,
    compute_supervised_task_losses,
    compute_unsupervised_task_losses,
)
from instance_seg.inference.embedding_to_instances import EmbeddingClusterer
from instance_seg.utils.instance_metrics import evaluate_batch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Semi-MoE with Instance Embedding')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--unsup_batch_size', type=int, default=0, help='Unlabeled batch size, 0 uses batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument(
        '--semi_supervised',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable unlabeled-data training on train_unsup'
    )
    parser.add_argument('--train_unsup_repeats', type=int, default=1, help='Repeat factor for train_unsup dataset')
    parser.add_argument(
        '--persistent_workers',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Keep DataLoader workers alive across epochs'
    )
    parser.add_argument(
        '--use_precomputed_labels',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use precomputed labels from split folders (sdf/ and boundary/)'
    )
    
    # Model
    parser.add_argument('--network', type=str, default='unet', help='Base network architecture')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes')
    parser.add_argument(
        '--seg_loss',
        type=str,
        default='dice',
        choices=['dice', 'crossentropy', 'dicece'],
        help='Segmentation/boundary supervision loss'
    )
    parser.add_argument('--seg_dice_weight', type=float, default=0.5, help='Dice weight for dicece loss')
    parser.add_argument('--seg_ce_weight', type=float, default=0.5, help='Cross-entropy weight for dicece loss')
    parser.add_argument(
        '--train_aug_preset',
        type=str,
        default='baseline',
        choices=['baseline', 'strong'],
        help='Training augmentation preset'
    )
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume or finetune from')
    parser.add_argument(
        '--finetune',
        action='store_true',
        help='Load model weights only and reset optimizer/scaler/epoch state'
    )
    parser.add_argument(
        '--amp',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable mixed precision training (AMP)'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='cosine',
        choices=['none', 'cosine'],
        help='Learning rate scheduler'
    )
    parser.add_argument('--warmup_epochs', type=float, default=5.0, help='Linear warmup duration in epochs')
    parser.add_argument('--min_lr_ratio', type=float, default=0.05, help='Minimum LR ratio for cosine decay')
    parser.add_argument('--lambda_unsup_max', type=float, default=5.0, help='Maximum weight for unlabeled loss')
    parser.add_argument('--unsup_warmup_epochs', type=float, default=5.0, help='Warmup duration for unlabeled loss')
    parser.add_argument('--adaptive_loss_gamma', type=float, default=0.4, help='Regularization strength for adaptive task weighting')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Max grad norm, <=0 disables clipping')
    parser.add_argument(
        '--ema',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Track exponential moving average of model weights'
    )
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay')
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=20,
        help='Stop after this many validation checks without improvement; <=0 disables'
    )
    
    # Loss weights
    parser.add_argument('--weight_seg', type=float, default=1.0, help='Weight for semantic loss')
    parser.add_argument('--weight_sdf', type=float, default=1.0, help='Weight for SDF loss')
    parser.add_argument('--weight_bnd', type=float, default=1.0, help='Weight for boundary loss')
    parser.add_argument('--weight_embed', type=float, default=1.0, help='Weight for embedding loss')
    
    # Embedding loss
    parser.add_argument('--delta_v', type=float, default=0.5, help='Variance margin for embedding loss')
    parser.add_argument('--delta_d', type=float, default=1.5, help='Distance margin for embedding loss')
    
    # Clustering
    parser.add_argument('--cluster_method', type=str, default='meanshift', help='Clustering method')
    parser.add_argument('--cluster_bandwidth', type=float, default=0.5, help='Mean shift bandwidth')
    parser.add_argument('--min_instance_area', type=int, default=50, help='Minimum instance area')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard log directory')
    parser.add_argument('--save_dir', type=str, default='trained_models', help='Model save directory')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval (iterations)')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval (epochs)')
    parser.add_argument('--experiment_name', type=str, default='autoresearch', help='Experiment series label')
    parser.add_argument(
        '--selection_metric',
        type=str,
        default='aji',
        choices=['aji', 'semantic_dice', 'semantic_jaccard'],
        help='Validation metric used for best-model selection and early stopping'
    )
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelEma:
    """Track EMA weights across multiple modules."""
    def __init__(self, modules, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = None

        for module_name, module in modules.items():
            for state_name, tensor in module.state_dict().items():
                key = f'{module_name}.{state_name}'
                self.shadow[key] = tensor.detach().clone()

    def update(self, modules):
        for module_name, module in modules.items():
            for state_name, tensor in module.state_dict().items():
                key = f'{module_name}.{state_name}'
                shadow_tensor = self.shadow[key]
                detached = tensor.detach()
                if torch.is_floating_point(detached):
                    shadow_tensor.mul_(self.decay).add_(detached, alpha=1.0 - self.decay)
                else:
                    shadow_tensor.copy_(detached)

    def apply(self, modules):
        self.backup = {}
        for module_name, module in modules.items():
            current_state = module.state_dict()
            self.backup[module_name] = {
                state_name: tensor.detach().clone()
                for state_name, tensor in current_state.items()
            }
            ema_state = {}
            for state_name in current_state:
                ema_state[state_name] = self.shadow[f'{module_name}.{state_name}'].clone()
            module.load_state_dict(ema_state, strict=True)

    def restore(self, modules):
        if self.backup is None:
            return
        for module_name, module in modules.items():
            module.load_state_dict(self.backup[module_name], strict=True)
        self.backup = None

    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow': {key: tensor.clone() for key, tensor in self.shadow.items()}
        }


def build_lr_scheduler(optimizer, steps_per_epoch: int, args):
    """Create per-step warmup + cosine scheduler."""
    if args.scheduler == 'none':
        return None

    total_steps = max(1, args.num_epochs * steps_per_epoch)
    warmup_steps = int(max(0, args.warmup_epochs) * steps_per_epoch)

    def lr_lambda(step_idx: int):
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0

        progress = float(step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def snapshot_module_states(modules):
    """Clone module state_dicts for checkpointing."""
    return {
        module_name: {
            state_name: tensor.detach().clone()
            for state_name, tensor in module.state_dict().items()
        }
        for module_name, module in modules.items()
    }


def build_train_modules(models, gating_net, task_weighting):
    """Return every trainable module that should share EMA/checkpoint handling."""
    return {**models, 'gating_net': gating_net, 'task_weighting': task_weighting}


def compute_semantic_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Compute semantic Dice and Jaccard on binary foreground masks."""
    pred_fg = pred_mask.astype(bool)
    gt_fg = gt_mask.astype(bool)

    intersection = float(np.logical_and(pred_fg, gt_fg).sum())
    pred_sum = float(pred_fg.sum())
    gt_sum = float(gt_fg.sum())
    union = float(np.logical_or(pred_fg, gt_fg).sum())

    dice = 1.0 if pred_sum + gt_sum == 0 else (2.0 * intersection) / (pred_sum + gt_sum)
    jaccard = 1.0 if union == 0 else intersection / union
    return dice, jaccard, intersection, pred_sum, gt_sum, union


def compute_unsup_weight(global_step: int, steps_per_epoch: int, args) -> float:
    """Linear warmup for the unlabeled loss weight."""
    if not args.semi_supervised or args.lambda_unsup_max <= 0:
        return 0.0

    warmup_steps = int(max(0.0, args.unsup_warmup_epochs) * steps_per_epoch)
    if warmup_steps <= 0:
        return args.lambda_unsup_max

    progress = min(1.0, float(global_step + 1) / float(warmup_steps))
    return args.lambda_unsup_max * progress


def build_experiment_paths(args):
    """Create stable, timestamped artifact paths for the current run."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f'{timestamp}_{args.experiment_name}'
    base_save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir) / experiment_id
    best_model_path = base_save_dir / f'{experiment_id}_best.pth'
    summary_path = base_save_dir / f'{experiment_id}_summary.json'
    research_best_path = base_save_dir / 'current_research_best.pth'
    tracker_path = Path('autoresearch_best_aji.txt')
    return {
        'timestamp': timestamp,
        'experiment_id': experiment_id,
        'log_dir': log_dir,
        'best_model_path': best_model_path,
        'summary_path': summary_path,
        'research_best_path': research_best_path,
        'tracker_path': tracker_path,
    }


def read_checkpoint_best_metric(checkpoint_path: Path) -> float:
    """Read the recorded best metric from a checkpoint if it exists."""
    if not checkpoint_path.exists():
        return float('-inf')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    value = checkpoint.get('best_metric', float('-inf'))
    return float(value) if value is not None else float('-inf')


def initialize_global_best_tracker(tracker_path: Path, protected_baseline_path: Path) -> float:
    """Initialize the research-best tracker, seeding from the protected baseline if needed."""
    if tracker_path.exists():
        try:
            return float(tracker_path.read_text().strip())
        except ValueError:
            pass

    baseline_aji = read_checkpoint_best_metric(protected_baseline_path)
    tracker_path.write_text(f'{baseline_aji:.16f}\n')
    return baseline_aji


def write_experiment_summary(summary_path: Path, payload: dict):
    """Persist a compact machine-readable experiment summary."""
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + '\n')


def forward_all_heads(models, gating_net, images):
    """Forward images through experts and the gating network."""
    seg_feat, seg_pred = models['segment'](images)
    sdf_feat, sdf_pred = models['sdf'](images)
    bnd_feat, bnd_pred = models['boundary'](images)
    inst_feat, inst_pred = models['instance'](images)

    concat_feat = torch.cat([seg_feat, sdf_feat, bnd_feat, inst_feat], dim=1)
    gated_seg, gated_sdf, gated_bnd, gated_instance = gating_net(concat_feat)

    return {
        'expert': {
            'seg': seg_pred,
            'sdf': sdf_pred,
            'bnd': bnd_pred,
            'instance': inst_pred,
        },
        'gated': {
            'seg': gated_seg,
            'sdf': gated_sdf,
            'bnd': gated_bnd,
            'instance': gated_instance,
        }
    }


def train_epoch(
    models,
    gating_net,
    dataloaders,
    optimizers,
    loss_functions,
    device,
    epoch,
    writer,
    args,
    scaler,
    use_amp,
    scheduler,
    trainable_params,
    ema,
    task_weighting
):
    """Train for one epoch."""
    # Set to train mode
    for model in models.values():
        model.train()
    gating_net.train()
    task_weighting.train()
    
    train_loader = dataloaders['train_sup']
    train_unsup_loader = dataloaders.get('train_unsup')
    unsup_iter = iter(train_unsup_loader) if train_unsup_loader is not None else None
    running_losses = {
        'total': 0.0,
        'seg': 0.0,
        'sdf': 0.0,
        'bnd': 0.0,
        'instance': 0.0,
        'unsup_weight': 0.0,
        'semantic_sup': 0.0,
        'semantic_unsup': 0.0,
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
    
    for i, sup_batch in enumerate(pbar):
        images = sup_batch['image'].to(device)
        seg_masks = sup_batch['mask'].to(device)
        sdf_maps = sup_batch['SDF'].to(device)
        bnd_masks = sup_batch['boundary'].to(device)
        inst_masks = sup_batch['instance_mask'].to(device)
        global_step = epoch * len(train_loader) + i
        lambda_unsup = compute_unsup_weight(global_step, len(train_loader), args)
        unsup_images = None
        if unsup_iter is not None:
            try:
                unsup_batch = next(unsup_iter)
            except StopIteration:
                unsup_iter = iter(train_unsup_loader)
                unsup_batch = next(unsup_iter)
            unsup_images = unsup_batch['image'].to(device)
        
        with autocast(enabled=use_amp):
            sup_outputs = forward_all_heads(models, gating_net, images)
            sup_task_losses, sup_details = compute_supervised_task_losses(
                sup_outputs, seg_masks, sdf_maps, bnd_masks, inst_masks, loss_functions
            )

            unsup_task_losses = {
                'seg': torch.zeros((), device=device),
                'sdf': torch.zeros((), device=device),
                'bnd': torch.zeros((), device=device),
                'instance': torch.zeros((), device=device),
            }
            if unsup_images is not None and lambda_unsup > 0:
                unsup_outputs = forward_all_heads(models, gating_net, unsup_images)
                unsup_task_losses = compute_unsupervised_task_losses(unsup_outputs, loss_functions)

            combined_task_losses = {
                'seg': args.weight_seg * (sup_task_losses['seg'] + lambda_unsup * unsup_task_losses['seg']),
                'sdf': args.weight_sdf * (sup_task_losses['sdf'] + lambda_unsup * unsup_task_losses['sdf']),
                'bnd': args.weight_bnd * (sup_task_losses['bnd'] + lambda_unsup * unsup_task_losses['bnd']),
                'instance': args.weight_embed * sup_task_losses['instance'],
            }
            total_loss, weighted_task_losses, adaptive_reg = task_weighting(combined_task_losses)
        
        # Backward
        for optimizer in optimizers:
            optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        if args.grad_clip_norm > 0:
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip_norm)
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(build_train_modules(models, gating_net, task_weighting))
        
        # Accumulate losses
        running_losses['total'] += total_loss.item()
        running_losses['seg'] += combined_task_losses['seg'].item()
        running_losses['sdf'] += combined_task_losses['sdf'].item()
        running_losses['bnd'] += combined_task_losses['bnd'].item()
        running_losses['instance'] += combined_task_losses['instance'].item()
        running_losses['semantic_sup'] += sup_task_losses['seg'].item()
        running_losses['semantic_unsup'] += unsup_task_losses['seg'].item()
        running_losses['unsup_weight'] += lambda_unsup
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'seg': f"{combined_task_losses['seg'].item():.4f}",
            'inst': f"{combined_task_losses['instance'].item():.4f}",
            'u': f"{lambda_unsup:.2f}"
        })
        
        # Log to TensorBoard
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('Train/total_loss', total_loss.item(), global_step)
            writer.add_scalar('Train/seg_loss', combined_task_losses['seg'].item(), global_step)
            writer.add_scalar('Train/sdf_loss', combined_task_losses['sdf'].item(), global_step)
            writer.add_scalar('Train/bnd_loss', combined_task_losses['bnd'].item(), global_step)
            writer.add_scalar('Train/embed_loss', combined_task_losses['instance'].item(), global_step)
            writer.add_scalar('Train/seg_loss_supervised', sup_task_losses['seg'].item(), global_step)
            writer.add_scalar('Train/seg_loss_unsupervised', unsup_task_losses['seg'].item(), global_step)
            writer.add_scalar('Train/lambda_unsup', lambda_unsup, global_step)
            writer.add_scalar('Train/adaptive_reg', adaptive_reg.item(), global_step)
            for task_name, weighted_loss in weighted_task_losses.items():
                writer.add_scalar(f'Train/weighted_{task_name}_loss', weighted_loss.item(), global_step)
            for task_name, sigma in task_weighting.log_sigma.items():
                writer.add_scalar(f'Train/log_sigma_{task_name}', sigma.item(), global_step)
            writer.add_scalar('Train/expert_instance_var_loss', sup_details['expert_instance_var'].item(), global_step)
            writer.add_scalar('Train/expert_instance_dist_loss', sup_details['expert_instance_dist'].item(), global_step)
            writer.add_scalar('Train/gated_instance_var_loss', sup_details['gated_instance_var'].item(), global_step)
            writer.add_scalar('Train/gated_instance_dist_loss', sup_details['gated_instance_dist'].item(), global_step)
            writer.add_scalar('Train/lr', optimizers[0].param_groups[0]['lr'], global_step)
    
    # Return average losses
    num_batches = len(train_loader)
    for key in running_losses:
        running_losses[key] /= num_batches
    
    return running_losses


def validate(models, gating_net, val_loader, clusterer, device, epoch, writer, args, use_amp):
    """Validate the model."""
    # Set to eval mode
    for model in models.values():
        model.eval()
    gating_net.eval()
    
    pred_masks = []
    gt_masks = []
    semantic_intersection = 0.0
    semantic_pred_sum = 0.0
    semantic_gt_sum = 0.0
    semantic_union = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            inst_masks_gt = batch['instance_mask'].cpu().numpy()
            semantic_gt = batch['semantic_mask'].cpu().numpy()
            
            with autocast(enabled=use_amp):
                outputs = forward_all_heads(models, gating_net, images)
            
            # Get semantic predictions
            seg_pred = torch.argmax(outputs['gated']['seg'], dim=1)
            seg_pred_np = seg_pred.cpu().numpy()
            semantic_dice, semantic_jaccard, intersection, pred_sum, gt_sum, union = compute_semantic_metrics(
                seg_pred_np > 0, semantic_gt > 0
            )
            semantic_intersection += intersection
            semantic_pred_sum += pred_sum
            semantic_gt_sum += gt_sum
            semantic_union += union
            
            # Cluster embeddings to get instance predictions
            inst_masks_pred = clusterer(outputs['gated']['instance'], seg_pred)
            inst_masks_pred = inst_masks_pred.cpu().numpy()
            
            pred_masks.extend(inst_masks_pred)
            gt_masks.extend(inst_masks_gt)
    
    # Compute metrics
    metrics = evaluate_batch(pred_masks, gt_masks, iou_threshold=0.5)
    metrics['semantic_dice'] = 1.0 if semantic_pred_sum + semantic_gt_sum == 0 else (
        2.0 * semantic_intersection / (semantic_pred_sum + semantic_gt_sum)
    )
    metrics['semantic_jaccard'] = 1.0 if semantic_union == 0 else semantic_intersection / semantic_union
    
    # Log metrics
    for metric_name, value in metrics.items():
        writer.add_scalar(f'Val/{metric_name}', value, epoch)
    
    print(f'\nValidation Results:')
    print(f'  AJI: {metrics["aji"]:.4f}')
    print(f'  PQ: {metrics["pq"]:.4f}')
    print(f'  F1: {metrics["f1"]:.4f}')
    print(f'  Instance Dice: {metrics["dice"]:.4f}')
    print(f'  Semantic Dice: {metrics["semantic_dice"]:.4f}')
    print(f'  Semantic Jaccard: {metrics["semantic_jaccard"]:.4f}')
    
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    start_epoch = 0
    checkpoint = None
    experiment_paths = build_experiment_paths(args)
    args.log_dir = str(experiment_paths['log_dir'])
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    experiment_paths['best_model_path'].parent.mkdir(parents=True, exist_ok=True)
    global_best_aji = initialize_global_best_tracker(
        experiment_paths['tracker_path'],
        Path(args.save_dir) / 'edmund_best_model.pth'
    )
    print(f'Experiment ID: {experiment_paths["experiment_id"]}')
    print(f'Global tracked AJI to beat: {global_best_aji:.4f}')
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Create datasets
    normalize = get_normalize_transform()
    train_transform = get_train_transforms(preset=args.train_aug_preset)
    val_transform = get_val_transforms()
    
    train_dataset = QuPathPatchDataset(
        data_dir=os.path.join(args.data_dir, 'train_sup'),
        transform=train_transform,
        normalize=normalize,
        supervised=True,
        generate_labels_online=not args.use_precomputed_labels,
        use_precomputed_labels=args.use_precomputed_labels
    )
    train_unsup_dataset = None
    if args.semi_supervised:
        train_unsup_dataset = QuPathUnsupervisedDataset(
            data_dir=os.path.join(args.data_dir, 'train_unsup'),
            transform=train_transform,
            normalize=normalize,
            num_repeats=max(1, args.train_unsup_repeats)
        )
    
    val_dataset = QuPathPatchDataset(
        data_dir=os.path.join(args.data_dir, 'val'),
        transform=val_transform,
        normalize=normalize,
        supervised=True,
        generate_labels_online=not args.use_precomputed_labels,
        use_precomputed_labels=args.use_precomputed_labels
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.persistent_workers and args.num_workers > 0
    )
    train_unsup_loader = None
    if train_unsup_dataset is not None:
        unsup_batch_size = args.unsup_batch_size if args.unsup_batch_size > 0 else args.batch_size
        train_unsup_loader = DataLoader(
            train_unsup_dataset,
            batch_size=unsup_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.persistent_workers and args.num_workers > 0
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.persistent_workers and args.num_workers > 0
    )
    
    dataloaders = {'train_sup': train_loader, 'val': val_loader}
    if train_unsup_loader is not None:
        dataloaders['train_unsup'] = train_unsup_loader
    
    if train_unsup_dataset is not None:
        print(f'Train sup: {len(train_dataset)} samples, Train unsup: {len(train_unsup_dataset)} samples, Val: {len(val_dataset)} samples')
    else:
        print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')
    
    # Create models
    models = create_all_experts(
        network_name=args.network,
        in_channels=3,
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim
    )
    
    gating_net = get_gating_network_4experts(
        feature_channels=64,
        num_experts=4,
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim
    )
    
    # Move to device
    for key in models:
        models[key] = models[key].to(device)
    gating_net = gating_net.to(device)
    task_weighting = AdaptiveTaskWeighting(gamma=args.adaptive_loss_gamma).to(device)

    if args.resume:
        print(f'Loading checkpoint from: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)

        if 'models' in checkpoint:
            for model_name, model in models.items():
                if model_name in checkpoint['models']:
                    model.load_state_dict(checkpoint['models'][model_name], strict=True)
                    print(f'Loaded expert weights: {model_name}')
                else:
                    print(f'Checkpoint missing expert weights for: {model_name}')

        if 'gating_net' in checkpoint:
            gating_net.load_state_dict(checkpoint['gating_net'], strict=True)
            print('Loaded gating network weights')
        else:
            print('Checkpoint missing gating network weights')

        if 'task_weighting' in checkpoint:
            task_weighting.load_state_dict(checkpoint['task_weighting'], strict=True)
            print('Loaded adaptive task weighting state')
        else:
            print('Checkpoint missing adaptive task weighting state')

        if args.finetune:
            start_epoch = 0
            print('Finetune mode enabled: optimizer, scaler, and epoch state will be reset')
        else:
            start_epoch = checkpoint.get('epoch', -1) + 1
            print(f'Resuming training from epoch {start_epoch}')
    
    # Setup optimizers
    all_params = []
    for model in models.values():
        all_params += list(model.parameters())
    all_params += list(gating_net.parameters())
    all_params += list(task_weighting.parameters())
    
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(all_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    optimizers = [optimizer]
    scheduler = build_lr_scheduler(optimizer, len(train_loader), args)
    
    # Setup loss functions
    seg_loss_fn = segmentation_loss(
        loss=args.seg_loss,
        aux=False,
        dice_weight=args.seg_dice_weight,
        ce_weight=args.seg_ce_weight,
    )
    embedding_loss = DiscriminativeLoss(
        delta_v=args.delta_v,
        delta_d=args.delta_d,
        alpha=1.0,
        beta=1.0,
        gamma=0.001
    ).to(device)
    
    loss_functions = {
        'segmentation': seg_loss_fn,
        'embedding': embedding_loss
    }
    
    # Setup clusterer for validation
    clusterer = EmbeddingClusterer(
        method=args.cluster_method,
        bandwidth=args.cluster_bandwidth,
        min_instance_area=args.min_instance_area,
        device=device
    )
    
    scaler = GradScaler(enabled=args.amp)
    ema = ModelEma(build_train_modules(models, gating_net, task_weighting), decay=args.ema_decay) if args.ema else None
    
    # Training loop
    best_metric = float('-inf')
    epochs_without_improvement = 0
    best_metrics_payload = None

    if checkpoint is not None and not args.finetune:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Loaded optimizer state')
        else:
            print('Checkpoint missing optimizer state')

        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print('Loaded scheduler state')
        elif scheduler is not None:
            print('Checkpoint missing scheduler state')

        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            print('Loaded AMP scaler state')
        else:
            print('Checkpoint missing AMP scaler state')

        if ema is not None and checkpoint.get('ema') is not None:
            ema_state = checkpoint['ema']
            ema.decay = ema_state.get('decay', ema.decay)
            ema.shadow = {
                key: tensor.detach().clone()
                for key, tensor in ema_state.get('shadow', {}).items()
            }
            print('Loaded EMA state')
        elif ema is not None:
            print('Checkpoint missing EMA state')

        best_metric = checkpoint.get('best_metric', best_metric)
        print(f'Loaded best metric: {best_metric:.4f}')
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_losses = train_epoch(
            models, gating_net, dataloaders, optimizers,
            loss_functions, device, epoch, writer, args, scaler, args.amp,
            scheduler, all_params, ema, task_weighting
        )
        
        print(f'\nEpoch {epoch+1}/{args.num_epochs} - Train Loss: {train_losses["total"]:.4f}')
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            if ema is not None:
                ema.apply(build_train_modules(models, gating_net, task_weighting))
            metrics = validate(models, gating_net, val_loader, clusterer, device, epoch, writer, args, args.amp)
            if ema is not None:
                ema.restore(build_train_modules(models, gating_net, task_weighting))
            
            # Save best model
            selection_value = metrics[args.selection_metric]
            if selection_value > best_metric:
                best_metric = selection_value
                epochs_without_improvement = 0
                if ema is not None:
                    ema.apply(build_train_modules(models, gating_net, task_weighting))
                checkpoint_models = snapshot_module_states(models)
                checkpoint_gating_net = snapshot_module_states({'gating_net': gating_net})['gating_net']
                checkpoint_task_weighting = snapshot_module_states({'task_weighting': task_weighting})['task_weighting']
                if ema is not None:
                    ema.restore(build_train_modules(models, gating_net, task_weighting))
                checkpoint = {
                    'epoch': epoch,
                    'models': checkpoint_models,
                    'gating_net': checkpoint_gating_net,
                    'task_weighting': checkpoint_task_weighting,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'scaler': scaler.state_dict(),
                    'ema': ema.state_dict() if ema is not None else None,
                    'best_metric': best_metric,
                    'selection_metric': args.selection_metric,
                    'args': args
                }
                torch.save(checkpoint, experiment_paths['best_model_path'])
                print(f'Saved experiment best model to {experiment_paths["best_model_path"].name} with {args.selection_metric}: {best_metric:.4f}')
                best_metrics_payload = {key: float(value) for key, value in metrics.items()}

                if args.selection_metric == 'aji' and best_metric > global_best_aji:
                    shutil.copy2(experiment_paths['best_model_path'], experiment_paths['research_best_path'])
                    experiment_paths['tracker_path'].write_text(f'{best_metric:.16f}\n')
                    global_best_aji = best_metric
                    print(f'Promoted experiment to current_research_best.pth with AJI: {best_metric:.4f}')
            else:
                epochs_without_improvement += 1

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f'Early stopping triggered after {epochs_without_improvement} validation checks without {args.selection_metric} improvement.')
                break
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            if ema is not None:
                ema.apply(build_train_modules(models, gating_net, task_weighting))
            checkpoint_models = snapshot_module_states(models)
            checkpoint_gating_net = snapshot_module_states({'gating_net': gating_net})['gating_net']
            checkpoint_task_weighting = snapshot_module_states({'task_weighting': task_weighting})['task_weighting']
            if ema is not None:
                ema.restore(build_train_modules(models, gating_net, task_weighting))
            checkpoint = {
                'epoch': epoch,
                'models': checkpoint_models,
                'gating_net': checkpoint_gating_net,
                'task_weighting': checkpoint_task_weighting,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'scaler': scaler.state_dict(),
                'ema': ema.state_dict() if ema is not None else None,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'{experiment_paths["experiment_id"]}_checkpoint_epoch_{epoch+1}.pth'))
    
    write_experiment_summary(experiment_paths['summary_path'], {
        'experiment_id': experiment_paths['experiment_id'],
        'timestamp': experiment_paths['timestamp'],
        'selection_metric': args.selection_metric,
        'best_metric': best_metric,
        'best_metrics': best_metrics_payload,
        'global_best_aji': global_best_aji,
        'best_model_path': str(experiment_paths['best_model_path']),
        'promoted_model_path': str(experiment_paths['research_best_path']) if experiment_paths['research_best_path'].exists() else None,
        'args': vars(args),
    })
    writer.close()
    print('Training complete!')


if __name__ == '__main__':
    main()
