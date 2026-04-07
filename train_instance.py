#!/usr/bin/env python3
"""
Training script for Semi-MoE with Instance Embedding Head.

This script extends the original Semi-MoE training to support instance segmentation
by adding a 4th expert for instance embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import os
import time
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add Semi-MoE to path
sys.path.insert(0, str(Path(__file__).parent / 'Semi-MoE'))

# Import from Semi-MoE
from loss.loss_function import segmentation_loss

# Import from instance_seg
from instance_seg.config.kidney_config import get_kidney_config, update_config
from instance_seg.dataload.qupath_dataset import (
    QuPathPatchDataset,
    get_train_transforms,
    get_val_transforms,
    get_normalize_transform
)
from instance_seg.models.instance_expert import create_all_experts
from instance_seg.models.gating_network_4experts import get_gating_network_4experts
from instance_seg.loss.embedding_loss import DiscriminativeLoss
from instance_seg.inference.embedding_to_instances import EmbeddingClusterer
from instance_seg.utils.instance_metrics import evaluate_batch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Semi-MoE with Instance Embedding')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Model
    parser.add_argument('--network', type=str, default='unet', help='Base network architecture')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer')
    
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


def train_epoch(models, gating_net, dataloaders, optimizers, loss_functions, device, epoch, writer, args):
    """Train for one epoch."""
    # Set to train mode
    for model in models.values():
        model.train()
    gating_net.train()
    
    train_loader = dataloaders['train']
    running_losses = {'total': 0.0, 'seg': 0.0, 'sdf': 0.0, 'bnd': 0.0, 'embed': 0.0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
    
    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        seg_masks = batch['mask'].to(device)
        sdf_maps = batch['SDF'].to(device)
        bnd_masks = batch['boundary'].to(device)
        inst_masks = batch['instance_mask'].to(device)
        
        # Forward pass through experts
        seg_feat, _ = models['segment'](images)
        sdf_feat, _ = models['sdf'](images)
        bnd_feat, _ = models['boundary'](images)
        inst_feat, _ = models['instance'](images)
        
        # Concatenate features for gating
        concat_feat = torch.cat([seg_feat, sdf_feat, bnd_feat, inst_feat], dim=1)
        
        # Gating network
        seg_out, sdf_out, bnd_out, embed_out = gating_net(concat_feat)
        
        # Compute losses
        loss_seg = segmentation_loss(seg_out, seg_masks)
        loss_sdf = nn.MSELoss()(sdf_out.squeeze(1), sdf_maps)
        loss_bnd = segmentation_loss(bnd_out, bnd_masks)
        
        embed_loss_fn = loss_functions['embedding']
        embed_loss_dict = embed_loss_fn(embed_out, inst_masks)
        loss_embed = embed_loss_dict['loss']
        
        # Total loss
        total_loss = (
            args.weight_seg * loss_seg +
            args.weight_sdf * loss_sdf +
            args.weight_bnd * loss_bnd +
            args.weight_embed * loss_embed
        )
        
        # Backward
        for optimizer in optimizers:
            optimizer.zero_grad()
        total_loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        
        # Accumulate losses
        running_losses['total'] += total_loss.item()
        running_losses['seg'] += loss_seg.item()
        running_losses['sdf'] += loss_sdf.item()
        running_losses['bnd'] += loss_bnd.item()
        running_losses['embed'] += loss_embed.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'seg': f"{loss_seg.item():.4f}",
            'embed': f"{loss_embed.item():.4f}"
        })
        
        # Log to TensorBoard
        global_step = epoch * len(train_loader) + i
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('Train/total_loss', total_loss.item(), global_step)
            writer.add_scalar('Train/seg_loss', loss_seg.item(), global_step)
            writer.add_scalar('Train/sdf_loss', loss_sdf.item(), global_step)
            writer.add_scalar('Train/bnd_loss', loss_bnd.item(), global_step)
            writer.add_scalar('Train/embed_loss', loss_embed.item(), global_step)
            writer.add_scalar('Train/embed_var_loss', embed_loss_dict['var_loss'].item(), global_step)
            writer.add_scalar('Train/embed_dist_loss', embed_loss_dict['dist_loss'].item(), global_step)
    
    # Return average losses
    num_batches = len(train_loader)
    for key in running_losses:
        running_losses[key] /= num_batches
    
    return running_losses


def validate(models, gating_net, val_loader, clusterer, device, epoch, writer, args):
    """Validate the model."""
    # Set to eval mode
    for model in models.values():
        model.eval()
    gating_net.eval()
    
    pred_masks = []
    gt_masks = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            inst_masks_gt = batch['instance_mask'].cpu().numpy()
            
            # Forward pass
            seg_feat, _ = models['segment'](images)
            sdf_feat, _ = models['sdf'](images)
            bnd_feat, _ = models['boundary'](images)
            inst_feat, _ = models['instance'](images)
            
            concat_feat = torch.cat([seg_feat, sdf_feat, bnd_feat, inst_feat], dim=1)
            seg_out, sdf_out, bnd_out, embed_out = gating_net(concat_feat)
            
            # Get semantic predictions
            seg_pred = torch.argmax(seg_out, dim=1)
            
            # Cluster embeddings to get instance predictions
            inst_masks_pred = clusterer(embed_out, seg_pred)
            inst_masks_pred = inst_masks_pred.cpu().numpy()
            
            pred_masks.extend(inst_masks_pred)
            gt_masks.extend(inst_masks_gt)
    
    # Compute metrics
    metrics = evaluate_batch(pred_masks, gt_masks, iou_threshold=0.5)
    
    # Log metrics
    for metric_name, value in metrics.items():
        writer.add_scalar(f'Val/{metric_name}', value, epoch)
    
    print(f'\nValidation Results:')
    print(f'  AJI: {metrics["aji"]:.4f}')
    print(f'  PQ: {metrics["pq"]:.4f}')
    print(f'  F1: {metrics["f1"]:.4f}')
    print(f'  Dice: {metrics["dice"]:.4f}')
    
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Create datasets
    normalize = get_normalize_transform()
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_dataset = QuPathPatchDataset(
        data_dir=os.path.join(args.data_dir, 'train_sup'),
        transform=train_transform,
        normalize=normalize,
        supervised=True
    )
    
    val_dataset = QuPathPatchDataset(
        data_dir=os.path.join(args.data_dir, 'val'),
        transform=val_transform,
        normalize=normalize,
        supervised=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    
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
    
    # Setup optimizers
    all_params = []
    for model in models.values():
        all_params += list(model.parameters())
    all_params += list(gating_net.parameters())
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(all_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    optimizers = [optimizer]
    
    # Setup loss functions
    embedding_loss = DiscriminativeLoss(
        delta_v=args.delta_v,
        delta_d=args.delta_d,
        alpha=1.0,
        beta=1.0,
        gamma=0.001
    ).to(device)
    
    loss_functions = {'embedding': embedding_loss}
    
    # Setup clusterer for validation
    clusterer = EmbeddingClusterer(
        method=args.cluster_method,
        bandwidth=args.cluster_bandwidth,
        min_instance_area=args.min_instance_area,
        device=device
    )
    
    # Training loop
    best_aji = 0.0
    
    for epoch in range(args.num_epochs):
        # Train
        train_losses = train_epoch(
            models, gating_net, dataloaders, optimizers,
            loss_functions, device, epoch, writer, args
        )
        
        print(f'\nEpoch {epoch+1}/{args.num_epochs} - Train Loss: {train_losses["total"]:.4f}')
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            metrics = validate(models, gating_net, val_loader, clusterer, device, epoch, writer, args)
            
            # Save best model
            if metrics['aji'] > best_aji:
                best_aji = metrics['aji']
                checkpoint = {
                    'epoch': epoch,
                    'models': {key: model.state_dict() for key, model in models.items()},
                    'gating_net': gating_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_aji': best_aji,
                    'args': args
                }
                torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
                print(f'Saved best model with AJI: {best_aji:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'models': {key: model.state_dict() for key, model in models.items()},
                'gating_net': gating_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print('Training complete!')


if __name__ == '__main__':
    main()
