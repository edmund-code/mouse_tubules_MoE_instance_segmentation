"""
Instance Segmentation Evaluation Metrics.

This module implements metrics for evaluating instance segmentation:
- Aggregated Jaccard Index (AJI)
- Panoptic Quality (PQ) and its components (SQ, RQ)
- F1 score, Precision, Recall
- Dice coefficient
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Compute IoU matrix between predicted and ground truth instances.
    
    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted instance mask of shape (H, W).
        Background = 0, instances = 1, 2, 3, ...
    gt_mask : np.ndarray
        Ground truth instance mask of shape (H, W).
    
    Returns
    -------
    tuple
        iou_matrix : np.ndarray of shape (num_pred, num_gt)
        pred_ids : list of predicted instance IDs (excluding 0)
        gt_ids : list of ground truth instance IDs (excluding 0)
    """
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0].tolist()
    
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0].tolist()
    
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return np.zeros((len(pred_ids), len(gt_ids))), pred_ids, gt_ids
    
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
    
    for i, pred_id in enumerate(pred_ids):
        pred_pixels = (pred_mask == pred_id)
        for j, gt_id in enumerate(gt_ids):
            gt_pixels = (gt_mask == gt_id)
            
            intersection = np.sum(pred_pixels & gt_pixels)
            union = np.sum(pred_pixels | gt_pixels)
            
            if union > 0:
                iou_matrix[i, j] = intersection / union
    
    return iou_matrix, pred_ids, gt_ids


def match_instances(
    iou_matrix: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Match predicted instances to ground truth using Hungarian algorithm.
    
    Parameters
    ----------
    iou_matrix : np.ndarray
        IoU matrix of shape (num_pred, num_gt).
    iou_threshold : float, default=0.5
        Minimum IoU to consider a match valid.
    
    Returns
    -------
    tuple
        matches : list of (pred_idx, gt_idx, iou) for valid matches
        unmatched_pred : list of unmatched prediction indices
        unmatched_gt : list of unmatched GT indices
    """
    num_pred, num_gt = iou_matrix.shape
    
    if num_pred == 0 or num_gt == 0:
        return [], list(range(num_pred)), list(range(num_gt))
    
    # Hungarian matching - minimize cost = maximize IoU
    cost_matrix = 1.0 - iou_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Filter matches by threshold
    matches = []
    matched_pred = set()
    matched_gt = set()
    
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[pred_idx, gt_idx]
        if iou >= iou_threshold:
            matches.append((pred_idx, gt_idx, iou))
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
    
    # Find unmatched
    unmatched_pred = [i for i in range(num_pred) if i not in matched_pred]
    unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
    
    return matches, unmatched_pred, unmatched_gt


def compute_aji(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """
    Compute Aggregated Jaccard Index (AJI).
    
    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted instance mask of shape (H, W).
    gt_mask : np.ndarray
        Ground truth instance mask of shape (H, W).
    
    Returns
    -------
    float
        AJI score in range [0, 1]. Higher is better.
    """
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]
    
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]
    
    if len(gt_ids) == 0:
        if len(pred_ids) == 0:
            return 1.0
        else:
            return 0.0
    
    total_intersection = 0
    total_union = 0
    matched_pred = set()
    
    # For each GT, find best matching pred
    for gt_id in gt_ids:
        gt_pixels = (gt_mask == gt_id)
        gt_area = np.sum(gt_pixels)
        
        best_iou = 0
        best_pred_id = None
        
        for pred_id in pred_ids:
            pred_pixels = (pred_mask == pred_id)
            intersection = np.sum(gt_pixels & pred_pixels)
            
            if intersection > 0:
                union = np.sum(gt_pixels | pred_pixels)
                iou = intersection / union
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_id
        
        if best_pred_id is not None:
            pred_pixels = (pred_mask == best_pred_id)
            intersection = np.sum(gt_pixels & pred_pixels)
            union = np.sum(gt_pixels | pred_pixels)
            
            total_intersection += intersection
            total_union += union
            matched_pred.add(best_pred_id)
        else:
            total_union += gt_area
    
    # Add unmatched predictions to union
    for pred_id in pred_ids:
        if pred_id not in matched_pred:
            pred_area = np.sum(pred_mask == pred_id)
            total_union += pred_area
    
    if total_union == 0:
        return 1.0
    
    aji = total_intersection / total_union
    return aji


def compute_panoptic_quality(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ) and its components.
    
    PQ = SQ * RQ
    
    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted instance mask.
    gt_mask : np.ndarray
        Ground truth instance mask.
    iou_threshold : float
        IoU threshold for matching.
    
    Returns
    -------
    dict
        Metrics including pq, sq, rq, tp, fp, fn
    """
    iou_matrix, pred_ids, gt_ids = compute_iou_matrix(pred_mask, gt_mask)
    matches, unmatched_pred, unmatched_gt = match_instances(iou_matrix, iou_threshold)
    
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    
    if tp == 0:
        sq = 0.0
    else:
        sq = np.mean([iou for _, _, iou in matches])
    
    denominator = tp + 0.5 * fp + 0.5 * fn
    if denominator == 0:
        rq = 0.0
    else:
        rq = tp / denominator
    
    pq = sq * rq
    
    return {
        'pq': pq,
        'sq': sq,
        'rq': rq,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_f1_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute detection F1 score at given IoU threshold.
    
    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted instance mask.
    gt_mask : np.ndarray
        Ground truth instance mask.
    iou_threshold : float
        IoU threshold for considering a detection correct.
    
    Returns
    -------
    dict
        Metrics including f1, precision, recall, tp, fp, fn
    """
    iou_matrix, pred_ids, gt_ids = compute_iou_matrix(pred_mask, gt_mask)
    matches, unmatched_pred, unmatched_gt = match_instances(iou_matrix, iou_threshold)
    
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_dice_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """
    Compute semantic Dice score (treating all instances as one class).
    
    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted mask (instance or semantic).
    gt_mask : np.ndarray
        Ground truth mask (instance or semantic).
    
    Returns
    -------
    float
        Dice score in range [0, 1].
    """
    pred_binary = (pred_mask > 0).astype(np.int32)
    gt_binary = (gt_mask > 0).astype(np.int32)
    
    intersection = np.sum(pred_binary & gt_binary)
    pred_area = np.sum(pred_binary)
    gt_area = np.sum(gt_binary)
    
    denominator = pred_area + gt_area
    if denominator == 0:
        return 1.0
    
    dice = 2.0 * intersection / denominator
    return dice


def evaluate_batch(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a batch of predictions.
    
    Parameters
    ----------
    pred_masks : list of np.ndarray
        List of predicted instance masks.
    gt_masks : list of np.ndarray
        List of ground truth instance masks.
    iou_threshold : float
        IoU threshold for PQ and F1.
    
    Returns
    -------
    dict
        Aggregated metrics including aji, pq, sq, rq, f1, dice
    """
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Number of predictions and ground truths must match")
    
    if len(pred_masks) == 0:
        return {
            'aji': 0.0,
            'pq': 0.0,
            'sq': 0.0,
            'rq': 0.0,
            'f1': 0.0,
            'dice': 0.0
        }
    
    aji_scores = []
    pq_scores = []
    sq_scores = []
    rq_scores = []
    f1_scores = []
    dice_scores = []
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        aji_scores.append(compute_aji(pred_mask, gt_mask))
        
        pq_result = compute_panoptic_quality(pred_mask, gt_mask, iou_threshold)
        pq_scores.append(pq_result['pq'])
        sq_scores.append(pq_result['sq'])
        rq_scores.append(pq_result['rq'])
        
        f1_result = compute_f1_score(pred_mask, gt_mask, iou_threshold)
        f1_scores.append(f1_result['f1'])
        
        dice_scores.append(compute_dice_score(pred_mask, gt_mask))
    
    return {
        'aji': np.mean(aji_scores),
        'pq': np.mean(pq_scores),
        'sq': np.mean(sq_scores),
        'rq': np.mean(rq_scores),
        'f1': np.mean(f1_scores),
        'dice': np.mean(dice_scores)
    }
